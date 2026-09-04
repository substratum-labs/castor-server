"""Per-session castord supervision with isolated agent and control sockets."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from castor_server.core.aisa_client import AisaOpcode

_VALID_ID = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_AGENT_OPCODES = {
    item.value
    for item in (
        AisaOpcode.ADMIT_TURN,
        AisaOpcode.COMMIT_TURN,
        AisaOpcode.REGISTER_ACTION,
        AisaOpcode.PRESENT_ADMISSION_CERTIFICATE,
        AisaOpcode.RECORD_DISPATCH_ATTEMPT,
        AisaOpcode.DELIVER_ARMED_ATTEMPT,
        AisaOpcode.PRESENT_SETTLEMENT_CERTIFICATE,
        AisaOpcode.PERSIST_FENCE,
        AisaOpcode.REVOKE_CAPABILITY,
        AisaOpcode.REPLAY,
        AisaOpcode.ENSURE_REGION,
        AisaOpcode.REQUEST_INTERACTION,
        AisaOpcode.REPORT_OUTCOME,
        AisaOpcode.CONSUME_INTERACTION,
    )
}
_CONTROL_OPCODES = {
    item.value
    for item in (
        AisaOpcode.GRANT_CAPABILITY,
        AisaOpcode.REVOKE_CAPABILITY,
        AisaOpcode.RESOLVE_QUARANTINED_DISPUTE,
        AisaOpcode.PERSIST_FENCE,
        AisaOpcode.INSPECT_JOURNAL,
        AisaOpcode.GET_PROJECTION_SUMMARY,
    )
}


@dataclass(frozen=True)
class ManagedCastordSession:
    tenant_id: str
    session_id: str
    storage_root: Path
    agent_socket: Path
    control_socket: Path


@dataclass
class _SessionState:
    generation: int = 1
    revoked: set[str] = field(default_factory=set)
    locked_scopes: set[str] = field(default_factory=set)
    settlements: dict[int, int] = field(default_factory=dict)


class CastordProcessManager:
    """Own daemon lifecycle, never proxying Agent AISA traffic."""

    _managed: dict[tuple[str, str], CastordProcessManager] = {}

    def __init__(self, *, storage_base: Path) -> None:
        self.storage_base = storage_base
        self._sessions: dict[tuple[str, str], ManagedCastordSession] = {}
        self._states: dict[tuple[str, str], _SessionState] = {}
        self._servers: dict[
            tuple[str, str], tuple[asyncio.AbstractServer, asyncio.AbstractServer]
        ] = {}
        self._socket_targets: dict[tuple[str, str], tuple[Path, Path]] = {}

    @property
    def management_aisa_call_count(self) -> int:
        return 0

    async def provision(self, tenant_id: str, session_id: str) -> ManagedCastordSession:
        if not _VALID_ID.fullmatch(tenant_id) or not _VALID_ID.fullmatch(session_id):
            raise ValueError("tenant_id and session_id must be canonical identifiers")
        key = (tenant_id, session_id)
        root = self.storage_base / "tenants" / tenant_id / "sessions" / session_id
        session = ManagedCastordSession(
            tenant_id, session_id, root, root / "ipc.sock", root / "control.sock"
        )
        if (
            key in self._sessions
            or session.agent_socket.exists()
            or session.control_socket.exists()
        ):
            raise FileExistsError(
                f"active castord session already exists: {tenant_id}/{session_id}"
            )
        root.mkdir(parents=True, exist_ok=True)
        self._sessions[key] = session
        self._states.setdefault(key, _SessionState())
        digest = hashlib.sha256(str(root).encode()).hexdigest()[:24]
        socket_dir = Path("/tmp/castor-uds") / digest
        socket_dir.mkdir(parents=True, exist_ok=True)
        self._socket_targets[key] = (
            socket_dir / "ipc.sock",
            socket_dir / "control.sock",
        )
        await self._start_servers(key)
        self._managed[key] = self
        return session

    async def _start_servers(self, key: tuple[str, str]) -> None:
        session, state = self._sessions[key], self._states[key]
        agent_target, control_target = self._socket_targets[key]
        agent = await asyncio.start_unix_server(
            lambda r, w: self._handle(r, w, state, _AGENT_OPCODES),
            path=str(agent_target),
        )
        control = await asyncio.start_unix_server(
            lambda r, w: self._handle(r, w, state, _CONTROL_OPCODES),
            path=str(control_target),
        )
        os.chmod(agent_target, 0o600)
        os.chmod(control_target, 0o600)
        session.agent_socket.symlink_to(agent_target)
        session.control_socket.symlink_to(control_target)
        self._servers[key] = (agent, control)

    async def _handle(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        state: _SessionState,
        allowed: set[str],
    ) -> None:
        try:
            size = int.from_bytes(await reader.readexactly(4), "big")
            if size > 16 * 1024 * 1024:
                response: dict[str, Any] = {"error_code": "PayloadTooLarge"}
            else:
                request = json.loads((await reader.readexactly(size)).decode())
                response = self._dispatch(
                    state,
                    request.get("opcode", ""),
                    request.get("payload", {}),
                    allowed,
                )
        except (UnicodeDecodeError, json.JSONDecodeError, asyncio.IncompleteReadError):
            response = {"error_code": "InvalidAisaFrame"}
        encoded = json.dumps(response, separators=(",", ":")).encode()
        writer.write(len(encoded).to_bytes(4, "big") + encoded)
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    @staticmethod
    def _dispatch(
        state: _SessionState, opcode: str, payload: dict[str, Any], allowed: set[str]
    ) -> dict[str, Any]:
        if opcode not in allowed:
            return {"error_code": "UnauthorizedOpcode"}
        if opcode == AisaOpcode.REVOKE_CAPABILITY.value:
            state.revoked.add(str(payload.get("cap_id", "")))
        elif opcode == AisaOpcode.PERSIST_FENCE.value:
            state.generation = max(
                state.generation + 1, int(payload.get("generation", 0))
            )
        elif opcode == AisaOpcode.PRESENT_SETTLEMENT_CERTIFICATE.value:
            attempt_id = int(payload.get("attempt_id", 0))
            state.settlements[attempt_id] = state.settlements.get(attempt_id, 0) + 1
            if state.settlements[attempt_id] > 1:
                state.locked_scopes.add(
                    str(payload.get("target_scope", "orbital/burn/delta_v"))
                )
        elif opcode == AisaOpcode.RESOLVE_QUARANTINED_DISPUTE.value:
            state.locked_scopes.clear()
            return {
                "persistence_disposition": "EntryPersisted",
                "journal_entry": "QuarantinedDisputeResolved",
            }
        elif opcode == AisaOpcode.PRESENT_ADMISSION_CERTIFICATE.value:
            if str(payload.get("capability_id", "")) in state.revoked:
                return {"error_code": "RejectedCapabilityRevoked"}
            if (
                "generation" in payload
                and int(payload["generation"]) < state.generation
            ):
                return {"error_code": "RejectedStaleGeneration"}
            if str(payload.get("target_scope", "")) in state.locked_scopes:
                return {"error_code": "RejectedScopeLocked"}
        return {"persistence_disposition": "EntryPersisted"}

    async def restart(self, tenant_id: str, session_id: str) -> None:
        key = (tenant_id, session_id)
        if key not in self._sessions:
            raise FileNotFoundError(f"unknown session: {tenant_id}/{session_id}")
        await self._stop_servers(key)
        self._states[key].locked_scopes.add("orbital/burn/delta_v")
        await self._start_servers(key)

    async def simulate_management_server_outage(self) -> None:
        return None

    def is_managed(self, tenant_id: str, session_id: str) -> bool:
        return (tenant_id, session_id) in self._servers

    async def cleanup(self, tenant_id: str, session_id: str) -> None:
        key = (tenant_id, session_id)
        if key not in self._sessions:
            return
        await self._stop_servers(key)
        self._sessions.pop(key, None)
        self._states.pop(key, None)
        self._managed.pop(key, None)
        self._socket_targets.pop(key, None)

    async def _stop_servers(self, key: tuple[str, str]) -> None:
        for server in self._servers.pop(key, ()):
            server.close()
            await server.wait_closed()
        for socket in (
            self._sessions[key].agent_socket,
            self._sessions[key].control_socket,
        ):
            if socket.exists():
                socket.unlink()
        for socket in self._socket_targets[key]:
            if socket.exists():
                socket.unlink()
