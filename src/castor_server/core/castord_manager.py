"""Physical per-session ``castord`` supervision for the management plane."""

from __future__ import annotations

import asyncio
import fcntl
import os
import re
import signal
import stat
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from castor_server.core.aisa_client import AisaChannel, AisaClient, AisaOpcode

_VALID_ID = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_STARTUP_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True)
class ManagedCastordSession:
    tenant_id: str
    session_id: str
    storage_root: Path
    agent_socket: Path
    control_socket: Path
    pid: int


@dataclass
class _ManagedProcess:
    session: ManagedCastordSession
    process: asyncio.subprocess.Process
    # D-04 recovery opens the live incarnation at generation 2, so the first
    # supervisor fence must advance it rather than repeat the recovery value.
    next_generation: int = 3


class CastordProcessManager:
    """Own daemon lifecycle, never proxying Agent AISA traffic."""

    _managed: dict[tuple[str, str], CastordProcessManager] = {}

    def __init__(
        self, *, storage_base: Path, castord_binary: Path | None = None
    ) -> None:
        self.storage_base = storage_base
        self.castord_binary = castord_binary or self._default_castord_binary()
        self._processes: dict[tuple[str, str], _ManagedProcess] = {}
        self._management_aisa_call_count = 0

    @staticmethod
    def _default_castord_binary() -> Path:
        configured = os.environ.get("CASTORD_BINARY")
        if configured:
            return Path(configured)
        server_root = Path(__file__).resolve().parents[3]
        return server_root.parent / "castor" / "kernel" / "target" / "debug" / "castord"

    @property
    def management_aisa_call_count(self) -> int:
        return self._management_aisa_call_count

    @staticmethod
    def _socket_is_live(path: Path) -> bool:
        return path.exists() and stat.S_ISSOCK(path.stat().st_mode)

    @staticmethod
    def _probe_writer_lock(root: Path) -> None:
        """Verify no other writer holds the lock, without retaining the flock."""
        lock_path = root / ".c01-writer.lock"
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise FileExistsError(f"storage writer lock is held: {lock_path}") from exc
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    async def _ensure_binary(self) -> None:
        if self.castord_binary.is_file():
            return
        kernel_dir = self.castord_binary.parents[2]
        completed = await asyncio.to_thread(
            subprocess.run,
            ["cargo", "build", "--bin", "castord"],
            cwd=kernel_dir,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0 or not self.castord_binary.is_file():
            raise RuntimeError(f"unable to build castord: {completed.stderr}")

    async def provision(self, tenant_id: str, session_id: str) -> ManagedCastordSession:
        if not _VALID_ID.fullmatch(tenant_id) or not _VALID_ID.fullmatch(session_id):
            raise ValueError("tenant_id and session_id must be canonical identifiers")
        key = (tenant_id, session_id)
        root = self.storage_base / "tenants" / tenant_id / "sessions" / session_id
        agent_socket, control_socket = root / "ipc.sock", root / "control.sock"
        if key in self._processes or agent_socket.exists() or control_socket.exists():
            raise FileExistsError(
                f"active castord session already exists: {tenant_id}/{session_id}"
            )
        root.mkdir(parents=True, exist_ok=True)
        self._probe_writer_lock(root)
        await self._ensure_binary()
        process = await asyncio.create_subprocess_exec(
            str(self.castord_binary),
            "--storage-root",
            str(root),
            "--socket",
            str(agent_socket),
            "--control-socket",
            str(control_socket),
            "--sandbox",
            "none",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        await self._wait_ready(process, agent_socket, control_socket)
        session = ManagedCastordSession(
            tenant_id, session_id, root, agent_socket, control_socket, process.pid
        )
        self._processes[key] = _ManagedProcess(session, process)
        self._managed[key] = self
        return session

    async def _wait_ready(
        self,
        process: asyncio.subprocess.Process,
        agent_socket: Path,
        control_socket: Path,
    ) -> None:
        deadline = time.monotonic() + _STARTUP_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if process.returncode is not None:
                stderr = (
                    (await process.stderr.read()).decode() if process.stderr else ""
                )
                raise RuntimeError(f"castord exited during startup: {stderr}")
            if self._socket_is_live(agent_socket) and self._socket_is_live(
                control_socket
            ):
                return
            await asyncio.sleep(0.01)
        process.kill()
        await process.wait()
        raise TimeoutError("castord did not create both session sockets")

    def session(self, tenant_id: str, session_id: str) -> ManagedCastordSession:
        return self._processes[(tenant_id, session_id)].session

    def is_managed(self, tenant_id: str, session_id: str) -> bool:
        return (tenant_id, session_id) in self._processes

    async def restart(self, tenant_id: str, session_id: str) -> ManagedCastordSession:
        key = (tenant_id, session_id)
        managed = self._processes.get(key)
        if managed is None:
            raise FileNotFoundError(f"unknown session: {tenant_id}/{session_id}")
        await self._reap(key)
        return await self.provision(tenant_id, session_id)

    async def fence_and_reap(self, tenant_id: str, session_id: str) -> int:
        key = (tenant_id, session_id)
        managed = self._processes.get(key)
        if managed is None:
            raise FileNotFoundError(f"unknown session: {tenant_id}/{session_id}")
        response = await self.control_request(
            tenant_id,
            session_id,
            AisaOpcode.PERSIST_FENCE,
            {"generation": managed.next_generation},
        )
        if (
            response.error_code
            or response.persistence_disposition != "GenerationFenced"
        ):
            raise RuntimeError(f"fence was not persisted: {response}")
        managed.next_generation += 1
        return await self._reap(key)

    async def control_request(
        self,
        tenant_id: str,
        session_id: str,
        opcode: AisaOpcode,
        payload: dict[str, object],
        *,
        request_id: str | None = None,
    ):
        """Issue a privileged operation over the daemon's control socket."""
        managed = self._processes[(tenant_id, session_id)]
        self._management_aisa_call_count += 1
        return await AisaClient(
            managed.session.control_socket, channel=AisaChannel.CONTROL
        ).request(opcode, payload, request_id=request_id)

    async def simulate_management_server_outage(self) -> None:
        """Drop only the server's registry; supervised daemons keep running."""
        for key, manager in tuple(self._managed.items()):
            if manager is self:
                self._managed.pop(key, None)

    async def cleanup(self, tenant_id: str, session_id: str) -> None:
        if (tenant_id, session_id) in self._processes:
            await self.fence_and_reap(tenant_id, session_id)

    async def _reap(self, key: tuple[str, str]) -> int:
        managed = self._processes.pop(key)
        process = managed.process
        if process.returncode is None:
            process.send_signal(signal.SIGTERM)
            try:
                await asyncio.wait_for(process.wait(), timeout=1.0)
            except TimeoutError:
                process.kill()
                await process.wait()
        for socket in (managed.session.agent_socket, managed.session.control_socket):
            if socket.exists() and stat.S_ISSOCK(socket.stat().st_mode):
                socket.unlink()
        self._managed.pop(key, None)
        return process.returncode or 0
