"""Fail-closed ``castord`` supervision contract for EPIC-30.

T-306-B needs stable names for tests without starting a daemon, acquiring a
writer lock, or touching sockets.  T-306-C supplies those privileged actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ManagedCastordSession:
    """Per-session paths that a future supervisor will own."""

    tenant_id: str
    session_id: str
    storage_root: Path
    agent_socket: Path
    control_socket: Path


class CastordProcessManager:
    """Contract boundary for isolated ``castord`` lifecycle management."""

    def __init__(self, *, storage_base: Path) -> None:
        self.storage_base = storage_base

    @property
    def management_aisa_call_count(self) -> int:
        """No management-path AISA proxy exists before T-306-C."""
        return 0

    async def provision(self, tenant_id: str, session_id: str) -> ManagedCastordSession:
        raise NotImplementedError(
            "castord provisioning is not implemented; refuse to create session"
        )

    async def restart(self, tenant_id: str, session_id: str) -> None:
        raise NotImplementedError(
            "castord recovery is not implemented; refuse to restart session"
        )

    async def simulate_management_server_outage(self) -> None:
        raise NotImplementedError(
            "outage simulation is not implemented; refuse lifecycle mutation"
        )

    def is_managed(self, tenant_id: str, session_id: str) -> bool:
        return False
