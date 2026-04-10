"""Sandbox manager: per-session Roche sandbox lifecycle."""

from __future__ import annotations

import logging
from typing import Any

from roche_sandbox.client import AsyncRoche
from roche_sandbox.sandbox import AsyncSandbox

from castor_server.config import settings
from castor_server.models.environments import EnvironmentResponse

logger = logging.getLogger("castor_server.sandbox_manager")


class SandboxManager:
    """Manages per-session Roche sandboxes."""

    def __init__(self) -> None:
        self._sandboxes: dict[str, AsyncSandbox] = {}
        self._client: AsyncRoche | None = None

    def _get_client(self) -> AsyncRoche:
        if self._client is None:
            kwargs: dict[str, Any] = {"provider": settings.roche_provider}
            if settings.roche_daemon_port is not None:
                kwargs["daemon_port"] = settings.roche_daemon_port
            self._client = AsyncRoche(**kwargs)
        return self._client

    async def get_or_create(
        self, session_id: str, env: EnvironmentResponse
    ) -> AsyncSandbox:
        """Get existing sandbox or create one for this session."""
        if session_id in self._sandboxes:
            return self._sandboxes[session_id]

        client = self._get_client()
        sandbox = await client.create(
            image=env.image,
            memory=env.memory,
            cpus=env.cpus,
            timeout_secs=env.timeout_secs,
            network=env.network,
            writable=env.writable,
            network_allowlist=env.network_allowlist or None,
        )
        self._sandboxes[session_id] = sandbox
        logger.info(
            "sandbox_created session=%s sandbox=%s image=%s",
            session_id,
            sandbox.id,
            env.image,
        )
        return sandbox

    def get_sandbox(self, session_id: str) -> AsyncSandbox | None:
        return self._sandboxes.get(session_id)

    async def destroy_sandbox(self, session_id: str) -> None:
        sandbox = self._sandboxes.pop(session_id, None)
        if sandbox:
            try:
                await sandbox.destroy()
                logger.info(
                    "sandbox_destroyed session=%s sandbox=%s",
                    session_id,
                    sandbox.id,
                )
            except Exception:
                logger.exception(
                    "sandbox_destroy_error session=%s sandbox=%s",
                    session_id,
                    sandbox.id,
                )

    async def destroy_all(self) -> None:
        """Destroy all active sandboxes (called on shutdown)."""
        for session_id in list(self._sandboxes.keys()):
            await self.destroy_sandbox(session_id)


# Singleton
sandbox_manager = SandboxManager()
