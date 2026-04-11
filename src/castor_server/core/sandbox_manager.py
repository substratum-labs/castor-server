"""Sandbox manager: per-session Roche sandbox lifecycle.

Roche is an optional dependency — install with ``pip install castor-server[sandbox]``
to enable isolated tool execution. Without it, sessions created with an
environment_id will raise a clear error at sandbox-creation time, but the
rest of the server (CRUD, events, agents without environments) works fine.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from castor_server.config import settings
from castor_server.models.environments import EnvironmentResponse

if TYPE_CHECKING:
    from roche_sandbox.sandbox import AsyncSandbox

logger = logging.getLogger("castor_server.sandbox_manager")


def _import_roche():
    """Lazy import of roche_sandbox so the package stays optional."""
    try:
        from roche_sandbox.client import AsyncRoche
        from roche_sandbox.sandbox import AsyncSandbox
    except ImportError as e:
        raise RuntimeError(
            "Roche sandbox is not installed. Install the sandbox extra to "
            "enable isolated tool execution:\n"
            "    pip install 'castor-server[sandbox]'\n"
            "Or omit environment_id when creating sessions to run tools on "
            "the host (development mode)."
        ) from e
    return AsyncRoche, AsyncSandbox


class SandboxManager:
    """Manages per-session Roche sandboxes."""

    def __init__(self) -> None:
        self._sandboxes: dict[str, Any] = {}
        self._client: Any | None = None

    def _get_client(self):
        if self._client is None:
            async_roche_cls, _ = _import_roche()
            kwargs: dict[str, Any] = {"provider": settings.roche_provider}
            if settings.roche_daemon_port is not None:
                kwargs["daemon_port"] = settings.roche_daemon_port
            self._client = async_roche_cls(**kwargs)
        return self._client

    async def get_or_create(
        self,
        session_id: str,
        env: EnvironmentResponse,
        resources: list[dict[str, Any]] | None = None,
    ) -> AsyncSandbox:
        """Get existing sandbox or create one for this session.

        On first creation, mounts any session ``resources`` into the
        sandbox (e.g. clones github_repository entries via ``git clone``
        inside the container). Resource mounting is one-shot — subsequent
        calls return the cached sandbox without re-mounting.
        """
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

        if resources:
            await self._mount_resources(sandbox, session_id, resources)

        return sandbox

    async def _mount_resources(
        self,
        sandbox: AsyncSandbox,
        session_id: str,
        resources: list[dict[str, Any]],
    ) -> None:
        """Materialize session resources inside the sandbox.

        Supports:
          - ``github_repository``: ``git clone`` the repo into ``mount_path``
            (defaults to ``/workspace/<repo-name>``). Optional
            ``authorization_token`` is injected as the URL user, and
            ``checkout`` (branch/tag/sha) is checked out after clone.
          - ``file``: ``copy_to`` the uploaded file blob from the host
            into ``mount_path`` (defaults to ``/workspace/<filename>``).
            Looks up the metadata and on-disk path via the Files API
            store.
        """
        for resource in resources:
            rtype = resource.get("type")
            if rtype == "github_repository":
                await self._mount_github_repository(sandbox, session_id, resource)
            elif rtype == "file":
                await self._mount_file_resource(sandbox, session_id, resource)
            else:
                logger.warning("unknown resource type=%r session=%s", rtype, session_id)

    async def _mount_file_resource(
        self,
        sandbox: AsyncSandbox,
        session_id: str,
        resource: dict[str, Any],
    ) -> None:
        """Copy an uploaded file from the host blob store into the sandbox.

        Looks up the file metadata via a fresh database session because
        sandbox_manager doesn't have access to the request session.
        """
        from castor_server.store.database import async_session
        from castor_server.store.repository import get_file as repo_get_file

        file_id = resource.get("file_id")
        if not file_id:
            logger.warning("file resource missing file_id session=%s", session_id)
            return

        async with async_session() as db:
            meta = await repo_get_file(db, file_id)

        if meta is None:
            logger.warning(
                "file resource not found in store session=%s file_id=%s",
                session_id,
                file_id,
            )
            return

        host_path = str(Path(settings.files_dir) / file_id)
        if not Path(host_path).exists():
            logger.warning(
                "file blob missing on disk session=%s file_id=%s path=%s",
                session_id,
                file_id,
                host_path,
            )
            return

        mount_path = resource.get("mount_path") or f"/workspace/{meta.filename}"

        # Make sure the parent directory exists.
        parent = "/".join(mount_path.rstrip("/").split("/")[:-1]) or "/"
        await sandbox.exec(["mkdir", "-p", parent])

        try:
            await sandbox.copy_to(host_path, mount_path)
        except Exception:
            logger.exception(
                "file copy_to failed session=%s file_id=%s mount_path=%s",
                session_id,
                file_id,
                mount_path,
            )
            return

        logger.info(
            "file_resource_mounted session=%s file_id=%s filename=%s path=%s",
            session_id,
            file_id,
            meta.filename,
            mount_path,
        )

    async def _mount_github_repository(
        self,
        sandbox: AsyncSandbox,
        session_id: str,
        resource: dict[str, Any],
    ) -> None:
        url = resource.get("url", "")
        token = resource.get("authorization_token")
        checkout = resource.get("checkout")
        mount_path = resource.get("mount_path") or self._default_mount_path(url)

        # Inject the token into the URL if provided. We use the
        # ``x-access-token`` GitHub convention which works for personal
        # access tokens, fine-grained tokens, and GitHub Apps.
        clone_url = url
        if token and url.startswith("https://"):
            clone_url = url.replace("https://", f"https://x-access-token:{token}@", 1)

        # Make sure the parent directory exists.
        parent = "/".join(mount_path.rstrip("/").split("/")[:-1]) or "/"
        await sandbox.exec(["mkdir", "-p", parent])

        result = await sandbox.exec(
            ["git", "clone", clone_url, mount_path],
            timeout_secs=300,
        )
        if getattr(result, "exit_code", 0) != 0:
            logger.error(
                "git clone failed session=%s url=%s exit=%s stderr=%s",
                session_id,
                url,
                getattr(result, "exit_code", "?"),
                getattr(result, "stderr", "")[:500],
            )
            return

        if checkout:
            checkout_result = await sandbox.exec(
                ["git", "-C", mount_path, "checkout", checkout],
                timeout_secs=60,
            )
            if getattr(checkout_result, "exit_code", 0) != 0:
                logger.warning(
                    "git checkout failed session=%s ref=%s",
                    session_id,
                    checkout,
                )

        logger.info(
            "github_repository_mounted session=%s url=%s path=%s checkout=%s",
            session_id,
            url,
            mount_path,
            checkout or "(default)",
        )

    @staticmethod
    def _default_mount_path(url: str) -> str:
        """Derive ``/workspace/<repo-name>`` from a git URL."""
        # Strip .git suffix and pull the last path segment
        clean = url.rstrip("/").removesuffix(".git")
        name = clean.split("/")[-1] or "repo"
        return f"/workspace/{name}"

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
