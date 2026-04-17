"""SessionManager: Ring 2 coordinator for stateless session lifecycle.

Each HTTP event triggers: load checkpoint → kernel.run() → save checkpoint.
The server is stateless — all state lives in the checkpoint (DB).

This module is purely Ring 2 (API/app layer). It coordinates:
- EventBus management (per-session pub/sub for SSE)
- Checkpoint serialization between DB and kernel
- Custom tool client interaction (protocol layering)
- Session status transitions

Execution logic (Ring 0/1) is delegated to the Castor kernel.
Agent logic (Ring 3) lives in agent_fn.py.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC
from typing import Any

from castor.kernel.journal import InMemoryJournal
from castor.models.checkpoint import AgentCheckpoint, SyscallRecord
from sqlalchemy.ext.asyncio import AsyncSession

from castor_server.core.agent_fn import build_agent_fn
from castor_server.core.event_bus import EventBus
from castor_server.core.kernel_adapter import build_kernel_for_agent
from castor_server.core.mcp_runtime import discover_mcp_tools
from castor_server.core.sandbox_manager import sandbox_manager
from castor_server.models.agents import AgentResponse
from castor_server.models.events import (
    SessionError,
    SessionErrorDetail,
    SessionStatusIdle,
    SessionStatusRunning,
    StopReasonEndTurn,
    StopReasonRequiresAction,
)
from castor_server.store import repository as repo

logger = logging.getLogger("castor_server.session_manager")


class SessionManager:
    """Ring 2 coordinator for session lifecycle."""

    def __init__(self) -> None:
        self._buses: dict[str, EventBus] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        # In-memory snapshot of the most recent agent_fn conversation per
        # session. Used to recover the real ``agent.tool_use`` id when the
        # kernel suspends for HITL. NOT persisted — the kernel checkpoint
        # is the authoritative resume state.
        self._latest_conversation_by_session: dict[str, list[dict[str, Any]]] = {}
        # Per-session MCP tool discovery cache. Populated lazily on the
        # first kernel run for a session and reused across turns.
        self._mcp_tools_by_session: dict[str, dict[str, list[dict[str, Any]]]] = {}
        self._background_tasks: set[asyncio.Task] = set()

    def get_bus(self, session_id: str) -> EventBus:
        if session_id not in self._buses:
            self._buses[session_id] = EventBus(session_id)
        return self._buses[session_id]

    def _get_lock(self, session_id: str) -> asyncio.Lock:
        if session_id not in self._locks:
            self._locks[session_id] = asyncio.Lock()
        return self._locks[session_id]

    def dispatch(self, coro) -> asyncio.Task:
        """Spawn a fire-and-forget task and track it for drain on shutdown."""
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    async def drain(self) -> None:
        """Wait for all in-flight background tasks to complete."""
        if not self._background_tasks:
            return
        await asyncio.gather(*self._background_tasks, return_exceptions=True)

    # ------------------------------------------------------------------
    # Synchronous execution (used by OpenAI-compat adaptor)
    # ------------------------------------------------------------------

    async def run_and_wait(
        self,
        db: AsyncSession,
        session_id: str,
        content: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Run the agent loop and return collected events (blocking).

        Unlike ``handle_user_message`` (fire-and-forget), this awaits
        the full agent loop and returns all emitted events as dicts.
        Used by the OpenAI Responses API adaptor which needs the result
        in the same HTTP request.
        """
        lock = self._get_lock(session_id)
        async with lock:
            session = await repo.get_session(db, session_id)
            if not session:
                return []

            bus = self.get_bus(session_id)

            # Subscribe to capture events emitted during this run
            queue = bus.subscribe()

            kernel_cp, messages = await self._load_checkpoint(db, session_id)

            user_text = " ".join(
                c.get("text", "") for c in content if c.get("type") == "text"
            )
            messages.append({"role": "user", "content": user_text})

            await self._emit_running(db, session_id, bus)

            kernel_cp = await self._run_kernel(
                db, session_id, session.agent, bus, kernel_cp, messages
            )

            await self._handle_kernel_result(db, session_id, bus, kernel_cp, messages)
            await self._save_checkpoint(db, session_id, kernel_cp, messages)

            # Drain all events from the queue
            bus.unsubscribe(queue)
            collected: list[dict[str, Any]] = []
            while not queue.empty():
                evt = queue.get_nowait()
                if evt is not None:
                    collected.append(evt)

            return collected

    # ------------------------------------------------------------------
    # Event handlers (called from api/events.py)
    # ------------------------------------------------------------------

    async def handle_user_message(
        self,
        db: AsyncSession,
        session_id: str,
        content: list[dict[str, Any]],
    ) -> None:
        """Process a user.message event: run the agent loop via kernel."""
        lock = self._get_lock(session_id)
        async with lock:
            session = await repo.get_session(db, session_id)
            if not session:
                return

            bus = self.get_bus(session_id)
            kernel_cp, messages = await self._load_checkpoint(db, session_id)

            # Append user message to conversation
            user_text = " ".join(
                c.get("text", "") for c in content if c.get("type") == "text"
            )
            messages.append({"role": "user", "content": user_text})

            # Emit running status
            await self._emit_running(db, session_id, bus)

            # Run agent via kernel
            kernel_cp = await self._run_kernel(
                db, session_id, session.agent, bus, kernel_cp, messages
            )

            # Handle result
            await self._handle_kernel_result(db, session_id, bus, kernel_cp, messages)
            await self._save_checkpoint(db, session_id, kernel_cp, messages)

    async def handle_tool_confirmation(
        self,
        db: AsyncSession,
        session_id: str,
        tool_use_id: str,
        result: str,
        deny_message: str | None = None,
        modify_feedback: str | None = None,
    ) -> None:
        """Process a user.tool_confirmation event (builtin tool HITL)."""
        from castor_server.tools.builtin import clear_sandbox, set_sandbox

        lock = self._get_lock(session_id)
        async with lock:
            session = await repo.get_session(db, session_id)
            if not session:
                return

            bus = self.get_bus(session_id)
            kernel_cp, messages = await self._load_checkpoint(db, session_id)

            if kernel_cp is None:
                return

            # Verify the incoming tool_use_id matches what's actually
            # pending. The kernel only suspends one syscall at a time, so
            # there's exactly one pending tool — we use the latest
            # in-memory conversation snapshot to recover its real id (the
            # LLM's tool_use id) and reject mismatches. Synthetic
            # 'hitl_<name>' fallback ids are also accepted for backwards
            # compatibility with older clients.
            expected_id = None
            if kernel_cp.pending_hitl:
                pending_tool_name = kernel_cp.pending_hitl.get("tool_name", "")
                snapshot = self._latest_conversation_by_session.get(session_id)
                expected_id = self._find_pending_tool_use_id(
                    snapshot, pending_tool_name
                )
                synthetic = f"hitl_{pending_tool_name}"
                if tool_use_id != expected_id and tool_use_id != synthetic:
                    logger.warning(
                        "tool_confirmation id mismatch: got %s expected %s "
                        "(session=%s)",
                        tool_use_id,
                        expected_id,
                        session_id,
                    )
                    return

            kernel = build_kernel_for_agent(session.agent)

            # Set up sandbox context BEFORE approve (which executes the tool)
            sandbox_token = None
            if session.environment_id:
                env = await repo.get_environment(db, session.environment_id)
                if env:
                    sbx = await sandbox_manager.get_or_create(session_id, env)
                    sandbox_token = set_sandbox(sbx)

            try:
                if result == "allow":
                    await kernel.approve(kernel_cp)
                elif result == "modify":
                    kernel.modify(kernel_cp, modify_feedback or "")
                else:
                    kernel.reject(kernel_cp, deny_message or "Tool use denied.")
            finally:
                if sandbox_token is not None:
                    clear_sandbox(sandbox_token)

            # Emit running and resume
            await self._emit_running(db, session_id, bus)
            kernel_cp = await self._run_kernel(
                db, session_id, session.agent, bus, kernel_cp, messages
            )
            await self._handle_kernel_result(db, session_id, bus, kernel_cp, messages)
            await self._save_checkpoint(db, session_id, kernel_cp, messages)

    async def handle_custom_tool_result(
        self,
        db: AsyncSession,
        session_id: str,
        custom_tool_use_id: str,
        content: list[dict[str, Any]] | None,
        is_error: bool,
    ) -> None:
        """Process a user.custom_tool_result — protocol layering server side."""
        lock = self._get_lock(session_id)
        async with lock:
            session = await repo.get_session(db, session_id)
            if not session:
                return

            bus = self.get_bus(session_id)
            kernel_cp, messages = await self._load_checkpoint(db, session_id)

            if kernel_cp is None:
                return

            # Extract result text
            result_text = ""
            if content:
                result_text = " ".join(
                    c.get("text", "") for c in content if c.get("type") == "text"
                )
            if not result_text:
                result_text = "Error" if is_error else "Done"

            # Protocol layering: inject result into kernel checkpoint
            # The external_input syscall is pending HITL — we inject the
            # client result as the syscall response in the journal.
            self._inject_external_result(kernel_cp, result_text)

            # Add tool result to conversation for LLM context
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": custom_tool_use_id,
                    "content": result_text,
                }
            )

            # Resume
            await self._emit_running(db, session_id, bus)
            kernel_cp = await self._run_kernel(
                db, session_id, session.agent, bus, kernel_cp, messages
            )
            await self._handle_kernel_result(db, session_id, bus, kernel_cp, messages)
            await self._save_checkpoint(db, session_id, kernel_cp, messages)

    async def handle_interrupt(self, db: AsyncSession, session_id: str) -> None:
        """Process a user.interrupt event."""
        bus = self.get_bus(session_id)
        await repo.update_session_status(db, session_id, "idle")
        idle_evt = SessionStatusIdle(stop_reason=StopReasonEndTurn())
        await bus.publish(idle_evt)
        await repo.store_event(
            db,
            session_id=session_id,
            event_id=idle_evt.id,
            event_type=idle_evt.type,
            data=idle_evt.model_dump(exclude_none=True),
        )

    # ------------------------------------------------------------------
    # Internal — kernel execution
    # ------------------------------------------------------------------

    async def _run_kernel(
        self,
        db: AsyncSession,
        session_id: str,
        agent: AgentResponse,
        bus: EventBus,
        kernel_cp: AgentCheckpoint | None,
        messages: list[dict[str, Any]],
    ) -> AgentCheckpoint:
        """Build kernel + agent function, execute via kernel.run().

        Mutates ``messages`` in place with the agent's latest conversation
        snapshot AFTER the kernel run completes (or suspends). This is how
        we recover the assistant's tool_calls when the kernel pauses for
        HITL — the snapshot is built via a side-channel list that the
        agent_fn updates without breaking replay determinism.
        """
        from castor_server.tools.builtin import clear_sandbox, set_sandbox

        kernel = build_kernel_for_agent(agent)

        # Side-channel for the latest conversation state. agent_fn writes
        # into this list at every checkpoint (assistant tool_call, tool
        # result, terminal message); we mirror it back into ``messages``
        # after the kernel returns so _handle_kernel_result and
        # _save_checkpoint see the right state.
        latest_conversation: list[dict[str, Any]] = []

        # Discover MCP tools so the agent loop can offer them to the LLM.
        # Cached per session to avoid re-discovering on every turn.
        mcp_tools_by_server = await self._get_mcp_tools(session_id, agent)

        # Resolve skill content — read SKILL.md files from disk for all
        # skills referenced by the agent. Cached per session.
        skill_contents = await self._resolve_skill_contents(db, agent)

        agent_fn = build_agent_fn(
            agent=agent,
            messages=messages,
            bus=bus,
            db=db,
            session_id=session_id,
            latest_conversation=latest_conversation,
            mcp_tools_by_server=mcp_tools_by_server,
            skill_contents=skill_contents,
        )

        # Brain-before-body: start sandbox provisioning concurrently
        # with kernel.run(). The first kernel step is a pure LLM call
        # (no tool invocation), so the sandbox has time to provision
        # while the LLM is thinking. The sandbox contextvar is set
        # once the provisioning task completes, before any tool executes.
        sandbox_token = None
        sandbox_task = None
        session_data = await repo.get_session(db, session_id)
        if session_data and session_data.environment_id:
            env = await repo.get_environment(db, session_data.environment_id)
            if env:
                resources = [
                    r if isinstance(r, dict) else r.model_dump(exclude_none=True)
                    for r in (session_data.resources or [])
                ]
                # If sandbox is already cached, set it immediately
                existing = sandbox_manager.get_sandbox(session_id)
                if existing:
                    sandbox_token = set_sandbox(existing)
                else:
                    # Start provisioning in background — will be awaited
                    # before the first tool call
                    async def _provision():
                        sbx = await sandbox_manager.get_or_create(
                            session_id, env, resources=resources
                        )
                        return set_sandbox(sbx)

                    sandbox_task = asyncio.create_task(_provision())

        # Set up MCP auth from vault credentials.
        mcp_auth_token = None
        vault_ids = session_data.vault_ids if session_data else []
        if vault_ids:
            mcp_auth_map = await self._build_mcp_auth_map(db, vault_ids)
            if mcp_auth_map:
                from castor_server.core.mcp_runtime import set_mcp_auth

                mcp_auth_token = set_mcp_auth(mcp_auth_map)

        try:
            # If sandbox is provisioning in background, ensure it's ready
            # before kernel.run starts tool execution. The first step is
            # always a pure LLM call, so we have a window.
            if sandbox_task is not None:
                sandbox_token = await sandbox_task

            kernel_cp = await kernel.run(agent_fn, checkpoint=kernel_cp)
        except Exception as e:
            logger.exception("Kernel run error session=%s", session_id)
            # Emit session.error here; status_idle is emitted by
            # _handle_kernel_result based on the FAILED checkpoint status.
            err_evt = SessionError(
                error=SessionErrorDetail(type="unknown_error", message=str(e))
            )
            await bus.publish(err_evt)
            await repo.store_event(
                db,
                session_id=session_id,
                event_id=err_evt.id,
                event_type=err_evt.type,
                data=err_evt.model_dump(exclude_none=True),
            )
            # Return a minimal checkpoint so save doesn't fail
            if kernel_cp is None:
                kernel_cp = AgentCheckpoint(
                    pid=f"error-{session_id}",
                    status="FAILED",
                    agent_function_name="anthropic_agent_loop",
                    capabilities={},
                )
            else:
                kernel_cp.status = "FAILED"
        finally:
            if sandbox_token is not None:
                clear_sandbox(sandbox_token)
            if mcp_auth_token is not None:
                from castor_server.core.mcp_runtime import clear_mcp_auth

                clear_mcp_auth(mcp_auth_token)
            # NOTE: do NOT mirror latest_conversation back into messages.
            # The kernel's replay-on-resume requires ``messages`` to be the
            # exact original input each time agent_fn is called. The
            # latest_conversation snapshot is for the caller to peek at
            # the in-progress state (e.g. to find tool_use ids for HITL),
            # not for persistence.
            self._latest_conversation_by_session[session_id] = list(latest_conversation)

        return kernel_cp

    async def _handle_kernel_result(
        self,
        db: AsyncSession,
        session_id: str,
        bus: EventBus,
        kernel_cp: AgentCheckpoint,
        messages: list[dict[str, Any]] | None = None,
    ) -> None:
        """Emit appropriate session events based on kernel checkpoint status.

        ``messages`` is the conversation history (used to recover the real
        ``agent.tool_use`` id when the kernel suspends for HITL — see
        ``_find_pending_tool_use_id``).
        """
        if kernel_cp.status == "COMPLETED":
            await repo.update_session_status(db, session_id, "idle")
            idle_evt = SessionStatusIdle(stop_reason=StopReasonEndTurn())
            await bus.publish(idle_evt)
            await repo.store_event(
                db,
                session_id=session_id,
                event_id=idle_evt.id,
                event_type=idle_evt.type,
                data=idle_evt.model_dump(exclude_none=True),
            )

        elif kernel_cp.status == "SUSPENDED_FOR_HITL":
            # Determine blocking event IDs from pending_hitl
            blocking_ids: list[str] = []
            if kernel_cp.pending_hitl:
                tool_name = kernel_cp.pending_hitl.get("tool_name", "")
                if tool_name == "external_input":
                    # Custom tool suspension — blocking ID from payload
                    payload = kernel_cp.pending_hitl.get("arguments", {}).get(
                        "payload", {}
                    )
                    tool_id = payload.get("tool_id", "")
                    if tool_id:
                        blocking_ids.append(tool_id)
                else:
                    # Builtin tool HITL — recover the real tool_use id from
                    # the latest in-progress conversation snapshot (built
                    # by agent_fn during the run). The SDK uses this id to
                    # send back a user.tool_confirmation.
                    snapshot = self._latest_conversation_by_session.get(session_id)
                    real_id = self._find_pending_tool_use_id(snapshot, tool_name)
                    if real_id:
                        blocking_ids.append(real_id)
                    else:
                        # Fallback to a synthetic id if we can't find the
                        # original — better than emitting empty event_ids.
                        logger.warning(
                            "could not find tool_use id for pending HITL "
                            "tool=%s session=%s",
                            tool_name,
                            session_id,
                        )
                        blocking_ids.append(f"hitl_{tool_name}")

            await repo.update_session_status(db, session_id, "idle")
            idle_evt = SessionStatusIdle(
                stop_reason=StopReasonRequiresAction(event_ids=blocking_ids)
            )
            await bus.publish(idle_evt)
            await repo.store_event(
                db,
                session_id=session_id,
                event_id=idle_evt.id,
                event_type=idle_evt.type,
                data=idle_evt.model_dump(exclude_none=True),
            )

        elif kernel_cp.status == "FAILED":
            await repo.update_session_status(db, session_id, "idle")
            idle_evt = SessionStatusIdle(stop_reason=StopReasonEndTurn())
            await bus.publish(idle_evt)
            await repo.store_event(
                db,
                session_id=session_id,
                event_id=idle_evt.id,
                event_type=idle_evt.type,
                data=idle_evt.model_dump(exclude_none=True),
            )

    # ------------------------------------------------------------------
    # Internal — HITL helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_pending_tool_use_id(
        messages: list[dict[str, Any]] | None,
        tool_name: str,
    ) -> str | None:
        """Find the LLM-assigned tool_use id for the most recent call to
        ``tool_name``.

        The conversation history (``messages``) carries the assistant's
        ``tool_calls`` from each LLM turn. When the kernel suspends for
        HITL, the pending syscall corresponds to one of the tool_calls in
        the most recent assistant message. We match by tool name.

        Returns ``None`` if no matching tool_call can be found.
        """
        if not messages:
            return None
        for msg in reversed(messages):
            if msg.get("role") != "assistant":
                continue
            tool_calls = msg.get("tool_calls") or []
            for tc in tool_calls:
                func = tc.get("function") or {}
                if func.get("name") == tool_name:
                    return tc.get("id")
            # Stop at the first assistant message — older ones are stale.
            return None
        return None

    # ------------------------------------------------------------------
    # Internal — MCP tool discovery
    # ------------------------------------------------------------------

    async def _get_mcp_tools(
        self, session_id: str, agent: AgentResponse
    ) -> dict[str, list[dict[str, Any]]]:
        """Discover (and cache) MCP tools for the agent's mcp_servers.

        First call for a session connects to each MCP server and lists
        its tools; subsequent calls return the cached result. Discovery
        failures are logged inside ``discover_mcp_tools`` and the failing
        servers are simply omitted from the returned dict.
        """
        if session_id in self._mcp_tools_by_session:
            return self._mcp_tools_by_session[session_id]
        if not agent.mcp_servers:
            self._mcp_tools_by_session[session_id] = {}
            return {}
        tools = await discover_mcp_tools(agent.mcp_servers)
        self._mcp_tools_by_session[session_id] = tools
        return tools

    # ------------------------------------------------------------------
    # Internal — status events
    # ------------------------------------------------------------------

    async def _emit_running(
        self, db: AsyncSession, session_id: str, bus: EventBus
    ) -> None:
        await repo.update_session_status(db, session_id, "running")
        running_evt = SessionStatusRunning()
        await bus.publish(running_evt)
        await repo.store_event(
            db,
            session_id=session_id,
            event_id=running_evt.id,
            event_type=running_evt.type,
            data=running_evt.model_dump(exclude_none=True),
        )

    # ------------------------------------------------------------------
    # Internal — checkpoint persistence (Step 5)
    # ------------------------------------------------------------------

    async def _load_checkpoint(
        self, db: AsyncSession, session_id: str
    ) -> tuple[AgentCheckpoint | None, list[dict[str, Any]]]:
        """Load kernel checkpoint and messages from DB."""
        row = await repo.get_session_row(db, session_id)
        if not row or not row.checkpoint_json:
            return None, []

        data = row.checkpoint_json

        # New format (version 2)
        if isinstance(data, dict) and data.get("version") == 2:
            kc_data = data.get("kernel_checkpoint")
            kernel_cp = AgentCheckpoint.model_validate(kc_data) if kc_data else None
            messages = data.get("messages", [])
            return kernel_cp, messages

        # Legacy format (Phase 1) — treat as fresh session
        return None, data.get("messages", [])

    async def _save_checkpoint(
        self,
        db: AsyncSession,
        session_id: str,
        kernel_cp: AgentCheckpoint,
        messages: list[dict[str, Any]],
    ) -> None:
        """Save kernel checkpoint and messages to DB."""
        data = {
            "version": 2,
            "kernel_checkpoint": kernel_cp.model_dump(),
            "messages": messages,
        }
        await repo.update_session_checkpoint(db, session_id, data)

    # ------------------------------------------------------------------
    # Internal — custom tool protocol layering
    # ------------------------------------------------------------------

    async def _build_mcp_auth_map(
        self, db: AsyncSession, vault_ids: list[str]
    ) -> dict[str, dict[str, str]]:
        """Build a URL → auth-headers mapping from vault credentials.

        Looks up all active credentials in the given vaults and returns
        a dict suitable for ``set_mcp_auth()``:
        ``{mcp_server_url: {"Authorization": "Bearer <token>"}}``

        For ``mcp_oauth`` credentials with an expired ``access_token``,
        attempts to refresh using the ``refresh_token`` (if available)
        and persists the new token back to the database.
        """
        cred_rows = await repo.get_credentials_for_vaults(db, vault_ids)
        auth_map: dict[str, dict[str, str]] = {}
        for row in cred_rows:
            url = row.mcp_server_url
            if row.auth_type == "static_bearer" and row.token:
                auth_map[url] = {"Authorization": f"Bearer {row.token}"}
            elif row.auth_type == "mcp_oauth" and row.access_token:
                token = row.access_token
                if self._is_token_expired(row.expires_at_str) and row.refresh_token:
                    refreshed = await self._refresh_oauth_token(db, row)
                    if refreshed:
                        token = refreshed
                auth_map[url] = {"Authorization": f"Bearer {token}"}
        return auth_map

    @staticmethod
    def _is_token_expired(expires_at_str: str | None) -> bool:
        if not expires_at_str:
            return False
        from datetime import datetime

        try:
            expires = datetime.fromisoformat(expires_at_str.replace("Z", "+00:00"))
            return datetime.now(UTC) >= expires
        except (ValueError, TypeError):
            return False

    @staticmethod
    async def _refresh_oauth_token(db: AsyncSession, row: Any) -> str | None:
        """Attempt to refresh an expired OAuth access token.

        Uses a simple OAuth2 token refresh request. On success, updates
        the credential row in the database and returns the new access token.
        Returns None on failure (the caller falls back to the old token).
        """
        import httpx

        # The refresh endpoint is typically the MCP server's token URL.
        # Convention: POST to mcp_server_url + "/oauth/token"
        token_url = row.mcp_server_url.rstrip("/") + "/oauth/token"

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(
                    token_url,
                    data={
                        "grant_type": "refresh_token",
                        "refresh_token": row.refresh_token,
                    },
                )
                if resp.status_code != 200:
                    logger.warning(
                        "oauth_refresh_failed url=%s status=%d",
                        token_url,
                        resp.status_code,
                    )
                    return None

                data = resp.json()
                new_access = data.get("access_token")
                if not new_access:
                    return None

                # Update the credential row in DB
                row.access_token = new_access
                if data.get("expires_in"):
                    from datetime import datetime, timedelta

                    new_expires = datetime.now(UTC) + timedelta(
                        seconds=int(data["expires_in"])
                    )
                    row.expires_at_str = new_expires.isoformat()
                if data.get("refresh_token"):
                    row.refresh_token = data["refresh_token"]
                row.updated_at = datetime.utcnow()
                await db.commit()

                logger.info("oauth_token_refreshed url=%s", row.mcp_server_url)
                return new_access
        except Exception:
            logger.exception("oauth_refresh_error url=%s", row.mcp_server_url)
            return None

    async def _resolve_skill_contents(
        self, db: AsyncSession, agent: AgentResponse
    ) -> list[str]:
        """Read SKILL.md content for each skill referenced by the agent.

        Returns a list of skill markdown strings to be injected into the
        LLM system prompt by the agent function.
        """
        if not agent.skills:
            return []

        from pathlib import Path

        from castor_server.config import settings

        contents: list[str] = []
        for skill_ref in agent.skills:
            sd = skill_ref if isinstance(skill_ref, dict) else skill_ref.model_dump()
            skill_id = sd.get("skill_id")
            version = sd.get("version")

            if not skill_id:
                continue

            # Resolve version — if not pinned, look up latest from DB
            if not version:
                skill_resp = await repo.get_skill(db, skill_id)
                if skill_resp is None:
                    logger.warning("skill not found id=%s", skill_id)
                    continue
                version = skill_resp.latest_version
                if not version:
                    logger.warning("skill has no versions id=%s", skill_id)
                    continue

            # Read SKILL.md from disk
            skill_dir = Path(settings.files_dir) / "skills" / skill_id / version
            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                logger.warning(
                    "SKILL.md not found skill=%s version=%s path=%s",
                    skill_id,
                    version,
                    skill_md,
                )
                continue

            try:
                text = skill_md.read_text(encoding="utf-8")
                if text.strip():
                    contents.append(text)
            except Exception:
                logger.exception(
                    "failed to read SKILL.md skill=%s version=%s", skill_id, version
                )

        return contents

    @staticmethod
    def _inject_external_result(kernel_cp: AgentCheckpoint, result_text: str) -> None:
        """Inject client tool result into kernel checkpoint.

        When external_input is suspended for HITL, we inject the client's
        result directly into the syscall_log (as if the tool returned it)
        and clear the pending state so replay can continue.
        """
        if kernel_cp.pending_hitl is None:
            return

        journal = InMemoryJournal(kernel_cp.syscall_log)
        journal.append(
            SyscallRecord(
                request=kernel_cp.pending_hitl,
                response=result_text,
                was_hitl=True,
            )
        )
        kernel_cp.pending_hitl = None
        kernel_cp.status = "RUNNING"


# Singleton
session_manager = SessionManager()
