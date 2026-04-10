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
from typing import Any

from castor.kernel.journal import InMemoryJournal
from castor.models.checkpoint import AgentCheckpoint, SyscallRecord
from sqlalchemy.ext.asyncio import AsyncSession

from castor_server.core.agent_fn import build_agent_fn
from castor_server.core.event_bus import EventBus
from castor_server.core.kernel_adapter import build_kernel_for_agent
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

    def get_bus(self, session_id: str) -> EventBus:
        if session_id not in self._buses:
            self._buses[session_id] = EventBus(session_id)
        return self._buses[session_id]

    def _get_lock(self, session_id: str) -> asyncio.Lock:
        if session_id not in self._locks:
            self._locks[session_id] = asyncio.Lock()
        return self._locks[session_id]

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
            await self._handle_kernel_result(db, session_id, bus, kernel_cp)
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
            await self._handle_kernel_result(db, session_id, bus, kernel_cp)
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
            await self._handle_kernel_result(db, session_id, bus, kernel_cp)
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
            data=idle_evt.model_dump(),
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
        """Build kernel + agent function, execute via kernel.run()."""
        from castor_server.tools.builtin import clear_sandbox, set_sandbox

        kernel = build_kernel_for_agent(agent)

        agent_fn = build_agent_fn(
            agent=agent,
            messages=messages,
            bus=bus,
            db=db,
            session_id=session_id,
        )

        # Set up sandbox context if session has an environment
        sandbox_token = None
        session_data = await repo.get_session(db, session_id)
        if session_data and session_data.environment_id:
            env = await repo.get_environment(db, session_data.environment_id)
            if env:
                sbx = await sandbox_manager.get_or_create(session_id, env)
                sandbox_token = set_sandbox(sbx)

        try:
            kernel_cp = await kernel.run(agent_fn, checkpoint=kernel_cp)
        except Exception as e:
            logger.exception("Kernel run error session=%s", session_id)
            err_evt = SessionError(
                error=SessionErrorDetail(type="unknown_error", message=str(e))
            )
            await bus.publish(err_evt)
            await repo.store_event(
                db,
                session_id=session_id,
                event_id=err_evt.id,
                event_type=err_evt.type,
                data=err_evt.model_dump(),
            )
            # Emit idle on error
            await repo.update_session_status(db, session_id, "idle")
            idle_evt = SessionStatusIdle(stop_reason=StopReasonEndTurn())
            await bus.publish(idle_evt)
            await repo.store_event(
                db,
                session_id=session_id,
                event_id=idle_evt.id,
                event_type=idle_evt.type,
                data=idle_evt.model_dump(),
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

        return kernel_cp

    async def _handle_kernel_result(
        self,
        db: AsyncSession,
        session_id: str,
        bus: EventBus,
        kernel_cp: AgentCheckpoint,
    ) -> None:
        """Emit appropriate session events based on kernel checkpoint status."""
        if kernel_cp.status == "COMPLETED":
            await repo.update_session_status(db, session_id, "idle")
            idle_evt = SessionStatusIdle(stop_reason=StopReasonEndTurn())
            await bus.publish(idle_evt)
            await repo.store_event(
                db,
                session_id=session_id,
                event_id=idle_evt.id,
                event_type=idle_evt.type,
                data=idle_evt.model_dump(),
            )

        elif kernel_cp.status == "SUSPENDED_FOR_HITL":
            # Determine blocking event IDs from pending_hitl
            blocking_ids = []
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
                    # Builtin tool HITL — use a generated event ID
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
                data=idle_evt.model_dump(),
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
                data=idle_evt.model_dump(),
            )

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
            data=running_evt.model_dump(),
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
