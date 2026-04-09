"""SessionManager: stateless session loop with checkpoint/replay.

Each HTTP event triggers: load checkpoint → kernel.run() → save checkpoint.
The server is stateless — all state lives in the checkpoint (DB).
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from castor_server.core.event_bus import EventBus
from castor_server.core.llm_adapter import litellm_chat
from castor_server.models.agents import AgentResponse
from castor_server.models.common import ModelConfig, TextBlock
from castor_server.models.events import (
    AgentCustomToolUseEvent,
    AgentMessageEvent,
    AgentToolResultEvent,
    AgentToolUseEvent,
    ModelUsage,
    SessionError,
    SessionErrorDetail,
    SessionStatusIdle,
    SessionStatusRunning,
    SpanModelRequestEnd,
    SpanModelRequestStart,
    StopReasonEndTurn,
    StopReasonRequiresAction,
)
from castor_server.store import repository as repo
from castor_server.tools.builtin import execute_builtin_tool

logger = logging.getLogger("castor_server.session_manager")


class SessionManager:
    """Manages the agent execution loop for a single session."""

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

    async def handle_user_message(
        self,
        db: AsyncSession,
        session_id: str,
        content: list[dict[str, Any]],
    ) -> None:
        """Process a user.message event: run the agent loop."""
        lock = self._get_lock(session_id)
        async with lock:
            await self._run_agent_loop(db, session_id, content)

    async def handle_tool_confirmation(
        self,
        db: AsyncSession,
        session_id: str,
        tool_use_id: str,
        result: str,
        deny_message: str | None = None,
    ) -> None:
        """Process a user.tool_confirmation event."""
        lock = self._get_lock(session_id)
        async with lock:
            session = await repo.get_session(db, session_id)
            if not session:
                return

            bus = self.get_bus(session_id)
            checkpoint = await self._load_checkpoint(db, session_id)

            if result == "allow":
                # Find the pending tool call and execute it
                pending = checkpoint.get("pending_tool_calls", [])
                for tc in pending:
                    if tc["id"] == tool_use_id:
                        tool_result = await execute_builtin_tool(
                            tc["name"], tc["input"]
                        )
                        evt = AgentToolResultEvent(
                            tool_use_id=tool_use_id,
                            content=[TextBlock(text=str(tool_result))],
                        )
                        await bus.publish(evt)
                        await repo.store_event(
                            db,
                            session_id=session_id,
                            event_id=evt.id,
                            event_type=evt.type,
                            data=evt.model_dump(),
                        )
                        # Remove from pending
                        pending.remove(tc)
                        break

                checkpoint["pending_tool_calls"] = pending
                if not pending:
                    # Resume agent loop with tool results
                    await self._resume_after_confirmation(db, session_id, checkpoint)
            else:
                # Denied - add denial to messages and resume
                pending = checkpoint.get("pending_tool_calls", [])
                pending = [tc for tc in pending if tc["id"] != tool_use_id]
                checkpoint["pending_tool_calls"] = pending

                denial_text = deny_message or "Tool use was denied by the user."
                evt = AgentToolResultEvent(
                    tool_use_id=tool_use_id,
                    content=[TextBlock(text=denial_text)],
                    is_error=True,
                )
                await bus.publish(evt)
                await repo.store_event(
                    db,
                    session_id=session_id,
                    event_id=evt.id,
                    event_type=evt.type,
                    data=evt.model_dump(),
                )

                if not pending:
                    await self._resume_after_confirmation(db, session_id, checkpoint)

            await self._save_checkpoint(db, session_id, checkpoint)

    async def handle_custom_tool_result(
        self,
        db: AsyncSession,
        session_id: str,
        custom_tool_use_id: str,
        content: list[dict[str, Any]] | None,
        is_error: bool,
    ) -> None:
        """Process a user.custom_tool_result event."""
        lock = self._get_lock(session_id)
        async with lock:
            checkpoint = await self._load_checkpoint(db, session_id)

            pending = checkpoint.get("pending_custom_tools", [])
            tool_result_text = ""
            if content:
                tool_result_text = " ".join(
                    c.get("text", "") for c in content if c.get("type") == "text"
                )

            # Add tool result to conversation
            checkpoint.setdefault("messages", []).append(
                {
                    "role": "tool",
                    "tool_call_id": custom_tool_use_id,
                    "content": tool_result_text or ("Error" if is_error else "Done"),
                }
            )

            pending = [tc for tc in pending if tc["id"] != custom_tool_use_id]
            checkpoint["pending_custom_tools"] = pending

            if not pending:
                await self._resume_after_confirmation(db, session_id, checkpoint)

            await self._save_checkpoint(db, session_id, checkpoint)

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
    # Internal
    # ------------------------------------------------------------------

    async def _load_checkpoint(self, db: AsyncSession, session_id: str) -> dict:
        row = await repo.get_session_row(db, session_id)
        if row and row.checkpoint_json:
            return row.checkpoint_json
        return {"messages": [], "pending_tool_calls": [], "pending_custom_tools": []}

    async def _save_checkpoint(
        self, db: AsyncSession, session_id: str, checkpoint: dict
    ) -> None:
        await repo.update_session_checkpoint(db, session_id, checkpoint)

    async def _run_agent_loop(
        self,
        db: AsyncSession,
        session_id: str,
        user_content: list[dict[str, Any]],
    ) -> None:
        session = await repo.get_session(db, session_id)
        if not session:
            return

        bus = self.get_bus(session_id)
        agent = session.agent
        checkpoint = await self._load_checkpoint(db, session_id)

        # Update status to running
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

        # Build user message text
        user_text = " ".join(
            c.get("text", "") for c in user_content if c.get("type") == "text"
        )

        # Add user message to conversation history
        checkpoint.setdefault("messages", []).append(
            {"role": "user", "content": user_text}
        )

        try:
            await self._agent_turn(db, session_id, agent, checkpoint, bus)
        except Exception as e:
            logger.exception("Agent loop error session=%s", session_id)
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

        await self._save_checkpoint(db, session_id, checkpoint)

    async def _agent_turn(
        self,
        db: AsyncSession,
        session_id: str,
        agent: AgentResponse,
        checkpoint: dict,
        bus: EventBus,
    ) -> None:
        """Run one or more LLM turns until the agent stops or needs input."""
        if isinstance(agent.model, ModelConfig):
            model_id = agent.model.id
        else:
            model_id = agent.model
        tools_for_llm = self._build_tools_for_llm(agent)
        max_turns = 50

        for _ in range(max_turns):
            messages = self._build_llm_messages(agent, checkpoint)

            # Emit span start
            span_start = SpanModelRequestStart()
            await bus.publish(span_start)
            await repo.store_event(
                db,
                session_id=session_id,
                event_id=span_start.id,
                event_type=span_start.type,
                data=span_start.model_dump(),
            )

            # Call LLM
            response = await litellm_chat(
                model=model_id,
                messages=messages,
                tools=tools_for_llm or None,
            )

            # Emit span end
            usage_data = response.get("usage", {})
            span_end = SpanModelRequestEnd(
                model_request_start_id=span_start.id,
                model_usage=ModelUsage(
                    input_tokens=usage_data.get("input_tokens", 0),
                    output_tokens=usage_data.get("output_tokens", 0),
                ),
            )
            await bus.publish(span_end)
            await repo.store_event(
                db,
                session_id=session_id,
                event_id=span_end.id,
                event_type=span_end.type,
                data=span_end.model_dump(),
            )

            # Update session usage
            await repo.update_session_usage(db, session_id, usage_data)

            content_blocks = response.get("content", [])
            has_tool_use = any(b["type"] == "tool_use" for b in content_blocks)

            # Add assistant response to history
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": []}
            if not has_tool_use:
                # Text-only response
                text_parts = [b["text"] for b in content_blocks if b["type"] == "text"]
                full_text = "\n".join(text_parts)
                if full_text:
                    msg_evt = AgentMessageEvent(content=[TextBlock(text=full_text)])
                    await bus.publish(msg_evt)
                    await repo.store_event(
                        db,
                        session_id=session_id,
                        event_id=msg_evt.id,
                        event_type=msg_evt.type,
                        data=msg_evt.model_dump(),
                    )
                    assistant_msg["content"].append({"type": "text", "text": full_text})

                checkpoint["messages"].append(
                    {"role": "assistant", "content": full_text}
                )

                # End turn
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
                return

            # Process tool uses
            # First emit any text blocks
            for block in content_blocks:
                if block["type"] == "text" and block.get("text"):
                    msg_evt = AgentMessageEvent(content=[TextBlock(text=block["text"])])
                    await bus.publish(msg_evt)
                    await repo.store_event(
                        db,
                        session_id=session_id,
                        event_id=msg_evt.id,
                        event_type=msg_evt.type,
                        data=msg_evt.model_dump(),
                    )

            # Build assistant message for history (OpenAI format with tool_calls)
            tool_calls_for_history = []
            tool_use_blocks = [b for b in content_blocks if b["type"] == "tool_use"]

            for block in tool_use_blocks:
                tool_calls_for_history.append(
                    {
                        "id": block["id"],
                        "type": "function",
                        "function": {
                            "name": block["name"],
                            "arguments": json.dumps(block["input"]),
                        },
                    }
                )

            text_content = " ".join(
                b["text"]
                for b in content_blocks
                if b["type"] == "text" and b.get("text")
            )
            checkpoint["messages"].append(
                {
                    "role": "assistant",
                    "content": text_content or None,
                    "tool_calls": tool_calls_for_history,
                }
            )

            # Execute each tool
            pending_asks: list[dict] = []
            pending_custom: list[dict] = []
            all_results_available = True

            for block in tool_use_blocks:
                tool_name = block["name"]
                tool_input = block["input"]
                tool_id = block["id"]

                # Determine tool type and permission
                perm = self._get_tool_permission(agent, tool_name)
                is_custom = self._is_custom_tool(agent, tool_name)

                if is_custom:
                    # Custom tool - ask client to execute
                    custom_evt = AgentCustomToolUseEvent(
                        name=tool_name, input=tool_input
                    )
                    custom_evt.id = tool_id
                    await bus.publish(custom_evt)
                    await repo.store_event(
                        db,
                        session_id=session_id,
                        event_id=custom_evt.id,
                        event_type=custom_evt.type,
                        data=custom_evt.model_dump(),
                    )
                    pending_custom.append(
                        {"id": tool_id, "name": tool_name, "input": tool_input}
                    )
                    all_results_available = False
                elif perm == "ask":
                    # Needs confirmation
                    tool_evt = AgentToolUseEvent(
                        name=tool_name,
                        input=tool_input,
                        evaluated_permission="ask",
                    )
                    tool_evt.id = tool_id
                    await bus.publish(tool_evt)
                    await repo.store_event(
                        db,
                        session_id=session_id,
                        event_id=tool_evt.id,
                        event_type=tool_evt.type,
                        data=tool_evt.model_dump(),
                    )
                    pending_asks.append(
                        {"id": tool_id, "name": tool_name, "input": tool_input}
                    )
                    all_results_available = False
                else:
                    # Execute built-in tool directly
                    tool_evt = AgentToolUseEvent(
                        name=tool_name,
                        input=tool_input,
                        evaluated_permission="allow",
                    )
                    tool_evt.id = tool_id
                    await bus.publish(tool_evt)
                    await repo.store_event(
                        db,
                        session_id=session_id,
                        event_id=tool_evt.id,
                        event_type=tool_evt.type,
                        data=tool_evt.model_dump(),
                    )

                    try:
                        result = await execute_builtin_tool(tool_name, tool_input)
                        result_evt = AgentToolResultEvent(
                            tool_use_id=tool_id,
                            content=[TextBlock(text=str(result))],
                        )
                    except Exception as e:
                        result_evt = AgentToolResultEvent(
                            tool_use_id=tool_id,
                            content=[TextBlock(text=f"Error: {e}")],
                            is_error=True,
                        )

                    await bus.publish(result_evt)
                    await repo.store_event(
                        db,
                        session_id=session_id,
                        event_id=result_evt.id,
                        event_type=result_evt.type,
                        data=result_evt.model_dump(),
                    )

                    # Add tool result to conversation
                    checkpoint["messages"].append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "content": str(result_evt.content[0].text)
                            if result_evt.content
                            else "",
                        }
                    )

            if not all_results_available:
                # Suspend - waiting for user input
                checkpoint["pending_tool_calls"] = pending_asks
                checkpoint["pending_custom_tools"] = pending_custom
                await self._save_checkpoint(db, session_id, checkpoint)

                blocking_ids = [tc["id"] for tc in pending_asks + pending_custom]
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
                return

            # All tools executed — loop back for next LLM turn

        # Max turns reached
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

    async def _resume_after_confirmation(
        self,
        db: AsyncSession,
        session_id: str,
        checkpoint: dict,
    ) -> None:
        """Resume the agent loop after all confirmations received."""
        session = await repo.get_session(db, session_id)
        if not session:
            return
        bus = self.get_bus(session_id)
        agent = session.agent

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

        try:
            await self._agent_turn(db, session_id, agent, checkpoint, bus)
        except Exception as e:
            logger.exception("Resume error session=%s", session_id)
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
            await repo.update_session_status(db, session_id, "idle")
            idle_evt = SessionStatusIdle(stop_reason=StopReasonEndTurn())
            await bus.publish(idle_evt)

    def _build_llm_messages(
        self, agent: AgentResponse, checkpoint: dict
    ) -> list[dict[str, Any]]:
        """Build the messages array for the LLM call."""
        messages: list[dict[str, Any]] = []

        # System prompt
        if agent.system:
            messages.append({"role": "system", "content": agent.system})

        # Conversation history
        for msg in checkpoint.get("messages", []):
            messages.append(msg)

        return messages

    def _build_tools_for_llm(self, agent: AgentResponse) -> list[dict[str, Any]]:
        """Convert agent tool definitions to OpenAI-format tool specs for LiteLLM."""
        tools: list[dict[str, Any]] = []

        for tool_def in agent.tools:
            if isinstance(tool_def, dict):
                tool_type = tool_def.get("type", "")
            else:
                tool_type = getattr(tool_def, "type", "")

            if tool_type == "agent_toolset_20260401":
                # Add built-in tools
                configs = (
                    tool_def.get("configs", [])
                    if isinstance(tool_def, dict)
                    else (tool_def.configs or [])
                )
                default_config = (
                    tool_def.get("default_config", {})
                    if isinstance(tool_def, dict)
                    else tool_def.default_config
                )
                default_enabled = True
                if default_config:
                    if isinstance(default_config, dict):
                        default_enabled = default_config.get("enabled", True)
                    else:
                        default_enabled = default_config.enabled

                enabled_map: dict[str, bool] = {}
                if configs:
                    for cfg in configs:
                        if isinstance(cfg, dict):
                            name = cfg.get("name")
                            enabled = cfg.get("enabled", True)
                        else:
                            name = cfg.name
                            enabled = cfg.enabled
                        enabled_map[name] = enabled

                for name, spec in _BUILTIN_TOOL_SPECS.items():
                    if enabled_map.get(name, default_enabled):
                        tools.append(spec)

            elif tool_type == "custom":
                if isinstance(tool_def, dict):
                    name = tool_def.get("name")
                    desc = tool_def.get("description", "")
                else:
                    name = tool_def.name
                    desc = tool_def.description
                schema = (
                    tool_def.get("input_schema", {})
                    if isinstance(tool_def, dict)
                    else tool_def.input_schema.model_dump()
                )
                tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": name,
                            "description": desc,
                            "parameters": schema,
                        },
                    }
                )

        return tools

    def _get_tool_permission(self, agent: AgentResponse, tool_name: str) -> str:
        """Return 'allow' or 'ask' for a given tool."""
        for tool_def in agent.tools:
            td = tool_def if isinstance(tool_def, dict) else tool_def.model_dump()
            if td.get("type") == "agent_toolset_20260401":
                configs = td.get("configs") or []
                for cfg in configs:
                    if cfg.get("name") == tool_name:
                        pp = cfg.get("permission_policy")
                        if pp and pp.get("type") == "always_ask":
                            return "ask"
                default_config = td.get("default_config") or {}
                pp = default_config.get("permission_policy")
                if pp and pp.get("type") == "always_ask":
                    return "ask"
        return "allow"

    def _is_custom_tool(self, agent: AgentResponse, tool_name: str) -> bool:
        """Check if a tool is a custom tool (executed client-side)."""
        for tool_def in agent.tools:
            td = tool_def if isinstance(tool_def, dict) else tool_def.model_dump()
            if td.get("type") == "custom" and td.get("name") == tool_name:
                return True
        return False


# ---------------------------------------------------------------------------
# Built-in tool specs (OpenAI function-calling format)
# ---------------------------------------------------------------------------

_BUILTIN_TOOL_SPECS: dict[str, dict[str, Any]] = {
    "bash": {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a bash command in a shell session.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute.",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds.",
                    },
                },
                "required": ["command"],
            },
        },
    },
    "read": {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read a file from the filesystem.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the file to read.",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Line number to start reading from.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of lines to read.",
                    },
                },
                "required": ["file_path"],
            },
        },
    },
    "write": {
        "type": "function",
        "function": {
            "name": "write",
            "description": "Write content to a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the file.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write.",
                    },
                },
                "required": ["file_path", "content"],
            },
        },
    },
    "edit": {
        "type": "function",
        "function": {
            "name": "edit",
            "description": "Perform string replacement in a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the file.",
                    },
                    "old_string": {
                        "type": "string",
                        "description": "The text to replace.",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "The replacement text.",
                    },
                },
                "required": ["file_path", "old_string", "new_string"],
            },
        },
    },
    "glob": {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "Find files matching a glob pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern (e.g. '**/*.py').",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search in.",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    "grep": {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search file contents using regex.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for.",
                    },
                    "path": {
                        "type": "string",
                        "description": "File or directory to search.",
                    },
                    "include": {
                        "type": "string",
                        "description": "Glob to filter files (e.g. '*.py').",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    "web_fetch": {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "Fetch content from a URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch.",
                    },
                },
                "required": ["url"],
            },
        },
    },
    "web_search": {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    },
                },
                "required": ["query"],
            },
        },
    },
}


# Singleton
session_manager = SessionManager()
