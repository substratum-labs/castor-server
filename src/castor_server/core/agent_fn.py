"""Agent function builder — standard LLM-tool loop (convenience layer).

This is a convenience helper that implements the Anthropic-style ReAct
pattern (LLM → parse tool calls → execute → loop) using kernel syscalls.
It is NOT a kernel feature — it doesn't belong to any Ring.  It's a
reusable template that castor-server uses to provide Anthropic API
compatibility.

The function also emits SSE events via the EventBus at each step
(skipped during replay via proxy.is_replaying).

Protocol layering for custom tools:
- Server layer (Ring 2): manages SSE push + client interaction
- Kernel layer (Ring 0/1): only sees ``external_input`` syscall
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

from castor.scheduler.proxy import SyscallProxy
from sqlalchemy.ext.asyncio import AsyncSession

from castor_server.core.event_bus import EventBus
from castor_server.models.agents import AgentResponse, ModelConfig
from castor_server.models.common import TextBlock
from castor_server.models.events import (
    AgentCustomToolUseEvent,
    AgentMessageEvent,
    AgentToolResultEvent,
    AgentToolUseEvent,
    ModelUsage,
    SpanModelRequestEnd,
    SpanModelRequestStart,
)
from castor_server.store import repository as repo
from castor_server.tools.builtin import BUILTIN_TOOL_SPECS

logger = logging.getLogger("castor_server.agent_fn")

MAX_TURNS = 50


def build_agent_fn(
    *,
    agent: AgentResponse,
    messages: list[dict[str, Any]],
    bus: EventBus,
    db: AsyncSession,
    session_id: str,
) -> Callable[[SyscallProxy], Any]:
    """Build a kernel-compatible agent function as a closure.

    The returned coroutine runs inside ``kernel.run(agent_fn)`` and has
    access to session context via closure variables.
    """
    model_id = agent.model.id if isinstance(agent.model, ModelConfig) else agent.model
    tool_specs = _build_tools_for_llm(agent)
    custom_tool_names = _get_custom_tool_names(agent)

    async def agent_fn(proxy: SyscallProxy) -> str:
        conversation = list(messages)

        for _turn in range(MAX_TURNS):
            # -- LLM call --
            llm_messages = _build_llm_messages(agent, conversation)

            span_start = SpanModelRequestStart()
            if not proxy.is_replaying:
                await _emit(bus, db, session_id, span_start)

            response = await proxy.syscall(
                "llm_inference",
                {
                    "model": model_id,
                    "messages": llm_messages,
                    "tools": tool_specs or None,
                },
            )

            if not proxy.is_replaying:
                usage = response.get("usage", {})
                span_end = SpanModelRequestEnd(
                    model_request_start_id=span_start.id,
                    model_usage=ModelUsage(
                        input_tokens=usage.get("input_tokens", 0),
                        output_tokens=usage.get("output_tokens", 0),
                    ),
                )
                await _emit(bus, db, session_id, span_end)
                await repo.update_session_usage(db, session_id, usage)

            # -- Parse response --
            content_blocks = response.get("content", [])
            tool_use_blocks = [b for b in content_blocks if b["type"] == "tool_use"]

            if not tool_use_blocks:
                # Text-only response — emit and return
                text = _extract_text(content_blocks)
                if text and not proxy.is_replaying:
                    msg_evt = AgentMessageEvent(content=[TextBlock(text=text)])
                    await _emit(bus, db, session_id, msg_evt)

                conversation.append({"role": "assistant", "content": text or ""})
                return text or ""

            # -- Emit text blocks before tool calls --
            if not proxy.is_replaying:
                for block in content_blocks:
                    if block["type"] == "text" and block.get("text"):
                        msg_evt = AgentMessageEvent(
                            content=[TextBlock(text=block["text"])]
                        )
                        await _emit(bus, db, session_id, msg_evt)

            # -- Add assistant message with tool_calls to history --
            conversation.append(_build_assistant_message(content_blocks))

            # -- Execute tool calls --
            for block in tool_use_blocks:
                tool_name = block["name"]
                tool_input = block["input"]
                tool_id = block["id"]

                if tool_name in custom_tool_names:
                    # Protocol layering: server layer event + kernel syscall
                    if not proxy.is_replaying:
                        custom_evt = AgentCustomToolUseEvent(
                            name=tool_name, input=tool_input
                        )
                        custom_evt.id = tool_id
                        await _emit(bus, db, session_id, custom_evt)

                    result = await proxy.syscall(
                        "external_input",
                        {"payload": {"tool_name": tool_name, "tool_id": tool_id}},
                    )
                else:
                    # Builtin tool via kernel syscall
                    if not proxy.is_replaying:
                        tool_evt = AgentToolUseEvent(
                            name=tool_name,
                            input=tool_input,
                            evaluated_permission="allow",
                        )
                        tool_evt.id = tool_id
                        await _emit(bus, db, session_id, tool_evt)

                    result = await proxy.syscall(tool_name, tool_input)

                    if not proxy.is_replaying:
                        result_evt = AgentToolResultEvent(
                            tool_use_id=tool_id,
                            content=[TextBlock(text=str(result))],
                        )
                        await _emit(bus, db, session_id, result_evt)

                # Add tool result to conversation for next LLM turn
                conversation.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": str(result),
                    }
                )

        return "max_turns_reached"

    agent_fn.__name__ = "anthropic_agent_loop"
    return agent_fn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _emit(bus: EventBus, db: AsyncSession, session_id: str, event: Any) -> None:
    """Publish an event to the bus and persist it."""
    await bus.publish(event)
    await repo.store_event(
        db,
        session_id=session_id,
        event_id=event.id,
        event_type=event.type,
        data=event.model_dump(exclude_none=True),
    )


def _extract_text(content_blocks: list[dict[str, Any]]) -> str:
    parts = [b["text"] for b in content_blocks if b["type"] == "text" and b.get("text")]
    return "\n".join(parts)


def _build_assistant_message(
    content_blocks: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build OpenAI-format assistant message with tool_calls."""
    tool_calls = []
    for b in content_blocks:
        if b["type"] == "tool_use":
            tool_calls.append(
                {
                    "id": b["id"],
                    "type": "function",
                    "function": {
                        "name": b["name"],
                        "arguments": json.dumps(b["input"]),
                    },
                }
            )

    text_content = " ".join(
        b["text"] for b in content_blocks if b["type"] == "text" and b.get("text")
    )
    msg: dict[str, Any] = {
        "role": "assistant",
        "content": text_content or None,
    }
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def _build_llm_messages(
    agent: AgentResponse, conversation: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Build the messages array for an LLM call."""
    messages: list[dict[str, Any]] = []
    if agent.system:
        messages.append({"role": "system", "content": agent.system})
    messages.extend(conversation)
    return messages


def _build_tools_for_llm(agent: AgentResponse) -> list[dict[str, Any]]:
    """Convert agent tool definitions to OpenAI function-calling format."""
    tools: list[dict[str, Any]] = []

    for tool_def in agent.tools:
        td = tool_def if isinstance(tool_def, dict) else tool_def.model_dump()
        tool_type = td.get("type", "")

        if tool_type == "agent_toolset_20260401":
            default_config = td.get("default_config") or {}
            default_enabled = (
                default_config.get("enabled", True)
                if isinstance(default_config, dict)
                else True
            )
            per_tool: dict[str, bool] = {}
            for cfg in td.get("configs") or []:
                if isinstance(cfg, dict):
                    per_tool[cfg["name"]] = cfg.get("enabled", True)
                else:
                    per_tool[cfg.name] = cfg.enabled

            for name, spec in BUILTIN_TOOL_SPECS.items():
                if per_tool.get(name, default_enabled):
                    tools.append(spec)

        elif tool_type == "custom":
            name = td.get("name")
            desc = td.get("description", "")
            schema = td.get("input_schema", {})
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


def _get_custom_tool_names(agent: AgentResponse) -> set[str]:
    """Extract the set of custom tool names from agent config."""
    names: set[str] = set()
    for tool_def in agent.tools:
        td = tool_def if isinstance(tool_def, dict) else tool_def.model_dump()
        if td.get("type") == "custom":
            name = td.get("name")
            if name:
                names.add(name)
    return names
