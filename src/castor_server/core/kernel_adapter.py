"""Kernel adapter: bridges Anthropic agent config to Castor kernel instances.

This module sits at the Ring 1 ↔ Ring 2 boundary. It translates an
AgentResponse (Anthropic API model) into a configured Castor kernel
instance with the correct tools, LLM callable, and destructive flags.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from castor.core import Castor

from castor_server.core.llm_adapter import litellm_chat_for_kernel
from castor_server.core.mcp_runtime import call_mcp_tool
from castor_server.models.agents import AgentResponse, AgentToolset, ModelConfig
from castor_server.tools.builtin import (
    BUILTIN_TOOL_NAMES,
    BUILTIN_TOOLS,
    DESTRUCTIVE_TOOL_NAMES,
    external_input,
)


async def mcp_call(
    mcp_server_url: str = "",
    tool_name: str = "",
    arguments: dict[str, Any] | None = None,
) -> str:
    """Kernel-registered tool that proxies a single MCP tool invocation.

    All MCP tool calls in agent_fn route through this single syscall so
    that the kernel's syscall_log captures them for replay. Each call
    opens a transient streamable HTTP session to the MCP server.
    """
    return await call_mcp_tool(
        mcp_server_url=mcp_server_url,
        tool_name=tool_name,
        arguments=arguments or {},
    )


def build_kernel_for_agent(agent: AgentResponse) -> Castor:
    """Create a Castor kernel configured for the given agent.

    Reads the agent's tool definitions and permission policies to determine:
    - Which builtin tools are enabled
    - Which tools require HITL (always_ask permission → destructive in kernel)
    - The LLM callable for inference syscalls
    """
    enabled_tools = _resolve_enabled_tools(agent)
    hitl_tool_names = resolve_hitl_tools(agent)

    model_id = agent.model.id if isinstance(agent.model, ModelConfig) else agent.model

    llm_callable = _make_llm_callable(model_id)

    return Castor(
        tools=[*enabled_tools, external_input, mcp_call],
        destructive=hitl_tool_names,
        llm=llm_callable,
        llm_cost=0.03,
        llm_resource="api_usd",
    )


def _resolve_enabled_tools(agent: AgentResponse) -> list[Callable]:
    """Determine which builtin tools are enabled based on agent config."""
    tools_by_name = {fn.__name__: fn for fn in BUILTIN_TOOLS}
    enabled: list[Callable] = []

    for tool_def in agent.tools:
        if not isinstance(tool_def, AgentToolset):
            td = tool_def if isinstance(tool_def, dict) else tool_def.model_dump()
            if td.get("type") != "agent_toolset_20260401":
                continue
        else:
            td = tool_def.model_dump()

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

        for name in BUILTIN_TOOL_NAMES:
            if per_tool.get(name, default_enabled):
                enabled.append(tools_by_name[name])

    return enabled


def resolve_hitl_tools(agent: AgentResponse) -> list[str]:
    """Determine which tools should trigger HITL suspension.

    A tool triggers HITL if:
    - It is in DESTRUCTIVE_TOOL_NAMES, OR
    - Its permission_policy is 'always_ask' in the agent config
    """
    hitl_names: set[str] = set(DESTRUCTIVE_TOOL_NAMES)

    for tool_def in agent.tools:
        if not isinstance(tool_def, AgentToolset):
            td = tool_def if isinstance(tool_def, dict) else tool_def.model_dump()
            if td.get("type") != "agent_toolset_20260401":
                continue
        else:
            td = tool_def.model_dump()

        default_config = td.get("default_config") or {}
        default_pp = (
            default_config.get("permission_policy")
            if isinstance(default_config, dict)
            else None
        )
        if default_pp and default_pp.get("type") == "always_ask":
            hitl_names.update(BUILTIN_TOOL_NAMES)

        for cfg in td.get("configs") or []:
            if isinstance(cfg, dict):
                pp = cfg.get("permission_policy")
                name = cfg.get("name")
            else:
                pp = cfg.permission_policy
                name = cfg.name
            if pp:
                pp_dict = pp if isinstance(pp, dict) else pp.model_dump()
                if pp_dict.get("type") == "always_ask" and name:
                    hitl_names.add(name)

    return sorted(hitl_names)


def _make_llm_callable(model_id: str) -> Callable[..., Any]:
    """Create an async LLM callable for the kernel's LLMSyscall."""

    async def llm_callable(
        model: str = "",
        messages: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return await litellm_chat_for_kernel(
            model=model or model_id,
            messages=messages or [],
            tools=tools,
        )

    return llm_callable
