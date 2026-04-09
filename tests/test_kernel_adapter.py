"""Tests for kernel adapter — agent config → Castor kernel bridge."""

from __future__ import annotations

from castor_server.core.kernel_adapter import (
    _resolve_enabled_tools,
    _resolve_hitl_tools,
    build_kernel_for_agent,
)
from castor_server.models.agents import (
    AgentResponse,
    AgentToolset,
    CustomTool,
    CustomToolInputSchema,
    DefaultToolConfig,
    ModelConfig,
    PermissionPolicy,
    ToolConfig,
)
from castor_server.tools.builtin import BUILTIN_TOOL_NAMES


def _make_agent(**overrides) -> AgentResponse:
    defaults = {
        "id": "agent_test",
        "name": "Test Agent",
        "model": ModelConfig(id="claude-sonnet-4-6"),
        "tools": [AgentToolset()],
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
    }
    defaults.update(overrides)
    return AgentResponse(**defaults)


def test_resolve_all_tools_enabled_by_default():
    agent = _make_agent()
    tools = _resolve_enabled_tools(agent)
    tool_names = {fn.__name__ for fn in tools}
    assert tool_names == set(BUILTIN_TOOL_NAMES)


def test_resolve_tool_disabled():
    agent = _make_agent(
        tools=[
            AgentToolset(
                configs=[ToolConfig(name="bash", enabled=False)],
            )
        ]
    )
    tools = _resolve_enabled_tools(agent)
    tool_names = {fn.__name__ for fn in tools}
    assert "bash" not in tool_names
    assert "read" in tool_names


def test_resolve_default_disabled():
    agent = _make_agent(
        tools=[
            AgentToolset(
                default_config=DefaultToolConfig(enabled=False),
                configs=[ToolConfig(name="bash", enabled=True)],
            )
        ]
    )
    tools = _resolve_enabled_tools(agent)
    tool_names = {fn.__name__ for fn in tools}
    assert tool_names == {"bash"}


def test_hitl_tools_include_destructive_by_default():
    agent = _make_agent()
    hitl = _resolve_hitl_tools(agent)
    assert "bash" in hitl
    assert "write" in hitl
    assert "edit" in hitl


def test_hitl_tools_always_ask_per_tool():
    agent = _make_agent(
        tools=[
            AgentToolset(
                configs=[
                    ToolConfig(
                        name="read",
                        permission_policy=PermissionPolicy(type="always_ask"),
                    )
                ],
            )
        ]
    )
    hitl = _resolve_hitl_tools(agent)
    assert "read" in hitl


def test_hitl_tools_always_ask_default():
    agent = _make_agent(
        tools=[
            AgentToolset(
                default_config=DefaultToolConfig(
                    permission_policy=PermissionPolicy(type="always_ask"),
                ),
            )
        ]
    )
    hitl = _resolve_hitl_tools(agent)
    # All builtin tools should be HITL
    for name in BUILTIN_TOOL_NAMES:
        assert name in hitl


def test_build_kernel_creates_instance():
    agent = _make_agent()
    kernel = build_kernel_for_agent(agent)
    # Kernel should have a gate with registered tools
    tool_names = kernel.gate.list_tools()
    assert "bash" in tool_names
    assert "external_input" in tool_names
    # LLM syscall should be registered
    assert "llm_inference" in tool_names


def test_build_kernel_custom_tools_ignored():
    """Custom tools are not registered in kernel — only external_input is."""
    agent = _make_agent(
        tools=[
            AgentToolset(),
            CustomTool(
                name="query_db",
                description="Query the database",
                input_schema=CustomToolInputSchema(
                    properties={"sql": {"type": "string"}},
                    required=["sql"],
                ),
            ),
        ]
    )
    kernel = build_kernel_for_agent(agent)
    tool_names = kernel.gate.list_tools()
    assert "query_db" not in tool_names
    assert "external_input" in tool_names
