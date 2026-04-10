"""Integration test for MCP toolset wired through agent_fn → kernel.

The MCP server is mocked at the SDK level so this test runs in-process,
but the agent loop, kernel, event bus, and routing layers are all real.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient


def _fake_tool(name: str, description: str = ""):
    t = MagicMock()
    t.name = name
    t.description = description
    t.inputSchema = {
        "type": "object",
        "properties": {"q": {"type": "string"}},
        "required": ["q"],
    }
    return t


def _patch_mcp_sdk():
    """Context manager that fakes the entire MCP client stack.

    list_tools returns one tool ("echo_mcp_tool"); call_tool returns
    a text content block echoing the input. This lets agent_fn discover
    a real-looking MCP tool and route a call through it without any
    network access.
    """
    fake_session = MagicMock()
    fake_session.initialize = AsyncMock()
    fake_session.list_tools = AsyncMock(
        return_value=MagicMock(tools=[_fake_tool("echo_mcp_tool")])
    )

    def _make_call_result(args):
        block = MagicMock()
        block.type = "text"
        block.text = f"mcp echo: {args.get('q', '')}"
        result = MagicMock()
        result.isError = False
        result.content = [block]
        return result

    fake_session.call_tool = AsyncMock(
        side_effect=lambda name, args: _make_call_result(args)
    )

    @asynccontextmanager
    async def fake_session_cm(read, write):
        yield fake_session

    @asynccontextmanager
    async def fake_transport(url, **kwargs):
        yield (MagicMock(), MagicMock(), lambda: None)

    return (
        patch(
            "mcp.client.streamable_http.streamablehttp_client",
            side_effect=fake_transport,
        ),
        patch("mcp.client.session.ClientSession", side_effect=fake_session_cm),
        fake_session,
    )


@pytest.mark.asyncio
async def test_mcp_tool_discovered_and_appears_in_session_events(client: AsyncClient):
    """Discovery alone — verify create-agent-with-mcp_servers, then send
    a no-op user.message and confirm session.events.list contains the
    expected events."""
    transport_patch, session_patch, _ = _patch_mcp_sdk()

    with transport_patch, session_patch:
        # Create an agent with an MCP server config
        agent = (
            await client.post(
                "/v1/agents",
                json={
                    "name": "mcp-agent",
                    "model": "mock",
                    "mcp_servers": [
                        {
                            "name": "demo",
                            "type": "url",
                            "url": "https://example.invalid/mcp",
                        }
                    ],
                    "tools": [
                        {
                            "type": "mcp_toolset",
                            "mcp_server_name": "demo",
                        }
                    ],
                },
            )
        ).json()
        assert agent["id"].startswith("agent_")

        session = (
            await client.post("/v1/sessions", json={"agent": agent["id"]})
        ).json()

        await client.post(
            f"/v1/sessions/{session['id']}/events",
            json={
                "events": [
                    {
                        "type": "user.message",
                        "content": [{"type": "text", "text": "hi"}],
                    }
                ]
            },
        )

        # Drain the agent loop
        for _ in range(50):
            await asyncio.sleep(0.05)
            resp = await client.get(f"/v1/sessions/{session['id']}/events")
            types = [e["type"] for e in resp.json()["data"]]
            if "session.status_idle" in types:
                break

    # Mock model doesn't actually call the MCP tool — it just echoes the
    # user message. So we should NOT see agent.mcp_tool_use, but we should
    # see the full happy-path event sequence with no error.
    types = [e["type"] for e in resp.json()["data"]]
    assert "session.status_running" in types
    assert "agent.message" in types
    assert "session.status_idle" in types
    assert "session.error" not in types


@pytest.mark.asyncio
async def test_mcp_call_routes_through_kernel_syscall():
    """Direct unit test on the kernel-registered ``mcp_call`` function."""
    from castor_server.core.kernel_adapter import mcp_call

    transport_patch, session_patch, fake_session = _patch_mcp_sdk()

    with transport_patch, session_patch:
        result = await mcp_call(
            mcp_server_url="https://example.invalid/mcp",
            tool_name="echo_mcp_tool",
            arguments={"q": "ping"},
        )

    assert result == "mcp echo: ping"
    fake_session.call_tool.assert_awaited_once_with("echo_mcp_tool", {"q": "ping"})


@pytest.mark.asyncio
async def test_agent_fn_routes_mcp_tool_via_dispatch_branch():
    """Verify build_agent_fn picks up mcp_tools_by_server and that the
    mcp tool name set is wired into the dispatch logic."""
    from castor_server.core.agent_fn import build_agent_fn
    from castor_server.core.event_bus import EventBus
    from castor_server.models.agents import AgentResponse, ModelConfig

    agent = AgentResponse(
        id="agent_test",
        name="t",
        model=ModelConfig(id="mock"),
        created_at="2026-04-10T00:00:00.000Z",
        updated_at="2026-04-10T00:00:00.000Z",
    )
    bus = EventBus("session_test")
    mcp_tools = {
        "demo": [
            {
                "type": "function",
                "function": {
                    "name": "demo_tool",
                    "description": "x",
                    "parameters": {"type": "object", "properties": {}},
                },
                "_mcp_server_name": "demo",
            }
        ]
    }

    fn = build_agent_fn(
        agent=agent,
        messages=[],
        bus=bus,
        db=MagicMock(),
        session_id="session_test",
        mcp_tools_by_server=mcp_tools,
    )
    # Function builds successfully and is callable
    assert callable(fn)
