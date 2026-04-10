"""Tests for the MCP toolset runtime.

These mock the official ``mcp`` Python SDK so we don't need a real MCP
server to verify the discovery + dispatch logic.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from castor_server.core.mcp_runtime import (
    _format_call_result,
    _mcp_tool_to_openai_spec,
    call_mcp_tool,
    discover_mcp_tools,
    find_mcp_server_for_tool,
    get_server_url,
)
from castor_server.models.agents import AgentResponse, MCPServer, ModelConfig

# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_mcp_tool_to_openai_spec_includes_routing_key():
    fake_tool = MagicMock()
    fake_tool.name = "github_create_issue"
    fake_tool.description = "Create a GitHub issue"
    fake_tool.inputSchema = {
        "type": "object",
        "properties": {"title": {"type": "string"}},
        "required": ["title"],
    }

    spec = _mcp_tool_to_openai_spec(fake_tool, mcp_server_name="github")
    assert spec["type"] == "function"
    assert spec["function"]["name"] == "github_create_issue"
    assert spec["function"]["description"] == "Create a GitHub issue"
    assert spec["function"]["parameters"]["properties"]["title"]["type"] == "string"
    assert spec["_mcp_server_name"] == "github"


def test_mcp_tool_to_openai_spec_handles_missing_schema():
    fake_tool = MagicMock()
    fake_tool.name = "noop"
    fake_tool.description = None
    fake_tool.inputSchema = None

    spec = _mcp_tool_to_openai_spec(fake_tool, mcp_server_name="x")
    assert spec["function"]["description"] == ""
    assert spec["function"]["parameters"] == {"type": "object", "properties": {}}


def test_format_call_result_text_content():
    fake_block = MagicMock()
    fake_block.type = "text"
    fake_block.text = "the answer is 42"
    fake_result = MagicMock()
    fake_result.isError = False
    fake_result.content = [fake_block]

    out = _format_call_result(fake_result)
    assert out == "the answer is 42"


def test_format_call_result_error():
    fake_block = MagicMock()
    fake_block.type = "text"
    fake_block.text = "boom"
    fake_result = MagicMock()
    fake_result.isError = True
    fake_result.content = [fake_block]

    out = _format_call_result(fake_result)
    assert out.startswith("Error: ")
    assert "boom" in out


def test_format_call_result_multiple_blocks():
    block1 = MagicMock()
    block1.type = "text"
    block1.text = "line one"
    block2 = MagicMock()
    block2.type = "text"
    block2.text = "line two"
    fake_result = MagicMock()
    fake_result.isError = False
    fake_result.content = [block1, block2]

    out = _format_call_result(fake_result)
    assert "line one" in out
    assert "line two" in out


def test_format_call_result_empty():
    fake_result = MagicMock()
    fake_result.isError = False
    fake_result.content = []
    out = _format_call_result(fake_result)
    assert "no content" in out


def test_find_mcp_server_for_tool():
    tools_by_server = {
        "github": [
            {"function": {"name": "create_issue"}, "_mcp_server_name": "github"},
            {"function": {"name": "list_repos"}, "_mcp_server_name": "github"},
        ],
        "slack": [
            {"function": {"name": "post_message"}, "_mcp_server_name": "slack"},
        ],
    }
    assert find_mcp_server_for_tool(tools_by_server, "create_issue") == "github"
    assert find_mcp_server_for_tool(tools_by_server, "post_message") == "slack"
    assert find_mcp_server_for_tool(tools_by_server, "no_such_tool") is None


def test_get_server_url_resolves_by_name():
    agent = AgentResponse(
        id="agent_x",
        name="t",
        model=ModelConfig(id="mock"),
        mcp_servers=[
            MCPServer(name="github", type="url", url="https://example.com/mcp"),
            MCPServer(name="slack", type="url", url="https://slack.example/mcp"),
        ],
        created_at="2026-04-10T00:00:00.000Z",
        updated_at="2026-04-10T00:00:00.000Z",
    )
    assert get_server_url(agent, "github") == "https://example.com/mcp"
    assert get_server_url(agent, "slack") == "https://slack.example/mcp"
    assert get_server_url(agent, "missing") is None


# ---------------------------------------------------------------------------
# Discovery + call (with mocked MCP SDK)
# ---------------------------------------------------------------------------


def _make_fake_tool(name: str, description: str = "", schema: dict | None = None):
    t = MagicMock()
    t.name = name
    t.description = description
    t.inputSchema = schema or {"type": "object", "properties": {}}
    return t


@pytest.mark.asyncio
async def test_discover_mcp_tools_lists_each_server():
    fake_session = MagicMock()
    fake_session.initialize = AsyncMock()
    fake_session.list_tools = AsyncMock(
        return_value=MagicMock(
            tools=[_make_fake_tool("greet"), _make_fake_tool("ping")]
        )
    )

    @asynccontextmanager
    async def fake_session_cm(read, write):
        yield fake_session

    @asynccontextmanager
    async def fake_transport(url, **kwargs):
        yield (MagicMock(), MagicMock(), lambda: None)

    with (
        patch(
            "mcp.client.streamable_http.streamablehttp_client",
            side_effect=fake_transport,
        ),
        patch("mcp.client.session.ClientSession", side_effect=fake_session_cm),
    ):
        result = await discover_mcp_tools(
            [
                MCPServer(name="hello", type="url", url="https://example.com/hello"),
            ]
        )

    assert "hello" in result
    names = [t["function"]["name"] for t in result["hello"]]
    assert names == ["greet", "ping"]
    assert all(t["_mcp_server_name"] == "hello" for t in result["hello"])


@pytest.mark.asyncio
async def test_discover_mcp_tools_skips_failed_server(caplog):
    @asynccontextmanager
    async def failing_transport(url, **kwargs):
        raise RuntimeError("connection refused")
        yield  # pragma: no cover

    with (
        patch(
            "mcp.client.streamable_http.streamablehttp_client",
            side_effect=failing_transport,
        ),
        caplog.at_level("WARNING"),
    ):
        result = await discover_mcp_tools(
            [
                MCPServer(name="broken", type="url", url="https://broken.example"),
            ]
        )

    assert result == {}
    assert any("mcp_discover_failed" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_call_mcp_tool_returns_text_content():
    fake_block = MagicMock()
    fake_block.type = "text"
    fake_block.text = "hello from MCP"
    fake_result = MagicMock()
    fake_result.isError = False
    fake_result.content = [fake_block]

    fake_session = MagicMock()
    fake_session.initialize = AsyncMock()
    fake_session.call_tool = AsyncMock(return_value=fake_result)

    @asynccontextmanager
    async def fake_session_cm(read, write):
        yield fake_session

    @asynccontextmanager
    async def fake_transport(url, **kwargs):
        yield (MagicMock(), MagicMock(), lambda: None)

    with (
        patch(
            "mcp.client.streamable_http.streamablehttp_client",
            side_effect=fake_transport,
        ),
        patch("mcp.client.session.ClientSession", side_effect=fake_session_cm),
    ):
        result = await call_mcp_tool(
            mcp_server_url="https://example.com/mcp",
            tool_name="greet",
            arguments={"name": "world"},
        )

    assert result == "hello from MCP"
    fake_session.call_tool.assert_awaited_once_with("greet", {"name": "world"})


@pytest.mark.asyncio
async def test_call_mcp_tool_returns_error_string_on_exception():
    @asynccontextmanager
    async def failing_transport(url, **kwargs):
        raise RuntimeError("connection died")
        yield  # pragma: no cover

    with patch(
        "mcp.client.streamable_http.streamablehttp_client",
        side_effect=failing_transport,
    ):
        result = await call_mcp_tool(
            mcp_server_url="https://broken.example",
            tool_name="x",
            arguments={},
        )

    assert result.startswith("Error: ")
    assert "connection died" in result
