"""MCP toolset runtime — discover and call tools on remote MCP servers.

Connects to URL-based MCP servers using the official `mcp` Python SDK
(streamable HTTP transport). Per-call connection: open → list/call →
close. This is slower than holding a persistent session but is much
simpler to integrate with the kernel's checkpoint/replay model and is
acceptable for the typical Anthropic-style usage where tools are called
once or twice per turn.

For each call we:
  1. Open a streamable HTTP transport to ``mcp_server.url``
  2. Initialize an MCP ClientSession
  3. Call ``list_tools()`` (for discovery) or ``call_tool(name, args)``
  4. Close everything

The design surface kept here is intentionally minimal — auth, vault,
SSE-based persistent sessions, and per-tool permission filtering are all
out of scope for the MVP and can be layered on later.
"""

from __future__ import annotations

import contextvars
import logging
from typing import Any

from castor_server.models.agents import AgentResponse, MCPServer

logger = logging.getLogger("castor_server.mcp_runtime")

# Contextvar holding a mapping of mcp_server_url → auth headers.
# Set by session_manager before kernel.run() based on vault credentials.
_mcp_auth: contextvars.ContextVar[dict[str, dict[str, str]]] = contextvars.ContextVar(
    "mcp_auth", default={}
)


def set_mcp_auth(auth_map: dict[str, dict[str, str]]) -> contextvars.Token:
    """Set per-URL auth headers for MCP calls. Returns a reset token."""
    return _mcp_auth.set(auth_map)


def clear_mcp_auth(token: contextvars.Token) -> None:
    _mcp_auth.reset(token)


# Tool spec format used by the LLM (OpenAI function-calling shape) plus
# our own metadata to route the call back to the right MCP server.
McpToolSpec = dict[str, Any]


async def discover_mcp_tools(
    mcp_servers: list[MCPServer | dict[str, Any]],
) -> dict[str, list[McpToolSpec]]:
    """Connect to each MCP server and list its tools.

    Returns a dict mapping ``mcp_server_name`` → list of OpenAI-format
    tool specs (with the original MCP tool name and an
    ``_mcp_server_name`` annotation so callers can route back).

    Servers that fail to connect or list are logged and skipped — the
    rest of the agent loop continues without them.
    """
    result: dict[str, list[McpToolSpec]] = {}
    for server in mcp_servers:
        sd = server if isinstance(server, dict) else server.model_dump()
        name = sd.get("name", "")
        url = sd.get("url", "")
        if not name or not url:
            continue

        try:
            tools = await _list_server_tools(url)
        except Exception as e:
            logger.warning(
                "mcp_discover_failed name=%s url=%s err=%s: %s",
                name,
                url,
                type(e).__name__,
                e,
            )
            continue

        specs: list[McpToolSpec] = []
        for tool in tools:
            specs.append(_mcp_tool_to_openai_spec(tool, mcp_server_name=name))
        result[name] = specs
        logger.info("mcp_discovered server=%s tools=%d", name, len(specs))
    return result


async def _list_server_tools(url: str) -> list[Any]:
    """Open a transient MCP session and call list_tools()."""
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async with streamablehttp_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            response = await session.list_tools()
            return list(response.tools)


def _mcp_tool_to_openai_spec(tool: Any, *, mcp_server_name: str) -> McpToolSpec:
    """Convert an MCP Tool to the OpenAI function-calling shape.

    Adds ``_mcp_server_name`` to the spec for our own routing — LiteLLM
    ignores keys it doesn't recognize so this is safe to send to the LLM.
    """
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.inputSchema or {"type": "object", "properties": {}},
        },
        "_mcp_server_name": mcp_server_name,
    }


async def call_mcp_tool(
    *,
    mcp_server_url: str,
    tool_name: str,
    arguments: dict[str, Any],
    auth_headers: dict[str, str] | None = None,
) -> str:
    """Open a transient MCP session and call ``tool_name`` with ``arguments``.

    If ``auth_headers`` is provided, they are passed to the transport as
    extra HTTP headers (e.g. ``{"Authorization": "Bearer <token>"}``).

    Returns the tool result as a string. If the tool returns multiple
    content blocks, they are concatenated. Errors are returned as text
    prefixed with ``Error:``.
    """
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    try:
        transport_kwargs: dict[str, Any] = {}
        if auth_headers:
            transport_kwargs["headers"] = auth_headers
        async with streamablehttp_client(mcp_server_url, **transport_kwargs) as (
            read,
            write,
            _,
        ):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments)
                return _format_call_result(result)
    except Exception as e:
        logger.exception("mcp_call_failed url=%s tool=%s", mcp_server_url, tool_name)
        return f"Error: MCP call failed ({type(e).__name__}): {e}"


def _format_call_result(result: Any) -> str:
    """Convert an MCP CallToolResult to a plain string for the LLM."""
    if getattr(result, "isError", False):
        prefix = "Error: "
    else:
        prefix = ""
    parts: list[str] = []
    for block in getattr(result, "content", []) or []:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            parts.append(getattr(block, "text", ""))
        elif block_type == "image":
            parts.append("[image content]")
        else:
            parts.append(str(block))
    return prefix + "\n".join(parts) if parts else prefix + "(no content)"


def find_mcp_server_for_tool(
    mcp_tools_by_server: dict[str, list[McpToolSpec]],
    tool_name: str,
) -> str | None:
    """Find which MCP server provides a tool with the given name.

    Returns the server name, or ``None`` if no server provides the tool.
    """
    for server_name, specs in mcp_tools_by_server.items():
        for spec in specs:
            if spec.get("function", {}).get("name") == tool_name:
                return server_name
    return None


def get_server_url(agent: AgentResponse, mcp_server_name: str) -> str | None:
    """Look up the URL for an MCP server by its name in the agent config."""
    for server in agent.mcp_servers:
        sd = server if isinstance(server, dict) else server.model_dump()
        if sd.get("name") == mcp_server_name:
            return sd.get("url")
    return None
