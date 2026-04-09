"""Built-in tools matching agent_toolset_20260401.

Each tool is implemented as an async function. In production, these execute
inside the session's working directory / sandbox.
"""

from __future__ import annotations

import asyncio
import glob as globlib
import os
from pathlib import Path
from typing import Any

import httpx

BUILTIN_TOOL_NAMES = frozenset(
    ["bash", "read", "write", "edit", "glob", "grep", "web_fetch", "web_search"]
)


async def execute_builtin_tool(name: str, params: dict[str, Any]) -> str:
    """Dispatch to the appropriate built-in tool implementation."""
    if name not in _TOOL_DISPATCH:
        raise ValueError(f"Unknown built-in tool: {name}")
    return await _TOOL_DISPATCH[name](params)


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


async def _tool_bash(params: dict[str, Any]) -> str:
    command = params["command"]
    timeout = params.get("timeout", 120)

    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env={**os.environ, "TERM": "dumb"},
    )
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        proc.kill()
        return f"Command timed out after {timeout}s"

    output = stdout.decode("utf-8", errors="replace")
    exit_code = proc.returncode
    if exit_code != 0:
        return f"Exit code: {exit_code}\n{output}"
    return output


async def _tool_read(params: dict[str, Any]) -> str:
    file_path = params["file_path"]
    offset = params.get("offset", 0)
    limit = params.get("limit")

    try:
        path = Path(file_path)
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines(keepends=True)

        if offset > 0:
            lines = lines[offset:]
        if limit is not None:
            lines = lines[:limit]

        # Format with line numbers
        result_lines = []
        start = offset if offset > 0 else 0
        for i, line in enumerate(lines, start=start + 1):
            result_lines.append(f"{i}\t{line.rstrip()}")
        return "\n".join(result_lines)
    except FileNotFoundError:
        return f"Error: File not found: {file_path}"
    except Exception as e:
        return f"Error reading file: {e}"


async def _tool_write(params: dict[str, Any]) -> str:
    file_path = params["file_path"]
    content = params["content"]

    try:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} bytes to {file_path}"
    except Exception as e:
        return f"Error writing file: {e}"


async def _tool_edit(params: dict[str, Any]) -> str:
    file_path = params["file_path"]
    old_string = params["old_string"]
    new_string = params["new_string"]

    try:
        path = Path(file_path)
        text = path.read_text(encoding="utf-8")

        count = text.count(old_string)
        if count == 0:
            return f"Error: old_string not found in {file_path}"
        if count > 1:
            return (
                f"Error: old_string matches {count} locations."
                " Provide more context to make it unique."
            )

        new_text = text.replace(old_string, new_string, 1)
        path.write_text(new_text, encoding="utf-8")
        return f"Successfully edited {file_path}"
    except FileNotFoundError:
        return f"Error: File not found: {file_path}"
    except Exception as e:
        return f"Error editing file: {e}"


async def _tool_glob(params: dict[str, Any]) -> str:
    pattern = params["pattern"]
    search_path = params.get("path", ".")

    try:
        full_pattern = os.path.join(search_path, pattern)
        matches = sorted(globlib.glob(full_pattern, recursive=True))
        if not matches:
            return "No files matched the pattern."
        return "\n".join(matches[:500])
    except Exception as e:
        return f"Error: {e}"


async def _tool_grep(params: dict[str, Any]) -> str:
    pattern = params["pattern"]
    search_path = params.get("path", ".")
    include = params.get("include")

    try:
        cmd = ["grep", "-rn", "--color=never"]
        if include:
            cmd.extend(["--include", include])
        cmd.extend([pattern, search_path])

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        output = stdout.decode("utf-8", errors="replace")
        if not output:
            return "No matches found."
        # Limit output
        lines = output.splitlines()
        if len(lines) > 200:
            return "\n".join(lines[:200]) + f"\n... ({len(lines) - 200} more lines)"
        return output
    except TimeoutError:
        return "Error: grep timed out"
    except Exception as e:
        return f"Error: {e}"


async def _tool_web_fetch(params: dict[str, Any]) -> str:
    url = params["url"]
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
            resp = await client.get(url)
            content_type = resp.headers.get("content-type", "")
            is_text = any(t in content_type for t in ("text", "json", "xml"))
            if is_text:
                text = resp.text
                # Limit to ~100KB
                if len(text) > 100_000:
                    text = text[:100_000] + "\n... (truncated)"
                return text
            else:
                return f"Binary content ({content_type}), {len(resp.content)} bytes"
    except Exception as e:
        return f"Error fetching URL: {e}"


async def _tool_web_search(params: dict[str, Any]) -> str:
    query = params["query"]
    # Web search requires an external search API.
    # Return a stub response indicating the tool is available but needs configuration.
    return (
        f"Web search for: {query}\n"
        "Note: Web search requires configuration of a search API backend "
        "(e.g., SerpAPI, Tavily, or similar). "
        "Set CASTOR_SEARCH_API_KEY and CASTOR_SEARCH_PROVIDER in environment."
    )


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_TOOL_DISPATCH: dict[str, Any] = {
    "bash": _tool_bash,
    "read": _tool_read,
    "write": _tool_write,
    "edit": _tool_edit,
    "glob": _tool_glob,
    "grep": _tool_grep,
    "web_fetch": _tool_web_fetch,
    "web_search": _tool_web_search,
}
