"""Built-in tools matching agent_toolset_20260401.

Each tool is an async function with typed parameters. The Castor kernel
registers them via ToolMetadata.from_function() — we do NOT use @castor_tool
here to avoid polluting the global default_registry.

Destructive flags:
  - bash, write, edit → destructive (can modify filesystem / run arbitrary code)
  - read, glob, grep, web_fetch, web_search → non-destructive
  - external_input → non-destructive (result injected by server layer)
"""

from __future__ import annotations

import asyncio
import glob as globlib
import os
from pathlib import Path
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Tool implementations (typed parameters, no dict unpacking)
# ---------------------------------------------------------------------------


async def bash(command: str, timeout: int = 120) -> str:
    """Execute a bash command in a shell session."""
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


async def read(file_path: str, offset: int = 0, limit: int | None = None) -> str:
    """Read a file from the filesystem."""
    try:
        path = Path(file_path)
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines(keepends=True)

        if offset > 0:
            lines = lines[offset:]
        if limit is not None:
            lines = lines[:limit]

        result_lines = []
        start = offset if offset > 0 else 0
        for i, line in enumerate(lines, start=start + 1):
            result_lines.append(f"{i}\t{line.rstrip()}")
        return "\n".join(result_lines)
    except FileNotFoundError:
        return f"Error: File not found: {file_path}"
    except Exception as e:
        return f"Error reading file: {e}"


async def write(file_path: str, content: str) -> str:
    """Write content to a file."""
    try:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} bytes to {file_path}"
    except Exception as e:
        return f"Error writing file: {e}"


async def edit(file_path: str, old_string: str, new_string: str) -> str:
    """Perform string replacement in a file."""
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


async def glob(pattern: str, path: str = ".") -> str:
    """Find files matching a glob pattern."""
    try:
        full_pattern = os.path.join(path, pattern)
        matches = sorted(globlib.glob(full_pattern, recursive=True))
        if not matches:
            return "No files matched the pattern."
        return "\n".join(matches[:500])
    except Exception as e:
        return f"Error: {e}"


async def grep(pattern: str, path: str = ".", include: str | None = None) -> str:
    """Search file contents using regex."""
    try:
        cmd = ["grep", "-rn", "--color=never"]
        if include:
            cmd.extend(["--include", include])
        cmd.extend([pattern, path])

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        output = stdout.decode("utf-8", errors="replace")
        if not output:
            return "No matches found."
        lines = output.splitlines()
        if len(lines) > 200:
            return "\n".join(lines[:200]) + f"\n... ({len(lines) - 200} more lines)"
        return output
    except TimeoutError:
        return "Error: grep timed out"
    except Exception as e:
        return f"Error: {e}"


async def web_fetch(url: str) -> str:
    """Fetch content from a URL."""
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
            resp = await client.get(url)
            content_type = resp.headers.get("content-type", "")
            is_text = any(t in content_type for t in ("text", "json", "xml"))
            if is_text:
                text = resp.text
                if len(text) > 100_000:
                    text = text[:100_000] + "\n... (truncated)"
                return text
            else:
                return f"Binary content ({content_type}), {len(resp.content)} bytes"
    except Exception as e:
        return f"Error fetching URL: {e}"


async def web_search(query: str) -> str:
    """Search the web for information."""
    return (
        f"Web search for: {query}\n"
        "Note: Web search requires configuration of a search API backend "
        "(e.g., SerpAPI, Tavily, or similar). "
        "Set CASTOR_SEARCH_API_KEY and CASTOR_SEARCH_PROVIDER in environment."
    )


async def external_input(payload: dict[str, Any] | None = None) -> str:
    """Receive external input (e.g., custom tool results from client).

    This tool is used as the kernel-layer representation of custom tools.
    The server layer handles the full Anthropic custom tool semantics;
    the kernel only sees this simplified syscall.

    In normal execution this tool suspends for HITL (the server injects the
    client result into the checkpoint). During replay the cached result is
    returned directly.
    """
    # This body only runs if the kernel executes it directly (shouldn't
    # happen in the normal custom-tool flow where HITL suspend/inject is used).
    return ""


# ---------------------------------------------------------------------------
# Exports for kernel registration
# ---------------------------------------------------------------------------

#: Tools to pass to Castor(tools=[...]).  The kernel creates its own registry.
BUILTIN_TOOLS: list = [bash, read, write, edit, glob, grep, web_fetch, web_search]

#: Tools that should be marked destructive when registering with the kernel.
DESTRUCTIVE_TOOL_NAMES: frozenset[str] = frozenset(["bash", "write", "edit"])

BUILTIN_TOOL_NAMES: frozenset[str] = frozenset(
    ["bash", "read", "write", "edit", "glob", "grep", "web_fetch", "web_search"]
)

# ---------------------------------------------------------------------------
# Backward-compatible dispatch (used by existing tests and Phase 1 code)
# ---------------------------------------------------------------------------

_TOOL_DISPATCH: dict[str, Any] = {
    "bash": lambda params: bash(**params),
    "read": lambda params: read(**params),
    "write": lambda params: write(**params),
    "edit": lambda params: edit(**params),
    "glob": lambda params: glob(**params),
    "grep": lambda params: grep(**params),
    "web_fetch": lambda params: web_fetch(**params),
    "web_search": lambda params: web_search(**params),
}


async def execute_builtin_tool(name: str, params: dict[str, Any]) -> str:
    """Dispatch to the appropriate built-in tool implementation.

    Backward-compatible entry point: accepts a dict of params and unpacks
    them into the typed function signature.
    """
    if name not in _TOOL_DISPATCH:
        raise ValueError(f"Unknown built-in tool: {name}")
    return await _TOOL_DISPATCH[name](params)
