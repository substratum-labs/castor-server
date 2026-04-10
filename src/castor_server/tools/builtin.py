"""Built-in tools matching agent_toolset_20260401.

Each tool is an async function with typed parameters. The Castor kernel
registers them via ToolMetadata.from_function() — we do NOT use @castor_tool
here to avoid polluting the global default_registry.

When a Roche sandbox is active (via ``set_sandbox``), tools execute inside
the sandbox. Otherwise they execute directly on the host (backward compat).

Destructive flags:
  - bash, write, edit → destructive (can modify filesystem / run arbitrary code)
  - read, glob, grep, web_fetch, web_search → non-destructive
  - external_input → non-destructive (result injected by server layer)
"""

from __future__ import annotations

import asyncio
import contextvars
import glob as globlib
import os
from pathlib import Path
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Sandbox context — set by session_manager before running the agent
# ---------------------------------------------------------------------------

_current_sandbox: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "current_sandbox", default=None
)


def set_sandbox(sandbox: Any) -> contextvars.Token:
    """Set the active sandbox for tool execution. Returns a reset token."""
    return _current_sandbox.set(sandbox)


def clear_sandbox(token: contextvars.Token) -> None:
    """Reset sandbox to previous value."""
    _current_sandbox.reset(token)


async def _sandbox_exec(command: str, timeout: int = 120) -> tuple[int, str, str]:
    """Execute a command in the active sandbox. Returns (exit_code, stdout, stderr)."""
    sandbox = _current_sandbox.get()
    result = await sandbox.exec(["bash", "-c", command], timeout_secs=timeout)
    return result.exit_code, result.stdout, result.stderr


# ---------------------------------------------------------------------------
# Tool implementations (typed parameters, no dict unpacking)
# ---------------------------------------------------------------------------


async def bash(command: str, timeout: int = 120) -> str:
    """Execute a bash command in a shell session."""
    if _current_sandbox.get() is not None:
        exit_code, stdout, stderr = await _sandbox_exec(command, timeout)
        output = stdout + stderr
        if exit_code != 0:
            return f"Exit code: {exit_code}\n{output}"
        return output

    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env={**os.environ, "TERM": "dumb"},
    )
    try:
        stdout_bytes, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        proc.kill()
        return f"Command timed out after {timeout}s"

    output = stdout_bytes.decode("utf-8", errors="replace")
    exit_code = proc.returncode
    if exit_code != 0:
        return f"Exit code: {exit_code}\n{output}"
    return output


async def read(file_path: str, offset: int = 0, limit: int | None = None) -> str:
    """Read a file from the filesystem."""
    if _current_sandbox.get() is not None:
        start = offset + 1
        if limit is not None:
            end = str(offset + limit)
        else:
            end = "$"
        cmd = f"awk 'NR>={start} && NR<={end}' '{file_path}'"
        cmd = f"{cmd} | awk '{{printf \"%d\\t%s\\n\", NR+{offset}, $0}}'"
        exit_code, stdout, stderr = await _sandbox_exec(cmd)
        if exit_code != 0:
            return f"Error: {stderr or stdout}"
        return stdout.rstrip()

    try:
        path = Path(file_path)
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines(keepends=True)

        if offset > 0:
            lines = lines[offset:]
        if limit is not None:
            lines = lines[:limit]

        result_lines = []
        start_line = offset if offset > 0 else 0
        for i, line in enumerate(lines, start=start_line + 1):
            result_lines.append(f"{i}\t{line.rstrip()}")
        return "\n".join(result_lines)
    except FileNotFoundError:
        return f"Error: File not found: {file_path}"
    except Exception as e:
        return f"Error reading file: {e}"


async def write(file_path: str, content: str) -> str:
    """Write content to a file."""
    if _current_sandbox.get() is not None:
        import base64

        b64 = base64.b64encode(content.encode()).decode()
        cmd = (
            f"mkdir -p \"$(dirname '{file_path}')\" && "
            f"echo '{b64}' | base64 -d > '{file_path}'"
        )
        exit_code, stdout, stderr = await _sandbox_exec(cmd)
        if exit_code != 0:
            return f"Error writing file: {stderr}"
        return f"Successfully wrote {len(content)} bytes to {file_path}"

    try:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} bytes to {file_path}"
    except Exception as e:
        return f"Error writing file: {e}"


async def edit(file_path: str, old_string: str, new_string: str) -> str:
    """Perform string replacement in a file."""
    if _current_sandbox.get() is not None:
        # Read file content from sandbox
        exit_code, text, stderr = await _sandbox_exec(f"cat '{file_path}'")
        if exit_code != 0:
            return f"Error: {stderr or 'File not found'}"

        count = text.count(old_string)
        if count == 0:
            return f"Error: old_string not found in {file_path}"
        if count > 1:
            return (
                f"Error: old_string matches {count} locations."
                " Provide more context to make it unique."
            )

        import base64

        new_text = text.replace(old_string, new_string, 1)
        b64 = base64.b64encode(new_text.encode()).decode()
        cmd = f"echo '{b64}' | base64 -d > '{file_path}'"
        exit_code, _, stderr = await _sandbox_exec(cmd)
        if exit_code != 0:
            return f"Error editing file: {stderr}"
        return f"Successfully edited {file_path}"

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
    if _current_sandbox.get() is not None:
        cmd = f"find {path} -path '{os.path.join(path, pattern)}' | sort | head -500"
        exit_code, stdout, stderr = await _sandbox_exec(cmd)
        if not stdout.strip():
            return "No files matched the pattern."
        return stdout.rstrip()

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
    if _current_sandbox.get() is not None:
        cmd = "grep -rn --color=never"
        if include:
            cmd += f" --include '{include}'"
        cmd += f" '{pattern}' {path} | head -200"
        exit_code, stdout, stderr = await _sandbox_exec(cmd, timeout=30)
        if not stdout.strip():
            return "No matches found."
        return stdout.rstrip()

    try:
        cmd_parts = ["grep", "-rn", "--color=never"]
        if include:
            cmd_parts.extend(["--include", include])
        cmd_parts.extend([pattern, path])

        proc = await asyncio.create_subprocess_exec(
            *cmd_parts,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=30
        )
        output = stdout_bytes.decode("utf-8", errors="replace")
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
    if _current_sandbox.get() is not None:
        cmd = f"curl -sL -m 30 '{url}' | head -c 100000"
        exit_code, stdout, stderr = await _sandbox_exec(cmd, timeout=35)
        if exit_code != 0:
            return f"Error fetching URL: {stderr}"
        return stdout

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


# ---------------------------------------------------------------------------
# OpenAI function-calling format specs (for LLM tool_choice)
# ---------------------------------------------------------------------------

BUILTIN_TOOL_SPECS: dict[str, dict[str, Any]] = {
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
