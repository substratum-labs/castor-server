"""Tests for built-in tool implementations."""

from __future__ import annotations

import inspect
import os
import tempfile

import pytest

from castor_server.tools.builtin import (
    BUILTIN_TOOL_NAMES,
    BUILTIN_TOOLS,
    DESTRUCTIVE_TOOL_NAMES,
    execute_builtin_tool,
    external_input,
)


@pytest.mark.asyncio
async def test_bash_echo():
    result = await execute_builtin_tool("bash", {"command": "echo hello"})
    assert "hello" in result


@pytest.mark.asyncio
async def test_bash_exit_code():
    result = await execute_builtin_tool("bash", {"command": "exit 1"})
    assert "Exit code: 1" in result


@pytest.mark.asyncio
async def test_read_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("line one\nline two\nline three\n")
        path = f.name

    try:
        result = await execute_builtin_tool("read", {"file_path": path})
        assert "1\tline one" in result
        assert "2\tline two" in result
    finally:
        os.unlink(path)


@pytest.mark.asyncio
async def test_read_file_with_offset():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("a\nb\nc\nd\ne\n")
        path = f.name

    try:
        result = await execute_builtin_tool(
            "read", {"file_path": path, "offset": 2, "limit": 2}
        )
        assert "3\tc" in result
        assert "4\td" in result
        assert "1\ta" not in result
    finally:
        os.unlink(path)


@pytest.mark.asyncio
async def test_read_file_not_found():
    result = await execute_builtin_tool("read", {"file_path": "/nonexistent"})
    assert "Error" in result


@pytest.mark.asyncio
async def test_write_file():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "output.txt")
        result = await execute_builtin_tool(
            "write", {"file_path": path, "content": "hello world"}
        )
        assert "Successfully" in result
        with open(path) as f:
            assert f.read() == "hello world"


@pytest.mark.asyncio
async def test_edit_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("foo bar baz")
        path = f.name

    try:
        result = await execute_builtin_tool(
            "edit",
            {"file_path": path, "old_string": "bar", "new_string": "qux"},
        )
        assert "Successfully" in result
        with open(path) as f:
            assert f.read() == "foo qux baz"
    finally:
        os.unlink(path)


@pytest.mark.asyncio
async def test_edit_file_not_found():
    result = await execute_builtin_tool(
        "edit",
        {
            "file_path": "/nonexistent",
            "old_string": "x",
            "new_string": "y",
        },
    )
    assert "Error" in result


@pytest.mark.asyncio
async def test_glob():
    result = await execute_builtin_tool(
        "glob", {"pattern": "*.py", "path": os.path.dirname(__file__)}
    )
    assert "test_tools.py" in result


@pytest.mark.asyncio
async def test_grep():
    result = await execute_builtin_tool(
        "grep",
        {"pattern": "def test_grep", "path": __file__},
    )
    assert "test_grep" in result


@pytest.mark.asyncio
async def test_web_fetch():
    # Test with a non-existent URL (should return error)
    result = await execute_builtin_tool(
        "web_fetch", {"url": "http://localhost:1/nonexistent"}
    )
    assert "Error" in result


@pytest.mark.asyncio
async def test_unknown_tool():
    with pytest.raises(ValueError, match="Unknown built-in tool"):
        await execute_builtin_tool("nonexistent_tool", {})


# ---------------------------------------------------------------------------
# Tool registration & metadata tests
# ---------------------------------------------------------------------------


def test_builtin_tools_list_matches_names():
    """BUILTIN_TOOLS list should contain functions matching BUILTIN_TOOL_NAMES."""
    tool_fn_names = {fn.__name__ for fn in BUILTIN_TOOLS}
    assert tool_fn_names == set(BUILTIN_TOOL_NAMES)


def test_all_tools_are_async():
    """Every builtin tool must be an async function."""
    for fn in BUILTIN_TOOLS:
        assert inspect.iscoroutinefunction(fn), f"{fn.__name__} is not async"


def test_all_tools_have_typed_params():
    """Every builtin tool should have type annotations on all params."""
    for fn in BUILTIN_TOOLS:
        sig = inspect.signature(fn)
        hints = fn.__annotations__
        for name, param in sig.parameters.items():
            assert name in hints, (
                f"{fn.__name__}() param '{name}' has no type annotation"
            )


def test_destructive_tools_subset():
    """Destructive tools should be a subset of all builtin tools."""
    assert DESTRUCTIVE_TOOL_NAMES <= BUILTIN_TOOL_NAMES


def test_destructive_tools_expected():
    """bash, write, edit should be destructive; others should not."""
    assert DESTRUCTIVE_TOOL_NAMES == {"bash", "write", "edit"}


def test_external_input_is_async():
    """external_input tool must be async."""
    assert inspect.iscoroutinefunction(external_input)


@pytest.mark.asyncio
async def test_external_input_returns_empty():
    """external_input default body returns empty string."""
    result = await external_input()
    assert result == ""


def test_kernel_metadata_generation():
    """ToolMetadata.from_function() should work on all builtin tools."""
    from castor.gate.registry import ToolMetadata

    for fn in [*BUILTIN_TOOLS, external_input]:
        meta = ToolMetadata.from_function(fn)
        assert meta.tool_name == fn.__name__
        assert meta.is_async is True
        assert "properties" in meta.input_schema or meta.input_schema.get("type")
