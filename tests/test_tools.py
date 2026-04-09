"""Tests for built-in tool implementations."""

from __future__ import annotations

import os
import tempfile

import pytest

from castor_server.tools.builtin import execute_builtin_tool


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
