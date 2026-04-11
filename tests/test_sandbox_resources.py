"""Tests for resource mounting in the Roche sandbox.

These verify the mount logic without spinning up a real sandbox — the
sandbox is mocked to capture the exec() calls that get issued.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from castor_server.core.sandbox_manager import SandboxManager
from castor_server.models.environments import EnvironmentResponse


def _fake_env() -> EnvironmentResponse:
    return EnvironmentResponse(
        id="env_test",
        name="t",
        image="python:3.12-slim",
        memory=None,
        cpus=None,
        timeout_secs=300,
        network=False,
        writable=True,
        network_allowlist=[],
        metadata={},
        created_at="2026-04-10T00:00:00.000Z",
        updated_at="2026-04-10T00:00:00.000Z",
    )


def _fake_exec_result(exit_code: int = 0, stderr: str = ""):
    result = MagicMock()
    result.exit_code = exit_code
    result.stderr = stderr
    return result


@pytest.mark.asyncio
async def test_mount_github_repository_default_path():
    """A github_repository resource without mount_path defaults to /workspace/<name>."""
    sm = SandboxManager()

    fake_sandbox = MagicMock()
    fake_sandbox.id = "sbx_1"
    fake_sandbox.exec = AsyncMock(return_value=_fake_exec_result())

    fake_client = MagicMock()
    fake_client.create = AsyncMock(return_value=fake_sandbox)
    sm._client = fake_client

    resources = [
        {
            "type": "github_repository",
            "url": "https://github.com/foo/bar.git",
        }
    ]

    sandbox = await sm.get_or_create("session_1", _fake_env(), resources=resources)
    assert sandbox is fake_sandbox

    # Should have made a mkdir + git clone
    exec_calls = [c.args[0] for c in fake_sandbox.exec.call_args_list]
    assert any(c[:1] == ["mkdir"] for c in exec_calls), f"no mkdir: {exec_calls}"
    clone_call = next(
        (c for c in exec_calls if c and c[0] == "git" and c[1] == "clone"),
        None,
    )
    assert clone_call is not None, f"no git clone: {exec_calls}"
    assert clone_call[2] == "https://github.com/foo/bar.git"
    assert clone_call[3] == "/workspace/bar"


@pytest.mark.asyncio
async def test_mount_github_repository_custom_path():
    sm = SandboxManager()

    fake_sandbox = MagicMock()
    fake_sandbox.id = "sbx_2"
    fake_sandbox.exec = AsyncMock(return_value=_fake_exec_result())

    fake_client = MagicMock()
    fake_client.create = AsyncMock(return_value=fake_sandbox)
    sm._client = fake_client

    resources = [
        {
            "type": "github_repository",
            "url": "https://github.com/foo/bar",
            "mount_path": "/code/bar",
        }
    ]
    await sm.get_or_create("session_2", _fake_env(), resources=resources)

    clone_call = next(
        c.args[0]
        for c in fake_sandbox.exec.call_args_list
        if c.args[0][:2] == ["git", "clone"]
    )
    assert clone_call[3] == "/code/bar"


@pytest.mark.asyncio
async def test_mount_github_repository_with_token():
    """authorization_token is injected as x-access-token in the URL."""
    sm = SandboxManager()

    fake_sandbox = MagicMock()
    fake_sandbox.id = "sbx_3"
    fake_sandbox.exec = AsyncMock(return_value=_fake_exec_result())

    fake_client = MagicMock()
    fake_client.create = AsyncMock(return_value=fake_sandbox)
    sm._client = fake_client

    resources = [
        {
            "type": "github_repository",
            "url": "https://github.com/foo/private.git",
            "authorization_token": "ghp_secret",
        }
    ]
    await sm.get_or_create("session_3", _fake_env(), resources=resources)

    clone_call = next(
        c.args[0]
        for c in fake_sandbox.exec.call_args_list
        if c.args[0][:2] == ["git", "clone"]
    )
    # The URL should now include the token
    assert "x-access-token:ghp_secret@" in clone_call[2]
    assert "github.com/foo/private.git" in clone_call[2]


@pytest.mark.asyncio
async def test_mount_github_repository_with_checkout():
    sm = SandboxManager()

    fake_sandbox = MagicMock()
    fake_sandbox.id = "sbx_4"
    fake_sandbox.exec = AsyncMock(return_value=_fake_exec_result())

    fake_client = MagicMock()
    fake_client.create = AsyncMock(return_value=fake_sandbox)
    sm._client = fake_client

    resources = [
        {
            "type": "github_repository",
            "url": "https://github.com/foo/bar.git",
            "checkout": "feature-branch",
        }
    ]
    await sm.get_or_create("session_4", _fake_env(), resources=resources)

    # There should be a `git -C <path> checkout feature-branch` call
    exec_calls = [c.args[0] for c in fake_sandbox.exec.call_args_list]
    checkout_call = next(
        (c for c in exec_calls if "checkout" in c and "feature-branch" in c),
        None,
    )
    assert checkout_call is not None, f"no checkout call: {exec_calls}"
    assert checkout_call[:2] == ["git", "-C"]
    assert checkout_call[3:] == ["checkout", "feature-branch"]


@pytest.mark.asyncio
async def test_mount_skips_unknown_resource_type(caplog):
    sm = SandboxManager()

    fake_sandbox = MagicMock()
    fake_sandbox.id = "sbx_5"
    fake_sandbox.exec = AsyncMock(return_value=_fake_exec_result())

    fake_client = MagicMock()
    fake_client.create = AsyncMock(return_value=fake_sandbox)
    sm._client = fake_client

    resources = [{"type": "weird_thing", "url": "??"}]

    with caplog.at_level("WARNING"):
        await sm.get_or_create("session_5", _fake_env(), resources=resources)

    # No git calls should have been made
    git_calls = [
        c.args[0]
        for c in fake_sandbox.exec.call_args_list
        if c.args[0] and c.args[0][0] == "git"
    ]
    assert git_calls == []
    assert any("unknown resource type" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_file_resource_mounts_via_copy_to(tmp_path, monkeypatch):
    """A file resource looks up the metadata, then copy_to's the blob."""
    from unittest.mock import patch as mpatch

    from castor_server.models.files import FileMetadata

    # Create a real blob on disk inside tmp_path
    monkeypatch.setattr("castor_server.config.settings.files_dir", str(tmp_path))
    blob = tmp_path / "file_abc"
    blob.write_bytes(b"hello world")

    fake_meta = FileMetadata(
        id="file_abc",
        filename="report.pdf",
        mime_type="application/pdf",
        size_bytes=11,
        created_at="2026-04-10T00:00:00.000Z",
    )

    sm = SandboxManager()
    fake_sandbox = MagicMock()
    fake_sandbox.id = "sbx_file"
    fake_sandbox.exec = AsyncMock(return_value=_fake_exec_result())
    fake_sandbox.copy_to = AsyncMock()

    fake_client = MagicMock()
    fake_client.create = AsyncMock(return_value=fake_sandbox)
    sm._client = fake_client

    resources = [{"type": "file", "file_id": "file_abc"}]

    # Patch the repository lookup so we don't need a real DB
    with mpatch(
        "castor_server.store.repository.get_file",
        new=AsyncMock(return_value=fake_meta),
    ):
        await sm.get_or_create("session_file", _fake_env(), resources=resources)

    fake_sandbox.copy_to.assert_awaited_once()
    args = fake_sandbox.copy_to.await_args.args
    # Host path should be tmp_path/file_abc
    assert args[0] == str(blob)
    # Sandbox path defaults to /workspace/<filename>
    assert args[1] == "/workspace/report.pdf"


@pytest.mark.asyncio
async def test_file_resource_missing_file_id_warns(caplog):
    sm = SandboxManager()
    fake_sandbox = MagicMock()
    fake_sandbox.id = "sbx_no_id"
    fake_sandbox.exec = AsyncMock(return_value=_fake_exec_result())
    fake_sandbox.copy_to = AsyncMock()

    fake_client = MagicMock()
    fake_client.create = AsyncMock(return_value=fake_sandbox)
    sm._client = fake_client

    resources = [{"type": "file"}]  # missing file_id

    with caplog.at_level("WARNING"):
        await sm.get_or_create("session_no_id", _fake_env(), resources=resources)

    assert any("missing file_id" in r.message for r in caplog.records)
    fake_sandbox.copy_to.assert_not_awaited()


@pytest.mark.asyncio
async def test_file_resource_blob_missing_warns(tmp_path, monkeypatch, caplog):
    from unittest.mock import patch as mpatch

    from castor_server.models.files import FileMetadata

    monkeypatch.setattr("castor_server.config.settings.files_dir", str(tmp_path))
    # Don't create the blob — only the metadata

    fake_meta = FileMetadata(
        id="file_missing_blob",
        filename="x.txt",
        mime_type="text/plain",
        size_bytes=1,
        created_at="2026-04-10T00:00:00.000Z",
    )

    sm = SandboxManager()
    fake_sandbox = MagicMock()
    fake_sandbox.id = "sbx_no_blob"
    fake_sandbox.exec = AsyncMock(return_value=_fake_exec_result())
    fake_sandbox.copy_to = AsyncMock()

    fake_client = MagicMock()
    fake_client.create = AsyncMock(return_value=fake_sandbox)
    sm._client = fake_client

    resources = [{"type": "file", "file_id": "file_missing_blob"}]

    with (
        mpatch(
            "castor_server.store.repository.get_file",
            new=AsyncMock(return_value=fake_meta),
        ),
        caplog.at_level("WARNING"),
    ):
        await sm.get_or_create("session_no_blob", _fake_env(), resources=resources)

    assert any("blob missing on disk" in r.message for r in caplog.records)
    fake_sandbox.copy_to.assert_not_awaited()


@pytest.mark.asyncio
async def test_resources_only_mounted_on_first_create():
    """Cached sandbox should NOT re-run mount on the second get_or_create call."""
    sm = SandboxManager()

    fake_sandbox = MagicMock()
    fake_sandbox.id = "sbx_7"
    fake_sandbox.exec = AsyncMock(return_value=_fake_exec_result())

    fake_client = MagicMock()
    fake_client.create = AsyncMock(return_value=fake_sandbox)
    sm._client = fake_client

    resources = [{"type": "github_repository", "url": "https://github.com/foo/bar.git"}]

    await sm.get_or_create("session_7", _fake_env(), resources=resources)
    initial_call_count = fake_sandbox.exec.call_count
    assert initial_call_count > 0

    # Second call returns cached sandbox without mounting again
    await sm.get_or_create("session_7", _fake_env(), resources=resources)
    assert fake_sandbox.exec.call_count == initial_call_count


def test_default_mount_path_strips_git_suffix():
    assert (
        SandboxManager._default_mount_path("https://github.com/foo/bar.git")
        == "/workspace/bar"
    )
    assert (
        SandboxManager._default_mount_path("https://github.com/foo/bar")
        == "/workspace/bar"
    )
    assert (
        SandboxManager._default_mount_path("https://github.com/foo/bar/")
        == "/workspace/bar"
    )
