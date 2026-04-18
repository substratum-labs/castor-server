"""Tests for Phase 4.2 — injectable agent_fn_factory."""

from __future__ import annotations

import pytest
from httpx import AsyncClient

from castor_server.core.session_manager import SessionManager


@pytest.mark.asyncio
async def test_default_agent_fn_unchanged(client: AsyncClient):
    """Agent without agent_fn_factory uses default ReAct loop."""
    resp = await client.post(
        "/v1/agents",
        json={
            "name": "default-loop",
            "model": "mock",
            "tools": [{"type": "agent_toolset_20260401"}],
        },
    )
    assert resp.status_code == 201
    assert resp.json()["agent_fn_factory"] is None


@pytest.mark.asyncio
async def test_agent_fn_factory_stored(client: AsyncClient):
    """agent_fn_factory field is persisted and returned."""
    resp = await client.post(
        "/v1/agents",
        json={
            "name": "custom-loop",
            "model": "mock",
            "agent_fn_factory": "castor.examples.react:run",
        },
    )
    assert resp.status_code == 201
    assert resp.json()["agent_fn_factory"] == "castor.examples.react:run"

    # Verify it persists on GET
    agent_id = resp.json()["id"]
    get_resp = await client.get(f"/v1/agents/{agent_id}")
    assert get_resp.json()["agent_fn_factory"] == "castor.examples.react:run"


def test_load_factory_valid():
    """Valid module:callable path resolves correctly."""
    from castor_server.config import settings

    old = settings.api_key
    try:
        settings.api_key = "test-key"
        fn = SessionManager._load_agent_fn_factory("castor.examples.react:run")
        assert callable(fn)
    finally:
        settings.api_key = old


def test_load_factory_bogus():
    """Non-existent module raises ValueError, not ImportError."""
    from castor_server.config import settings

    old = settings.api_key
    try:
        settings.api_key = "test-key"
        with pytest.raises(ValueError, match="Failed to load"):
            SessionManager._load_agent_fn_factory("bogus.does.not:exist")
    finally:
        settings.api_key = old


def test_load_factory_bad_format():
    """Missing colon separator raises ValueError."""
    from castor_server.config import settings

    old = settings.api_key
    try:
        settings.api_key = "test-key"
        with pytest.raises(ValueError, match="module:callable"):
            SessionManager._load_agent_fn_factory("no_colon_here")
    finally:
        settings.api_key = old


def test_trust_gate_blocks_without_api_key():
    """Custom factory rejected when no API key set (open-access mode)."""
    from castor_server.config import settings

    old = settings.api_key
    try:
        settings.api_key = None
        with pytest.raises(ValueError, match="CASTOR_API_KEY"):
            SessionManager._load_agent_fn_factory("castor.examples.react:run")
    finally:
        settings.api_key = old
