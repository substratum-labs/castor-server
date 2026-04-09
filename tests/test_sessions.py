"""Tests for session CRUD endpoints."""

from __future__ import annotations

import pytest
from httpx import AsyncClient


async def _create_agent(client: AsyncClient) -> str:
    resp = await client.post(
        "/v1/agents",
        json={
            "name": "session-test-agent",
            "model": "claude-sonnet-4-6",
            "tools": [{"type": "agent_toolset_20260401"}],
        },
    )
    return resp.json()["id"]


@pytest.mark.asyncio
async def test_create_session(client: AsyncClient):
    agent_id = await _create_agent(client)
    resp = await client.post(
        "/v1/sessions",
        json={"agent": agent_id, "title": "Test session"},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["type"] == "session"
    assert data["id"].startswith("session_")
    assert data["status"] == "idle"
    assert data["agent"]["id"] == agent_id
    assert data["title"] == "Test session"


@pytest.mark.asyncio
async def test_create_session_with_pinned_version(client: AsyncClient):
    agent_id = await _create_agent(client)
    resp = await client.post(
        "/v1/sessions",
        json={"agent": {"type": "agent", "id": agent_id, "version": 1}},
    )
    assert resp.status_code == 201
    assert resp.json()["agent"]["version"] == 1


@pytest.mark.asyncio
async def test_create_session_agent_not_found(client: AsyncClient):
    resp = await client.post("/v1/sessions", json={"agent": "nonexistent"})
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_list_sessions(client: AsyncClient):
    agent_id = await _create_agent(client)
    await client.post("/v1/sessions", json={"agent": agent_id})
    await client.post("/v1/sessions", json={"agent": agent_id})

    resp = await client.get("/v1/sessions")
    assert resp.status_code == 200
    assert len(resp.json()["data"]) >= 2


@pytest.mark.asyncio
async def test_list_sessions_by_agent(client: AsyncClient):
    agent_id = await _create_agent(client)
    await client.post("/v1/sessions", json={"agent": agent_id})

    resp = await client.get(f"/v1/sessions?agent_id={agent_id}")
    assert resp.status_code == 200
    for s in resp.json()["data"]:
        assert s["agent"]["id"] == agent_id


@pytest.mark.asyncio
async def test_get_session(client: AsyncClient):
    agent_id = await _create_agent(client)
    create_resp = await client.post(
        "/v1/sessions", json={"agent": agent_id, "title": "find me"}
    )
    session_id = create_resp.json()["id"]

    resp = await client.get(f"/v1/sessions/{session_id}")
    assert resp.status_code == 200
    assert resp.json()["title"] == "find me"


@pytest.mark.asyncio
async def test_get_session_not_found(client: AsyncClient):
    resp = await client.get("/v1/sessions/nonexistent")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_update_session(client: AsyncClient):
    agent_id = await _create_agent(client)
    create_resp = await client.post(
        "/v1/sessions", json={"agent": agent_id, "title": "old"}
    )
    session_id = create_resp.json()["id"]

    resp = await client.post(f"/v1/sessions/{session_id}", json={"title": "new"})
    assert resp.status_code == 200
    assert resp.json()["title"] == "new"


@pytest.mark.asyncio
async def test_delete_session(client: AsyncClient):
    agent_id = await _create_agent(client)
    create_resp = await client.post("/v1/sessions", json={"agent": agent_id})
    session_id = create_resp.json()["id"]

    resp = await client.delete(f"/v1/sessions/{session_id}")
    assert resp.status_code == 200
    assert resp.json()["type"] == "session_deleted"

    # Verify it's gone
    resp = await client.get(f"/v1/sessions/{session_id}")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_archive_session(client: AsyncClient):
    agent_id = await _create_agent(client)
    create_resp = await client.post("/v1/sessions", json={"agent": agent_id})
    session_id = create_resp.json()["id"]

    resp = await client.post(f"/v1/sessions/{session_id}/archive")
    assert resp.status_code == 200
    assert resp.json()["archived_at"] is not None


@pytest.mark.asyncio
async def test_session_metadata(client: AsyncClient):
    agent_id = await _create_agent(client)
    create_resp = await client.post(
        "/v1/sessions",
        json={
            "agent": agent_id,
            "metadata": {"project": "demo"},
        },
    )
    session_id = create_resp.json()["id"]
    assert create_resp.json()["metadata"]["project"] == "demo"

    # Patch metadata
    resp = await client.post(
        f"/v1/sessions/{session_id}",
        json={"metadata": {"project": None, "env": "prod"}},
    )
    assert resp.status_code == 200
    meta = resp.json()["metadata"]
    assert "project" not in meta
    assert meta["env"] == "prod"
