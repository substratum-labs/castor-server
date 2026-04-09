"""Tests for agent CRUD endpoints."""

from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_create_agent(client: AsyncClient):
    resp = await client.post(
        "/v1/agents",
        json={
            "name": "test-agent",
            "model": "claude-sonnet-4-6",
            "system": "You are a helpful assistant.",
            "description": "A test agent",
            "tools": [{"type": "agent_toolset_20260401"}],
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["type"] == "agent"
    assert data["name"] == "test-agent"
    assert data["model"]["id"] == "claude-sonnet-4-6"
    assert data["version"] == 1
    assert data["id"].startswith("agent_")
    assert data["archived_at"] is None


@pytest.mark.asyncio
async def test_create_agent_with_model_config(client: AsyncClient):
    resp = await client.post(
        "/v1/agents",
        json={
            "name": "fast-agent",
            "model": {"id": "claude-sonnet-4-6", "speed": "fast"},
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["model"]["id"] == "claude-sonnet-4-6"
    assert data["model"]["speed"] == "fast"


@pytest.mark.asyncio
async def test_list_agents(client: AsyncClient):
    # Create two agents
    await client.post("/v1/agents", json={"name": "a1", "model": "claude-sonnet-4-6"})
    await client.post("/v1/agents", json={"name": "a2", "model": "claude-sonnet-4-6"})

    resp = await client.get("/v1/agents")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["data"]) >= 2


@pytest.mark.asyncio
async def test_get_agent(client: AsyncClient):
    create_resp = await client.post(
        "/v1/agents", json={"name": "get-me", "model": "claude-sonnet-4-6"}
    )
    agent_id = create_resp.json()["id"]

    resp = await client.get(f"/v1/agents/{agent_id}")
    assert resp.status_code == 200
    assert resp.json()["name"] == "get-me"


@pytest.mark.asyncio
async def test_get_agent_not_found(client: AsyncClient):
    resp = await client.get("/v1/agents/nonexistent")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_update_agent(client: AsyncClient):
    create_resp = await client.post(
        "/v1/agents",
        json={"name": "update-me", "model": "claude-sonnet-4-6", "system": "v1"},
    )
    agent_id = create_resp.json()["id"]

    resp = await client.post(
        f"/v1/agents/{agent_id}",
        json={"version": 1, "name": "updated-name", "system": "v2"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "updated-name"
    assert data["system"] == "v2"
    assert data["version"] == 2


@pytest.mark.asyncio
async def test_update_agent_version_conflict(client: AsyncClient):
    create_resp = await client.post(
        "/v1/agents", json={"name": "conflict", "model": "claude-sonnet-4-6"}
    )
    agent_id = create_resp.json()["id"]

    resp = await client.post(
        f"/v1/agents/{agent_id}", json={"version": 99, "name": "bad"}
    )
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_archive_agent(client: AsyncClient):
    create_resp = await client.post(
        "/v1/agents", json={"name": "archive-me", "model": "claude-sonnet-4-6"}
    )
    agent_id = create_resp.json()["id"]

    resp = await client.post(f"/v1/agents/{agent_id}/archive")
    assert resp.status_code == 200
    assert resp.json()["archived_at"] is not None


@pytest.mark.asyncio
async def test_list_agent_versions(client: AsyncClient):
    create_resp = await client.post(
        "/v1/agents", json={"name": "versioned", "model": "claude-sonnet-4-6"}
    )
    agent_id = create_resp.json()["id"]

    # Create v2
    await client.post(
        f"/v1/agents/{agent_id}", json={"version": 1, "name": "versioned-v2"}
    )

    resp = await client.get(f"/v1/agents/{agent_id}/versions")
    assert resp.status_code == 200
    versions = resp.json()["data"]
    assert len(versions) == 2
    assert versions[0]["version"] == 2
    assert versions[1]["version"] == 1


@pytest.mark.asyncio
async def test_create_agent_with_custom_tool(client: AsyncClient):
    resp = await client.post(
        "/v1/agents",
        json={
            "name": "custom-tools",
            "model": "claude-sonnet-4-6",
            "tools": [
                {
                    "type": "custom",
                    "name": "lookup_user",
                    "description": "Look up a user by ID",
                    "input_schema": {
                        "type": "object",
                        "properties": {"user_id": {"type": "string"}},
                        "required": ["user_id"],
                    },
                }
            ],
        },
    )
    assert resp.status_code == 201
    tools = resp.json()["tools"]
    assert len(tools) == 1
    assert tools[0]["name"] == "lookup_user"


@pytest.mark.asyncio
async def test_create_agent_with_metadata(client: AsyncClient):
    resp = await client.post(
        "/v1/agents",
        json={
            "name": "meta-agent",
            "model": "claude-sonnet-4-6",
            "metadata": {"env": "staging", "team": "platform"},
        },
    )
    assert resp.status_code == 201
    assert resp.json()["metadata"]["env"] == "staging"
    assert resp.json()["metadata"]["team"] == "platform"
