"""Tests for OpenAI Responses API compatibility layer."""

from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_create_response_basic(client: AsyncClient):
    """POST /v1/responses with simple text input."""
    resp = await client.post(
        "/v1/responses",
        json={
            "model": "mock",
            "input": "hello",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "response"
    assert data["status"] in ("completed", "failed")
    assert data["id"].startswith("resp_")
    assert isinstance(data["output"], list)
    assert isinstance(data["usage"], dict)


@pytest.mark.asyncio
async def test_create_response_with_instructions(client: AsyncClient):
    """Instructions map to agent system prompt."""
    resp = await client.post(
        "/v1/responses",
        json={
            "model": "mock",
            "input": "ping",
            "instructions": "You are a test assistant.",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("completed", "failed")


@pytest.mark.asyncio
async def test_create_response_array_input(client: AsyncClient):
    """Input as array of message objects."""
    resp = await client.post(
        "/v1/responses",
        json={
            "model": "mock",
            "input": [{"type": "text", "text": "hello from array"}],
        },
    )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_response_has_output(client: AsyncClient):
    """Mock model should produce an agent.message → output message."""
    resp = await client.post(
        "/v1/responses",
        json={"model": "mock", "input": "test output"},
    )
    data = resp.json()
    # Mock model echoes input as agent.message
    messages = [o for o in data["output"] if o.get("type") == "message"]
    assert len(messages) >= 1
    text = messages[0]["content"][0]["text"]
    assert "test output" in text.lower() or "mock" in text.lower()


@pytest.mark.asyncio
async def test_previous_response_id_chains(client: AsyncClient):
    """Chaining via previous_response_id reuses the same session."""
    r1 = await client.post(
        "/v1/responses",
        json={"model": "mock", "input": "first turn"},
    )
    resp_id = r1.json()["id"]

    r2 = await client.post(
        "/v1/responses",
        json={
            "model": "mock",
            "input": "second turn",
            "previous_response_id": resp_id,
        },
    )
    assert r2.status_code == 200
    assert r2.json()["previous_response_id"] == resp_id


@pytest.mark.asyncio
async def test_get_response(client: AsyncClient):
    """GET /v1/responses/{id} retrieves a previously created response."""
    create = await client.post(
        "/v1/responses",
        json={"model": "mock", "input": "get test"},
    )
    resp_id = create.json()["id"]

    get = await client.get(f"/v1/responses/{resp_id}")
    assert get.status_code == 200
    assert get.json()["id"] == resp_id


@pytest.mark.asyncio
async def test_get_response_not_found(client: AsyncClient):
    resp = await client.get("/v1/responses/nonexistent")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_response(client: AsyncClient):
    create = await client.post(
        "/v1/responses",
        json={"model": "mock", "input": "delete me"},
    )
    resp_id = create.json()["id"]

    delete = await client.delete(f"/v1/responses/{resp_id}")
    assert delete.status_code == 200
    assert delete.json()["deleted"] is True

    # Should be gone
    get = await client.get(f"/v1/responses/{resp_id}")
    assert get.status_code == 404


@pytest.mark.asyncio
async def test_agent_reuse(client: AsyncClient):
    """Same model + tools + instructions should reuse the same agent."""
    r1 = await client.post(
        "/v1/responses",
        json={
            "model": "mock",
            "input": "first",
            "instructions": "be concise",
        },
    )
    r2 = await client.post(
        "/v1/responses",
        json={
            "model": "mock",
            "input": "second",
            "instructions": "be concise",
        },
    )
    # Both should succeed (agent reused internally)
    assert r1.status_code == 200
    assert r2.status_code == 200
