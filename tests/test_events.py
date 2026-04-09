"""Tests for event endpoints."""

from __future__ import annotations

import pytest
from httpx import AsyncClient


async def _create_session(client: AsyncClient) -> tuple[str, str]:
    agent_resp = await client.post(
        "/v1/agents",
        json={
            "name": "event-test-agent",
            "model": "claude-sonnet-4-6",
            "tools": [{"type": "agent_toolset_20260401"}],
        },
    )
    agent_id = agent_resp.json()["id"]
    session_resp = await client.post("/v1/sessions", json={"agent": agent_id})
    return agent_id, session_resp.json()["id"]


@pytest.mark.asyncio
async def test_send_events(client: AsyncClient):
    _, session_id = await _create_session(client)
    resp = await client.post(
        f"/v1/sessions/{session_id}/events",
        json={
            "events": [
                {
                    "type": "user.message",
                    "content": [{"type": "text", "text": "Hello!"}],
                }
            ]
        },
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert len(data) == 1
    assert data[0]["type"] == "user.message"
    assert "id" in data[0]


@pytest.mark.asyncio
async def test_send_interrupt(client: AsyncClient):
    _, session_id = await _create_session(client)
    resp = await client.post(
        f"/v1/sessions/{session_id}/events",
        json={"events": [{"type": "user.interrupt"}]},
    )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_list_events(client: AsyncClient):
    _, session_id = await _create_session(client)

    # Send a message
    await client.post(
        f"/v1/sessions/{session_id}/events",
        json={
            "events": [
                {
                    "type": "user.message",
                    "content": [{"type": "text", "text": "test"}],
                }
            ]
        },
    )

    resp = await client.get(f"/v1/sessions/{session_id}/events")
    assert resp.status_code == 200
    events = resp.json()["data"]
    assert len(events) >= 1


@pytest.mark.asyncio
async def test_send_events_session_not_found(client: AsyncClient):
    resp = await client.post(
        "/v1/sessions/nonexistent/events",
        json={
            "events": [
                {
                    "type": "user.message",
                    "content": [{"type": "text", "text": "hello"}],
                }
            ]
        },
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_send_tool_confirmation(client: AsyncClient):
    _, session_id = await _create_session(client)
    resp = await client.post(
        f"/v1/sessions/{session_id}/events",
        json={
            "events": [
                {
                    "type": "user.tool_confirmation",
                    "tool_use_id": "fake_tool_123",
                    "result": "allow",
                }
            ]
        },
    )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_send_custom_tool_result(client: AsyncClient):
    _, session_id = await _create_session(client)
    resp = await client.post(
        f"/v1/sessions/{session_id}/events",
        json={
            "events": [
                {
                    "type": "user.custom_tool_result",
                    "custom_tool_use_id": "fake_custom_123",
                    "content": [{"type": "text", "text": "result data"}],
                    "is_error": False,
                }
            ]
        },
    )
    assert resp.status_code == 200
