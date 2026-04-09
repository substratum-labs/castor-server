"""Tests for Phase 2 — Castor extension endpoints (budget, scan, fork, HITL modify)."""

from __future__ import annotations

import pytest
from httpx import AsyncClient


async def _create_session(client: AsyncClient) -> tuple[str, str]:
    agent_resp = await client.post(
        "/v1/agents",
        json={
            "name": "phase2-test-agent",
            "model": "claude-sonnet-4-6",
            "tools": [{"type": "agent_toolset_20260401"}],
        },
    )
    agent_id = agent_resp.json()["id"]
    session_resp = await client.post("/v1/sessions", json={"agent": agent_id})
    return agent_id, session_resp.json()["id"]


# ---------------------------------------------------------------------------
# Budget endpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_budget_empty_session(client: AsyncClient):
    """Budget on a fresh session with no checkpoint returns empty list."""
    _, session_id = await _create_session(client)
    resp = await client.get(f"/v1/sessions/{session_id}/budget")
    assert resp.status_code == 200
    data = resp.json()
    assert data["budgets"] == []


@pytest.mark.asyncio
async def test_budget_not_found(client: AsyncClient):
    resp = await client.get("/v1/sessions/nonexistent/budget")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Scan endpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scan_empty_session(client: AsyncClient):
    """Scan on a fresh session returns zero steps."""
    _, session_id = await _create_session(client)
    resp = await client.post(f"/v1/sessions/{session_id}/scan")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_steps"] == 0
    assert data["flagged"] == []


@pytest.mark.asyncio
async def test_scan_not_found(client: AsyncClient):
    resp = await client.post("/v1/sessions/nonexistent/scan")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Fork endpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fork_no_checkpoint(client: AsyncClient):
    """Fork on a session with no checkpoint returns 400."""
    _, session_id = await _create_session(client)
    resp = await client.post(
        f"/v1/sessions/{session_id}/fork",
        json={"at_step": 0},
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_fork_not_found(client: AsyncClient):
    resp = await client.post(
        "/v1/sessions/nonexistent/fork",
        json={"at_step": 0},
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# HITL modify (event model)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_confirmation_modify(client: AsyncClient):
    """Verify modify is accepted as a tool_confirmation result."""
    _, session_id = await _create_session(client)
    resp = await client.post(
        f"/v1/sessions/{session_id}/events",
        json={
            "events": [
                {
                    "type": "user.tool_confirmation",
                    "tool_use_id": "fake_tool_123",
                    "result": "modify",
                    "modify_feedback": "Use a different approach",
                }
            ]
        },
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data[0]["result"] == "modify"
    assert data[0]["modify_feedback"] == "Use a different approach"
