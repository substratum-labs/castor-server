"""Tests for the built-in 'mock' model.

The mock model lets users (and CI) run the full agent pipeline end-to-end
with zero external dependencies — no LLM API key required. It's the
foundation of the 30-second offline demo.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient

from castor_server.core.llm_adapter import _mock_chat_response, litellm_chat


@pytest.mark.asyncio
async def test_mock_chat_response_echoes_last_user_message():
    """Direct unit test on the helper — independent of the agent loop."""
    response = _mock_chat_response(
        [
            {"role": "system", "content": "ignored"},
            {"role": "user", "content": "hello world"},
        ]
    )
    assert response["role"] == "assistant"
    assert response["stop_reason"] == "end_turn"
    assert response["usage"] == {"input_tokens": 0, "output_tokens": 0}
    assert response["content"][0]["type"] == "text"
    assert "hello world" in response["content"][0]["text"]
    assert response["content"][0]["text"].startswith("[mock]")


@pytest.mark.asyncio
async def test_mock_chat_response_handles_block_content():
    """User message can be a list of content blocks rather than a string."""
    response = _mock_chat_response(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "block one"},
                    {"type": "text", "text": "block two"},
                ],
            }
        ]
    )
    text = response["content"][0]["text"]
    assert "block one" in text
    assert "block two" in text


@pytest.mark.asyncio
async def test_mock_chat_response_no_user_message():
    """Edge case — no user message in context yet."""
    response = _mock_chat_response([{"role": "system", "content": "system only"}])
    assert "no user message" in response["content"][0]["text"]


@pytest.mark.asyncio
async def test_litellm_chat_dispatches_to_mock_when_model_is_mock():
    """litellm_chat short-circuits to the mock helper for model='mock'.

    Critically, this MUST NOT call litellm.acompletion — if it does, the test
    would either hit the network or raise an auth error.
    """
    response = await litellm_chat(
        model="mock",
        messages=[{"role": "user", "content": "ping"}],
    )
    assert response["stop_reason"] == "end_turn"
    assert "ping" in response["content"][0]["text"]


@pytest.mark.asyncio
async def test_mock_model_end_to_end_via_api(client: AsyncClient):
    """Full pipeline: create agent with model='mock', send message, list events.

    This is the 30-second demo's smoke test — proves the entire stack
    (agents → sessions → events → kernel → LLM adapter) works without any
    external dependency.
    """
    # 1. Create an agent using the mock model
    agent_resp = await client.post(
        "/v1/agents",
        json={
            "name": "mock-demo",
            "model": "mock",
            "system": "ignored by mock",
            "tools": [],  # no tools — pure text loop
        },
    )
    assert agent_resp.status_code == 201
    agent_id = agent_resp.json()["id"]

    # 2. Create a session
    session_resp = await client.post("/v1/sessions", json={"agent": agent_id})
    assert session_resp.status_code == 201
    session_id = session_resp.json()["id"]

    # 3. Send a user message — fire and forget
    send_resp = await client.post(
        f"/v1/sessions/{session_id}/events",
        json={
            "events": [
                {
                    "type": "user.message",
                    "content": [{"type": "text", "text": "hello castor"}],
                }
            ]
        },
    )
    assert send_resp.status_code == 200

    # 4. Wait for the agent loop to drain (fire-and-forget create_task in events.py)
    import asyncio

    for _ in range(50):
        await asyncio.sleep(0.05)
        events_resp = await client.get(f"/v1/sessions/{session_id}/events")
        types = [e["type"] for e in events_resp.json()["data"]]
        if "session.status_idle" in types:
            break

    # 5. Verify the full event sequence is present
    events = events_resp.json()["data"]
    types = [e["type"] for e in events]

    assert "user.message" in types, f"missing user.message, got: {types}"
    assert "session.status_running" in types, (
        f"missing session.status_running, got: {types}"
    )
    assert "session.status_idle" in types, f"missing session.status_idle, got: {types}"
    assert "session.error" not in types, (
        f"unexpected session.error in mock flow: {events}"
    )

    # 6. Verify the agent.message contains our echo
    agent_messages = [e for e in events if e["type"] == "agent.message"]
    assert len(agent_messages) >= 1, "expected at least one agent.message event"
    text_blocks = agent_messages[-1]["content"]
    full_text = " ".join(b.get("text", "") for b in text_blocks)
    assert "hello castor" in full_text, (
        f"mock echo did not include user text, got: {full_text!r}"
    )
    assert "[mock]" in full_text


@pytest.mark.asyncio
async def test_mock_model_session_reaches_idle_with_end_turn(client: AsyncClient):
    """The mock model finishes with stop_reason=end_turn (not requires_action)."""
    agent = (
        await client.post(
            "/v1/agents",
            json={"name": "mock-end-turn", "model": "mock", "tools": []},
        )
    ).json()
    session = (await client.post("/v1/sessions", json={"agent": agent["id"]})).json()

    await client.post(
        f"/v1/sessions/{session['id']}/events",
        json={
            "events": [
                {
                    "type": "user.message",
                    "content": [{"type": "text", "text": "test"}],
                }
            ]
        },
    )

    import asyncio

    idle_event = None
    for _ in range(50):
        await asyncio.sleep(0.05)
        resp = await client.get(f"/v1/sessions/{session['id']}/events")
        for e in resp.json()["data"]:
            if e["type"] == "session.status_idle":
                idle_event = e
                break
        if idle_event:
            break

    assert idle_event is not None, "session never reached idle"
    assert idle_event["stop_reason"]["type"] == "end_turn"
