"""Tests for Memory API and related components."""

from __future__ import annotations

import pytest
from httpx import AsyncClient


async def _create_session(client: AsyncClient) -> tuple[str, str]:
    agent_resp = await client.post(
        "/v1/agents",
        json={
            "name": "memory-test-agent",
            "model": "mock",
            "tools": [{"type": "agent_toolset_20260401"}],
        },
    )
    agent_id = agent_resp.json()["id"]
    session_resp = await client.post("/v1/sessions", json={"agent": agent_id})
    return agent_id, session_resp.json()["id"]


# ---------------------------------------------------------------------------
# Memory state endpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_memory_state_empty_session(client: AsyncClient):
    _, session_id = await _create_session(client)
    resp = await client.get(f"/v1/sessions/{session_id}/memory")
    assert resp.status_code == 200
    data = resp.json()
    assert data["token_count"] == 0
    assert data["message_count"] == 0
    assert data["pinned_indices"] == []


@pytest.mark.asyncio
async def test_memory_state_not_found(client: AsyncClient):
    resp = await client.get("/v1/sessions/nonexistent/memory")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Eviction endpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_evict(client: AsyncClient):
    _, session_id = await _create_session(client)
    resp = await client.post(
        f"/v1/sessions/{session_id}/memory/evict",
        json={"indices": [0, 1], "summary": "test summary"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["evicted"] == 2
    assert "event_id" in data


@pytest.mark.asyncio
async def test_evict_session_not_found(client: AsyncClient):
    resp = await client.post(
        "/v1/sessions/nonexistent/memory/evict",
        json={"indices": [0]},
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Recall endpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recall_empty(client: AsyncClient):
    _, session_id = await _create_session(client)
    resp = await client.post(
        f"/v1/sessions/{session_id}/memory/recall",
        json={"query": "what happened before?"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    assert data["results"] == []
    assert "event_id" in data


# ---------------------------------------------------------------------------
# Pin endpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pin(client: AsyncClient):
    _, session_id = await _create_session(client)
    resp = await client.post(
        f"/v1/sessions/{session_id}/memory/pin",
        json={"index": 3},
    )
    assert resp.status_code == 200
    assert resp.json()["pinned_index"] == 3


# ---------------------------------------------------------------------------
# Store endpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_store(client: AsyncClient):
    _, session_id = await _create_session(client)
    resp = await client.post(
        f"/v1/sessions/{session_id}/memory/store",
        json={
            "content": "The user prefers Python over JavaScript",
            "metadata": {"source": "conversation"},
        },
    )
    assert resp.status_code == 200
    assert "event_id" in resp.json()


# ---------------------------------------------------------------------------
# Cold storage list endpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cold_list_empty(client: AsyncClient):
    _, session_id = await _create_session(client)
    resp = await client.get(f"/v1/sessions/{session_id}/memory/cold")
    assert resp.status_code == 200
    assert resp.json()["data"] == []


@pytest.mark.asyncio
async def test_cold_list_after_store(client: AsyncClient):
    agent_id, session_id = await _create_session(client)
    # Store something
    await client.post(
        f"/v1/sessions/{session_id}/memory/store",
        json={"content": "important fact"},
    )
    # List it
    resp = await client.get(f"/v1/sessions/{session_id}/memory/cold")
    assert resp.status_code == 200
    entries = resp.json()["data"]
    assert len(entries) >= 1
    assert entries[0]["source"] == "explicit"


# ---------------------------------------------------------------------------
# Cross-session recall
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cross_session_recall(client: AsyncClient):
    """Session A stores, session B (same agent) recalls."""
    agent_id, session_a = await _create_session(client)

    # Store in session A
    await client.post(
        f"/v1/sessions/{session_a}/memory/store",
        json={"content": "The capital of France is Paris"},
    )

    # Create session B for the same agent
    session_b_resp = await client.post("/v1/sessions", json={"agent": agent_id})
    session_b = session_b_resp.json()["id"]

    # Recall from session B
    resp = await client.post(
        f"/v1/sessions/{session_b}/memory/recall",
        json={"query": "capital of France"},
    )
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) >= 1
    # Should contain the Paris fact
    content = str(results[0])
    assert "Paris" in content or "capital" in content.lower()


# ---------------------------------------------------------------------------
# Memory events in event list
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_evict_event_in_event_list(client: AsyncClient):
    """Evict should create a memory.evict event visible in event list."""
    _, session_id = await _create_session(client)
    await client.post(
        f"/v1/sessions/{session_id}/memory/evict",
        json={"indices": [0]},
    )
    # Check events
    resp = await client.get(f"/v1/sessions/{session_id}/events?order=asc")
    events = resp.json()["data"]
    evict_events = [e for e in events if e.get("type") == "memory.evict"]
    assert len(evict_events) >= 1
    assert evict_events[0]["indices"] == [0]


# ---------------------------------------------------------------------------
# DefaultMemoryPolicy unit tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_policy_should_evict_below_budget():
    from castor_server.core.memory_policy import DefaultMemoryPolicy

    policy = DefaultMemoryPolicy()
    context = [
        {"role": "user", "content": "short msg"},
    ]
    result = await policy.should_evict(context, token_budget=10000)
    assert result is None  # Below budget, no eviction


@pytest.mark.asyncio
async def test_policy_should_evict_over_budget():
    from castor_server.core.memory_policy import DefaultMemoryPolicy

    policy = DefaultMemoryPolicy()
    # Create a large context
    context = [{"role": "assistant", "content": "x" * 4000} for _ in range(10)]
    # Add 2 user messages at the end (anchored, should not be evicted)
    context.append({"role": "user", "content": "recent q1"})
    context.append({"role": "user", "content": "recent q2"})

    result = await policy.should_evict(context, token_budget=2000)
    assert result is not None
    assert len(result) > 0
    # Last two user messages should not be in eviction list
    assert 10 not in result
    assert 11 not in result


@pytest.mark.asyncio
async def test_policy_should_recall_with_cue():
    from castor_server.core.memory_policy import DefaultMemoryPolicy

    policy = DefaultMemoryPolicy()
    result = await policy.should_recall(
        [{"role": "user", "content": "hi"}],
        "do you remember what we discussed earlier?",
    )
    assert result is not None  # "earlier" is a recall cue


@pytest.mark.asyncio
async def test_policy_should_recall_short_context():
    from castor_server.core.memory_policy import DefaultMemoryPolicy

    policy = DefaultMemoryPolicy()
    result = await policy.should_recall(
        [{"role": "user", "content": "hi"}],
        "what is 2+2?",
    )
    # Short context (< 5) → always recall
    assert result is not None


@pytest.mark.asyncio
async def test_policy_no_recall_long_context_no_cue():
    from castor_server.core.memory_policy import DefaultMemoryPolicy

    policy = DefaultMemoryPolicy()
    context = [{"role": "user", "content": f"msg {i}"} for i in range(10)]
    result = await policy.should_recall(context, "what is 2+2?")
    assert result is None  # Long context, no cue → no recall


# ---------------------------------------------------------------------------
# SQLiteVec ColdStorage unit tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cold_storage_store_and_search():
    import tempfile

    from castor_server.store.cold_storage import SQLiteVecColdStorage

    with tempfile.TemporaryDirectory() as tmpdir:
        cold = SQLiteVecColdStorage(db_path=f"{tmpdir}/test.db")
        try:
            await cold.store(
                "agent_1",
                [{"role": "user", "content": "The sky is blue"}],
                summary="sky color",
            )
            results = await cold.search("agent_1", "sky color")
            assert len(results) >= 1
        finally:
            cold.close()


@pytest.mark.asyncio
async def test_cold_storage_store_explicit():
    import tempfile

    from castor_server.store.cold_storage import SQLiteVecColdStorage

    with tempfile.TemporaryDirectory() as tmpdir:
        cold = SQLiteVecColdStorage(db_path=f"{tmpdir}/test.db")
        try:
            await cold.store_explicit(
                "agent_1",
                "Python is preferred",
                metadata={"source": "user"},
            )
            entries = await cold.list_entries("agent_1")
            assert len(entries) == 1
            assert entries[0]["source"] == "explicit"
        finally:
            cold.close()


@pytest.mark.asyncio
async def test_cold_storage_agent_isolation():
    """Different agent_ids should not see each other's entries."""
    import tempfile

    from castor_server.store.cold_storage import SQLiteVecColdStorage

    with tempfile.TemporaryDirectory() as tmpdir:
        cold = SQLiteVecColdStorage(db_path=f"{tmpdir}/test.db")
        try:
            await cold.store_explicit("agent_A", "secret A")
            await cold.store_explicit("agent_B", "secret B")

            results_a = await cold.search("agent_A", "secret")
            results_b = await cold.search("agent_B", "secret")

            # Each agent only sees its own entries
            a_content = str(results_a)
            b_content = str(results_b)
            assert "secret A" in a_content
            assert "secret B" not in a_content
            assert "secret B" in b_content
            assert "secret A" not in b_content
        finally:
            cold.close()
