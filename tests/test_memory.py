"""Tests for Memory API (AISA §2.2 syscall surface)."""

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
    assert data["pinned_ids"] == []


@pytest.mark.asyncio
async def test_memory_state_not_found(client: AsyncClient):
    resp = await client.get("/v1/sessions/nonexistent/memory")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# mem_write
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_write(client: AsyncClient):
    _, session_id = await _create_session(client)
    resp = await client.post(
        f"/v1/sessions/{session_id}/memory/write",
        json={"content": "Python is preferred", "metadata": {"source": "user"}},
    )
    assert resp.status_code == 200
    assert "event_id" in resp.json()


# ---------------------------------------------------------------------------
# mem_read
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_read_not_found(client: AsyncClient):
    _, session_id = await _create_session(client)
    resp = await client.get(
        f"/v1/sessions/{session_id}/memory/read/999999",
    )
    assert resp.status_code == 200
    assert resp.json()["entry"] is None


# ---------------------------------------------------------------------------
# mem_search
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_empty(client: AsyncClient):
    _, session_id = await _create_session(client)
    resp = await client.post(
        f"/v1/sessions/{session_id}/memory/search",
        json={"query": "hello"},
    )
    assert resp.status_code == 200
    assert resp.json()["results"] == []


# ---------------------------------------------------------------------------
# mem_delete
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_nonexistent(client: AsyncClient):
    _, session_id = await _create_session(client)
    resp = await client.delete(
        f"/v1/sessions/{session_id}/memory/delete/999999",
    )
    assert resp.status_code == 200
    assert resp.json()["deleted"] is False


# ---------------------------------------------------------------------------
# mem_evict
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_evict(client: AsyncClient):
    _, session_id = await _create_session(client)
    resp = await client.post(
        f"/v1/sessions/{session_id}/memory/evict",
        json={"memory_id": "msg_abc123", "summary": "test summary"},
    )
    assert resp.status_code == 200
    assert resp.json()["memory_id"] == "msg_abc123"


# ---------------------------------------------------------------------------
# mem_promote
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_promote(client: AsyncClient):
    _, session_id = await _create_session(client)
    resp = await client.post(
        f"/v1/sessions/{session_id}/memory/promote",
        json={"memory_id": "msg_abc123"},
    )
    assert resp.status_code == 200
    assert resp.json()["memory_id"] == "msg_abc123"


# ---------------------------------------------------------------------------
# mem_protect
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_protect(client: AsyncClient):
    _, session_id = await _create_session(client)
    resp = await client.post(
        f"/v1/sessions/{session_id}/memory/protect",
        json={"memory_id": "msg_abc123", "protect": True},
    )
    assert resp.status_code == 200
    assert resp.json()["protected"] is True


@pytest.mark.asyncio
async def test_unprotect(client: AsyncClient):
    _, session_id = await _create_session(client)
    resp = await client.post(
        f"/v1/sessions/{session_id}/memory/protect",
        json={"memory_id": "msg_abc123", "protect": False},
    )
    assert resp.status_code == 200
    assert resp.json()["protected"] is False


# ---------------------------------------------------------------------------
# Cold storage list
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cold_list_empty(client: AsyncClient):
    _, session_id = await _create_session(client)
    resp = await client.get(f"/v1/sessions/{session_id}/memory/cold")
    assert resp.status_code == 200
    assert resp.json()["data"] == []


@pytest.mark.asyncio
async def test_cold_list_after_write(client: AsyncClient):
    agent_id, session_id = await _create_session(client)
    await client.post(
        f"/v1/sessions/{session_id}/memory/write",
        json={"content": "important fact"},
    )
    resp = await client.get(f"/v1/sessions/{session_id}/memory/cold")
    assert resp.status_code == 200
    entries = resp.json()["data"]
    assert len(entries) >= 1
    assert entries[0]["source"] == "explicit"


# ---------------------------------------------------------------------------
# Cross-session recall via search
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cross_session_search(client: AsyncClient):
    """Session A writes, session B (same agent) searches."""
    agent_id, session_a = await _create_session(client)

    await client.post(
        f"/v1/sessions/{session_a}/memory/write",
        json={"content": "The capital of France is Paris"},
    )

    session_b_resp = await client.post("/v1/sessions", json={"agent": agent_id})
    session_b = session_b_resp.json()["id"]

    resp = await client.post(
        f"/v1/sessions/{session_b}/memory/search",
        json={"query": "capital of France"},
    )
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) >= 1
    content = str(results[0])
    assert "Paris" in content or "capital" in content.lower()


# ---------------------------------------------------------------------------
# Memory events in event list
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_evict_event_in_event_list(client: AsyncClient):
    _, session_id = await _create_session(client)
    await client.post(
        f"/v1/sessions/{session_id}/memory/evict",
        json={"memory_id": "msg_test"},
    )
    resp = await client.get(f"/v1/sessions/{session_id}/events?order=asc")
    events = resp.json()["data"]
    evict_events = [e for e in events if e.get("type") == "memory.evict"]
    assert len(evict_events) >= 1
    assert evict_events[0]["memory_id"] == "msg_test"


# ---------------------------------------------------------------------------
# DefaultMemoryPolicy unit tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_policy_should_evict_below_budget():
    from castor_server.core.memory_policy import DefaultMemoryPolicy

    policy = DefaultMemoryPolicy()
    context = [{"role": "user", "content": "short msg", "id": "m1"}]
    result = await policy.should_evict(context, token_budget=10000)
    assert result is None


@pytest.mark.asyncio
async def test_policy_should_evict_over_budget():
    from castor_server.core.memory_policy import DefaultMemoryPolicy

    policy = DefaultMemoryPolicy()
    context = [
        {"role": "assistant", "content": "x" * 4000, "id": f"m{i}"} for i in range(10)
    ]
    context.append({"role": "user", "content": "recent q1", "id": "u1"})
    context.append({"role": "user", "content": "recent q2", "id": "u2"})

    result = await policy.should_evict(context, token_budget=2000)
    assert result is not None
    assert len(result) > 0
    # Result should be memory_ids (strings), not indices
    assert all(isinstance(mid, str) for mid in result)
    # Anchored user messages should not be evicted
    assert "u1" not in result
    assert "u2" not in result


@pytest.mark.asyncio
async def test_policy_should_recall_with_cue():
    from castor_server.core.memory_policy import DefaultMemoryPolicy

    policy = DefaultMemoryPolicy()
    result = await policy.should_recall(
        [{"role": "user", "content": "hi"}],
        "do you remember what we discussed earlier?",
    )
    assert result is not None


@pytest.mark.asyncio
async def test_policy_no_recall_long_context_no_cue():
    from castor_server.core.memory_policy import DefaultMemoryPolicy

    policy = DefaultMemoryPolicy()
    context = [{"role": "user", "content": f"msg {i}"} for i in range(10)]
    result = await policy.should_recall(context, "what is 2+2?")
    assert result is None


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
async def test_cold_storage_read_and_delete():
    import tempfile

    from castor_server.store.cold_storage import SQLiteVecColdStorage

    with tempfile.TemporaryDirectory() as tmpdir:
        cold = SQLiteVecColdStorage(db_path=f"{tmpdir}/test.db")
        try:
            await cold.store_explicit("agent_1", "test content")
            entries = await cold.list_entries("agent_1")
            assert len(entries) == 1
            entry_id = str(entries[0]["id"])

            # Read
            entry = await cold.read("agent_1", entry_id)
            assert entry is not None

            # Delete
            deleted = await cold.delete("agent_1", entry_id)
            assert deleted is True

            # Verify gone
            entry = await cold.read("agent_1", entry_id)
            assert entry is None
        finally:
            cold.close()


@pytest.mark.asyncio
async def test_cold_storage_agent_isolation():
    import tempfile

    from castor_server.store.cold_storage import SQLiteVecColdStorage

    with tempfile.TemporaryDirectory() as tmpdir:
        cold = SQLiteVecColdStorage(db_path=f"{tmpdir}/test.db")
        try:
            await cold.store_explicit("agent_A", "secret A")
            await cold.store_explicit("agent_B", "secret B")

            results_a = await cold.search("agent_A", "secret")
            results_b = await cold.search("agent_B", "secret")

            a_content = str(results_a)
            b_content = str(results_b)
            assert "secret A" in a_content
            assert "secret B" not in a_content
            assert "secret B" in b_content
            assert "secret A" not in b_content
        finally:
            cold.close()
