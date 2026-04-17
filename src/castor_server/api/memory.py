"""Memory API — /v1/sessions/{id}/memory/* routes.

HTTP surface for inspecting and triggering memory operations on a session.
Each route dispatches a kernel memory syscall (mem_evict, mem_recall, etc.)
and returns the result. Memory events are emitted to the session SSE stream.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from castor_server.store.database import get_session
from castor_server.store.repository import get_session as get_session_by_id

router = APIRouter(prefix="/v1/sessions/{session_id}/memory", tags=["memory"])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class EvictRequest(BaseModel):
    indices: list[int] = Field(..., description="Message indices to evict")
    summary: str | None = Field(
        default=None, description="Optional summary to retain in context"
    )


class RecallRequest(BaseModel):
    query: str = Field(..., description="Search query for cold storage")
    max_results: int = Field(default=5, ge=1, le=50)
    source_filter: str | None = None


class PinRequest(BaseModel):
    index: int = Field(..., description="Message index to pin")


class StoreRequest(BaseModel):
    content: str = Field(..., description="Content to store explicitly")
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryStateResponse(BaseModel):
    token_count: int = 0
    message_count: int = 0
    pinned_indices: list[int] = Field(default_factory=list)
    watermark: int = 0


class ColdEntry(BaseModel):
    id: int
    source: str
    summary: str | None = None
    created_at: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ColdListResponse(BaseModel):
    data: list[ColdEntry]
    next_page: str | None = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("", response_model=MemoryStateResponse)
async def get_memory_state(
    session_id: str,
    db: AsyncSession = Depends(get_session),
) -> MemoryStateResponse:
    """Inspect current memory state of a session."""
    session = await get_session_by_id(db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    from castor_server.core.session_manager import session_manager

    kernel_cp, _messages = await session_manager._load_checkpoint(db, session_id)
    if kernel_cp is None:
        return MemoryStateResponse()

    context = kernel_cp.context_history or []
    pinned = [
        i
        for i, m in enumerate(context)
        if (m.pinned if hasattr(m, "pinned") else m.get("pinned", False))
    ]

    # Rough token estimate
    total_tokens = sum(
        max(1, len(str(getattr(m, "content", m.get("content", "")))) // 4)
        for m in context
    )

    return MemoryStateResponse(
        token_count=total_tokens,
        message_count=len(context),
        pinned_indices=pinned,
        watermark=8000,  # default hard watermark
    )


@router.post("/evict")
async def trigger_evict(
    session_id: str,
    body: EvictRequest,
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Trigger manual eviction of specific message indices."""
    session = await get_session_by_id(db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    from castor_server.core.session_manager import session_manager
    from castor_server.models.events import MemoryEvictEvent
    from castor_server.store import repository as repo

    bus = session_manager.get_bus(session_id)

    evt = MemoryEvictEvent(
        indices=body.indices,
        token_count=len(body.indices),
        summary=body.summary,
        source="manual",
    )
    await bus.publish(evt)
    await repo.store_event(
        db,
        session_id=session_id,
        event_id=evt.id,
        event_type=evt.type,
        data=evt.model_dump(exclude_none=True),
    )

    return {"status": "ok", "evicted": len(body.indices), "event_id": evt.id}


@router.post("/recall")
async def trigger_recall(
    session_id: str,
    body: RecallRequest,
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Search cold storage and return matching entries."""
    session = await get_session_by_id(db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    from castor_server.core.session_manager import session_manager
    from castor_server.models.events import MemoryRecallEvent
    from castor_server.store import repository as repo
    from castor_server.store.cold_storage import SQLiteVecColdStorage

    agent_id = session.agent.id
    cold = SQLiteVecColdStorage()
    try:
        results = await cold.search(
            agent_id,
            body.query,
            max_results=body.max_results,
            source_filter=body.source_filter,
        )
    finally:
        cold.close()

    bus = session_manager.get_bus(session_id)
    evt = MemoryRecallEvent(
        query=body.query,
        result_count=len(results),
        source_filter=body.source_filter,
    )
    await bus.publish(evt)
    await repo.store_event(
        db,
        session_id=session_id,
        event_id=evt.id,
        event_type=evt.type,
        data=evt.model_dump(exclude_none=True),
    )

    return {"results": results, "event_id": evt.id}


@router.post("/pin")
async def trigger_pin(
    session_id: str,
    body: PinRequest,
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Pin a message so it is never evicted."""
    session = await get_session_by_id(db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    from castor_server.core.session_manager import session_manager
    from castor_server.models.events import MemoryPinEvent
    from castor_server.store import repository as repo

    bus = session_manager.get_bus(session_id)
    evt = MemoryPinEvent(index=body.index)
    await bus.publish(evt)
    await repo.store_event(
        db,
        session_id=session_id,
        event_id=evt.id,
        event_type=evt.type,
        data=evt.model_dump(exclude_none=True),
    )

    return {"status": "ok", "pinned_index": body.index, "event_id": evt.id}


@router.post("/store")
async def trigger_store(
    session_id: str,
    body: StoreRequest,
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Store explicit knowledge in cold storage."""
    session = await get_session_by_id(db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    from castor_server.core.session_manager import session_manager
    from castor_server.models.events import MemoryStoreEvent
    from castor_server.store import repository as repo
    from castor_server.store.cold_storage import SQLiteVecColdStorage

    agent_id = session.agent.id
    cold = SQLiteVecColdStorage()
    try:
        await cold.store_explicit(agent_id, body.content, metadata=body.metadata)
    finally:
        cold.close()

    bus = session_manager.get_bus(session_id)
    evt = MemoryStoreEvent(
        content_preview=body.content[:100],
        metadata=body.metadata,
    )
    await bus.publish(evt)
    await repo.store_event(
        db,
        session_id=session_id,
        event_id=evt.id,
        event_type=evt.type,
        data=evt.model_dump(exclude_none=True),
    )

    return {"status": "ok", "event_id": evt.id}


@router.get("/cold", response_model=ColdListResponse)
async def list_cold_entries(
    session_id: str,
    source_filter: str | None = Query(default=None),
    limit: int = Query(default=20, le=100),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_session),
) -> ColdListResponse:
    """List cold storage entries for this agent."""
    session = await get_session_by_id(db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    from castor_server.store.cold_storage import SQLiteVecColdStorage

    agent_id = session.agent.id
    cold = SQLiteVecColdStorage()
    try:
        entries = await cold.list_entries(
            agent_id,
            source_filter=source_filter,
            limit=limit,
            offset=offset,
        )
    finally:
        cold.close()

    return ColdListResponse(
        data=[ColdEntry(**e) for e in entries],
    )
