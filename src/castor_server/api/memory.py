"""Memory API — /v1/sessions/{id}/memory/* routes.

HTTP surface for AISA §2.2 memory operations. Each route maps to a
kernel memory syscall (mem_write, mem_read, mem_search, mem_delete,
mem_evict, mem_promote, mem_protect) using memory_id-based addressing.
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


class WriteRequest(BaseModel):
    content: str = Field(..., description="Content to store")
    metadata: dict[str, Any] = Field(default_factory=dict)
    pin: bool = False


class ReadRequest(BaseModel):
    memory_id: str


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query for cold storage")
    limit: int = Field(default=5, ge=1, le=50)
    filter: dict[str, Any] | None = None


class DeleteRequest(BaseModel):
    memory_id: str


class EvictRequest(BaseModel):
    memory_id: str = Field(..., description="Memory ID to evict to cold storage")
    summary: str | None = None


class PromoteRequest(BaseModel):
    memory_id: str = Field(..., description="Memory ID to promote from cold to context")


class ProtectRequest(BaseModel):
    memory_id: str = Field(..., description="Memory ID to protect/unprotect")
    protect: bool = True


class MemoryStateResponse(BaseModel):
    token_count: int = 0
    message_count: int = 0
    pinned_ids: list[str] = Field(default_factory=list)
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
# Helper
# ---------------------------------------------------------------------------


async def _emit(bus, db, session_id, event):
    from castor_server.store import repository as repo

    await bus.publish(event)
    await repo.store_event(
        db,
        session_id=session_id,
        event_id=event.id,
        event_type=event.type,
        data=event.model_dump(exclude_none=True),
    )


def _get_cold():
    from castor_server.store.cold_storage import SQLiteVecColdStorage

    return SQLiteVecColdStorage()


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
    pinned_ids = []
    total_tokens = 0
    for m in context:
        pinned = m.pinned if hasattr(m, "pinned") else m.get("pinned", False)
        mid = getattr(m, "id", None) or m.get("id", "")
        if pinned and mid:
            pinned_ids.append(mid)
        content = getattr(m, "content", m.get("content", ""))
        total_tokens += max(1, len(str(content)) // 4)

    return MemoryStateResponse(
        token_count=total_tokens,
        message_count=len(context),
        pinned_ids=pinned_ids,
        watermark=8000,
    )


@router.post("/write")
async def memory_write(
    session_id: str,
    body: WriteRequest,
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Store explicit knowledge (mem_write syscall)."""
    session = await get_session_by_id(db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    from castor_server.core.session_manager import session_manager
    from castor_server.models.events import MemoryWriteEvent

    cold = _get_cold()
    try:
        await cold.store_explicit(
            session.agent.id, body.content, metadata=body.metadata
        )
    finally:
        cold.close()

    bus = session_manager.get_bus(session_id)
    evt = MemoryWriteEvent(content_preview=body.content[:100], metadata=body.metadata)
    await _emit(bus, db, session_id, evt)

    return {"status": "ok", "event_id": evt.id}


@router.post("/read")
async def memory_read(
    session_id: str,
    body: ReadRequest,
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Read a specific memory entry by ID (mem_read syscall)."""
    session = await get_session_by_id(db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    from castor_server.core.session_manager import session_manager
    from castor_server.models.events import MemoryReadEvent

    cold = _get_cold()
    try:
        entry = await cold.read(session.agent.id, body.memory_id)
    finally:
        cold.close()

    bus = session_manager.get_bus(session_id)
    evt = MemoryReadEvent(memory_id=body.memory_id, found=entry is not None)
    await _emit(bus, db, session_id, evt)

    return {"entry": entry, "event_id": evt.id}


@router.post("/search")
async def memory_search(
    session_id: str,
    body: SearchRequest,
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Search cold storage (mem_search syscall)."""
    session = await get_session_by_id(db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    from castor_server.core.session_manager import session_manager
    from castor_server.models.events import MemorySearchEvent

    cold = _get_cold()
    try:
        results = await cold.search(
            session.agent.id,
            body.query,
            max_results=body.limit,
            filter=body.filter,
        )
    finally:
        cold.close()

    bus = session_manager.get_bus(session_id)
    evt = MemorySearchEvent(query=body.query, result_count=len(results))
    await _emit(bus, db, session_id, evt)

    return {"results": results, "event_id": evt.id}


@router.post("/delete")
async def memory_delete(
    session_id: str,
    body: DeleteRequest,
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Delete a memory entry (mem_delete syscall, irreversible)."""
    session = await get_session_by_id(db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    from castor_server.core.session_manager import session_manager
    from castor_server.models.events import MemoryDeleteEvent

    cold = _get_cold()
    try:
        deleted = await cold.delete(session.agent.id, body.memory_id)
    finally:
        cold.close()

    bus = session_manager.get_bus(session_id)
    evt = MemoryDeleteEvent(memory_id=body.memory_id, deleted=deleted)
    await _emit(bus, db, session_id, evt)

    return {"deleted": deleted, "event_id": evt.id}


@router.post("/evict")
async def memory_evict(
    session_id: str,
    body: EvictRequest,
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Evict a message from context to cold storage (mem_evict)."""
    session = await get_session_by_id(db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    from castor_server.core.session_manager import session_manager
    from castor_server.models.events import MemoryEvictEvent

    bus = session_manager.get_bus(session_id)
    evt = MemoryEvictEvent(memory_id=body.memory_id, summary=body.summary)
    await _emit(bus, db, session_id, evt)

    return {
        "status": "ok",
        "memory_id": body.memory_id,
        "event_id": evt.id,
    }


@router.post("/promote")
async def memory_promote(
    session_id: str,
    body: PromoteRequest,
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Promote from cold storage back to context (mem_promote)."""
    session = await get_session_by_id(db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    from castor_server.core.session_manager import session_manager
    from castor_server.models.events import MemoryPromoteEvent

    bus = session_manager.get_bus(session_id)
    evt = MemoryPromoteEvent(memory_id=body.memory_id)
    await _emit(bus, db, session_id, evt)

    return {
        "status": "ok",
        "memory_id": body.memory_id,
        "event_id": evt.id,
    }


@router.post("/protect")
async def memory_protect(
    session_id: str,
    body: ProtectRequest,
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Set/clear protection on a message (mem_protect syscall)."""
    session = await get_session_by_id(db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    from castor_server.core.session_manager import session_manager
    from castor_server.models.events import MemoryProtectEvent

    bus = session_manager.get_bus(session_id)
    evt = MemoryProtectEvent(memory_id=body.memory_id, protect=body.protect)
    await _emit(bus, db, session_id, evt)

    return {
        "status": "ok",
        "memory_id": body.memory_id,
        "protected": body.protect,
        "event_id": evt.id,
    }


@router.get("/cold", response_model=ColdListResponse)
async def list_cold_entries(
    session_id: str,
    source: str | None = Query(default=None),
    limit: int = Query(default=20, le=100),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_session),
) -> ColdListResponse:
    """List cold storage entries for this agent."""
    session = await get_session_by_id(db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    cold = _get_cold()
    try:
        f = {"source": source} if source else None
        entries = await cold.list_entries(
            session.agent.id, filter=f, limit=limit, offset=offset
        )
    finally:
        cold.close()

    return ColdListResponse(data=[ColdEntry(**e) for e in entries])
