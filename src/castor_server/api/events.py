"""Event routes — /v1/sessions/{id}/events and SSE stream."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from castor_server.core.session_manager import session_manager
from castor_server.models.common import gen_id, now_rfc3339
from castor_server.models.events import (
    EventListResponse,
    SendEventsRequest,
    SendEventsResponse,
    UserCustomToolResult,
    UserInterrupt,
    UserMessage,
    UserToolConfirmation,
)
from castor_server.store.database import get_session
from castor_server.store.repository import (
    get_session as get_session_by_id,
)
from castor_server.store.repository import (
    list_events,
    store_event,
)

router = APIRouter(prefix="/v1/sessions/{session_id}", tags=["events"])


@router.post("/events", response_model=SendEventsResponse)
async def send_events(
    session_id: str,
    body: SendEventsRequest,
    db: AsyncSession = Depends(get_session),
) -> SendEventsResponse:
    session = await get_session_by_id(db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    sent: list[dict[str, Any]] = []

    for event in body.events:
        event_id = gen_id("evt")
        event_data = event.model_dump()
        event_data["id"] = event_id
        event_data["processed_at"] = now_rfc3339()

        # Persist event
        await store_event(
            db,
            session_id=session_id,
            event_id=event_id,
            event_type=event.type,
            data=event_data,
        )
        sent.append(event_data)

        # Dispatch to session manager (fire and forget for async processing)
        if isinstance(event, UserMessage):
            asyncio.create_task(
                session_manager.handle_user_message(
                    db, session_id, [c.model_dump() for c in event.content]
                )
            )
        elif isinstance(event, UserInterrupt):
            asyncio.create_task(session_manager.handle_interrupt(db, session_id))
        elif isinstance(event, UserToolConfirmation):
            asyncio.create_task(
                session_manager.handle_tool_confirmation(
                    db,
                    session_id,
                    event.tool_use_id,
                    event.result,
                    event.deny_message,
                )
            )
        elif isinstance(event, UserCustomToolResult):
            asyncio.create_task(
                session_manager.handle_custom_tool_result(
                    db,
                    session_id,
                    event.custom_tool_use_id,
                    [c.model_dump() for c in event.content] if event.content else None,
                    event.is_error,
                )
            )

    return SendEventsResponse(data=sent)


@router.get("/events", response_model=EventListResponse)
async def list_events_endpoint(
    session_id: str,
    limit: int = Query(default=100, le=1000),
    order: str = Query(default="asc"),
    db: AsyncSession = Depends(get_session),
) -> EventListResponse:
    session = await get_session_by_id(db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    events = await list_events(db, session_id, limit=limit, order=order)
    return EventListResponse(data=events)


@router.get("/events/stream")
async def stream_events(
    session_id: str,
    request: Request,
    db: AsyncSession = Depends(get_session),
) -> EventSourceResponse:
    session = await get_session_by_id(db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    bus = session_manager.get_bus(session_id)
    queue = bus.subscribe()

    async def event_generator():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    if event is None:
                        break
                    yield {"data": json.dumps(event)}
                except TimeoutError:
                    # Send keepalive
                    yield {"comment": "keepalive"}
        finally:
            bus.unsubscribe(queue)

    return EventSourceResponse(event_generator())


# Also support /stream as an alias (used in quickstart examples)
@router.get("/stream")
async def stream_events_alias(
    session_id: str,
    request: Request,
    db: AsyncSession = Depends(get_session),
) -> EventSourceResponse:
    return await stream_events(session_id, request, db)
