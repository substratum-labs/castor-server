"""Session CRUD routes — /v1/sessions."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from castor_server.models.sessions import (
    CreateSessionRequest,
    SessionDeletedResponse,
    SessionListResponse,
    SessionResponse,
    UpdateSessionRequest,
)
from castor_server.store.database import get_session
from castor_server.store.repository import (
    archive_session,
    create_session,
    delete_session,
    get_agent,
    list_sessions,
    update_session,
)
from castor_server.store.repository import (
    get_session as get_session_by_id,
)

router = APIRouter(prefix="/v1/sessions", tags=["sessions"])


@router.post("", status_code=201, response_model=SessionResponse)
async def create_session_endpoint(
    body: CreateSessionRequest,
    db: AsyncSession = Depends(get_session),
) -> SessionResponse:
    # Resolve agent
    if isinstance(body.agent, str):
        agent = await get_agent(db, body.agent)
    else:
        agent = await get_agent(db, body.agent.id, version=body.agent.version)

    if agent is None:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Treat empty string as None (anthropic-python SDK requires
    # environment_id as a non-optional str, so users pass "" for
    # sessions without an environment).
    env_id = body.environment_id if body.environment_id else None

    return await create_session(
        db,
        agent=agent,
        environment_id=env_id,
        title=body.title,
        metadata=body.metadata,
        resources=body.resources,
        vault_ids=body.vault_ids,
    )


@router.get("", response_model=SessionListResponse)
async def list_sessions_endpoint(
    agent_id: str | None = Query(default=None),
    limit: int = Query(default=20, le=100),
    include_archived: bool = Query(default=False),
    order: str = Query(default="desc"),
    db: AsyncSession = Depends(get_session),
) -> SessionListResponse:
    sessions = await list_sessions(
        db,
        agent_id=agent_id,
        limit=limit,
        include_archived=include_archived,
        order=order,
    )
    return SessionListResponse(data=sessions)


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session_endpoint(
    session_id: str,
    db: AsyncSession = Depends(get_session),
) -> SessionResponse:
    session = await get_session_by_id(db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.post("/{session_id}", response_model=SessionResponse)
async def update_session_endpoint(
    session_id: str,
    body: UpdateSessionRequest,
    db: AsyncSession = Depends(get_session),
) -> SessionResponse:
    raw = body.model_dump(exclude_unset=True)
    kwargs: dict = {}
    if "title" in raw:
        kwargs["title"] = body.title
    if "metadata" in raw:
        kwargs["metadata"] = body.metadata

    result = await update_session(db, session_id, **kwargs)
    if result is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return result


@router.delete("/{session_id}", response_model=SessionDeletedResponse)
async def delete_session_endpoint(
    session_id: str,
    db: AsyncSession = Depends(get_session),
) -> SessionDeletedResponse:
    deleted = await delete_session(db, session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionDeletedResponse(id=session_id)


@router.post("/{session_id}/archive", response_model=SessionResponse)
async def archive_session_endpoint(
    session_id: str,
    db: AsyncSession = Depends(get_session),
) -> SessionResponse:
    result = await archive_session(db, session_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return result
