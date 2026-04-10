"""Environment routes — /v1/environments CRUD."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from castor_server.models.environments import (
    CreateEnvironmentRequest,
    EnvironmentListResponse,
    EnvironmentResponse,
    UpdateEnvironmentRequest,
)
from castor_server.store.database import get_session
from castor_server.store.repository import (
    archive_environment,
    create_environment,
    get_environment,
    list_environments,
    update_environment,
)

router = APIRouter(prefix="/v1/environments", tags=["environments"])


@router.post("", response_model=EnvironmentResponse, status_code=201)
async def create_env(
    body: CreateEnvironmentRequest,
    db: AsyncSession = Depends(get_session),
) -> EnvironmentResponse:
    return await create_environment(
        db,
        name=body.name,
        image=body.image,
        memory=body.memory,
        cpus=body.cpus,
        timeout_secs=body.timeout_secs,
        network=body.network,
        writable=body.writable,
        network_allowlist=body.network_allowlist,
        metadata=body.metadata,
    )


@router.get("", response_model=EnvironmentListResponse)
async def list_envs(
    limit: int = Query(default=20, le=100),
    include_archived: bool = Query(default=False),
    db: AsyncSession = Depends(get_session),
) -> EnvironmentListResponse:
    envs = await list_environments(db, limit=limit, include_archived=include_archived)
    return EnvironmentListResponse(data=envs)


@router.get("/{env_id}", response_model=EnvironmentResponse)
async def get_env(
    env_id: str,
    db: AsyncSession = Depends(get_session),
) -> EnvironmentResponse:
    env = await get_environment(db, env_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Environment not found")
    return env


@router.post("/{env_id}", response_model=EnvironmentResponse)
async def update_env(
    env_id: str,
    body: UpdateEnvironmentRequest,
    db: AsyncSession = Depends(get_session),
) -> EnvironmentResponse:
    updates = body.model_dump(exclude_none=True)
    if not updates:
        env = await get_environment(db, env_id)
        if env is None:
            raise HTTPException(status_code=404, detail="Environment not found")
        return env

    env = await update_environment(db, env_id, **updates)
    if env is None:
        raise HTTPException(status_code=404, detail="Environment not found")
    return env


@router.post("/{env_id}/archive", response_model=EnvironmentResponse)
async def archive_env(
    env_id: str,
    db: AsyncSession = Depends(get_session),
) -> EnvironmentResponse:
    env = await archive_environment(db, env_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Environment not found")
    return env
