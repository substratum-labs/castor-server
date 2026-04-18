"""Agent CRUD routes — /v1/agents."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from castor_server.models.agents import (
    AgentListResponse,
    AgentResponse,
    AgentVersionsResponse,
    CreateAgentRequest,
    UpdateAgentRequest,
)
from castor_server.store.database import get_session
from castor_server.store.repository import (
    archive_agent,
    create_agent,
    get_agent,
    list_agent_versions,
    list_agents,
    update_agent,
)

router = APIRouter(prefix="/v1/agents", tags=["agents"])


@router.post("", status_code=201, response_model=AgentResponse)
async def create_agent_endpoint(
    body: CreateAgentRequest,
    db: AsyncSession = Depends(get_session),
) -> AgentResponse:
    return await create_agent(
        db,
        name=body.name,
        model=body.model,
        system=body.system,
        description=body.description,
        tools=body.tools,
        mcp_servers=body.mcp_servers,
        skills=body.skills,
        metadata=body.metadata,
        agent_fn_factory=body.agent_fn_factory,
    )


@router.get("", response_model=AgentListResponse)
async def list_agents_endpoint(
    limit: int = Query(default=20, le=100),
    include_archived: bool = Query(default=False),
    db: AsyncSession = Depends(get_session),
) -> AgentListResponse:
    agents = await list_agents(db, limit=limit, include_archived=include_archived)
    return AgentListResponse(data=agents)


@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent_endpoint(
    agent_id: str,
    version: int | None = Query(default=None),
    db: AsyncSession = Depends(get_session),
) -> AgentResponse:
    agent = await get_agent(db, agent_id, version=version)
    if agent is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent


@router.post("/{agent_id}", response_model=AgentResponse)
async def update_agent_endpoint(
    agent_id: str,
    body: UpdateAgentRequest,
    db: AsyncSession = Depends(get_session),
) -> AgentResponse:
    # Determine which fields were explicitly set
    update_kwargs: dict = {"expected_version": body.version}

    if body.name is not None:
        update_kwargs["name"] = body.name
    if body.model is not None:
        update_kwargs["model"] = body.model
    if body.tools is not None:
        update_kwargs["tools"] = body.tools
    if body.mcp_servers is not None:
        update_kwargs["mcp_servers"] = body.mcp_servers
    if body.skills is not None:
        update_kwargs["skills"] = body.skills
    if body.metadata is not None:
        update_kwargs["metadata"] = body.metadata

    # Handle clearable fields (empty string clears, None means not set)
    raw = body.model_dump(exclude_unset=True)
    if "system" in raw:
        update_kwargs["system"] = body.system if body.system else None
    if "description" in raw:
        update_kwargs["description"] = body.description if body.description else None

    result = await update_agent(db, agent_id, **update_kwargs)
    if result is None:
        raise HTTPException(
            status_code=409, detail="Version conflict or agent not found"
        )
    return result


@router.post("/{agent_id}/archive", response_model=AgentResponse)
async def archive_agent_endpoint(
    agent_id: str,
    db: AsyncSession = Depends(get_session),
) -> AgentResponse:
    result = await archive_agent(db, agent_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    return result


@router.get("/{agent_id}/versions", response_model=AgentVersionsResponse)
async def list_agent_versions_endpoint(
    agent_id: str,
    limit: int = Query(default=20, le=100),
    db: AsyncSession = Depends(get_session),
) -> AgentVersionsResponse:
    versions = await list_agent_versions(db, agent_id, limit=limit)
    return AgentVersionsResponse(data=versions)
