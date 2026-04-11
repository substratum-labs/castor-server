"""Data access layer for agents, sessions, and events."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from castor_server.models.agents import AgentResponse
from castor_server.models.common import ModelConfig, gen_id
from castor_server.models.environments import EnvironmentResponse
from castor_server.models.files import FileMetadata
from castor_server.models.sessions import (
    SessionResponse,
    SessionStats,
    SessionUsage,
)

from .db_models import AgentRow, EnvironmentRow, EventRow, FileRow, SessionRow

# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------


def _dump_list(items: list | None) -> list:
    """Dump pydantic models to dicts, omitting None sub-fields.

    Used for tool definitions, MCP servers, skills, resources — none of these
    have meaningful None sub-fields, and stripping them keeps the agent/session
    response close to the Anthropic wire format.
    """
    return [
        i.model_dump(exclude_none=True) if hasattr(i, "model_dump") else i
        for i in (items or [])
    ]


def _model_to_config(model: str | ModelConfig) -> dict:
    if isinstance(model, str):
        return {"id": model, "speed": "standard"}
    return model.model_dump()


def _agent_row_to_response(row: AgentRow) -> AgentResponse:
    return AgentResponse(
        id=row.id,
        name=row.name,
        description=row.description,
        model=ModelConfig(**row.model_config_json),
        system=row.system,
        tools=row.tools_json,
        mcp_servers=row.mcp_servers_json,
        skills=row.skills_json,
        metadata={k: v for k, v in (row.metadata_json or {}).items() if v is not None},
        version=row.version,
        created_at=row.created_at.isoformat(timespec="milliseconds") + "Z",
        updated_at=row.updated_at.isoformat(timespec="milliseconds") + "Z",
        archived_at=(
            row.archived_at.isoformat(timespec="milliseconds") + "Z"
            if row.archived_at
            else None
        ),
    )


async def create_agent(
    db: AsyncSession,
    *,
    name: str,
    model: str | ModelConfig,
    system: str | None = None,
    description: str | None = None,
    tools: list | None = None,
    mcp_servers: list | None = None,
    skills: list | None = None,
    metadata: dict | None = None,
) -> AgentResponse:
    agent_id = gen_id("agent")
    now = datetime.utcnow()
    row = AgentRow(
        id=agent_id,
        version=1,
        name=name,
        description=description,
        model_config_json=_model_to_config(model),
        system=system,
        tools_json=_dump_list(tools),
        mcp_servers_json=_dump_list(mcp_servers),
        skills_json=_dump_list(skills),
        metadata_json=metadata or {},
        created_at=now,
        updated_at=now,
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return _agent_row_to_response(row)


async def get_agent(
    db: AsyncSession, agent_id: str, *, version: int | None = None
) -> AgentResponse | None:
    if version is not None:
        stmt = select(AgentRow).where(
            AgentRow.id == agent_id, AgentRow.version == version
        )
    else:
        stmt = (
            select(AgentRow)
            .where(AgentRow.id == agent_id)
            .order_by(AgentRow.version.desc())
            .limit(1)
        )
    result = await db.execute(stmt)
    row = result.scalar_one_or_none()
    return _agent_row_to_response(row) if row else None


async def list_agents(
    db: AsyncSession,
    *,
    limit: int = 20,
    include_archived: bool = False,
) -> list[AgentResponse]:
    # Get latest version of each agent via subquery
    from sqlalchemy import func as sqlfunc

    latest = (
        select(AgentRow.id, sqlfunc.max(AgentRow.version).label("max_ver"))
        .group_by(AgentRow.id)
        .subquery()
    )
    stmt = (
        select(AgentRow)
        .join(
            latest,
            (AgentRow.id == latest.c.id) & (AgentRow.version == latest.c.max_ver),
        )
        .order_by(AgentRow.created_at.desc())
        .limit(limit)
    )
    if not include_archived:
        stmt = stmt.where(AgentRow.archived_at.is_(None))
    result = await db.execute(stmt)
    return [_agent_row_to_response(r) for r in result.scalars().all()]


async def update_agent(
    db: AsyncSession,
    agent_id: str,
    *,
    expected_version: int,
    name: str | None = None,
    model: str | ModelConfig | None = None,
    system: str | None = ...,  # type: ignore[assignment]
    description: str | None = ...,  # type: ignore[assignment]
    tools: list | None = None,
    mcp_servers: list | None = None,
    skills: list | None = None,
    metadata: dict | None = None,
) -> AgentResponse | None:
    current = await get_agent(db, agent_id)
    if current is None or current.version != expected_version:
        return None

    # Get current row
    stmt = select(AgentRow).where(
        AgentRow.id == agent_id, AgentRow.version == expected_version
    )
    result = await db.execute(stmt)
    current_row = result.scalar_one()

    now = datetime.utcnow()
    new_version = expected_version + 1

    new_row = AgentRow(
        id=agent_id,
        version=new_version,
        name=name if name is not None else current_row.name,
        description=(
            description if description is not ... else current_row.description
        ),
        model_config_json=(
            _model_to_config(model)
            if model is not None
            else current_row.model_config_json
        ),
        system=(system if system is not ... else current_row.system),
        tools_json=(_dump_list(tools) if tools is not None else current_row.tools_json),
        mcp_servers_json=(
            _dump_list(mcp_servers)
            if mcp_servers is not None
            else current_row.mcp_servers_json
        ),
        skills_json=(
            _dump_list(skills) if skills is not None else current_row.skills_json
        ),
        metadata_json=_merge_metadata(current_row.metadata_json or {}, metadata),
        created_at=current_row.created_at,
        updated_at=now,
        archived_at=current_row.archived_at,
    )
    db.add(new_row)
    await db.commit()
    await db.refresh(new_row)
    return _agent_row_to_response(new_row)


def _merge_metadata(current: dict, patch: dict | None) -> dict:
    if patch is None:
        return current
    merged = dict(current)
    for k, v in patch.items():
        if v is None:
            merged.pop(k, None)
        else:
            merged[k] = v
    return merged


async def archive_agent(db: AsyncSession, agent_id: str) -> AgentResponse | None:
    current = await get_agent(db, agent_id)
    if current is None:
        return None
    stmt = select(AgentRow).where(
        AgentRow.id == agent_id, AgentRow.version == current.version
    )
    result = await db.execute(stmt)
    row = result.scalar_one()
    row.archived_at = datetime.utcnow()
    await db.commit()
    await db.refresh(row)
    return _agent_row_to_response(row)


async def list_agent_versions(
    db: AsyncSession, agent_id: str, *, limit: int = 20
) -> list[AgentResponse]:
    stmt = (
        select(AgentRow)
        .where(AgentRow.id == agent_id)
        .order_by(AgentRow.version.desc())
        .limit(limit)
    )
    result = await db.execute(stmt)
    return [_agent_row_to_response(r) for r in result.scalars().all()]


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


def _session_row_to_response(row: SessionRow, agent: AgentResponse) -> SessionResponse:
    return SessionResponse(
        id=row.id,
        agent=agent,
        environment_id=row.environment_id,
        title=row.title,
        status=row.status,
        metadata={k: v for k, v in (row.metadata_json or {}).items() if v is not None},
        resources=row.resources_json or [],
        stats=SessionStats(**(row.stats_json or {})),
        usage=SessionUsage(**(row.usage_json or {})),
        vault_ids=row.vault_ids_json or [],
        created_at=row.created_at.isoformat(timespec="milliseconds") + "Z",
        updated_at=row.updated_at.isoformat(timespec="milliseconds") + "Z",
        archived_at=(
            row.archived_at.isoformat(timespec="milliseconds") + "Z"
            if row.archived_at
            else None
        ),
    )


async def create_session(
    db: AsyncSession,
    *,
    agent: AgentResponse,
    environment_id: str | None = None,
    title: str | None = None,
    metadata: dict | None = None,
    resources: list | None = None,
    vault_ids: list[str] | None = None,
) -> SessionResponse:
    session_id = gen_id("session")
    now = datetime.utcnow()
    row = SessionRow(
        id=session_id,
        agent_id=agent.id,
        agent_version=agent.version,
        environment_id=environment_id,
        title=title,
        status="idle",
        metadata_json=metadata or {},
        resources_json=_dump_list(resources),
        vault_ids_json=vault_ids or [],
        stats_json={},
        usage_json={},
        created_at=now,
        updated_at=now,
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return _session_row_to_response(row, agent)


async def get_session(db: AsyncSession, session_id: str) -> SessionResponse | None:
    stmt = select(SessionRow).where(SessionRow.id == session_id)
    result = await db.execute(stmt)
    row = result.scalar_one_or_none()
    if row is None:
        return None
    agent = await get_agent(db, row.agent_id, version=row.agent_version)
    if agent is None:
        return None
    return _session_row_to_response(row, agent)


async def get_session_row(db: AsyncSession, session_id: str) -> SessionRow | None:
    stmt = select(SessionRow).where(SessionRow.id == session_id)
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def update_session_status(db: AsyncSession, session_id: str, status: str) -> None:
    row = await get_session_row(db, session_id)
    if row:
        row.status = status
        row.updated_at = datetime.utcnow()
        await db.commit()


async def update_session_checkpoint(
    db: AsyncSession, session_id: str, checkpoint_json: dict
) -> None:
    row = await get_session_row(db, session_id)
    if row:
        row.checkpoint_json = checkpoint_json
        row.updated_at = datetime.utcnow()
        await db.commit()


async def update_session_usage(db: AsyncSession, session_id: str, usage: dict) -> None:
    row = await get_session_row(db, session_id)
    if row:
        row.usage_json = usage
        row.updated_at = datetime.utcnow()
        await db.commit()


async def list_sessions(
    db: AsyncSession,
    *,
    agent_id: str | None = None,
    limit: int = 20,
    include_archived: bool = False,
    order: str = "desc",
) -> list[SessionResponse]:
    stmt = select(SessionRow)
    if agent_id:
        stmt = stmt.where(SessionRow.agent_id == agent_id)
    if not include_archived:
        stmt = stmt.where(SessionRow.archived_at.is_(None))
    if order == "asc":
        stmt = stmt.order_by(SessionRow.created_at.asc())
    else:
        stmt = stmt.order_by(SessionRow.created_at.desc())
    stmt = stmt.limit(limit)
    result = await db.execute(stmt)
    sessions = []
    for row in result.scalars().all():
        agent = await get_agent(db, row.agent_id, version=row.agent_version)
        if agent:
            sessions.append(_session_row_to_response(row, agent))
    return sessions


async def update_session(
    db: AsyncSession,
    session_id: str,
    *,
    title: str | None = ...,  # type: ignore[assignment]
    metadata: dict | None = None,
) -> SessionResponse | None:
    row = await get_session_row(db, session_id)
    if row is None:
        return None
    if title is not ...:
        row.title = title
    if metadata is not None:
        row.metadata_json = _merge_metadata(row.metadata_json or {}, metadata)
    row.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(row)
    agent = await get_agent(db, row.agent_id, version=row.agent_version)
    return _session_row_to_response(row, agent) if agent else None


async def delete_session(db: AsyncSession, session_id: str) -> bool:
    row = await get_session_row(db, session_id)
    if row is None:
        return False
    await db.delete(row)
    await db.commit()
    return True


async def archive_session(db: AsyncSession, session_id: str) -> SessionResponse | None:
    row = await get_session_row(db, session_id)
    if row is None:
        return None
    row.archived_at = datetime.utcnow()
    await db.commit()
    await db.refresh(row)
    agent = await get_agent(db, row.agent_id, version=row.agent_version)
    return _session_row_to_response(row, agent) if agent else None


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


async def store_event(
    db: AsyncSession,
    *,
    session_id: str,
    event_id: str,
    event_type: str,
    data: dict[str, Any],
) -> None:
    row = EventRow(
        id=event_id,
        session_id=session_id,
        type=event_type,
        data_json=data,
    )
    db.add(row)
    await db.commit()


async def list_events(
    db: AsyncSession,
    session_id: str,
    *,
    limit: int = 100,
    order: str = "asc",
) -> list[dict[str, Any]]:
    stmt = select(EventRow).where(EventRow.session_id == session_id)
    if order == "asc":
        stmt = stmt.order_by(EventRow.processed_at.asc())
    else:
        stmt = stmt.order_by(EventRow.processed_at.desc())
    stmt = stmt.limit(limit)
    result = await db.execute(stmt)
    return [row.data_json for row in result.scalars().all()]


# ---------------------------------------------------------------------------
# Environments
# ---------------------------------------------------------------------------


def _env_row_to_response(row: EnvironmentRow) -> EnvironmentResponse:
    return EnvironmentResponse(
        id=row.id,
        name=row.name,
        image=row.image,
        memory=row.memory,
        cpus=row.cpus,
        timeout_secs=row.timeout_secs,
        network=row.network,
        writable=row.writable,
        network_allowlist=row.network_allowlist_json or [],
        metadata={k: v for k, v in (row.metadata_json or {}).items() if v is not None},
        created_at=row.created_at.isoformat(timespec="milliseconds") + "Z",
        updated_at=row.updated_at.isoformat(timespec="milliseconds") + "Z",
        archived_at=(
            row.archived_at.isoformat(timespec="milliseconds") + "Z"
            if row.archived_at
            else None
        ),
    )


async def create_environment(
    db: AsyncSession,
    *,
    name: str,
    image: str = "python:3.12-slim",
    memory: str | None = None,
    cpus: float | None = None,
    timeout_secs: int = 300,
    network: bool = False,
    writable: bool = True,
    network_allowlist: list[str] | None = None,
    metadata: dict | None = None,
) -> EnvironmentResponse:
    env_id = gen_id("env")
    now = datetime.utcnow()
    row = EnvironmentRow(
        id=env_id,
        name=name,
        image=image,
        memory=memory,
        cpus=cpus,
        timeout_secs=timeout_secs,
        network=network,
        writable=writable,
        network_allowlist_json=network_allowlist or [],
        metadata_json=metadata or {},
        created_at=now,
        updated_at=now,
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return _env_row_to_response(row)


async def get_environment(db: AsyncSession, env_id: str) -> EnvironmentResponse | None:
    stmt = select(EnvironmentRow).where(EnvironmentRow.id == env_id)
    result = await db.execute(stmt)
    row = result.scalar_one_or_none()
    if row is None:
        return None
    return _env_row_to_response(row)


async def list_environments(
    db: AsyncSession,
    *,
    limit: int = 20,
    include_archived: bool = False,
) -> list[EnvironmentResponse]:
    stmt = select(EnvironmentRow)
    if not include_archived:
        stmt = stmt.where(EnvironmentRow.archived_at.is_(None))
    stmt = stmt.order_by(EnvironmentRow.created_at.desc()).limit(limit)
    result = await db.execute(stmt)
    return [_env_row_to_response(row) for row in result.scalars().all()]


async def update_environment(
    db: AsyncSession,
    env_id: str,
    **updates: Any,
) -> EnvironmentResponse | None:
    stmt = select(EnvironmentRow).where(EnvironmentRow.id == env_id)
    result = await db.execute(stmt)
    row = result.scalar_one_or_none()
    if row is None:
        return None

    field_map = {
        "name": "name",
        "image": "image",
        "memory": "memory",
        "cpus": "cpus",
        "timeout_secs": "timeout_secs",
        "network": "network",
        "writable": "writable",
        "network_allowlist": "network_allowlist_json",
        "metadata": "metadata_json",
    }
    for key, value in updates.items():
        if value is not None and key in field_map:
            setattr(row, field_map[key], value)

    row.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(row)
    return _env_row_to_response(row)


async def archive_environment(
    db: AsyncSession, env_id: str
) -> EnvironmentResponse | None:
    stmt = select(EnvironmentRow).where(EnvironmentRow.id == env_id)
    result = await db.execute(stmt)
    row = result.scalar_one_or_none()
    if row is None:
        return None
    row.archived_at = datetime.utcnow()
    row.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(row)
    return _env_row_to_response(row)


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


def _file_row_to_response(row: FileRow) -> FileMetadata:
    return FileMetadata(
        id=row.id,
        filename=row.filename,
        mime_type=row.mime_type,
        size_bytes=row.size_bytes,
        scope=row.scope,
        created_at=row.created_at.isoformat(timespec="milliseconds") + "Z",
    )


async def create_file(
    db: AsyncSession,
    *,
    file_id: str,
    filename: str,
    mime_type: str,
    size_bytes: int,
    scope: str | None = None,
) -> FileMetadata:
    row = FileRow(
        id=file_id,
        filename=filename,
        mime_type=mime_type,
        size_bytes=size_bytes,
        scope=scope,
        created_at=datetime.utcnow(),
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return _file_row_to_response(row)


async def get_file(db: AsyncSession, file_id: str) -> FileMetadata | None:
    stmt = select(FileRow).where(FileRow.id == file_id)
    result = await db.execute(stmt)
    row = result.scalar_one_or_none()
    return _file_row_to_response(row) if row else None


async def list_files(db: AsyncSession, *, limit: int = 100) -> list[FileMetadata]:
    stmt = select(FileRow).order_by(FileRow.created_at.desc()).limit(limit)
    result = await db.execute(stmt)
    return [_file_row_to_response(r) for r in result.scalars().all()]


async def delete_file(db: AsyncSession, file_id: str) -> bool:
    stmt = select(FileRow).where(FileRow.id == file_id)
    result = await db.execute(stmt)
    row = result.scalar_one_or_none()
    if row is None:
        return False
    await db.delete(row)
    await db.commit()
    return True
