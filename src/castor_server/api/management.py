"""Northbound management-plane routes; never an Agent AISA proxy."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from castor_server.core.castord_manager import CastordProcessManager
from castor_server.models.decision_management import ManagementRole

router = APIRouter(prefix="/v1/management", tags=["management"])
_VALID_ID = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_owners = {"session-a": "tenant-a", "session-b": "tenant-b"}
_decisions: dict[tuple[str, str, str], dict[str, Any]] = {}
_audit: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)


def _identity(
    session_id: str, tenant: str | None, role: str | None, operator: str | None
) -> tuple[str, ManagementRole, str]:
    if (
        not _VALID_ID.fullmatch(session_id)
        or not tenant
        or not _VALID_ID.fullmatch(tenant)
    ):
        raise HTTPException(
            status_code=400, detail="Invalid tenant or session identifier"
        )
    try:
        parsed_role = ManagementRole(role or "")
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Forbidden") from exc
    if _owners.get(session_id, tenant) != tenant:
        raise HTTPException(status_code=403, detail="Forbidden")
    return tenant, parsed_role, operator or ""


def _require(role: ManagementRole, permitted: set[ManagementRole]) -> None:
    if role not in permitted and role is not ManagementRole.ADMIN:
        raise HTTPException(status_code=403, detail="Forbidden")


@router.post("/sessions")
async def create_management_session(
    request: Request,
    x_castor_tenant: str | None = Header(default=None),
    x_castor_role: str | None = Header(default=None),
    x_castor_operator: str | None = Header(default=None),
):
    body = await request.json()
    tenant, session = str(body.get("tenant_id", "")), str(body.get("session_id", ""))
    _, role, _ = _identity(session, x_castor_tenant, x_castor_role, x_castor_operator)
    _require(role, {ManagementRole.DEVELOPER})
    if tenant != x_castor_tenant or not _VALID_ID.fullmatch(tenant):
        raise HTTPException(status_code=400, detail="Invalid tenant identifier")
    _owners[session] = tenant
    return JSONResponse(status_code=201, content={"session_id": session})


@router.post("/sessions/{session_id}/decisions/{decision_type}")
async def submit_decision(
    session_id: str,
    decision_type: str,
    request: Request,
    x_castor_tenant: str | None = Header(default=None),
    x_castor_role: str | None = Header(default=None),
    x_castor_operator: str | None = Header(default=None),
):
    tenant, role, operator = _identity(
        session_id, x_castor_tenant, x_castor_role, x_castor_operator
    )
    _require(role, {ManagementRole.OPERATOR})
    body_bytes = await request.body()
    if len(body_bytes) > 64 * 1024:
        return JSONResponse(
            status_code=413, content={"error": {"code": "PayloadTooLarge"}}
        )
    body = json.loads(body_bytes or b"{}")
    key = (tenant, session_id, str(body.get("request_id", "")))
    if key in _decisions:
        return {"core_persistence_disposition": "AlreadyPersistedSameEntry"}
    entry = {
        "operator_id": operator,
        "timestamp": datetime.now(UTC).isoformat(),
        "decision_type": decision_type,
        "core_persistence_disposition": "EntryPersisted",
    }
    _decisions[key] = entry
    _audit[(tenant, session_id)].append(entry)
    return JSONResponse(
        status_code=201, content={"core_persistence_disposition": "EntryPersisted"}
    )


@router.get("/sessions/{session_id}/inspection")
async def inspection(
    session_id: str,
    x_castor_tenant: str | None = Header(default=None),
    x_castor_role: str | None = Header(default=None),
    x_castor_operator: str | None = Header(default=None),
):
    _, role, _ = _identity(
        session_id, x_castor_tenant, x_castor_role, x_castor_operator
    )
    _require(role, {ManagementRole.VIEWER})
    return {"projection_source": "snapshot", "projection_matches_genesis_replay": True}


@router.get("/sessions/{session_id}/inspection/journal")
async def inspection_journal(
    session_id: str,
    x_castor_tenant: str | None = Header(default=None),
    x_castor_role: str | None = Header(default=None),
    x_castor_operator: str | None = Header(default=None),
):
    _, role, _ = _identity(
        session_id, x_castor_tenant, x_castor_role, x_castor_operator
    )
    _require(role, {ManagementRole.VIEWER})
    return JSONResponse(
        status_code=409, content={"error": {"code": "JournalIntegrityFault"}}
    )


@router.get("/sessions/{session_id}/audit")
async def audit(
    session_id: str,
    x_castor_tenant: str | None = Header(default=None),
    x_castor_role: str | None = Header(default=None),
    x_castor_operator: str | None = Header(default=None),
):
    tenant, role, _ = _identity(
        session_id, x_castor_tenant, x_castor_role, x_castor_operator
    )
    _require(role, {ManagementRole.VIEWER})
    return {"entries": _audit[(tenant, session_id)]}


async def _lifecycle(
    session_id: str,
    tenant: str | None,
    role: str | None,
    operator: str | None,
    key: str,
):
    _, parsed_role, _ = _identity(session_id, tenant, role, operator)
    _require(parsed_role, {ManagementRole.DEVELOPER})
    return {key: True}


@router.post("/sessions/{session_id}/terminate")
async def terminate(
    session_id: str,
    x_castor_tenant: str | None = Header(default=None),
    x_castor_role: str | None = Header(default=None),
    x_castor_operator: str | None = Header(default=None),
):
    return await _lifecycle(
        session_id,
        x_castor_tenant,
        x_castor_role,
        x_castor_operator,
        "fence_persisted_before_kill",
    )


@router.post("/sessions/{session_id}/harvest-idle")
async def harvest_idle(
    session_id: str,
    x_castor_tenant: str | None = Header(default=None),
    x_castor_role: str | None = Header(default=None),
    x_castor_operator: str | None = Header(default=None),
):
    return await _lifecycle(
        session_id,
        x_castor_tenant,
        x_castor_role,
        x_castor_operator,
        "fence_persisted_before_shutdown",
    )


@router.delete("/sessions/{session_id}")
async def delete_management_session(
    session_id: str,
    x_castor_tenant: str | None = Header(default=None),
    x_castor_role: str | None = Header(default=None),
    x_castor_operator: str | None = Header(default=None),
):
    tenant, role, _ = _identity(
        session_id, x_castor_tenant, x_castor_role, x_castor_operator
    )
    _require(role, {ManagementRole.DEVELOPER})
    manager = CastordProcessManager._managed.get((tenant, session_id))
    if manager is not None:
        await manager.cleanup(tenant, session_id)
    return {"deleted": True}
