"""Northbound management-plane routes; never an Agent AISA proxy."""

from __future__ import annotations

import json
import re
import zlib
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from castor_server.core.aisa_client import AisaOpcode
from castor_server.core.castord_manager import CastordProcessManager
from castor_server.models.decision_management import ManagementRole

router = APIRouter(prefix="/v1/management", tags=["management"])
_VALID_ID = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_owners = {"session-a": "tenant-a", "session-b": "tenant-b"}
_decisions: dict[tuple[str, str, str], dict[str, Any]] = {}
_audit: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)


def _manager(tenant: str, session_id: str) -> CastordProcessManager:
    manager = CastordProcessManager._managed.get((tenant, session_id))
    if manager is None:
        raise HTTPException(status_code=404, detail="Managed session not found")
    return manager


def _decision_opcode(decision_type: str) -> AisaOpcode:
    try:
        return {
            "grant": AisaOpcode.GRANT_CAPABILITY,
            "revoke": AisaOpcode.REVOKE_CAPABILITY,
            "resolve": AisaOpcode.RESOLVE_QUARANTINED_DISPUTE,
        }[decision_type]
    except KeyError as exc:
        raise HTTPException(status_code=400, detail="Unknown decision type") from exc


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
    response = await _manager(tenant, session_id).control_request(
        tenant,
        session_id,
        _decision_opcode(decision_type),
        body,
        request_id=str(body.get("request_id", "")) or None,
    )
    disposition = response.persistence_disposition
    if response.error_code or disposition not in {
        "EntryPersisted",
        "CapabilityGranted",
        "CapabilityRevoked",
    }:
        return JSONResponse(
            status_code=409,
            content={
                "error": {"code": response.error_code or disposition or "CoreRejected"}
            },
        )
    entry = {
        "operator_id": operator,
        "timestamp": datetime.now(UTC).isoformat(),
        "decision_type": decision_type,
        "core_persistence_disposition": disposition,
    }
    _decisions[key] = entry
    _audit[(tenant, session_id)].append(entry)
    return JSONResponse(
        status_code=201, content={"core_persistence_disposition": disposition}
    )


@router.get("/sessions/{session_id}/inspection")
async def inspection(
    session_id: str,
    x_castor_tenant: str | None = Header(default=None),
    x_castor_role: str | None = Header(default=None),
    x_castor_operator: str | None = Header(default=None),
):
    tenant, role, _ = _identity(
        session_id, x_castor_tenant, x_castor_role, x_castor_operator
    )
    _require(role, {ManagementRole.VIEWER})
    response = await _manager(tenant, session_id).control_request(
        tenant, session_id, AisaOpcode.GET_PROJECTION_SUMMARY, {}
    )
    if response.error_code:
        return JSONResponse(
            status_code=409, content={"error": {"code": response.error_code}}
        )
    return {"projection_source": "core", "projection": response.outcome}


@router.get("/sessions/{session_id}/inspection/journal")
async def inspection_journal(
    session_id: str,
    x_castor_tenant: str | None = Header(default=None),
    x_castor_role: str | None = Header(default=None),
    x_castor_operator: str | None = Header(default=None),
):
    tenant, role, _ = _identity(
        session_id, x_castor_tenant, x_castor_role, x_castor_operator
    )
    _require(role, {ManagementRole.VIEWER})
    path = (
        _manager(tenant, session_id).session(tenant, session_id).storage_root
        / "core-journal.log"
    )
    try:
        data = path.read_bytes()
        offset = 0
        frames = 0
        while offset < len(data):
            if len(data) - offset < 8:
                raise ValueError("incomplete journal frame")
            size = int.from_bytes(data[offset : offset + 4], "little")
            offset += 4
            if size == 0 or len(data) - offset < size + 4:
                raise ValueError("invalid journal frame")
            payload = data[offset : offset + size]
            offset += size
            crc = int.from_bytes(data[offset : offset + 4], "little")
            offset += 4
            if zlib.crc32(payload) & 0xFFFFFFFF != crc:
                raise ValueError("journal CRC mismatch")
            frames += 1
    except (OSError, ValueError):
        return JSONResponse(
            status_code=409, content={"error": {"code": "JournalIntegrityFault"}}
        )
    return {"frames": frames}


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
    owner, parsed_role, _ = _identity(session_id, tenant, role, operator)
    _require(parsed_role, {ManagementRole.DEVELOPER})
    await _manager(owner, session_id).fence_and_reap(owner, session_id)
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
