"""RED contracts for EPIC-30 decision management.

These tests deliberately target the Phase 3 management-plane interfaces.  They
must stay red until the server owns tenant/RBAC policy and delegates durable
control transitions to a per-session ``castord`` over ``control.sock``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from httpx import AsyncClient

from castor_server.core.aisa_client import AisaChannel, AisaClient, AisaOpcode
from castor_server.core.castord_manager import CastordProcessManager
from castor_server.models.decision_management import ManagementRole

TENANT_A = "tenant-a"
TENANT_B = "tenant-b"
SESSION_A = "session-a"
SESSION_B = "session-b"


def _headers(tenant_id: str, role: ManagementRole) -> dict[str, str]:
    return {
        "X-Castor-Tenant": tenant_id,
        "X-Castor-Role": role.value,
        "X-Castor-Operator": "operator-1",
    }


def _manager(tmp_path: Path) -> CastordProcessManager:
    return CastordProcessManager(storage_base=tmp_path / "castor")


async def _decision(
    client: AsyncClient,
    path: str,
    payload: dict[str, object],
    *,
    role: ManagementRole = ManagementRole.OPERATOR,
    tenant_id: str = TENANT_A,
):
    return await client.post(path, json=payload, headers=_headers(tenant_id, role))


@pytest.mark.asyncio
async def test_m1_cross_tenant_http_and_path_traversal_denial(client: AsyncClient):
    other_tenant = await client.get(
        f"/v1/management/sessions/{SESSION_B}/inspection",
        headers=_headers(TENANT_A, ManagementRole.VIEWER),
    )
    traversal = await client.post(
        "/v1/management/sessions",
        json={"tenant_id": "../../etc", "session_id": SESSION_A},
        headers=_headers(TENANT_A, ManagementRole.ADMIN),
    )

    assert other_tenant.status_code == 403
    assert traversal.status_code == 400


@pytest.mark.asyncio
async def test_m2_stale_generation_admission_certificate_rejected(
    client: AsyncClient, tmp_path: Path
):
    manager = _manager(tmp_path)
    session = await manager.provision(TENANT_A, SESSION_A)
    control = AisaClient(session.control_socket, channel=AisaChannel.CONTROL)
    agent = AisaClient(session.agent_socket, channel=AisaChannel.AGENT)

    await control.request(AisaOpcode.GRANT_CAPABILITY, {"cap_id": "cap-1"})
    await control.request(AisaOpcode.PERSIST_FENCE, {"generation": 2})
    result = await agent.request(
        AisaOpcode.PRESENT_ADMISSION_CERTIFICATE,
        {"capability_id": "cap-1", "generation": 1},
    )

    assert result.error_code == "RejectedStaleGeneration"


@pytest.mark.asyncio
async def test_m3_quarantined_dispute_operator_resolution_unlocks_mutex(
    client: AsyncClient, tmp_path: Path
):
    manager = _manager(tmp_path)
    session = await manager.provision(TENANT_A, SESSION_A)
    control = AisaClient(session.control_socket, channel=AisaChannel.CONTROL)
    agent = AisaClient(session.agent_socket, channel=AisaChannel.AGENT)

    await agent.request(AisaOpcode.PRESENT_SETTLEMENT_CERTIFICATE, {"attempt_id": 7})
    await agent.request(AisaOpcode.PRESENT_SETTLEMENT_CERTIFICATE, {"attempt_id": 7})
    resolved = await control.request(
        AisaOpcode.RESOLVE_QUARANTINED_DISPUTE,
        {
            "attempt_id": 7,
            "resolution": "NotApplied",
            "evidence_region_digest": "digest-1",
            "operator_id": "operator-1",
        },
    )
    successor = await agent.request(
        AisaOpcode.PRESENT_ADMISSION_CERTIFICATE,
        {"action_id": "successor", "target_scope": "orbital/burn/delta_v"},
    )

    assert resolved.persistence_disposition == "EntryPersisted"
    assert resolved.journal_entry == "QuarantinedDisputeResolved"
    assert successor.error_code is None


@pytest.mark.asyncio
async def test_m4_corrupted_journal_inspection_fails_closed(client: AsyncClient):
    response = await client.get(
        f"/v1/management/sessions/{SESSION_A}/inspection/journal",
        headers=_headers(TENANT_A, ManagementRole.VIEWER),
    )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "JournalIntegrityFault"


@pytest.mark.asyncio
async def test_m5_session_delete_reaps_daemon_and_cleans_sockets(
    client: AsyncClient, tmp_path: Path
):
    manager = _manager(tmp_path)
    session = await manager.provision(TENANT_A, SESSION_A)
    response = await client.delete(
        f"/v1/management/sessions/{SESSION_A}",
        headers=_headers(TENANT_A, ManagementRole.DEVELOPER),
    )

    assert response.status_code == 200
    assert not session.agent_socket.exists()
    assert not session.control_socket.exists()
    assert not manager.is_managed(TENANT_A, SESSION_A)


@pytest.mark.asyncio
async def test_m6_host_socket_collision_fails_closed_without_unlinking(tmp_path: Path):
    manager = _manager(tmp_path)
    session = await manager.provision(TENANT_A, SESSION_A)
    original_inode = session.agent_socket.stat().st_ino

    with pytest.raises(FileExistsError):
        await manager.provision(TENANT_A, SESSION_A)

    assert session.agent_socket.stat().st_ino == original_inode


@pytest.mark.asyncio
async def test_m7_d02_direct_datapath_unmediated_by_server(tmp_path: Path):
    manager = _manager(tmp_path)
    session = await manager.provision(TENANT_A, SESSION_A)
    agent = AisaClient(session.agent_socket, channel=AisaChannel.AGENT)

    result = await agent.request(AisaOpcode.ADMIT_TURN, {"turn_id": "turn-1"})

    assert result.error_code is None
    assert manager.management_aisa_call_count == 0


@pytest.mark.asyncio
async def test_m8_operator_capability_revocation_blocks_subsequent_arm(
    client: AsyncClient, tmp_path: Path
):
    manager = _manager(tmp_path)
    session = await manager.provision(TENANT_A, SESSION_A)
    control = AisaClient(session.control_socket, channel=AisaChannel.CONTROL)
    agent = AisaClient(session.agent_socket, channel=AisaChannel.AGENT)

    await control.request(AisaOpcode.REVOKE_CAPABILITY, {"cap_id": "cap-1"})
    result = await agent.request(
        AisaOpcode.PRESENT_ADMISSION_CERTIFICATE,
        {"capability_id": "cap-1", "generation": 1},
    )

    assert result.error_code == "RejectedCapabilityRevoked"


@pytest.mark.asyncio
async def test_m9_crash_recovery_preserves_quarantine_lock_until_resolved(
    tmp_path: Path,
):
    manager = _manager(tmp_path)
    session = await manager.provision(TENANT_A, SESSION_A)
    await manager.restart(TENANT_A, SESSION_A)
    agent = AisaClient(session.agent_socket, channel=AisaChannel.AGENT)

    blocked = await agent.request(
        AisaOpcode.PRESENT_ADMISSION_CERTIFICATE,
        {"action_id": "overlap", "target_scope": "orbital/burn/delta_v"},
    )

    assert blocked.error_code == "RejectedScopeLocked"


@pytest.mark.asyncio
async def test_m10_duplicate_operator_decision_idempotence(client: AsyncClient):
    payload = {"request_id": "decision-1", "attempt_id": 7, "resolution": "NotApplied"}
    first = await _decision(
        client, f"/v1/management/sessions/{SESSION_A}/decisions/resolve", payload
    )
    second = await _decision(
        client, f"/v1/management/sessions/{SESSION_A}/decisions/resolve", payload
    )

    assert first.status_code == 201
    assert first.json()["core_persistence_disposition"] == "EntryPersisted"
    assert second.status_code == 200
    assert second.json()["core_persistence_disposition"] == "AlreadyPersistedSameEntry"


@pytest.mark.asyncio
async def test_m11_oversized_management_payload_fails_closed(client: AsyncClient):
    response = await _decision(
        client,
        f"/v1/management/sessions/{SESSION_A}/decisions/grant",
        {"request_id": "too-large", "evidence": "x" * (64 * 1024 + 1)},
    )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "PayloadTooLarge"


@pytest.mark.asyncio
async def test_m12_rbac_unauthorized_operator_decision_rejected(client: AsyncClient):
    response = await _decision(
        client,
        f"/v1/management/sessions/{SESSION_A}/decisions/resolve",
        {"request_id": "decision-1", "attempt_id": 7, "resolution": "NotApplied"},
        role=ManagementRole.DEVELOPER,
    )

    assert response.status_code == 403


@pytest.mark.asyncio
async def test_m13_audit_log_append_only_immutability(client: AsyncClient):
    response = await _decision(
        client,
        f"/v1/management/sessions/{SESSION_A}/decisions/revoke",
        {"request_id": "revoke-1", "capability_id": "cap-1"},
    )
    audit = await client.get(
        f"/v1/management/sessions/{SESSION_A}/audit",
        headers=_headers(TENANT_A, ManagementRole.VIEWER),
    )

    assert response.status_code == 201
    assert audit.status_code == 200
    assert audit.json()["entries"][0]["operator_id"] == "operator-1"
    assert (
        audit.json()["entries"][0]["core_persistence_disposition"] == "EntryPersisted"
    )


@pytest.mark.asyncio
async def test_m14_snapshot_accelerated_inspection_loading(client: AsyncClient):
    response = await client.get(
        f"/v1/management/sessions/{SESSION_A}/inspection",
        headers=_headers(TENANT_A, ManagementRole.VIEWER),
    )

    assert response.status_code == 200
    assert response.json()["projection_source"] == "snapshot"
    assert response.json()["projection_matches_genesis_replay"] is True


@pytest.mark.asyncio
async def test_m15_session_termination_fence_before_kill(client: AsyncClient):
    response = await client.post(
        f"/v1/management/sessions/{SESSION_A}/terminate",
        headers=_headers(TENANT_A, ManagementRole.DEVELOPER),
    )

    assert response.status_code == 200
    assert response.json()["fence_persisted_before_kill"] is True


@pytest.mark.asyncio
async def test_m16_privilege_channel_containment_blocks_agent_grant(tmp_path: Path):
    manager = _manager(tmp_path)
    session = await manager.provision(TENANT_A, SESSION_A)
    agent = AisaClient(session.agent_socket, channel=AisaChannel.AGENT)

    for opcode in (
        AisaOpcode.GRANT_CAPABILITY,
        AisaOpcode.RESOLVE_QUARANTINED_DISPUTE,
        AisaOpcode.INSPECT_JOURNAL,
        AisaOpcode.GET_PROJECTION_SUMMARY,
    ):
        result = await agent.request(opcode, {})
        assert result.error_code == "UnauthorizedOpcode"


@pytest.mark.asyncio
async def test_m17_c08_server_outage_tolerance(tmp_path: Path):
    manager = _manager(tmp_path)
    session = await manager.provision(TENANT_A, SESSION_A)
    await manager.simulate_management_server_outage()
    agent = AisaClient(session.agent_socket, channel=AisaChannel.AGENT)

    result = await agent.request(AisaOpcode.COMMIT_TURN, {"turn_id": "turn-1"})

    assert result.error_code is None


@pytest.mark.asyncio
async def test_m18_idle_harvest_fences_before_shutdown(client: AsyncClient):
    response = await client.post(
        f"/v1/management/sessions/{SESSION_A}/harvest-idle",
        headers=_headers(TENANT_A, ManagementRole.DEVELOPER),
    )

    assert response.status_code == 200
    assert response.json()["fence_persisted_before_shutdown"] is True
