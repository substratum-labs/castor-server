"""RED contracts for EPIC-30 decision management.

These tests deliberately target the Phase 3 management-plane interfaces.  They
must stay red until the server owns tenant/RBAC policy and delegates durable
control transitions to a per-session ``castord`` over ``control.sock``.
"""

from __future__ import annotations

import hashlib
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
DIGEST = "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def _headers(tenant_id: str, role: ManagementRole) -> dict[str, str]:
    return {
        "X-Castor-Tenant": tenant_id,
        "X-Castor-Role": role.value,
        "X-Castor-Operator": "operator-1",
    }


def _manager(tmp_path: Path) -> CastordProcessManager:
    # Darwin's Unix-domain socket path limit is shorter than pytest's default
    # temporary-directory prefix.  The storage root remains unique per test.
    suffix = hashlib.sha256(str(tmp_path).encode()).hexdigest()[:16]
    return CastordProcessManager(storage_base=Path("/tmp") / f"castor-t306-{suffix}")


async def _decision(
    client: AsyncClient,
    path: str,
    payload: dict[str, object],
    *,
    role: ManagementRole = ManagementRole.OPERATOR,
    tenant_id: str = TENANT_A,
):
    return await client.post(path, json=payload, headers=_headers(tenant_id, role))


async def _grant(control: AisaClient, cap_id: str = "capability-1") -> None:
    response = await control.request(
        AisaOpcode.GRANT_CAPABILITY,
        {
            "grant": {
                "cap_id": cap_id,
                "subject": "agent-1",
                "object_ref": "scope-1",
                "rights": ["RegisterAction"],
                "constraints": [{"ScopePrefix": {"prefix": "scope-1"}}],
                "parent_cap_id": None,
                "revocation_domain": None,
                "delegation_allowed": False,
                "max_turns": None,
            }
        },
    )
    assert response.persistence_disposition == "CapabilityGranted"


async def _committed_turn(agent: AisaClient, actions: list[str]) -> None:
    manifest = ("\n".join(actions) + "\n").encode()
    manifest_digest = "sha256:" + hashlib.sha256(manifest).hexdigest()
    for opcode, payload in (
        (
            AisaOpcode.ENSURE_REGION,
            {
                "region_ref": "region://observation",
                "content_digest": DIGEST,
                "content": [],
            },
        ),
        (
            AisaOpcode.ADMIT_TURN,
            {
                "agent_id": "agent-1",
                "turn_id": 1,
                "lease_epoch": 0,
                "base_projection_digest": DIGEST,
            },
        ),
        (
            AisaOpcode.REQUEST_INTERACTION,
            {
                "interaction_id": "interaction-1",
                "lease_epoch": 0,
                "request_digest": DIGEST,
            },
        ),
        (
            AisaOpcode.REPORT_OUTCOME,
            {
                "interaction_id": "interaction-1",
                "observation_region_id": "region://observation",
                "observation_digest": DIGEST,
            },
        ),
        (
            AisaOpcode.CONSUME_INTERACTION,
            {"interaction_id": "interaction-1", "lease_epoch": 1},
        ),
        (
            AisaOpcode.ENSURE_REGION,
            {
                "region_ref": "region://manifest",
                "content_digest": manifest_digest,
                "content": list(manifest),
            },
        ),
        (
            AisaOpcode.COMMIT_TURN,
            {
                "lease_epoch": 1,
                "base_projection_digest": DIGEST,
                "successor_region_id": "region://observation",
                "successor_digest": DIGEST,
                "action_manifest_region_id": "region://manifest",
                "action_manifest_digest": manifest_digest,
                "action_manifest": actions,
            },
        ),
    ):
        assert (await agent.request(opcode, payload)).error_code is None


async def _arm(agent: AisaClient, action_id: str, *, generation: int = 1) -> None:
    registered = await agent.request(
        AisaOpcode.REGISTER_ACTION,
        {
            "action_id": action_id,
            "agent_id": "agent-1",
            "action_family": "scope-1",
            "cap_id": "capability-1",
            "target_scope": "scope-1",
        },
    )
    assert registered.persistence_disposition == "ActionRegistered"
    armed = await agent.request(
        AisaOpcode.PRESENT_ADMISSION_CERTIFICATE,
        {
            "action_id": action_id,
            "target_scope": "scope-1",
            "capability_id": "capability-1",
            "generation": generation,
        },
    )
    assert armed.persistence_disposition == "AttemptArmed"


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

    await _grant(control)
    await _committed_turn(agent, ["action-1", "action-2"])
    await _arm(agent, "action-1")
    await control.request(AisaOpcode.PERSIST_FENCE, {"generation": 2})
    result = await agent.request(
        AisaOpcode.PRESENT_ADMISSION_CERTIFICATE,
        {
            "action_id": "action-1",
            "target_scope": "scope-1",
            "capability_id": "capability-1",
            "generation": 1,
        },
    )

    assert result.persistence_disposition == "RejectedStaleGeneration"


@pytest.mark.asyncio
async def test_m3_quarantined_dispute_operator_resolution_unlocks_mutex(
    client: AsyncClient, tmp_path: Path
):
    manager = _manager(tmp_path)
    session = await manager.provision(TENANT_A, SESSION_A)
    control = AisaClient(session.control_socket, channel=AisaChannel.CONTROL)
    agent = AisaClient(session.agent_socket, channel=AisaChannel.AGENT)

    await _grant(control)
    await _committed_turn(agent, ["action-1", "action-2"])
    await _arm(agent, "action-1")
    await agent.request(
        AisaOpcode.RECORD_DISPATCH_ATTEMPT,
        {"attempt_id": 1, "dispatch_identity": "dispatch-1"},
    )
    await agent.request(
        AisaOpcode.DELIVER_ARMED_ATTEMPT,
        {"attempt_id": 1, "dispatch_identity": "dispatch-1"},
    )
    settlement = {
        "attempt_id": 1,
        "dispatch_identity": "dispatch-1",
        "evidence_region_id": "region://observation",
        "evidence_digest": DIGEST,
        "proof_class": "ProviderConfirmation",
        "resolution": "Confirmed",
    }
    await agent.request(AisaOpcode.PRESENT_SETTLEMENT_CERTIFICATE, settlement)
    conflict = b"conflict"
    conflict_digest = "sha256:" + hashlib.sha256(conflict).hexdigest()
    await agent.request(
        AisaOpcode.ENSURE_REGION,
        {
            "region_ref": "region://conflict",
            "content_digest": conflict_digest,
            "content": list(conflict),
        },
    )
    await agent.request(
        AisaOpcode.PRESENT_SETTLEMENT_CERTIFICATE,
        {
            **settlement,
            "evidence_region_id": "region://conflict",
            "evidence_digest": conflict_digest,
        },
    )
    resolved = await control.request(
        AisaOpcode.RESOLVE_QUARANTINED_DISPUTE,
        {
            "attempt_id": 1,
            "resolution": "NotApplied",
            "evidence_region_digest": "digest-1",
            "operator_id": "operator-1",
        },
    )
    successor = await agent.request(
        AisaOpcode.REGISTER_ACTION,
        {
            "action_id": "action-2",
            "agent_id": "agent-1",
            "action_family": "scope-1",
            "cap_id": "capability-1",
            "target_scope": "scope-1",
        },
    )
    armed_successor = await agent.request(
        AisaOpcode.PRESENT_ADMISSION_CERTIFICATE,
        {
            "action_id": "action-2",
            "target_scope": "scope-1",
            "capability_id": "capability-1",
            "generation": 1,
        },
    )

    assert resolved.persistence_disposition == "EntryPersisted"
    journal = await control.request(AisaOpcode.INSPECT_JOURNAL, {})
    assert "QuarantinedDisputeResolved" in str(journal.outcome)
    assert successor.persistence_disposition == "ActionRegistered"
    assert armed_successor.persistence_disposition == "AttemptArmed"


@pytest.mark.asyncio
async def test_m4_corrupted_journal_inspection_fails_closed(
    client: AsyncClient, tmp_path: Path
):
    manager = _manager(tmp_path)
    session = await manager.provision(TENANT_A, SESSION_A)
    journal = session.storage_root / "core-journal.log"
    journal.write_bytes(b"\x01\x00\x00\x00x\x00\x00\x00\x00")
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
    assert session.pid > 0
    assert session.agent_socket.is_socket()
    assert session.control_socket.is_socket()
    original_inode = session.agent_socket.stat().st_ino

    with pytest.raises(FileExistsError):
        await manager.provision(TENANT_A, SESSION_A)

    assert session.agent_socket.stat().st_ino == original_inode


@pytest.mark.asyncio
async def test_m7_d02_direct_datapath_unmediated_by_server(tmp_path: Path):
    manager = _manager(tmp_path)
    session = await manager.provision(TENANT_A, SESSION_A)
    agent = AisaClient(session.agent_socket, channel=AisaChannel.AGENT)

    result = await agent.request(
        AisaOpcode.ADMIT_TURN,
        {
            "agent_id": "agent-1",
            "turn_id": 1,
            "lease_epoch": 0,
            "base_projection_digest": DIGEST,
        },
    )

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

    await _grant(control)
    await _committed_turn(agent, ["action-1"])
    registered = await agent.request(
        AisaOpcode.REGISTER_ACTION,
        {
            "action_id": "action-1",
            "agent_id": "agent-1",
            "action_family": "scope-1",
            "cap_id": "capability-1",
            "target_scope": "scope-1",
        },
    )
    assert registered.persistence_disposition == "ActionRegistered"
    await control.request(
        AisaOpcode.REVOKE_CAPABILITY, {"capability_id": "capability-1"}
    )
    result = await agent.request(
        AisaOpcode.PRESENT_ADMISSION_CERTIFICATE,
        {
            "action_id": "action-1",
            "target_scope": "scope-1",
            "capability_id": "capability-1",
            "generation": 1,
        },
    )

    assert result.persistence_disposition == "RejectedCapabilityRevoked"


@pytest.mark.asyncio
async def test_m9_crash_recovery_preserves_quarantine_lock_until_resolved(
    tmp_path: Path,
):
    manager = _manager(tmp_path)
    session = await manager.provision(TENANT_A, SESSION_A)
    control = AisaClient(session.control_socket, channel=AisaChannel.CONTROL)
    agent = AisaClient(session.agent_socket, channel=AisaChannel.AGENT)
    await _grant(control)
    await _committed_turn(agent, ["action-1", "action-2"])
    await _arm(agent, "action-1")
    await agent.request(
        AisaOpcode.RECORD_DISPATCH_ATTEMPT,
        {"attempt_id": 1, "dispatch_identity": "dispatch-1"},
    )
    await agent.request(
        AisaOpcode.DELIVER_ARMED_ATTEMPT,
        {"attempt_id": 1, "dispatch_identity": "dispatch-1"},
    )
    settlement = {
        "attempt_id": 1,
        "dispatch_identity": "dispatch-1",
        "evidence_region_id": "region://observation",
        "evidence_digest": DIGEST,
        "proof_class": "ProviderConfirmation",
        "resolution": "Confirmed",
    }
    await agent.request(AisaOpcode.PRESENT_SETTLEMENT_CERTIFICATE, settlement)
    conflict = b"conflict"
    conflict_digest = "sha256:" + hashlib.sha256(conflict).hexdigest()
    await agent.request(
        AisaOpcode.ENSURE_REGION,
        {
            "region_ref": "region://conflict",
            "content_digest": conflict_digest,
            "content": list(conflict),
        },
    )
    await agent.request(
        AisaOpcode.PRESENT_SETTLEMENT_CERTIFICATE,
        {
            **settlement,
            "evidence_region_id": "region://conflict",
            "evidence_digest": conflict_digest,
        },
    )
    session = await manager.restart(TENANT_A, SESSION_A)
    agent = AisaClient(session.agent_socket, channel=AisaChannel.AGENT)

    registered = await agent.request(
        AisaOpcode.REGISTER_ACTION,
        {
            "action_id": "action-2",
            "agent_id": "agent-1",
            "action_family": "scope-1",
            "cap_id": "capability-1",
            "target_scope": "scope-1",
        },
    )
    blocked = await agent.request(
        AisaOpcode.PRESENT_ADMISSION_CERTIFICATE,
        {
            "action_id": "action-2",
            "target_scope": "scope-1",
            "capability_id": "capability-1",
            "generation": 1,
        },
    )

    assert registered.persistence_disposition == "ActionRegistered"
    assert blocked.persistence_disposition == "RejectedCurrentState"

    control = AisaClient(session.control_socket, channel=AisaChannel.CONTROL)
    resolved = await control.request(
        AisaOpcode.RESOLVE_QUARANTINED_DISPUTE,
        {
            "attempt_id": 1,
            "resolution": "NotApplied",
            "evidence_region_digest": "digest-post-restart",
            "operator_id": "operator-1",
        },
    )
    assert resolved.persistence_disposition == "EntryPersisted"
    unblocked = await agent.request(
        AisaOpcode.PRESENT_ADMISSION_CERTIFICATE,
        {
            "action_id": "action-2",
            "target_scope": "scope-1",
            "capability_id": "capability-1",
            "generation": 1,
        },
    )
    assert unblocked.persistence_disposition == "AttemptArmed"


@pytest.mark.asyncio
async def test_m10_duplicate_operator_decision_idempotence(
    client: AsyncClient, tmp_path: Path
):
    manager = _manager(tmp_path)
    await manager.provision(TENANT_A, SESSION_A)
    payload = {
        "request_id": "decision-1",
        "grant": {
            "cap_id": "cap-m10",
            "subject": "agent-1",
            "object_ref": "scope-1",
            "rights": ["RegisterAction"],
            "constraints": [],
            "parent_cap_id": None,
            "revocation_domain": None,
            "delegation_allowed": False,
            "max_turns": None,
        },
    }
    first = await _decision(
        client, f"/v1/management/sessions/{SESSION_A}/decisions/grant", payload
    )
    second = await _decision(
        client, f"/v1/management/sessions/{SESSION_A}/decisions/grant", payload
    )

    assert first.status_code == 201
    assert first.json()["core_persistence_disposition"] == "CapabilityGranted"
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
async def test_m13_audit_log_append_only_immutability(
    client: AsyncClient, tmp_path: Path
):
    manager = _manager(tmp_path)
    session = await manager.provision(TENANT_A, SESSION_A)
    control = AisaClient(session.control_socket, channel=AisaChannel.CONTROL)
    await _grant(control, "cap-1")
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
    assert audit.json()["entries"][-1]["operator_id"] == "operator-1"
    assert (
        audit.json()["entries"][-1]["core_persistence_disposition"]
        == "CapabilityRevoked"
    )


@pytest.mark.asyncio
async def test_m14_snapshot_accelerated_inspection_loading(
    client: AsyncClient, tmp_path: Path
):
    manager = _manager(tmp_path)
    await manager.provision(TENANT_A, SESSION_A)
    response = await client.get(
        f"/v1/management/sessions/{SESSION_A}/inspection",
        headers=_headers(TENANT_A, ManagementRole.VIEWER),
    )

    assert response.status_code == 200
    assert response.json()["projection_source"] == "core"
    assert "generation" in response.json()["projection"]


@pytest.mark.asyncio
async def test_m15_session_termination_fence_before_kill(
    client: AsyncClient, tmp_path: Path
):
    manager = _manager(tmp_path)
    await manager.provision(TENANT_A, SESSION_A)
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

    result = await agent.request(
        AisaOpcode.ADMIT_TURN,
        {
            "agent_id": "agent-1",
            "turn_id": 1,
            "lease_epoch": 0,
            "base_projection_digest": DIGEST,
        },
    )

    assert result.error_code is None


@pytest.mark.asyncio
async def test_m18_idle_harvest_fences_before_shutdown(
    client: AsyncClient, tmp_path: Path
):
    manager = _manager(tmp_path)
    await manager.provision(TENANT_A, SESSION_A)
    response = await client.post(
        f"/v1/management/sessions/{SESSION_A}/harvest-idle",
        headers=_headers(TENANT_A, ManagementRole.DEVELOPER),
    )

    assert response.status_code == 200
    assert response.json()["fence_persisted_before_shutdown"] is True
