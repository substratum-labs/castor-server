"""Fail-closed AISA client contract for the EPIC-30 management plane.

The Phase 2 contract suite imports these declarations.  Transport and protocol
implementation are intentionally deferred to T-306-C.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


class AisaChannel(StrEnum):
    """Physical endpoint selected by the caller."""

    AGENT = "agent"
    CONTROL = "control"


class AisaOpcode(StrEnum):
    """Frozen T-302-A and EPIC-30 control opcode vocabulary."""

    ADMIT_TURN = "AdmitTurn"
    COMMIT_TURN = "CommitTurn"
    REGISTER_ACTION = "RegisterAction"
    PRESENT_ADMISSION_CERTIFICATE = "PresentAdmissionCertificate"
    RECORD_DISPATCH_ATTEMPT = "RecordDispatchAttempt"
    DELIVER_ARMED_ATTEMPT = "DeliverArmedAttempt"
    PRESENT_SETTLEMENT_CERTIFICATE = "PresentSettlementCertificate"
    PERSIST_FENCE = "PersistFence"
    REVOKE_CAPABILITY = "RevokeCapability"
    REPLAY = "Replay"
    ENSURE_REGION = "EnsureRegion"
    REQUEST_INTERACTION = "RequestInteraction"
    REPORT_OUTCOME = "ReportOutcome"
    CONSUME_INTERACTION = "ConsumeInteraction"

    # The following operations are admitted only on the host control channel.
    GRANT_CAPABILITY = "GrantCapability"
    RESOLVE_QUARANTINED_DISPUTE = "ResolveQuarantinedDispute"
    INSPECT_JOURNAL = "InspectJournal"
    GET_PROJECTION_SUMMARY = "GetProjectionSummary"


@dataclass(frozen=True)
class AisaResponse:
    """Result envelope expected from a future AISA v0.1 request."""

    error_code: str | None = None
    persistence_disposition: str | None = None
    journal_entry: str | None = None
    payload: dict[str, Any] | None = None


class AisaClient:
    """Future Unix-domain-socket AISA client.

    The constructor intentionally performs no I/O.  Every request fails closed
    until the real framed socket implementation is introduced in T-306-C.
    """

    def __init__(self, socket_path: Path, *, channel: AisaChannel) -> None:
        self.socket_path = socket_path
        self.channel = channel

    async def request(
        self, opcode: AisaOpcode, payload: dict[str, Any]
    ) -> AisaResponse:
        raise NotImplementedError(
            "AISA UDS transport is not implemented; refuse management action"
        )
