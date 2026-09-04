"""AISA v0.1 Unix-domain-socket client."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


class AisaChannel(StrEnum):
    AGENT = "agent"
    CONTROL = "control"


class AisaOpcode(StrEnum):
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

    GRANT_CAPABILITY = "GrantCapability"
    RESOLVE_QUARANTINED_DISPUTE = "ResolveQuarantinedDispute"
    INSPECT_JOURNAL = "InspectJournal"
    GET_PROJECTION_SUMMARY = "GetProjectionSummary"


@dataclass(frozen=True)
class AisaResponse:
    error_code: str | None = None
    persistence_disposition: str | None = None
    journal_entry: str | None = None
    payload: dict[str, Any] | None = None


class AisaClient:
    """Send one bounded, length-prefixed AISA request over a UDS."""

    def __init__(self, socket_path: Path, *, channel: AisaChannel) -> None:
        self.socket_path = socket_path
        self.channel = channel

    async def request(
        self, opcode: AisaOpcode, payload: dict[str, Any]
    ) -> AisaResponse:
        request = json.dumps(
            {"opcode": opcode.value, "payload": payload}, separators=(",", ":")
        ).encode()
        if len(request) > 16 * 1024 * 1024:
            return AisaResponse(error_code="PayloadTooLarge")
        reader, writer = await asyncio.open_unix_connection(
            str(self.socket_path.resolve())
        )
        try:
            writer.write(len(request).to_bytes(4, "big") + request)
            await writer.drain()
            size = int.from_bytes(await reader.readexactly(4), "big")
            if size > 16 * 1024 * 1024:
                return AisaResponse(error_code="PayloadTooLarge")
            raw = json.loads((await reader.readexactly(size)).decode())
            return AisaResponse(
                error_code=raw.get("error_code"),
                persistence_disposition=raw.get("persistence_disposition"),
                journal_entry=raw.get("journal_entry"),
                payload=raw.get("payload"),
            )
        finally:
            writer.close()
            await writer.wait_closed()
