"""Minimal standalone reproduction of the anthropic-python 0.93.0 Stream bug.

No HTTP server, no network, no API key. Pure SDK + mocked response.
Paste into a bug report directly.

THE BUG
-------
``client.beta.sessions.events.stream()`` is typed as
``Stream[BetaManagedAgentsStreamSessionEvents]`` but its iterator yields
nothing because ``Stream.__stream__`` is hardcoded to match only the
Messages API event names (``completion``, ``message_start``,
``content_block_*``, ``ping``, ``error``). Managed agents event types
(``session.status_*``, ``agent.message``, ``agent.tool_use``,
``span.model_request_*``, ``session.error``, etc.) are all silently dropped.

RUN
---
    uv run python scripts/sdk_bug_repro.py
"""

from __future__ import annotations

from unittest.mock import MagicMock

from anthropic._streaming import SSEDecoder, Stream

# Three managed-agents events as they appear on the wire from any
# Anthropic-compatible managed agents server.
SSE_WIRE_BYTES = (
    b'data: {"id":"evt_1","type":"session.status_running",'
    b'"processed_at":"2026-04-10T00:00:00.000Z"}\n\n'
    b'data: {"id":"evt_2","type":"agent.message",'
    b'"content":[{"type":"text","text":"hello"}],'
    b'"processed_at":"2026-04-10T00:00:00.100Z"}\n\n'
    b'data: {"id":"evt_3","type":"session.status_idle",'
    b'"stop_reason":{"type":"end_turn"},'
    b'"processed_at":"2026-04-10T00:00:00.200Z"}\n\n'
)


class _FakeResponse:
    """Minimal stand-in for an httpx.Response that yields the bytes above."""

    def iter_bytes(self):
        yield SSE_WIRE_BYTES

    def close(self) -> None:
        pass


def main() -> int:
    # Bypass Stream.__init__ since we don't have a real client/response.
    stream = Stream.__new__(Stream)
    stream.response = _FakeResponse()
    stream._cast_to = object
    stream._client = MagicMock()
    stream._client._process_response_data = lambda data, cast_to, response: data
    stream._decoder = SSEDecoder()
    stream._iterator = stream.__stream__()

    received = list(stream)

    print("Events sent on wire:    3")
    print("  - session.status_running")
    print("  - agent.message")
    print("  - session.status_idle")
    print()
    print(f"Events yielded by Stream: {len(received)}")
    for evt in received:
        print(f"  - {evt}")
    print()

    if len(received) == 3:
        print("✅ Stream yielded all events — bug is FIXED in this SDK version.")
        return 0

    print("❌ BUG CONFIRMED: Stream dropped all managed agents events.")
    print()
    print("Root cause: Stream.__stream__ (anthropic/_streaming.py) matches")
    print("only Messages API event names. Paste from that file:")
    print()
    print("    if sse.event == 'completion': ...")
    print("    if sse.event in ('message_start', 'message_delta',")
    print("                     'message_stop', 'content_block_start',")
    print("                     'content_block_delta', 'content_block_stop',")
    print("                     'message'): ...")
    print("    if sse.event == 'ping': continue")
    print("    if sse.event == 'error': raise ...")
    print("    # No branch for session.*, agent.*, span.* — silently dropped.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
