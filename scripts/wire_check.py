"""Wire-format sanity check.

Pretends to be the Anthropic Python SDK by sending requests with the
expected headers and body shapes, and walks the full
create-agent → create-session → send-event → stream-events loop.

Run with: uv run python scripts/wire_check.py
"""

from __future__ import annotations

import asyncio
import json
import sys

import httpx

BASE_URL = "http://127.0.0.1:9090"

# Standard headers that anthropic-python sends
HEADERS = {
    "x-api-key": "local",
    "anthropic-version": "2023-06-01",
    "anthropic-beta": "managed-agents-2026-04-01",
    "content-type": "application/json",
    "user-agent": "anthropic-python-wire-check/0.1",
}


async def main() -> int:
    failures: list[str] = []

    async with httpx.AsyncClient(base_url=BASE_URL, headers=HEADERS, timeout=30) as c:
        # ── Health ──
        print("=" * 60)
        print("STEP 0: Health check")
        r = await c.get("/health")
        print(f"  GET /health → {r.status_code}")
        if r.status_code != 200:
            failures.append("/health did not return 200")

        # ── Create agent ──
        print("\n" + "=" * 60)
        print("STEP 1: POST /v1/agents")
        agent_body = {
            "name": "wire-check-agent",
            "model": "claude-sonnet-4-6",
            "system": "You are a helpful assistant. Be concise.",
            "tools": [{"type": "agent_toolset_20260401"}],
        }
        r = await c.post("/v1/agents", json=agent_body)
        print(f"  status: {r.status_code}")
        try:
            agent = r.json()
            print(f"  body: {json.dumps(agent, indent=2)[:500]}")
        except Exception as e:
            failures.append(f"agent response not JSON: {e}")
            print(f"  raw body: {r.text}")
            return _report(failures)

        if r.status_code != 201:
            failures.append(f"create agent: expected 201, got {r.status_code}")

        # Required fields per Anthropic spec
        for field in ("id", "type", "name", "model", "version", "created_at"):
            if field not in agent:
                failures.append(f"agent missing required field: {field}")

        if agent.get("type") != "agent":
            failures.append(f"agent.type expected 'agent', got {agent.get('type')!r}")

        if not isinstance(agent.get("model"), dict):
            failures.append("agent.model should be an object {id, speed}")

        agent_id = agent["id"]

        # ── Create session ──
        print("\n" + "=" * 60)
        print("STEP 2: POST /v1/sessions")
        session_body = {"agent": agent_id, "title": "wire check session"}
        r = await c.post("/v1/sessions", json=session_body)
        print(f"  status: {r.status_code}")
        try:
            session = r.json()
            print(f"  body keys: {list(session.keys())}")
        except Exception as e:
            failures.append(f"session response not JSON: {e}")
            print(f"  raw body: {r.text}")
            return _report(failures)

        if r.status_code != 201:
            failures.append(f"create session: expected 201, got {r.status_code}")

        for field in ("id", "type", "agent", "status", "created_at"):
            if field not in session:
                failures.append(f"session missing required field: {field}")

        if session.get("status") not in ("idle", "running"):
            failures.append(
                f"session.status expected 'idle' or 'running', got {session.get('status')!r}"
            )

        session_id = session["id"]

        # ── Open SSE stream BEFORE sending the event (required per spec) ──
        print("\n" + "=" * 60)
        print("STEP 3: GET /v1/sessions/{id}/events/stream  (SSE)")

        events_received: list[dict] = []
        stream_done = asyncio.Event()

        async def consume_stream():
            try:
                async with httpx.AsyncClient(
                    base_url=BASE_URL, headers=HEADERS, timeout=60
                ) as sc:
                    async with sc.stream(
                        "GET", f"/v1/sessions/{session_id}/events/stream"
                    ) as resp:
                        print(f"  stream open: status={resp.status_code}")
                        if resp.status_code != 200:
                            failures.append(
                                f"stream: expected 200, got {resp.status_code}"
                            )
                            stream_done.set()
                            return

                        ct = resp.headers.get("content-type", "")
                        print(f"  content-type: {ct}")
                        if "text/event-stream" not in ct:
                            failures.append(
                                f"stream content-type expected text/event-stream, got {ct}"
                            )

                        buffer = ""
                        async for chunk in resp.aiter_text():
                            buffer += chunk
                            while "\n\n" in buffer:
                                raw, buffer = buffer.split("\n\n", 1)
                                for line in raw.splitlines():
                                    if line.startswith("data:"):
                                        data = line[5:].strip()
                                        if not data:
                                            continue
                                        try:
                                            evt = json.loads(data)
                                            events_received.append(evt)
                                            print(
                                                f"    ← {evt.get('type', '?'):30} "
                                                f"id={evt.get('id', '?')[:16]}"
                                            )
                                            if evt.get("type") in (
                                                "session.status_idle",
                                                "session.status_terminated",
                                                "session.error",
                                            ):
                                                stream_done.set()
                                                return
                                        except json.JSONDecodeError as e:
                                            failures.append(
                                                f"stream emitted non-JSON data: {data[:100]} ({e})"
                                            )
            except Exception as e:
                failures.append(f"stream consumer error: {type(e).__name__}: {e}")
                stream_done.set()

        stream_task = asyncio.create_task(consume_stream())
        await asyncio.sleep(0.3)  # let the subscriber attach

        # ── Send a user message ──
        print("\n" + "=" * 60)
        print("STEP 4: POST /v1/sessions/{id}/events")
        send_body = {
            "events": [
                {
                    "type": "user.message",
                    "content": [{"type": "text", "text": "Say 'pong' and nothing else."}],
                }
            ]
        }
        r = await c.post(f"/v1/sessions/{session_id}/events", json=send_body)
        print(f"  status: {r.status_code}")
        try:
            sent = r.json()
            print(f"  data length: {len(sent.get('data', []))}")
            if sent.get("data"):
                first = sent["data"][0]
                print(f"  first event id: {first.get('id')}")
                if "id" not in first:
                    failures.append("sent event missing id")
                if "processed_at" not in first:
                    failures.append("sent event missing processed_at")
        except Exception as e:
            failures.append(f"send event response not JSON: {e}")
            print(f"  raw: {r.text}")

        if r.status_code != 200:
            failures.append(f"send events: expected 200, got {r.status_code}")

        # ── Wait for stream to complete ──
        print("\n" + "=" * 60)
        print("STEP 5: Waiting for stream to reach idle...")
        try:
            await asyncio.wait_for(stream_done.wait(), timeout=45)
        except asyncio.TimeoutError:
            failures.append("stream did not reach idle within 45s")

        stream_task.cancel()
        try:
            await stream_task
        except (asyncio.CancelledError, Exception):
            pass

        print(f"  total events received: {len(events_received)}")
        event_types = [e.get("type") for e in events_received]
        print(f"  event types: {event_types}")

        # Verify expected event sequence
        if "session.status_running" not in event_types:
            failures.append("missing session.status_running event")
        if not any(t == "session.status_idle" for t in event_types):
            failures.append("missing session.status_idle event")

        # ── List events (history) ──
        print("\n" + "=" * 60)
        print("STEP 6: GET /v1/sessions/{id}/events")
        r = await c.get(f"/v1/sessions/{session_id}/events")
        print(f"  status: {r.status_code}")
        if r.status_code == 200:
            history = r.json().get("data", [])
            print(f"  history length: {len(history)}")
        else:
            failures.append(f"list events: expected 200, got {r.status_code}")

    return _report(failures)


def _report(failures: list[str]) -> int:
    print("\n" + "=" * 60)
    if failures:
        print(f"❌ FAILED: {len(failures)} issue(s)")
        for i, f in enumerate(failures, 1):
            print(f"  {i}. {f}")
        return 1
    else:
        print("✅ ALL CHECKS PASSED")
        return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
