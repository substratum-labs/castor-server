"""Tests for the client-side SSE streaming helper.

The unit tests below cover the SSE parser thoroughly. The end-to-end
test runs against a real subprocess castor-server (not ASGITransport)
because httpx's in-process ASGI transport buffers streaming responses
in a way that breaks SSE — a known limitation, not a bug in our code.
"""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator

import httpx
import pytest

from castor_server.client import astream_events, stream_events
from castor_server.client.stream import _parse_sse_lines

# ---------------------------------------------------------------------------
# Pure unit tests for the SSE parser
# ---------------------------------------------------------------------------


def test_parse_sse_single_event():
    lines = [
        'data: {"id": "evt_1", "type": "agent.message"}',
        "",
    ]
    events = list(_parse_sse_lines(iter(lines)))
    assert events == [{"id": "evt_1", "type": "agent.message"}]


def test_parse_sse_multiple_events():
    lines = [
        'data: {"type": "session.status_running"}',
        "",
        'data: {"type": "session.status_idle"}',
        "",
    ]
    events = list(_parse_sse_lines(iter(lines)))
    assert [e["type"] for e in events] == [
        "session.status_running",
        "session.status_idle",
    ]


def test_parse_sse_skips_comments():
    lines = [
        ":keepalive",
        'data: {"type": "agent.message"}',
        "",
    ]
    events = list(_parse_sse_lines(iter(lines)))
    assert len(events) == 1


def test_parse_sse_handles_optional_space():
    """Per spec, a single space after 'data:' is stripped."""
    lines = [
        'data:{"type":"a"}',  # no space
        "",
        'data: {"type":"b"}',  # one space
        "",
    ]
    events = list(_parse_sse_lines(iter(lines)))
    assert [e["type"] for e in events] == ["a", "b"]


def test_parse_sse_skips_invalid_json():
    lines = [
        "data: not json at all",
        "",
        'data: {"type": "valid"}',
        "",
    ]
    events = list(_parse_sse_lines(iter(lines)))
    assert events == [{"type": "valid"}]


def test_parse_sse_multiline_data():
    """SSE spec: multiple data: lines on one event are joined with \\n."""
    lines = [
        'data: {"type":',
        'data: "split"}',
        "",
    ]
    events = list(_parse_sse_lines(iter(lines)))
    assert events == [{"type": "split"}]


# ---------------------------------------------------------------------------
# End-to-end tests against a real subprocess server
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def live_server() -> Iterator[str]:
    """Spawn castor-server in a subprocess on a random port for streaming tests.

    Uses a per-fixture SQLite file so it doesn't collide with the in-memory
    DB used by other tests.
    """
    port = _free_port()
    db_dir = tempfile.mkdtemp(prefix="castor-stream-test-")
    db_path = os.path.join(db_dir, "test.db")
    env = {
        **os.environ,
        "CASTOR_DATABASE_URL": f"sqlite+aiosqlite:///{db_path}",
        "CASTOR_API_KEY": "",  # disable auth
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "castor_server", "run", "--port", str(port)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        # Wait for the server to be ready
        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                if httpx.get(f"{base_url}/health", timeout=0.5).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.1)
        else:
            raise RuntimeError("castor-server subprocess did not become ready")
        yield base_url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_stream_events_full_pipeline(live_server: str):
    """stream_events yields all events from a mock-model session.

    This is the test that locks in the workaround: if it ever breaks,
    castor-server users lose their streaming story.
    """
    import threading

    # Set up agent + session
    agent = httpx.post(
        f"{live_server}/v1/agents", json={"name": "stream-test", "model": "mock"}
    ).json()
    session = httpx.post(
        f"{live_server}/v1/sessions", json={"agent": agent["id"]}
    ).json()

    received: list[dict] = []

    def consume() -> None:
        for event in stream_events(
            base_url=live_server, session_id=session["id"], timeout=15
        ):
            received.append(event)
            if event.get("type") == "session.status_idle":
                return

    t = threading.Thread(target=consume, daemon=True)
    t.start()
    time.sleep(0.4)  # let the subscriber attach

    httpx.post(
        f"{live_server}/v1/sessions/{session['id']}/events",
        json={
            "events": [
                {
                    "type": "user.message",
                    "content": [{"type": "text", "text": "ping castor"}],
                }
            ]
        },
    )

    t.join(timeout=15)
    assert not t.is_alive(), "stream consumer did not finish"

    types = [e.get("type") for e in received]
    assert "session.status_running" in types, f"missing running, got: {types}"
    assert "agent.message" in types, f"missing agent.message, got: {types}"
    assert "session.status_idle" in types, f"missing idle, got: {types}"

    agent_msgs = [e for e in received if e.get("type") == "agent.message"]
    text = "".join(
        b.get("text", "") for b in agent_msgs[-1]["content"] if b.get("type") == "text"
    )
    assert "ping castor" in text and "[mock]" in text


@pytest.mark.asyncio
async def test_astream_events_full_pipeline(live_server: str):
    """Async variant — same flow via astream_events."""
    async with httpx.AsyncClient(base_url=live_server) as c:
        agent = (
            await c.post("/v1/agents", json={"name": "astream-test", "model": "mock"})
        ).json()
        session = (await c.post("/v1/sessions", json={"agent": agent["id"]})).json()

    received: list[dict] = []
    stream_done = asyncio.Event()

    async def consume() -> None:
        async for event in astream_events(
            base_url=live_server, session_id=session["id"], timeout=15
        ):
            received.append(event)
            if event.get("type") == "session.status_idle":
                stream_done.set()
                return

    consumer = asyncio.create_task(consume())
    await asyncio.sleep(0.4)

    async with httpx.AsyncClient(base_url=live_server) as c:
        await c.post(
            f"/v1/sessions/{session['id']}/events",
            json={
                "events": [
                    {
                        "type": "user.message",
                        "content": [{"type": "text", "text": "async ping"}],
                    }
                ]
            },
        )

    try:
        await asyncio.wait_for(stream_done.wait(), timeout=15.0)
    finally:
        consumer.cancel()
        try:
            await consumer
        except (asyncio.CancelledError, Exception):
            pass

    types = [e.get("type") for e in received]
    assert "session.status_running" in types
    assert "agent.message" in types
    assert "session.status_idle" in types
