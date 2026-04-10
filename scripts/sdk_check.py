"""End-to-end SDK compatibility check.

Uses the real ``anthropic`` Python SDK pointed at a running castor-server,
exercises the full pipeline with the built-in ``mock`` model, and reports
any wire-format gaps.

Usage:
    # Terminal 1
    castor-server run --port 9092

    # Terminal 2
    uv run python scripts/sdk_check.py
"""

from __future__ import annotations

import sys

from anthropic import Anthropic

BASE_URL = "http://127.0.0.1:9092"


def main() -> int:
    client = Anthropic(base_url=BASE_URL, api_key="local")
    failures: list[str] = []

    # ---- Step 1: Create agent ----
    print("=" * 60)
    print("STEP 1: client.beta.agents.create(model='mock')")
    try:
        agent = client.beta.agents.create(name="sdk-check", model="mock")
        print(f"  ✓ id={agent.id}")
        print(f"    type={agent.type}")
        print(f"    name={agent.name}")
        print(f"    model.id={agent.model.id}")
        print(f"    version={agent.version}")
    except Exception as e:
        failures.append(f"agents.create: {type(e).__name__}: {e}")
        return _report(failures)

    # ---- Step 2: Retrieve agent ----
    print("\n" + "=" * 60)
    print("STEP 2: client.beta.agents.retrieve(agent.id)")
    try:
        retrieved = client.beta.agents.retrieve(agent.id)
        print(f"  ✓ id={retrieved.id}")
    except Exception as e:
        failures.append(f"agents.retrieve: {type(e).__name__}: {e}")

    # ---- Step 3: List agents ----
    print("\n" + "=" * 60)
    print("STEP 3: client.beta.agents.list()")
    try:
        page = client.beta.agents.list()
        print(f"  ✓ data length={len(page.data)}")
    except Exception as e:
        failures.append(f"agents.list: {type(e).__name__}: {e}")

    # ---- Step 3.5: Create environment (required by sessions.create) ----
    print("\n" + "=" * 60)
    print("STEP 3.5: client.beta.environments.create(name='sdk-check-env')")
    try:
        env = client.beta.environments.create(name="sdk-check-env")
        print(f"  ✓ id={env.id}")
    except Exception as e:
        failures.append(f"environments.create: {type(e).__name__}: {e}")
        return _report(failures)

    # ---- Step 4: Create session ----
    print("\n" + "=" * 60)
    print("STEP 4: client.beta.sessions.create(agent=agent.id, environment_id=env.id)")
    try:
        session = client.beta.sessions.create(agent=agent.id, environment_id=env.id)
        print(f"  ✓ id={session.id}")
        print(f"    type={session.type}")
        print(f"    status={session.status}")
        print(f"    agent.id={session.agent.id}")
    except Exception as e:
        failures.append(f"sessions.create: {type(e).__name__}: {e}")
        return _report(failures)

    # ---- Step 5: Open SSE stream BEFORE sending message ----
    print("\n" + "=" * 60)
    print("STEP 5: client.beta.sessions.events.stream(session_id) + send")
    received_types: list[str] = []
    received_events: list = []

    try:
        # The SDK's stream() returns a context manager. Open it in a thread
        # since the SDK is sync and we need to send while it's open.
        import threading
        import time

        def consume() -> None:
            try:
                with client.beta.sessions.events.stream(session_id=session.id) as stream:
                    for event in stream:
                        received_events.append(event)
                        evt_type = getattr(event, "type", None) or type(event).__name__
                        received_types.append(evt_type)
                        if evt_type in (
                            "session.status_idle",
                            "session.status_terminated",
                            "session.error",
                        ):
                            # Got a terminal-ish event; close
                            break
            except Exception as e:
                failures.append(f"stream consumer: {type(e).__name__}: {e}")

        t = threading.Thread(target=consume, daemon=True)
        t.start()
        time.sleep(0.4)  # let the subscriber attach

        # Send a user message via the SDK
        sent = client.beta.sessions.events.send(
            session_id=session.id,
            events=[
                {
                    "type": "user.message",
                    "content": [{"type": "text", "text": "ping from SDK"}],
                }
            ],
        )
        print(f"  ✓ events.send returned {len(sent.data)} event(s)")

        # Wait for stream to finish
        t.join(timeout=15)
        if t.is_alive():
            failures.append("stream did not reach idle in 15s")

        print(f"  ✓ stream received {len(received_types)} events: {received_types}")

        # Verify expected events
        if "session.status_running" not in received_types:
            failures.append("missing session.status_running on stream")
        if "agent.message" not in received_types:
            failures.append("missing agent.message on stream")
        if "session.status_idle" not in received_types:
            failures.append("missing session.status_idle on stream")

        # Verify the agent.message has our echo
        agent_msgs = [e for e in received_events if getattr(e, "type", "") == "agent.message"]
        if agent_msgs:
            text = ""
            content = agent_msgs[-1].content
            for block in content:
                if getattr(block, "type", "") == "text":
                    text += block.text
            if "ping from SDK" not in text:
                failures.append(f"agent.message did not echo user input: {text!r}")
            else:
                print(f"  ✓ agent echoed: {text!r}")
    except Exception as e:
        failures.append(f"stream/send: {type(e).__name__}: {e}")

    # ---- Step 6: List events (history) ----
    print("\n" + "=" * 60)
    print("STEP 6: client.beta.sessions.events.list(session_id)")
    try:
        history = client.beta.sessions.events.list(session_id=session.id)
        print(f"  ✓ data length={len(history.data)}")
        history_types = [getattr(e, "type", "?") for e in history.data]
        print(f"  history types: {history_types}")
    except Exception as e:
        failures.append(f"events.list: {type(e).__name__}: {e}")

    # ---- Step 7: Update agent (versioning) ----
    print("\n" + "=" * 60)
    print("STEP 7: client.beta.agents.update(agent.id, version=1, name=...)")
    try:
        updated = client.beta.agents.update(
            agent_id=agent.id, version=1, name="sdk-check-v2"
        )
        print(f"  ✓ new version={updated.version}, name={updated.name}")
        if updated.version != 2:
            failures.append(f"update should bump version to 2, got {updated.version}")
    except Exception as e:
        failures.append(f"agents.update: {type(e).__name__}: {e}")

    return _report(failures)


def _report(failures: list[str]) -> int:
    print("\n" + "=" * 60)
    if failures:
        print(f"❌ FAILED: {len(failures)} issue(s)")
        for i, f in enumerate(failures, 1):
            print(f"  {i}. {f}")
        return 1
    print("✅ ALL SDK CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
