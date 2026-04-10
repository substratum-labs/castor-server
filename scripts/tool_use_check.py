"""Tool-use round-trip validation against a real LLM.

Validates that ``agent.tool_use`` → ``agent.tool_result`` events flow
correctly through castor-server with the SDK as the client and a real
tool-calling LLM (OpenRouter routing).

Usage:
    OPENROUTER_API_KEY=... uv run python scripts/tool_use_check.py

Pre-requisites:
    castor-server run --port 9093  (in another terminal)
"""

from __future__ import annotations

import os
import sys
import threading
import time

from anthropic import Anthropic

from castor_server.client import stream_events

BASE_URL = "http://127.0.0.1:9097"
# Cheap, fast, supports tool use. Routed to openrouter via the explicit prefix.
TOOL_MODEL = "openrouter/anthropic/claude-3.5-haiku"


def main() -> int:
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("OPENROUTER_API_KEY not set in environment", file=sys.stderr)
        return 2

    client = Anthropic(base_url=BASE_URL, api_key="local")
    failures: list[str] = []

    # ── Setup ──
    print("=" * 60)
    print(f"STEP 1: create agent with model={TOOL_MODEL!r}")
    agent = client.beta.agents.create(
        name="tool-use-check",
        model=TOOL_MODEL,
        system=(
            "You are a helpful assistant with shell access. "
            "When the user asks you to run a command, use the bash tool. "
            "Be concise."
        ),
        tools=[{"type": "agent_toolset_20260401"}],
    )
    print(f"  ✓ {agent.id}")

    print("\n" + "=" * 60)
    print("STEP 2: create environment + session")
    env = client.beta.environments.create(name="tool-use-env")
    session = client.beta.sessions.create(agent=agent.id, environment_id=env.id)
    print(f"  ✓ env={env.id}")
    print(f"  ✓ session={session.id}")

    # ── Open stream BEFORE sending message ──
    received: list[dict] = []
    stream_done = threading.Event()
    stream_err: list[Exception] = []
    auto_approve = True  # auto-approve any requires_action and continue draining

    def consume() -> None:
        try:
            for event in stream_events(
                base_url=BASE_URL,
                session_id=session.id,
                api_key="local",
                timeout=120,
            ):
                received.append(event)
                etype = event.get("type", "")
                # Print live so we can watch it happen
                print(f"    ← {etype}")
                if etype == "session.status_idle":
                    stop = event.get("stop_reason") or {}
                    if stop.get("type") == "requires_action" and auto_approve:
                        # Auto-approve any blocked tool calls so the agent
                        # loop continues. We do this from within the consumer
                        # because the stream stays open across the resume.
                        for evt_id in stop.get("event_ids", []):
                            print(
                                f"    → user.tool_confirmation tool_use_id="
                                f"{evt_id} result=allow"
                            )
                            client.beta.sessions.events.send(
                                session_id=session.id,
                                events=[
                                    {
                                        "type": "user.tool_confirmation",
                                        "tool_use_id": evt_id,
                                        "result": "allow",
                                    }
                                ],
                            )
                        continue  # keep draining
                    # Terminal idle (end_turn / retries_exhausted)
                    stream_done.set()
                    return
                if etype == "session.error":
                    stream_done.set()
                    return
        except Exception as e:
            stream_err.append(e)
            stream_done.set()

    print("\n" + "=" * 60)
    print("STEP 3: open SSE stream")
    t = threading.Thread(target=consume, daemon=True)
    t.start()
    time.sleep(0.4)

    # ── Send a message that should trigger tool use ──
    print("\n" + "=" * 60)
    print("STEP 4: send user.message asking the agent to run a bash command")
    client.beta.sessions.events.send(
        session_id=session.id,
        events=[
            {
                "type": "user.message",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Please run `echo hello-from-castor` and tell me "
                            "what it printed."
                        ),
                    }
                ],
            }
        ],
    )

    print("\n" + "=" * 60)
    print("STEP 5: drain stream (live event log)")
    stream_done.wait(timeout=120)
    t.join(timeout=5)

    if stream_err:
        failures.append(f"stream consumer raised: {stream_err[0]!r}")

    # ── Analyze ──
    print("\n" + "=" * 60)
    print("ANALYSIS")
    types = [e.get("type", "") for e in received]
    print(f"  total events: {len(types)}")
    print(f"  unique types: {sorted(set(types))}")

    # Required events for the happy path
    required = [
        "session.status_running",
        "agent.tool_use",
        "agent.tool_result",
        "agent.message",
        "session.status_idle",
    ]
    for t_name in required:
        if t_name not in types:
            failures.append(f"missing event type: {t_name}")

    # Check that the tool was actually bash
    tool_uses = [e for e in received if e.get("type") == "agent.tool_use"]
    if tool_uses:
        names = [tu.get("name") for tu in tool_uses]
        print(f"  tool calls: {names}")
        if "bash" not in names:
            failures.append(f"expected bash tool call, got {names}")
    else:
        failures.append("no agent.tool_use events at all — model didn't call tools")

    # Check that the tool result mentioned our echoed string
    tool_results = [e for e in received if e.get("type") == "agent.tool_result"]
    if tool_results:
        for tr in tool_results:
            content = tr.get("content") or []
            for block in content:
                text = block.get("text", "")
                if "hello-from-castor" in text:
                    print(f"  ✓ tool result contains expected output: {text[:80]!r}")
                    break
            else:
                continue
            break
        else:
            failures.append("no tool result contained 'hello-from-castor'")
    else:
        failures.append("no agent.tool_result events")

    # Check the final assistant message references the output
    agent_msgs = [e for e in received if e.get("type") == "agent.message"]
    if agent_msgs:
        last = agent_msgs[-1]
        full_text = " ".join(
            b.get("text", "")
            for b in (last.get("content") or [])
            if b.get("type") == "text"
        )
        print(f"  final agent message: {full_text[:200]!r}")
        has_echo = "hello-from-castor" in full_text or "hello" in full_text.lower()
        if not has_echo:
            failures.append(
                f"final agent.message missing echoed output: {full_text[:100]!r}"
            )
    else:
        failures.append("no final agent.message event")

    # Verify exactly one TERMINAL idle (end_turn). HITL pauses also emit
    # idle events with stop_reason=requires_action — those don't count.
    terminal_idles = [
        e
        for e in received
        if e.get("type") == "session.status_idle"
        and (e.get("stop_reason") or {}).get("type") == "end_turn"
    ]
    if len(terminal_idles) != 1:
        failures.append(
            f"expected exactly 1 terminal idle (end_turn), got {len(terminal_idles)}"
        )

    # Verify the requires_action idle uses real tool_use ids (not synthetic)
    pause_idles = [
        e
        for e in received
        if e.get("type") == "session.status_idle"
        and (e.get("stop_reason") or {}).get("type") == "requires_action"
    ]
    for pause in pause_idles:
        ids = pause["stop_reason"].get("event_ids", [])
        for ev_id in ids:
            if ev_id.startswith("hitl_"):
                failures.append(
                    f"requires_action used synthetic id {ev_id!r} "
                    "instead of real tool_use id"
                )

    # Verify agent.tool_use uses evaluated_permission='ask' for HITL tools
    for tu in tool_uses:
        if tu.get("evaluated_permission") != "ask":
            failures.append(
                f"agent.tool_use for HITL tool should be 'ask', "
                f"got {tu.get('evaluated_permission')!r}"
            )

    # ── Report ──
    print("\n" + "=" * 60)
    if failures:
        print(f"❌ FAILED: {len(failures)} issue(s)")
        for i, f in enumerate(failures, 1):
            print(f"  {i}. {f}")
        return 1
    print("✅ TOOL-USE ROUND-TRIP VERIFIED")
    print("   The full agent loop works end-to-end via the Anthropic SDK:")
    print("   user.message → agent.tool_use → agent.tool_result → agent.message")
    return 0


if __name__ == "__main__":
    sys.exit(main())
