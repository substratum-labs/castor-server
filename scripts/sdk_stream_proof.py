"""Proof that Anthropic SDK 0.93.0 ``Stream`` drops managed agents events.

The SDK's ``Stream.__stream__`` only matches Messages API event names
(``message_start``, ``content_block_*``, ``ping``, ``error``). Managed
agents events (``session.status_*``, ``agent.message``, etc.) hit none
of those branches and get silently dropped.

This script:
  1. Uses the SDK to set up agent + environment + session (proves CRUD works)
  2. Bypasses the SDK's Stream and uses raw httpx on the same URL the SDK
     would have hit, with the same headers
  3. Prints whatever the server actually sends on the wire

If the server is sending events, they'll show up here even though
``client.beta.sessions.events.stream()`` returns empty.
"""

from __future__ import annotations

import threading
import time

import httpx
from anthropic import Anthropic

BASE_URL = "http://127.0.0.1:9092"


def main() -> None:
    client = Anthropic(base_url=BASE_URL, api_key="local")

    agent = client.beta.agents.create(name="proof", model="mock")
    env = client.beta.environments.create(name="proof-env")
    session = client.beta.sessions.create(agent=agent.id, environment_id=env.id)
    print(f"session={session.id}")

    sse_lines: list[str] = []

    def consume() -> None:
        # Same path the SDK hits, with the same beta header
        url = (
            f"{BASE_URL}/v1/sessions/{session.id}/events/stream?beta=true"
        )
        headers = {
            "x-api-key": "local",
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "managed-agents-2026-04-01",
        }
        try:
            with httpx.stream(
                "GET", url, headers=headers, timeout=10
            ) as resp:
                print(f"  raw GET status={resp.status_code} ct={resp.headers.get('content-type')}")
                for line in resp.iter_lines():
                    if line:
                        sse_lines.append(line)
                        print(f"    {line[:120]}")
                    if "session.status_idle" in (line or "") or "session.error" in (line or ""):
                        break
        except Exception as e:
            print(f"  consumer error: {type(e).__name__}: {e}")

    t = threading.Thread(target=consume, daemon=True)
    t.start()
    time.sleep(0.4)

    client.beta.sessions.events.send(
        session_id=session.id,
        events=[
            {
                "type": "user.message",
                "content": [{"type": "text", "text": "proof ping"}],
            }
        ],
    )

    t.join(timeout=10)

    print()
    print(f"raw SSE lines received from server: {len(sse_lines)}")
    data_lines = [line for line in sse_lines if line.startswith("data:")]
    print(f"data lines: {len(data_lines)}")
    if data_lines:
        print()
        print("CONCLUSION: Server IS sending events on the wire.")
        print("            anthropic-python 0.93.0 Stream class drops them ")
        print("            because Stream.__stream__ has no branch for ")
        print("            managed-agents event types.")
    else:
        print()
        print("CONCLUSION: Server is not sending events — different bug.")


if __name__ == "__main__":
    main()
