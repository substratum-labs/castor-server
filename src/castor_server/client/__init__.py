"""Client-side helpers for talking to a castor-server.

Most users will use the official ``anthropic`` Python SDK pointed at their
castor-server (HTTP CRUD operations all work). The one exception is
``client.beta.sessions.events.stream()`` — anthropic-python 0.93.0 ships
with a Stream class hardcoded for the Messages API event names, so it
silently drops managed agents events.

Until that's fixed upstream, this module provides a thin SSE consumer
that you can use directly:

    from castor_server.client import stream_events

    for event in stream_events(
        base_url="http://localhost:8080",
        session_id=session.id,
        api_key="local",
    ):
        print(event["type"], event)
        if event["type"] == "session.status_idle":
            break

Async variant:

    from castor_server.client import astream_events

    async for event in astream_events(...):
        ...
"""

from __future__ import annotations

from .stream import astream_events, stream_events

__all__ = ["astream_events", "stream_events"]
