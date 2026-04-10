"""EventBus: internal event distribution and SSE stream conversion.

Each session gets its own EventBus. Events emitted by the kernel are published
here and fan out to all SSE subscribers for that session.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger("castor_server.event_bus")


class EventBus:
    """Per-session event bus that bridges kernel events to SSE subscribers."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._subscribers: list[asyncio.Queue[dict[str, Any] | None]] = []

    def subscribe(self) -> asyncio.Queue[dict[str, Any] | None]:
        """Create a new subscriber queue. Returns a queue that yields event dicts.

        Send ``None`` to signal stream end.
        """
        q: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        try:
            self._subscribers.remove(q)
        except ValueError:
            pass

    async def publish(self, event: dict[str, Any] | BaseModel) -> None:
        """Publish an event to all subscribers.

        BaseModel events are dumped with ``exclude_none=True`` so optional
        sub-fields (like ``SessionErrorDetail.retry_status``) are omitted
        rather than serialized as ``null`` — matches the Anthropic wire format.
        """
        if isinstance(event, BaseModel):
            data = event.model_dump(exclude_none=True)
        else:
            data = event
        logger.debug("publish session=%s type=%s", self.session_id, data.get("type"))
        for q in self._subscribers:
            await q.put(data)

    async def close(self) -> None:
        """Signal all subscribers that the stream has ended."""
        for q in self._subscribers:
            await q.put(None)
        self._subscribers.clear()

    def sse_serialize(self, event: dict[str, Any]) -> str:
        """Serialize an event dict to SSE wire format."""
        return f"data: {json.dumps(event)}\n\n"
