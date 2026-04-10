"""SSE stream helpers for managed agents events.

These wrap ``httpx.stream`` / ``httpx.AsyncClient.stream`` and parse the
SSE protocol so users can iterate parsed event dicts. Use these if your
client is the official ``anthropic`` SDK and you've found that
``client.beta.sessions.events.stream()`` returns no events
(SDK 0.93.0 limitation — see castor-server README).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx

# Terminal events that signal the end of a turn — callers usually break here.
TERMINAL_EVENT_TYPES = frozenset(
    [
        "session.status_idle",
        "session.status_terminated",
        "session.deleted",
    ]
)


def _build_headers(api_key: str | None) -> dict[str, str]:
    headers = {
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "managed-agents-2026-04-01",
        "accept": "text/event-stream",
    }
    if api_key is not None:
        headers["x-api-key"] = api_key
        # castor-server uses Authorization: Bearer for its own auth path
        headers["authorization"] = f"Bearer {api_key}"
    return headers


def _parse_sse_lines(lines: Iterator[str]) -> Iterator[dict[str, Any]]:
    """Parse a stream of SSE wire-format lines into event dicts.

    Yields one dict per ``data:`` line. Skips comments and unknown fields.
    Handles multi-line ``data:`` continuations per the SSE spec.
    """
    data_buf: list[str] = []
    for raw in lines:
        line = raw.rstrip("\r")
        if line == "":
            # End of event
            if data_buf:
                joined = "\n".join(data_buf)
                data_buf = []
                try:
                    yield json.loads(joined)
                except json.JSONDecodeError:
                    continue
            continue
        if line.startswith(":"):
            # SSE comment / keepalive
            continue
        if line.startswith("data:"):
            # Spec: strip a single optional space after the colon
            value = line[5:]
            if value.startswith(" "):
                value = value[1:]
            data_buf.append(value)
        # Other fields (event:, id:, retry:) ignored — type is in the JSON


def stream_events(
    *,
    base_url: str,
    session_id: str,
    api_key: str | None = None,
    timeout: float = 60.0,
    httpx_client: httpx.Client | None = None,
) -> Iterator[dict[str, Any]]:
    """Iterate over managed agents events for a session, synchronously.

    Args:
        base_url: castor-server base URL, e.g. ``http://localhost:8080``.
        session_id: ID of the session to stream.
        api_key: Optional API key. Only required if the server has
            ``CASTOR_API_KEY`` set.
        timeout: HTTP read timeout. The connection stays open for the
            life of the session, so this should be larger than the
            longest expected gap between events.
        httpx_client: Optional pre-configured ``httpx.Client``. If
            provided, ``base_url`` is still used for the request URL but
            the client's settings (proxies, headers, etc.) take effect.

    Yields:
        Parsed event dicts. The caller is responsible for breaking on
        terminal events (``session.status_idle``, ``session.status_terminated``).
    """
    url = f"{base_url.rstrip('/')}/v1/sessions/{session_id}/events/stream"
    headers = _build_headers(api_key)

    if httpx_client is not None:
        ctx = httpx_client.stream("GET", url, headers=headers, timeout=timeout)
    else:
        client = httpx.Client(timeout=timeout)
        ctx = client.stream("GET", url, headers=headers, timeout=timeout)

    with ctx as resp:
        resp.raise_for_status()
        yield from _parse_sse_lines(resp.iter_lines())


async def astream_events(
    *,
    base_url: str,
    session_id: str,
    api_key: str | None = None,
    timeout: float = 60.0,
    httpx_client: httpx.AsyncClient | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Async version of :func:`stream_events`.

    Yields parsed event dicts. The caller is responsible for breaking
    on terminal events.
    """
    url = f"{base_url.rstrip('/')}/v1/sessions/{session_id}/events/stream"
    headers = _build_headers(api_key)

    owns_client = httpx_client is None
    client = httpx_client or httpx.AsyncClient(timeout=timeout)

    try:
        async with client.stream("GET", url, headers=headers, timeout=timeout) as resp:
            resp.raise_for_status()
            data_buf: list[str] = []
            async for raw in resp.aiter_lines():
                line = raw.rstrip("\r")
                if line == "":
                    if data_buf:
                        joined = "\n".join(data_buf)
                        data_buf = []
                        try:
                            yield json.loads(joined)
                        except json.JSONDecodeError:
                            continue
                    continue
                if line.startswith(":"):
                    continue
                if line.startswith("data:"):
                    value = line[5:]
                    if value.startswith(" "):
                        value = value[1:]
                    data_buf.append(value)
    finally:
        if owns_client:
            await client.aclose()
