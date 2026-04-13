"""Simple in-memory rate limiter.

Enforces a per-IP sliding-window limit on API requests. When
``settings.rate_limit_rpm`` is set (requests per minute), returns
429 Too Many Requests if the limit is exceeded.

If ``settings.rate_limit_rpm`` is 0 or unset, rate limiting is disabled.
"""

from __future__ import annotations

import time
from collections import defaultdict

from fastapi import HTTPException, Request

from castor_server.config import settings

# IP → list of request timestamps (sliding window)
_windows: dict[str, list[float]] = defaultdict(list)


async def check_rate_limit(request: Request) -> None:
    """FastAPI dependency that enforces per-IP rate limiting."""
    rpm = settings.rate_limit_rpm
    if not rpm:
        return

    ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    window = _windows[ip]

    # Prune entries older than 60 seconds
    cutoff = now - 60.0
    while window and window[0] < cutoff:
        window.pop(0)

    if len(window) >= rpm:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded ({rpm} requests/minute)",
            headers={"Retry-After": "60"},
        )

    window.append(now)
