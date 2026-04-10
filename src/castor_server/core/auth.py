"""Global API key authentication.

Simple bearer-token auth for protecting the server. If ``settings.api_key``
is set, all protected endpoints require ``Authorization: Bearer <key>``.
If unset, auth is disabled (development mode).

This is intentionally NOT multi-tenant — there's a single global key.
For real multi-tenancy with isolated tenants, see the Phase 2+ roadmap.
"""

from __future__ import annotations

import hmac

from fastapi import Header, HTTPException

from castor_server.config import settings


async def require_api_key(
    authorization: str | None = Header(default=None),
) -> None:
    """FastAPI dependency that enforces the global API key.

    - If ``settings.api_key`` is unset → no-op (dev mode).
    - Otherwise → expects ``Authorization: Bearer <key>`` matching settings.api_key.
    - Uses constant-time comparison to prevent timing attacks.
    """
    expected = settings.api_key
    if not expected:
        return  # Auth disabled

    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(
            status_code=401,
            detail="Invalid Authorization header (expected 'Bearer <token>')",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not hmac.compare_digest(token, expected):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
