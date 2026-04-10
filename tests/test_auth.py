"""Tests for global API key authentication."""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from castor_server.app import create_app
from castor_server.config import settings
from castor_server.store.database import get_session

API_KEY = "test-secret-abc123"


@pytest_asyncio.fixture
async def auth_client(db_engine, monkeypatch) -> AsyncGenerator[AsyncClient, None]:
    """Client with auth enabled (API_KEY set)."""
    monkeypatch.setattr(settings, "api_key", API_KEY)

    session_factory = async_sessionmaker(
        db_engine, class_=AsyncSession, expire_on_commit=False
    )

    from castor_server.store import database as db_module

    original_factory = db_module.async_session
    db_module.set_session_factory(session_factory)

    app = create_app()

    async def override_get_session():
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_session] = override_get_session
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c

    db_module.set_session_factory(original_factory)


@pytest.mark.asyncio
async def test_health_no_auth_required(auth_client: AsyncClient):
    """Health endpoint should be accessible without auth."""
    resp = await auth_client.get("/health")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_protected_endpoint_no_auth(auth_client: AsyncClient):
    """Requests without Authorization header are rejected."""
    resp = await auth_client.get("/v1/agents")
    assert resp.status_code == 401
    assert "Missing Authorization" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_protected_endpoint_wrong_scheme(auth_client: AsyncClient):
    """Non-Bearer schemes are rejected."""
    resp = await auth_client.get(
        "/v1/agents", headers={"Authorization": f"Basic {API_KEY}"}
    )
    assert resp.status_code == 401
    assert "Bearer" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_protected_endpoint_invalid_key(auth_client: AsyncClient):
    """Invalid API keys are rejected."""
    resp = await auth_client.get(
        "/v1/agents", headers={"Authorization": "Bearer wrong-key"}
    )
    assert resp.status_code == 401
    assert "Invalid API key" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_protected_endpoint_valid_key(auth_client: AsyncClient):
    """Valid API key allows access."""
    resp = await auth_client.get(
        "/v1/agents", headers={"Authorization": f"Bearer {API_KEY}"}
    )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_create_agent_with_auth(auth_client: AsyncClient):
    """Full flow: create an agent with valid auth."""
    resp = await auth_client.post(
        "/v1/agents",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "name": "auth-test-agent",
            "model": "claude-sonnet-4-6",
            "tools": [{"type": "agent_toolset_20260401"}],
        },
    )
    assert resp.status_code == 201
    assert resp.json()["name"] == "auth-test-agent"


@pytest.mark.asyncio
async def test_environments_protected(auth_client: AsyncClient):
    """Environments endpoint also requires auth."""
    resp = await auth_client.get("/v1/environments")
    assert resp.status_code == 401

    resp = await auth_client.get(
        "/v1/environments", headers={"Authorization": f"Bearer {API_KEY}"}
    )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_auth_disabled_by_default(client: AsyncClient):
    """When api_key is unset, no auth is required (dev mode)."""
    # The default `client` fixture has no api_key set
    resp = await client.get("/v1/agents")
    assert resp.status_code == 200
