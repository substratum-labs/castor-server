"""Tests for /v1/models endpoint."""

from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_list_models(client: AsyncClient):
    resp = await client.get("/v1/models")
    assert resp.status_code == 200
    body = resp.json()
    assert "data" in body
    ids = [m["id"] for m in body["data"]]
    # The mock model is always available
    assert "mock" in ids
    # The default model_map entries should appear
    assert "claude-sonnet-4-6" in ids
    # Each entry must have the required Anthropic fields
    for m in body["data"]:
        assert m["type"] == "model"
        assert "id" in m
        assert "display_name" in m
        assert "created_at" in m


@pytest.mark.asyncio
async def test_retrieve_model_by_id(client: AsyncClient):
    resp = await client.get("/v1/models/mock")
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == "mock"
    assert body["type"] == "model"
    assert "Mock" in body["display_name"]


@pytest.mark.asyncio
async def test_retrieve_model_not_found(client: AsyncClient):
    resp = await client.get("/v1/models/no-such-model")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_list_models_via_sdk_path(client: AsyncClient):
    """The SDK hits /v1/models?beta=true. The query string is ignored
    by FastAPI but the path must match."""
    resp = await client.get("/v1/models?beta=true")
    assert resp.status_code == 200
    assert "data" in resp.json()
