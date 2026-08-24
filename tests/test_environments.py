"""Tests for environment CRUD endpoints."""

from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_create_environment(client: AsyncClient):
    resp = await client.post(
        "/v1/environments",
        json={
            "name": "test-env",
            "image": "python:3.12-slim",
            "network": True,
            "writable": True,
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "test-env"
    assert data["image"] == "python:3.12-slim"
    assert data["network"] is True
    assert data["writable"] is True
    assert data["type"] == "environment"
    assert data["id"].startswith("env_")


@pytest.mark.asyncio
async def test_create_environment_defaults(client: AsyncClient):
    resp = await client.post(
        "/v1/environments",
        json={"name": "minimal"},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["image"] == "python:3.12-slim"
    assert data["timeout_secs"] == 300
    assert data["network"] is False
    assert data["writable"] is True


@pytest.mark.asyncio
async def test_list_environments(client: AsyncClient):
    await client.post("/v1/environments", json={"name": "env-1"})
    await client.post("/v1/environments", json={"name": "env-2"})

    resp = await client.get("/v1/environments")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert len(data) >= 2


@pytest.mark.asyncio
async def test_get_environment(client: AsyncClient):
    create_resp = await client.post("/v1/environments", json={"name": "get-test"})
    env_id = create_resp.json()["id"]

    resp = await client.get(f"/v1/environments/{env_id}")
    assert resp.status_code == 200
    assert resp.json()["name"] == "get-test"


@pytest.mark.asyncio
async def test_get_environment_not_found(client: AsyncClient):
    resp = await client.get("/v1/environments/nonexistent")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_update_environment(client: AsyncClient):
    create_resp = await client.post(
        "/v1/environments",
        json={"name": "update-test", "image": "python:3.11"},
    )
    env_id = create_resp.json()["id"]

    resp = await client.post(
        f"/v1/environments/{env_id}",
        json={"image": "python:3.12", "network": True},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["image"] == "python:3.12"
    assert data["network"] is True
    assert data["name"] == "update-test"


@pytest.mark.asyncio
async def test_archive_environment(client: AsyncClient):
    create_resp = await client.post("/v1/environments", json={"name": "archive-test"})
    env_id = create_resp.json()["id"]

    resp = await client.post(f"/v1/environments/{env_id}/archive")
    assert resp.status_code == 200
    assert resp.json()["archived_at"] is not None

    # Archived envs excluded from default list
    list_resp = await client.get("/v1/environments")
    ids = [e["id"] for e in list_resp.json()["data"]]
    assert env_id not in ids

    # But included with flag
    list_resp = await client.get("/v1/environments?include_archived=true")
    ids = [e["id"] for e in list_resp.json()["data"]]
    assert env_id in ids


@pytest.mark.asyncio
async def test_environment_with_metadata(client: AsyncClient):
    resp = await client.post(
        "/v1/environments",
        json={
            "name": "meta-test",
            "metadata": {"team": "infra", "tier": "standard"},
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["metadata"]["team"] == "infra"


@pytest.mark.asyncio
async def test_environment_with_allowlist(client: AsyncClient):
    resp = await client.post(
        "/v1/environments",
        json={
            "name": "net-test",
            "network": True,
            "network_allowlist": ["api.example.com", "cdn.example.com"],
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert len(data["network_allowlist"]) == 2


@pytest.mark.asyncio
async def test_registers_roche_space_ops_environment(client: AsyncClient):
    payload = {
        "environment_id": "argonavis-triage-env",
        "name": "Roche Space Ops Sandbox",
        "image": "substratum/roche-space-ops:v0.1.0",
        "provider": "docker",
        "resource_limits": {
            "cpu_cores": 2.0,
            "memory_mb": 2048,
            "pids_limit": 100,
            "read_only_rootfs": True,
            "network_mode": "none",
        },
        "env_vars": {
            "OREKIT_DATA_PATH": "/opt/space-data/orekit-data",
            "ASTROPY_OFFLINE": "1",
            "OFFLINE_MODE": "1",
        },
        "pre_warmed_instances": 1,
    }

    create_resp = await client.post("/v1/environments", json=payload)
    assert create_resp.status_code == 201
    data = create_resp.json()
    assert data["id"] == "argonavis-triage-env"
    assert data["image"] == payload["image"]
    assert data["provider"] == "docker"
    assert data["resource_limits"] == payload["resource_limits"]
    assert data["env_vars"] == payload["env_vars"]
    assert data["pre_warmed_instances"] == 1
    assert data["network"] is False
    assert data["writable"] is False

    get_resp = await client.get("/v1/environments/argonavis-triage-env")
    assert get_resp.status_code == 200
    assert get_resp.json()["resource_limits"] == payload["resource_limits"]
