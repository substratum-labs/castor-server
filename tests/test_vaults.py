"""Tests for vault and credential CRUD endpoints."""

from __future__ import annotations

import pytest
from httpx import AsyncClient

# ---------------------------------------------------------------------------
# Vault CRUD
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_vault(client: AsyncClient):
    resp = await client.post(
        "/v1/vaults",
        json={"display_name": "test-vault"},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["display_name"] == "test-vault"
    assert data["type"] == "vault"
    assert data["id"].startswith("vault_")


@pytest.mark.asyncio
async def test_create_vault_with_metadata(client: AsyncClient):
    resp = await client.post(
        "/v1/vaults",
        json={"display_name": "meta-vault", "metadata": {"team": "infra"}},
    )
    assert resp.status_code == 201
    assert resp.json()["metadata"]["team"] == "infra"


@pytest.mark.asyncio
async def test_list_vaults(client: AsyncClient):
    await client.post("/v1/vaults", json={"display_name": "v1"})
    await client.post("/v1/vaults", json={"display_name": "v2"})
    resp = await client.get("/v1/vaults")
    assert resp.status_code == 200
    assert len(resp.json()["data"]) >= 2


@pytest.mark.asyncio
async def test_get_vault(client: AsyncClient):
    create = await client.post("/v1/vaults", json={"display_name": "get-test"})
    vid = create.json()["id"]
    resp = await client.get(f"/v1/vaults/{vid}")
    assert resp.status_code == 200
    assert resp.json()["display_name"] == "get-test"


@pytest.mark.asyncio
async def test_get_vault_not_found(client: AsyncClient):
    resp = await client.get("/v1/vaults/nonexistent")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_update_vault(client: AsyncClient):
    create = await client.post("/v1/vaults", json={"display_name": "old"})
    vid = create.json()["id"]
    resp = await client.post(f"/v1/vaults/{vid}", json={"display_name": "new"})
    assert resp.status_code == 200
    assert resp.json()["display_name"] == "new"


@pytest.mark.asyncio
async def test_delete_vault(client: AsyncClient):
    create = await client.post("/v1/vaults", json={"display_name": "del"})
    vid = create.json()["id"]
    resp = await client.delete(f"/v1/vaults/{vid}")
    assert resp.status_code == 200
    assert resp.json()["type"] == "vault_deleted"
    assert (await client.get(f"/v1/vaults/{vid}")).status_code == 404


@pytest.mark.asyncio
async def test_archive_vault(client: AsyncClient):
    create = await client.post("/v1/vaults", json={"display_name": "arch"})
    vid = create.json()["id"]
    resp = await client.post(f"/v1/vaults/{vid}/archive")
    assert resp.status_code == 200
    assert resp.json()["archived_at"] is not None

    # Excluded from default list
    listed = await client.get("/v1/vaults")
    ids = [v["id"] for v in listed.json()["data"]]
    assert vid not in ids

    # Included with flag
    listed = await client.get("/v1/vaults?include_archived=true")
    ids = [v["id"] for v in listed.json()["data"]]
    assert vid in ids


# ---------------------------------------------------------------------------
# Credential CRUD
# ---------------------------------------------------------------------------


async def _create_vault(client: AsyncClient) -> str:
    resp = await client.post("/v1/vaults", json={"display_name": "cred-vault"})
    return resp.json()["id"]


@pytest.mark.asyncio
async def test_create_credential_static_bearer(client: AsyncClient):
    vid = await _create_vault(client)
    resp = await client.post(
        f"/v1/vaults/{vid}/credentials",
        json={
            "auth": {
                "type": "static_bearer",
                "mcp_server_url": "https://mcp.example.com",
                "token": "secret-123",
            },
            "display_name": "my-cred",
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["type"] == "vault_credential"
    assert data["vault_id"] == vid
    assert data["display_name"] == "my-cred"
    # Token must NOT be in the response
    assert data["auth"]["type"] == "static_bearer"
    assert data["auth"]["mcp_server_url"] == "https://mcp.example.com"
    assert "token" not in data["auth"]


@pytest.mark.asyncio
async def test_create_credential_mcp_oauth(client: AsyncClient):
    vid = await _create_vault(client)
    resp = await client.post(
        f"/v1/vaults/{vid}/credentials",
        json={
            "auth": {
                "type": "mcp_oauth",
                "mcp_server_url": "https://oauth.example.com",
                "access_token": "at-xxx",
                "expires_at": "2026-12-31T00:00:00Z",
                "refresh": {
                    "token": "rt-xxx",
                    "expires_at": "2027-06-30T00:00:00Z",
                },
            },
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["auth"]["type"] == "mcp_oauth"
    assert "access_token" not in data["auth"]


@pytest.mark.asyncio
async def test_list_credentials(client: AsyncClient):
    vid = await _create_vault(client)
    await client.post(
        f"/v1/vaults/{vid}/credentials",
        json={
            "auth": {
                "type": "static_bearer",
                "mcp_server_url": "https://a.com",
                "token": "t1",
            }
        },
    )
    await client.post(
        f"/v1/vaults/{vid}/credentials",
        json={
            "auth": {
                "type": "static_bearer",
                "mcp_server_url": "https://b.com",
                "token": "t2",
            }
        },
    )
    resp = await client.get(f"/v1/vaults/{vid}/credentials")
    assert resp.status_code == 200
    assert len(resp.json()["data"]) >= 2


@pytest.mark.asyncio
async def test_get_credential(client: AsyncClient):
    vid = await _create_vault(client)
    create = await client.post(
        f"/v1/vaults/{vid}/credentials",
        json={
            "auth": {
                "type": "static_bearer",
                "mcp_server_url": "https://get.com",
                "token": "t",
            },
            "display_name": "get-cred",
        },
    )
    cid = create.json()["id"]
    resp = await client.get(f"/v1/vaults/{vid}/credentials/{cid}")
    assert resp.status_code == 200
    assert resp.json()["display_name"] == "get-cred"


@pytest.mark.asyncio
async def test_delete_credential(client: AsyncClient):
    vid = await _create_vault(client)
    create = await client.post(
        f"/v1/vaults/{vid}/credentials",
        json={
            "auth": {
                "type": "static_bearer",
                "mcp_server_url": "https://del.com",
                "token": "t",
            }
        },
    )
    cid = create.json()["id"]
    resp = await client.delete(f"/v1/vaults/{vid}/credentials/{cid}")
    assert resp.status_code == 200
    assert resp.json()["type"] == "vault_credential_deleted"
    assert (await client.get(f"/v1/vaults/{vid}/credentials/{cid}")).status_code == 404


@pytest.mark.asyncio
async def test_archive_credential(client: AsyncClient):
    vid = await _create_vault(client)
    create = await client.post(
        f"/v1/vaults/{vid}/credentials",
        json={
            "auth": {
                "type": "static_bearer",
                "mcp_server_url": "https://arch.com",
                "token": "t",
            }
        },
    )
    cid = create.json()["id"]
    resp = await client.post(f"/v1/vaults/{vid}/credentials/{cid}/archive")
    assert resp.status_code == 200
    assert resp.json()["archived_at"] is not None


@pytest.mark.asyncio
async def test_credential_on_nonexistent_vault(client: AsyncClient):
    resp = await client.post(
        "/v1/vaults/nonexistent/credentials",
        json={
            "auth": {
                "type": "static_bearer",
                "mcp_server_url": "https://x.com",
                "token": "t",
            }
        },
    )
    assert resp.status_code == 404
