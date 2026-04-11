"""Tests for the Files API (/v1/files)."""

from __future__ import annotations

from io import BytesIO

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_upload_and_retrieve_metadata(client: AsyncClient, tmp_path, monkeypatch):
    monkeypatch.setattr("castor_server.config.settings.files_dir", str(tmp_path))

    payload = b"Hello, this is a test PDF body."
    resp = await client.post(
        "/v1/files",
        files={"file": ("hello.pdf", BytesIO(payload), "application/pdf")},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["type"] == "file"
    assert data["filename"] == "hello.pdf"
    assert data["mime_type"] == "application/pdf"
    assert data["size_bytes"] == len(payload)
    assert data["id"].startswith("file_")
    file_id = data["id"]

    # Metadata GET round-trip
    meta_resp = await client.get(f"/v1/files/{file_id}")
    assert meta_resp.status_code == 200
    assert meta_resp.json()["id"] == file_id

    # Blob is on disk
    blob = tmp_path / file_id
    assert blob.exists()
    assert blob.read_bytes() == payload


@pytest.mark.asyncio
async def test_download_file_content(client: AsyncClient, tmp_path, monkeypatch):
    monkeypatch.setattr("castor_server.config.settings.files_dir", str(tmp_path))

    payload = b"line one\nline two\n"
    upload = await client.post(
        "/v1/files",
        files={"file": ("notes.txt", BytesIO(payload), "text/plain")},
    )
    file_id = upload.json()["id"]

    resp = await client.get(f"/v1/files/{file_id}/content")
    assert resp.status_code == 200
    assert resp.content == payload
    assert resp.headers["content-type"].startswith("text/plain")


@pytest.mark.asyncio
async def test_list_files(client: AsyncClient, tmp_path, monkeypatch):
    monkeypatch.setattr("castor_server.config.settings.files_dir", str(tmp_path))

    for name in ("a.txt", "b.txt", "c.txt"):
        await client.post(
            "/v1/files",
            files={"file": (name, BytesIO(b"x"), "text/plain")},
        )

    resp = await client.get("/v1/files")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["data"]) >= 3
    filenames = {f["filename"] for f in data["data"]}
    assert {"a.txt", "b.txt", "c.txt"} <= filenames


@pytest.mark.asyncio
async def test_delete_file(client: AsyncClient, tmp_path, monkeypatch):
    monkeypatch.setattr("castor_server.config.settings.files_dir", str(tmp_path))

    upload = await client.post(
        "/v1/files",
        files={
            "file": (
                "doomed.bin",
                BytesIO(b"\x00\x01\x02"),
                "application/octet-stream",
            )
        },
    )
    file_id = upload.json()["id"]
    blob = tmp_path / file_id
    assert blob.exists()

    resp = await client.delete(f"/v1/files/{file_id}")
    assert resp.status_code == 200
    assert resp.json()["type"] == "file_deleted"
    assert resp.json()["id"] == file_id

    # Metadata 404
    assert (await client.get(f"/v1/files/{file_id}")).status_code == 404
    # Blob removed from disk
    assert not blob.exists()


@pytest.mark.asyncio
async def test_retrieve_file_not_found(client: AsyncClient):
    resp = await client.get("/v1/files/nonexistent_file_id")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_download_file_not_found(client: AsyncClient):
    resp = await client.get("/v1/files/nonexistent/content")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_file_not_found(client: AsyncClient):
    resp = await client.delete("/v1/files/nonexistent")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_upload_size_limit(client: AsyncClient, tmp_path, monkeypatch):
    monkeypatch.setattr("castor_server.config.settings.files_dir", str(tmp_path))
    monkeypatch.setattr("castor_server.config.settings.files_max_bytes", 100)

    big = b"x" * 500
    resp = await client.post(
        "/v1/files",
        files={"file": ("big.bin", BytesIO(big), "application/octet-stream")},
    )
    assert resp.status_code == 413
    # No leftover blob
    assert not any(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_upload_via_sdk_compatible_path(
    client: AsyncClient, tmp_path, monkeypatch
):
    """The SDK hits /v1/files?beta=true. Verify the query string is ignored."""
    monkeypatch.setattr("castor_server.config.settings.files_dir", str(tmp_path))

    resp = await client.post(
        "/v1/files?beta=true",
        files={"file": ("x.txt", BytesIO(b"y"), "text/plain")},
    )
    assert resp.status_code == 200
    assert resp.json()["filename"] == "x.txt"
