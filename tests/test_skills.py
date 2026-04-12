"""Tests for the Skills API (/v1/skills)."""

from __future__ import annotations

from io import BytesIO

import pytest
from httpx import AsyncClient

SKILL_MD = b"""# My Custom Skill

This skill helps with testing.

It does various useful things.
"""


@pytest.mark.asyncio
async def test_create_skill_with_skill_md(client: AsyncClient, tmp_path, monkeypatch):
    monkeypatch.setattr("castor_server.config.settings.files_dir", str(tmp_path))

    resp = await client.post(
        "/v1/skills",
        data={"display_title": "Test Skill"},
        files=[("files", ("SKILL.md", BytesIO(SKILL_MD), "text/markdown"))],
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["type"] == "skill"
    assert data["id"].startswith("skill_")
    assert data["display_title"] == "Test Skill"
    assert data["source"] == "custom"
    assert data["latest_version"] is not None
    assert data["created_at"]
    assert data["updated_at"]


@pytest.mark.asyncio
async def test_create_skill_without_display_title(
    client: AsyncClient, tmp_path, monkeypatch
):
    monkeypatch.setattr("castor_server.config.settings.files_dir", str(tmp_path))

    resp = await client.post(
        "/v1/skills",
        files=[("files", ("SKILL.md", BytesIO(SKILL_MD), "text/markdown"))],
    )
    assert resp.status_code == 200
    assert resp.json()["display_title"] is None


@pytest.mark.asyncio
async def test_retrieve_skill(client: AsyncClient, tmp_path, monkeypatch):
    monkeypatch.setattr("castor_server.config.settings.files_dir", str(tmp_path))

    create_resp = await client.post(
        "/v1/skills",
        files=[("files", ("SKILL.md", BytesIO(SKILL_MD), "text/markdown"))],
    )
    skill_id = create_resp.json()["id"]

    resp = await client.get(f"/v1/skills/{skill_id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == skill_id


@pytest.mark.asyncio
async def test_retrieve_skill_not_found(client: AsyncClient):
    resp = await client.get("/v1/skills/nonexistent")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_list_skills(client: AsyncClient, tmp_path, monkeypatch):
    monkeypatch.setattr("castor_server.config.settings.files_dir", str(tmp_path))

    for i in range(3):
        await client.post(
            "/v1/skills",
            data={"display_title": f"Skill {i}"},
            files=[("files", ("SKILL.md", BytesIO(SKILL_MD), "text/markdown"))],
        )

    resp = await client.get("/v1/skills")
    assert resp.status_code == 200
    assert len(resp.json()["data"]) >= 3


@pytest.mark.asyncio
async def test_delete_skill(client: AsyncClient, tmp_path, monkeypatch):
    monkeypatch.setattr("castor_server.config.settings.files_dir", str(tmp_path))

    create_resp = await client.post(
        "/v1/skills",
        files=[("files", ("SKILL.md", BytesIO(SKILL_MD), "text/markdown"))],
    )
    skill_id = create_resp.json()["id"]

    resp = await client.delete(f"/v1/skills/{skill_id}")
    assert resp.status_code == 200
    assert resp.json()["type"] == "skill_deleted"

    # Should be gone now
    assert (await client.get(f"/v1/skills/{skill_id}")).status_code == 404


@pytest.mark.asyncio
async def test_list_skill_versions(client: AsyncClient, tmp_path, monkeypatch):
    monkeypatch.setattr("castor_server.config.settings.files_dir", str(tmp_path))

    create_resp = await client.post(
        "/v1/skills",
        data={"display_title": "Versioned"},
        files=[("files", ("SKILL.md", BytesIO(SKILL_MD), "text/markdown"))],
    )
    skill_id = create_resp.json()["id"]

    resp = await client.get(f"/v1/skills/{skill_id}/versions")
    assert resp.status_code == 200
    versions = resp.json()["data"]
    assert len(versions) >= 1
    v = versions[0]
    assert v["type"] == "skill_version"
    assert v["skill_id"] == skill_id
    assert v["name"] == "My Custom Skill"
    assert "testing" in v["description"]


@pytest.mark.asyncio
async def test_skill_md_parsing():
    """Unit test for SKILL.md parsing."""
    from castor_server.api.skills import _parse_skill_md

    name, desc = _parse_skill_md(
        "# Hello World\n\nThis is a description.\n\n## Details\n\nMore."
    )
    assert name == "Hello World"
    assert desc == "This is a description."


@pytest.mark.asyncio
async def test_skill_md_parsing_empty():
    from castor_server.api.skills import _parse_skill_md

    name, desc = _parse_skill_md("")
    assert name == ""
    assert desc == ""


@pytest.mark.asyncio
async def test_sdk_compatible_path(client: AsyncClient, tmp_path, monkeypatch):
    """SDK hits /v1/skills?beta=true."""
    monkeypatch.setattr("castor_server.config.settings.files_dir", str(tmp_path))

    resp = await client.get("/v1/skills?beta=true")
    assert resp.status_code == 200
    assert "data" in resp.json()
