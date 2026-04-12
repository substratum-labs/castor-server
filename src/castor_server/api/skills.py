"""Skills API — /v1/skills.

Wire-compatible with anthropic-python's ``client.beta.skills`` namespace.
Skills are bundles of files (including a mandatory ``SKILL.md`` at root)
that extend an agent's capabilities. Each create produces a new version.
File blobs are stored on disk under ``settings.files_dir/skills/<id>/<ver>/``.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

from fastapi import APIRouter, Depends, Form, HTTPException, Query, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from castor_server.config import settings
from castor_server.models.common import gen_id
from castor_server.models.skills import (
    SkillDeleteResponse,
    SkillListResult,
    SkillResponse,
    SkillVersionListResult,
)
from castor_server.store.database import get_session
from castor_server.store.repository import (
    create_skill,
    create_skill_version,
    delete_skill,
    get_skill,
    list_skill_versions,
    list_skills,
)

router = APIRouter(prefix="/v1/skills", tags=["skills"])


def _skills_dir() -> Path:
    p = Path(settings.files_dir) / "skills"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _parse_skill_md(content: str) -> tuple[str, str]:
    """Extract name and description from SKILL.md content.

    Looks for a ``# Title`` line as the name and everything after it
    (up to the next heading) as the description. Falls back to empty
    strings if parsing fails.
    """
    name = ""
    description = ""
    lines = content.splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"^#\s+(.+)", line)
        if m:
            name = m.group(1).strip()
            # Everything after the heading until the next heading
            desc_lines = []
            for rest in lines[i + 1 :]:
                if rest.startswith("#"):
                    break
                desc_lines.append(rest)
            description = "\n".join(desc_lines).strip()
            break
    return name, description


@router.post("", response_model=SkillResponse)
async def create_skill_endpoint(
    display_title: str | None = Form(default=None),
    files: list[UploadFile] = [],
    db: AsyncSession = Depends(get_session),
) -> SkillResponse:
    """Create a new skill from uploaded files.

    The files must include a ``SKILL.md`` at the root of the top-level
    directory. Each call creates a new skill and a first version.
    """
    skill_id = gen_id("skill")
    version = str(int(time.time() * 1_000_000))
    version_id = gen_id("skillver")

    # Store files on disk
    version_dir = _skills_dir() / skill_id / version
    version_dir.mkdir(parents=True, exist_ok=True)

    skill_md_content = ""
    directory = ""

    for upload in files:
        filename = upload.filename or "unknown"
        # Determine directory from the first file's path
        parts = filename.replace("\\", "/").split("/")
        if len(parts) > 1 and not directory:
            directory = parts[0]
        target = version_dir / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        data = await upload.read()
        target.write_bytes(data)

        # Check for SKILL.md
        basename = parts[-1] if parts else filename
        if basename.upper() == "SKILL.MD":
            skill_md_content = data.decode("utf-8", errors="replace")

    name, description = _parse_skill_md(skill_md_content)

    skill = await create_skill(
        db,
        skill_id=skill_id,
        display_title=display_title,
        source="custom",
    )

    await create_skill_version(
        db,
        version_id=version_id,
        skill_id=skill_id,
        version=version,
        name=name or display_title or skill_id,
        description=description,
        directory=directory or skill_id,
    )

    # Refresh to get latest_version
    skill = await get_skill(db, skill_id)
    return skill  # type: ignore[return-value]


@router.get("", response_model=SkillListResult)
async def list_skills_endpoint(
    limit: int = Query(default=20, le=1000),
    db: AsyncSession = Depends(get_session),
) -> SkillListResult:
    skills = await list_skills(db, limit=limit)
    return SkillListResult(data=skills)


@router.get("/{skill_id}", response_model=SkillResponse)
async def retrieve_skill(
    skill_id: str,
    db: AsyncSession = Depends(get_session),
) -> SkillResponse:
    skill = await get_skill(db, skill_id)
    if skill is None:
        raise HTTPException(status_code=404, detail="Skill not found")
    return skill


@router.delete("/{skill_id}", response_model=SkillDeleteResponse)
async def delete_skill_endpoint(
    skill_id: str,
    db: AsyncSession = Depends(get_session),
) -> SkillDeleteResponse:
    deleted = await delete_skill(db, skill_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Skill not found")
    # Best-effort cleanup of files on disk
    skill_dir = _skills_dir() / skill_id
    if skill_dir.exists():
        import shutil

        shutil.rmtree(skill_dir, ignore_errors=True)
    return SkillDeleteResponse(id=skill_id)


@router.get("/{skill_id}/versions", response_model=SkillVersionListResult)
async def list_skill_versions_endpoint(
    skill_id: str,
    limit: int = Query(default=20, le=1000),
    db: AsyncSession = Depends(get_session),
) -> SkillVersionListResult:
    skill = await get_skill(db, skill_id)
    if skill is None:
        raise HTTPException(status_code=404, detail="Skill not found")
    versions = await list_skill_versions(db, skill_id, limit=limit)
    return SkillVersionListResult(data=versions)
