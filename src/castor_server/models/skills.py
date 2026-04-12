"""Skills API schemas matching anthropic-python's Skill* response types."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class SkillResponse(BaseModel):
    """Mirrors SkillCreateResponse / SkillRetrieveResponse / SkillListResponse."""

    id: str
    type: Literal["skill"] = "skill"
    display_title: str | None = None
    source: Literal["custom", "anthropic"] = "custom"
    created_at: str
    updated_at: str
    latest_version: str | None = None


class SkillListResult(BaseModel):
    data: list[SkillResponse]
    next_page: str | None = None


class SkillDeleteResponse(BaseModel):
    id: str
    type: Literal["skill_deleted"] = "skill_deleted"


class SkillVersionResponse(BaseModel):
    """Mirrors ``VersionListResponse`` / ``VersionRetrieveResponse``."""

    id: str
    type: Literal["skill_version"] = "skill_version"
    skill_id: str
    version: str
    name: str
    description: str
    directory: str
    created_at: str


class SkillVersionListResult(BaseModel):
    data: list[SkillVersionResponse]
    next_page: str | None = None


class SkillVersionDeleteResponse(BaseModel):
    id: str
    type: Literal["skill_version_deleted"] = "skill_version_deleted"
