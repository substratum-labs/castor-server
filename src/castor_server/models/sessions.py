"""Session models matching Anthropic Managed Agents API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .agents import AgentResponse
from .common import Metadata

# ---------------------------------------------------------------------------
# Resource types
# ---------------------------------------------------------------------------


class GitHubResource(BaseModel):
    type: Literal["github_repository"] = "github_repository"
    url: str
    authorization_token: str | None = None
    checkout: str | None = None
    mount_path: str | None = None


class FileResource(BaseModel):
    type: Literal["file"] = "file"
    file_id: str
    mount_path: str | None = None


Resource = GitHubResource | FileResource


# ---------------------------------------------------------------------------
# Agent reference in session create
# ---------------------------------------------------------------------------


class AgentRef(BaseModel):
    type: Literal["agent"] = "agent"
    id: str
    version: int | None = None


# ---------------------------------------------------------------------------
# Stats and usage
# ---------------------------------------------------------------------------


class SessionStats(BaseModel):
    active_seconds: float | None = None
    duration_seconds: float | None = None


class CacheCreation(BaseModel):
    ephemeral_1h_input_tokens: int | None = None
    ephemeral_5m_input_tokens: int | None = None


class SessionUsage(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    cache_creation: CacheCreation | None = None


# ---------------------------------------------------------------------------
# Create / Update requests
# ---------------------------------------------------------------------------


class CreateSessionRequest(BaseModel):
    agent: str | AgentRef
    environment_id: str | None = None
    title: str | None = None
    metadata: Metadata = Field(default_factory=dict)
    resources: list[Resource] = Field(default_factory=list)
    vault_ids: list[str] = Field(default_factory=list)


class UpdateSessionRequest(BaseModel):
    title: str | None = None
    metadata: Metadata | None = None


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------


class SessionResponse(BaseModel):
    id: str
    type: Literal["session"] = "session"
    agent: AgentResponse
    environment_id: str | None = None
    title: str | None = None
    status: Literal["idle", "running", "rescheduling", "terminated"] = "idle"
    metadata: dict[str, str] = Field(default_factory=dict)
    resources: list[Resource] = Field(default_factory=list)
    stats: SessionStats = Field(default_factory=SessionStats)
    usage: SessionUsage = Field(default_factory=SessionUsage)
    vault_ids: list[str] = Field(default_factory=list)
    created_at: str
    updated_at: str
    archived_at: str | None = None


class SessionListResponse(BaseModel):
    data: list[SessionResponse]
    next_page: str | None = None


class SessionDeletedResponse(BaseModel):
    id: str
    type: Literal["session_deleted"] = "session_deleted"


# ---------------------------------------------------------------------------
# Phase 2 — Castor extensions
# ---------------------------------------------------------------------------


class ForkRequest(BaseModel):
    at_step: int = Field(..., ge=0, description="Rewind to this syscall step index")


class BudgetItem(BaseModel):
    resource: str
    max_budget: float
    current_usage: float
    remaining: float


class BudgetResponse(BaseModel):
    budgets: list[BudgetItem]


class FlaggedStep(BaseModel):
    index: int
    tool_name: str
    arguments: dict
    response: object | None = None
    reason: str


class ScanResponse(BaseModel):
    total_steps: int
    auto_verified: int
    flagged_count: int
    flagged: list[FlaggedStep]
    tools_used: dict[str, int]
