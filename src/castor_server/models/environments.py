"""Environment models for sandbox configuration."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .common import Metadata


class CreateEnvironmentRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=256)
    image: str = "python:3.12-slim"
    memory: str | None = None
    cpus: float | None = None
    timeout_secs: int = 300
    network: bool = False
    writable: bool = True
    network_allowlist: list[str] = Field(default_factory=list)
    metadata: Metadata = Field(default_factory=dict)


class UpdateEnvironmentRequest(BaseModel):
    name: str | None = None
    image: str | None = None
    memory: str | None = None
    cpus: float | None = None
    timeout_secs: int | None = None
    network: bool | None = None
    writable: bool | None = None
    network_allowlist: list[str] | None = None
    metadata: Metadata | None = None


class EnvironmentResponse(BaseModel):
    id: str
    type: Literal["environment"] = "environment"
    name: str
    image: str
    memory: str | None = None
    cpus: float | None = None
    timeout_secs: int = 300
    network: bool = False
    writable: bool = True
    network_allowlist: list[str] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)
    created_at: str
    updated_at: str
    archived_at: str | None = None


class EnvironmentListResponse(BaseModel):
    data: list[EnvironmentResponse]
    next_page: str | None = None
