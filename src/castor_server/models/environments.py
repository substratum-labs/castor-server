"""Environment models for sandbox configuration."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .common import Metadata


class CreateEnvironmentRequest(BaseModel):
    environment_id: str | None = Field(default=None, min_length=1, max_length=64)
    name: str = Field(..., min_length=1, max_length=256)
    image: str = "python:3.12-slim"
    provider: Literal["docker"] | None = None
    resource_limits: ResourceLimits | None = None
    env_vars: dict[str, str] = Field(default_factory=dict)
    pre_warmed_instances: int | None = Field(default=None, ge=0)
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
    provider: Literal["docker"] | None = None
    resource_limits: ResourceLimits | None = None
    env_vars: dict[str, str] = Field(default_factory=dict)
    pre_warmed_instances: int | None = None
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


class ResourceLimits(BaseModel):
    cpu_cores: float = Field(gt=0)
    memory_mb: int = Field(gt=0)
    pids_limit: int = Field(gt=0)
    read_only_rootfs: bool
    network_mode: Literal["none"]
