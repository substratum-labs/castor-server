"""Vault and credential models matching Anthropic Managed Agents API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .common import Metadata

# ---------------------------------------------------------------------------
# Auth types for credentials
# ---------------------------------------------------------------------------


class StaticBearerAuth(BaseModel):
    type: Literal["static_bearer"] = "static_bearer"
    mcp_server_url: str
    token: str


class MCPOAuthRefresh(BaseModel):
    token: str
    expires_at: str | None = None


class MCPOAuthAuth(BaseModel):
    type: Literal["mcp_oauth"] = "mcp_oauth"
    mcp_server_url: str
    access_token: str
    expires_at: str | None = None
    refresh: MCPOAuthRefresh | None = None


CredentialAuth = StaticBearerAuth | MCPOAuthAuth


# ---------------------------------------------------------------------------
# Vault requests / responses
# ---------------------------------------------------------------------------


class CreateVaultRequest(BaseModel):
    display_name: str = Field(..., min_length=1, max_length=255)
    metadata: Metadata = Field(default_factory=dict)


class UpdateVaultRequest(BaseModel):
    display_name: str | None = None
    metadata: Metadata | None = None


class VaultResponse(BaseModel):
    id: str
    type: Literal["vault"] = "vault"
    display_name: str
    metadata: dict[str, str] = Field(default_factory=dict)
    created_at: str
    updated_at: str
    archived_at: str | None = None


class VaultListResponse(BaseModel):
    data: list[VaultResponse]
    next_page: str | None = None


class VaultDeletedResponse(BaseModel):
    id: str
    type: Literal["vault_deleted"] = "vault_deleted"


# ---------------------------------------------------------------------------
# Credential requests / responses
# ---------------------------------------------------------------------------


class CreateCredentialRequest(BaseModel):
    auth: CredentialAuth
    display_name: str | None = Field(default=None, max_length=255)
    metadata: Metadata = Field(default_factory=dict)


class UpdateCredentialRequest(BaseModel):
    auth: CredentialAuth | None = None
    display_name: str | None = None
    metadata: Metadata | None = None


class CredentialAuthResponse(BaseModel):
    """Auth in responses — sensitive fields (token, access_token) are masked."""

    type: str
    mcp_server_url: str


class CredentialResponse(BaseModel):
    id: str
    vault_id: str
    type: Literal["vault_credential"] = "vault_credential"
    display_name: str | None = None
    auth: CredentialAuthResponse
    metadata: dict[str, str] = Field(default_factory=dict)
    created_at: str
    updated_at: str
    archived_at: str | None = None


class CredentialListResponse(BaseModel):
    data: list[CredentialResponse]
    next_page: str | None = None


class CredentialDeletedResponse(BaseModel):
    id: str
    type: Literal["vault_credential_deleted"] = "vault_credential_deleted"
