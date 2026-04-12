"""Vault and credential routes — /v1/vaults CRUD."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from castor_server.models.vaults import (
    CreateCredentialRequest,
    CreateVaultRequest,
    CredentialDeletedResponse,
    CredentialListResponse,
    CredentialResponse,
    UpdateCredentialRequest,
    UpdateVaultRequest,
    VaultDeletedResponse,
    VaultListResponse,
    VaultResponse,
)
from castor_server.store.database import get_session
from castor_server.store.repository import (
    archive_credential,
    archive_vault,
    create_credential,
    create_vault,
    delete_credential,
    delete_vault,
    get_credential,
    get_vault,
    list_credentials,
    list_vaults,
    update_credential,
    update_vault,
)

router = APIRouter(prefix="/v1/vaults", tags=["vaults"])


# ---------------------------------------------------------------------------
# Vault CRUD
# ---------------------------------------------------------------------------


@router.post("", response_model=VaultResponse, status_code=201)
async def create_vault_endpoint(
    body: CreateVaultRequest,
    db: AsyncSession = Depends(get_session),
) -> VaultResponse:
    return await create_vault(
        db, display_name=body.display_name, metadata=body.metadata
    )


@router.get("", response_model=VaultListResponse)
async def list_vaults_endpoint(
    limit: int = Query(default=20, le=100),
    include_archived: bool = Query(default=False),
    db: AsyncSession = Depends(get_session),
) -> VaultListResponse:
    vaults = await list_vaults(db, limit=limit, include_archived=include_archived)
    return VaultListResponse(data=vaults)


@router.get("/{vault_id}", response_model=VaultResponse)
async def get_vault_endpoint(
    vault_id: str,
    db: AsyncSession = Depends(get_session),
) -> VaultResponse:
    vault = await get_vault(db, vault_id)
    if vault is None:
        raise HTTPException(status_code=404, detail="Vault not found")
    return vault


@router.post("/{vault_id}", response_model=VaultResponse)
async def update_vault_endpoint(
    vault_id: str,
    body: UpdateVaultRequest,
    db: AsyncSession = Depends(get_session),
) -> VaultResponse:
    updates = body.model_dump(exclude_none=True)
    if not updates:
        vault = await get_vault(db, vault_id)
        if vault is None:
            raise HTTPException(status_code=404, detail="Vault not found")
        return vault
    vault = await update_vault(db, vault_id, **updates)
    if vault is None:
        raise HTTPException(status_code=404, detail="Vault not found")
    return vault


@router.delete("/{vault_id}", response_model=VaultDeletedResponse)
async def delete_vault_endpoint(
    vault_id: str,
    db: AsyncSession = Depends(get_session),
) -> VaultDeletedResponse:
    if not await delete_vault(db, vault_id):
        raise HTTPException(status_code=404, detail="Vault not found")
    return VaultDeletedResponse(id=vault_id)


@router.post("/{vault_id}/archive", response_model=VaultResponse)
async def archive_vault_endpoint(
    vault_id: str,
    db: AsyncSession = Depends(get_session),
) -> VaultResponse:
    vault = await archive_vault(db, vault_id)
    if vault is None:
        raise HTTPException(status_code=404, detail="Vault not found")
    return vault


# ---------------------------------------------------------------------------
# Credential CRUD (nested under vault)
# ---------------------------------------------------------------------------


def _extract_auth_fields(auth) -> dict:
    """Flatten CredentialAuth union into repository function kwargs."""
    fields = {
        "auth_type": auth.type,
        "mcp_server_url": auth.mcp_server_url,
    }
    if auth.type == "static_bearer":
        fields["token"] = auth.token
    elif auth.type == "mcp_oauth":
        fields["access_token"] = auth.access_token
        fields["expires_at"] = auth.expires_at
        if auth.refresh:
            fields["refresh_token"] = auth.refresh.token
            fields["refresh_expires_at"] = auth.refresh.expires_at
    return fields


@router.post(
    "/{vault_id}/credentials",
    response_model=CredentialResponse,
    status_code=201,
)
async def create_credential_endpoint(
    vault_id: str,
    body: CreateCredentialRequest,
    db: AsyncSession = Depends(get_session),
) -> CredentialResponse:
    vault = await get_vault(db, vault_id)
    if vault is None:
        raise HTTPException(status_code=404, detail="Vault not found")
    auth_fields = _extract_auth_fields(body.auth)
    return await create_credential(
        db,
        vault_id=vault_id,
        display_name=body.display_name,
        metadata=body.metadata,
        **auth_fields,
    )


@router.get("/{vault_id}/credentials", response_model=CredentialListResponse)
async def list_credentials_endpoint(
    vault_id: str,
    limit: int = Query(default=20, le=100),
    include_archived: bool = Query(default=False),
    db: AsyncSession = Depends(get_session),
) -> CredentialListResponse:
    creds = await list_credentials(
        db, vault_id, limit=limit, include_archived=include_archived
    )
    return CredentialListResponse(data=creds)


@router.get(
    "/{vault_id}/credentials/{credential_id}",
    response_model=CredentialResponse,
)
async def get_credential_endpoint(
    vault_id: str,
    credential_id: str,
    db: AsyncSession = Depends(get_session),
) -> CredentialResponse:
    cred = await get_credential(db, vault_id, credential_id)
    if cred is None:
        raise HTTPException(status_code=404, detail="Credential not found")
    return cred


@router.post(
    "/{vault_id}/credentials/{credential_id}",
    response_model=CredentialResponse,
)
async def update_credential_endpoint(
    vault_id: str,
    credential_id: str,
    body: UpdateCredentialRequest,
    db: AsyncSession = Depends(get_session),
) -> CredentialResponse:
    updates = {}
    if body.display_name is not None:
        updates["display_name"] = body.display_name
    if body.metadata is not None:
        updates["metadata"] = body.metadata
    if body.auth is not None:
        updates.update(_extract_auth_fields(body.auth))
    if not updates:
        cred = await get_credential(db, vault_id, credential_id)
        if cred is None:
            raise HTTPException(status_code=404, detail="Credential not found")
        return cred
    cred = await update_credential(db, vault_id, credential_id, **updates)
    if cred is None:
        raise HTTPException(status_code=404, detail="Credential not found")
    return cred


@router.delete(
    "/{vault_id}/credentials/{credential_id}",
    response_model=CredentialDeletedResponse,
)
async def delete_credential_endpoint(
    vault_id: str,
    credential_id: str,
    db: AsyncSession = Depends(get_session),
) -> CredentialDeletedResponse:
    if not await delete_credential(db, vault_id, credential_id):
        raise HTTPException(status_code=404, detail="Credential not found")
    return CredentialDeletedResponse(id=credential_id)


@router.post(
    "/{vault_id}/credentials/{credential_id}/archive",
    response_model=CredentialResponse,
)
async def archive_credential_endpoint(
    vault_id: str,
    credential_id: str,
    db: AsyncSession = Depends(get_session),
) -> CredentialResponse:
    cred = await archive_credential(db, vault_id, credential_id)
    if cred is None:
        raise HTTPException(status_code=404, detail="Credential not found")
    return cred
