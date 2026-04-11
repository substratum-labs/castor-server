"""Files API — /v1/files.

Wire-compatible with anthropic-python's ``client.beta.files`` namespace.
File metadata lives in SQLAlchemy; the actual bytes are stored on disk
under ``settings.files_dir/<file_id>`` so the database stays small.
"""

from __future__ import annotations

import mimetypes
import os
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from castor_server.config import settings
from castor_server.models.common import gen_id
from castor_server.models.files import (
    DeletedFile,
    FileListResponse,
    FileMetadata,
)
from castor_server.store.database import get_session
from castor_server.store.repository import (
    create_file,
    delete_file,
    get_file,
    list_files,
)

router = APIRouter(prefix="/v1/files", tags=["files"])


def _files_dir() -> Path:
    p = Path(settings.files_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _blob_path(file_id: str) -> Path:
    return _files_dir() / file_id


@router.post("", response_model=FileMetadata)
async def upload_file(
    file: UploadFile,
    db: AsyncSession = Depends(get_session),
) -> FileMetadata:
    """Upload a file. Body is multipart/form-data with a single ``file`` part."""
    file_id = gen_id("file")
    mime_type = (
        file.content_type
        or mimetypes.guess_type(file.filename or "")[0]
        or "application/octet-stream"
    )

    # Stream to disk so large files don't blow the process memory.
    target = _blob_path(file_id)
    size = 0
    with target.open("wb") as out:
        while True:
            chunk = await file.read(64 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > settings.files_max_bytes:
                out.close()
                target.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"File too large (>{settings.files_max_bytes} bytes). "
                        "Increase CASTOR_FILES_MAX_BYTES to allow larger uploads."
                    ),
                )
            out.write(chunk)

    return await create_file(
        db,
        file_id=file_id,
        filename=file.filename or file_id,
        mime_type=mime_type,
        size_bytes=size,
    )


@router.get("", response_model=FileListResponse)
async def list_files_endpoint(
    limit: int = 100,
    db: AsyncSession = Depends(get_session),
) -> FileListResponse:
    files = await list_files(db, limit=limit)
    return FileListResponse(
        data=files,
        has_more=False,
        first_id=files[0].id if files else None,
        last_id=files[-1].id if files else None,
    )


@router.get("/{file_id}", response_model=FileMetadata)
async def retrieve_file_metadata(
    file_id: str,
    db: AsyncSession = Depends(get_session),
) -> FileMetadata:
    meta = await get_file(db, file_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="File not found")
    return meta


@router.get("/{file_id}/content")
async def download_file(
    file_id: str,
    db: AsyncSession = Depends(get_session),
) -> FileResponse:
    meta = await get_file(db, file_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="File not found")
    blob = _blob_path(file_id)
    if not blob.exists():
        raise HTTPException(
            status_code=404,
            detail=f"File metadata exists but blob is missing on disk: {blob}",
        )
    return FileResponse(
        path=blob,
        media_type=meta.mime_type,
        filename=meta.filename,
    )


@router.delete("/{file_id}", response_model=DeletedFile)
async def delete_file_endpoint(
    file_id: str,
    db: AsyncSession = Depends(get_session),
) -> DeletedFile:
    deleted = await delete_file(db, file_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="File not found")
    # Best-effort delete of the blob too — leaving it on disk is wasteful
    # but not catastrophic.
    blob = _blob_path(file_id)
    try:
        if blob.exists():
            os.unlink(blob)
    except OSError:
        pass
    return DeletedFile(id=file_id)
