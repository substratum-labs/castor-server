"""Files API schemas matching anthropic-python's BetaFileMetadata."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class FileMetadata(BaseModel):
    """Mirror of ``anthropic.types.beta.FileMetadata``.

    Returned from ``POST /v1/files`` (upload), ``GET /v1/files/{id}``
    (retrieve metadata), and as elements of ``GET /v1/files`` (list).
    """

    id: str
    type: Literal["file"] = "file"
    filename: str
    mime_type: str
    size_bytes: int
    created_at: str
    downloadable: bool = True
    scope: str | None = None


class FileListResponse(BaseModel):
    data: list[FileMetadata]
    has_more: bool = False
    first_id: str | None = None
    last_id: str | None = None


class DeletedFile(BaseModel):
    id: str
    type: Literal["file_deleted"] = "file_deleted"


class FileResource(BaseModel):
    """Subset of session resource fields used in repository helpers."""

    type: Literal["file"] = "file"
    file_id: str
    mount_path: str | None = Field(default=None)
