"""Common model types shared across the API."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


def gen_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:24]}"


def now_rfc3339() -> str:
    return datetime.utcnow().isoformat(timespec="milliseconds") + "Z"


# ---------------------------------------------------------------------------
# Content blocks
# ---------------------------------------------------------------------------


class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


class Base64ImageSource(BaseModel):
    type: Literal["base64"] = "base64"
    data: str
    media_type: str


class URLImageSource(BaseModel):
    type: Literal["url"] = "url"
    url: str


class ImageBlock(BaseModel):
    type: Literal["image"] = "image"
    source: Base64ImageSource | URLImageSource


class DocumentBlock(BaseModel):
    type: Literal["document"] = "document"
    title: str | None = None
    context: str | None = None
    source: dict[str, Any]


ContentBlock = Annotated[
    TextBlock | ImageBlock | DocumentBlock,
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

Metadata = dict[str, str | None]


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------


class PaginatedResponse(BaseModel):
    data: list[Any]
    next_page: str | None = None


# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------


class ModelConfig(BaseModel):
    id: str
    speed: Literal["standard", "fast"] = "standard"
