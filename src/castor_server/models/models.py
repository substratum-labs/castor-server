"""Model info schemas matching anthropic-python `BetaModelInfo`."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class ModelInfo(BaseModel):
    id: str
    type: Literal["model"] = "model"
    display_name: str
    created_at: str
    max_input_tokens: int | None = None
    max_tokens: int | None = None


class ModelListResponse(BaseModel):
    data: list[ModelInfo]
    has_more: bool = False
    first_id: str | None = None
    last_id: str | None = None
