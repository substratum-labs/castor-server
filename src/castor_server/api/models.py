"""Models endpoint — /v1/models.

Reports which model identifiers castor-server accepts in agent configs.
The list is derived from ``settings.litellm_model_map`` plus the built-in
``mock`` model. SDK clients use this to populate model pickers etc.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from castor_server.config import settings
from castor_server.models.models import ModelInfo, ModelListResponse

router = APIRouter(prefix="/v1/models", tags=["models"])

# Server start time — used as the placeholder created_at for built-in
# models. Agents using these names always resolve via the model map at
# call time, so the timestamp is purely informational.
_FIXED_CREATED_AT = "2026-01-01T00:00:00.000Z"

# Pretty display names for the well-known Anthropic models.
_DISPLAY_NAMES: dict[str, str] = {
    "claude-opus-4-6": "Claude Opus 4.6",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "claude-haiku-4-5": "Claude Haiku 4.5",
    "claude-haiku-4-5-20251001": "Claude Haiku 4.5 (2025-10-01)",
    "claude-opus-4-5": "Claude Opus 4.5",
    "claude-sonnet-4-5": "Claude Sonnet 4.5",
    "mock": "Mock (offline echo)",
}


def _build_model_list() -> list[ModelInfo]:
    """Build the list of models advertised by this server.

    Includes everything in ``litellm_model_map`` (the configured model
    aliases) plus the built-in ``mock`` model. Order matches map insertion
    order so newer models stay near the top — matches Anthropic's
    "more recent first" convention as much as we can.
    """
    seen: set[str] = set()
    models: list[ModelInfo] = []
    for model_id in settings.litellm_model_map:
        if model_id in seen:
            continue
        seen.add(model_id)
        models.append(
            ModelInfo(
                id=model_id,
                display_name=_DISPLAY_NAMES.get(model_id, model_id),
                created_at=_FIXED_CREATED_AT,
            )
        )
    return models


@router.get("", response_model=ModelListResponse)
async def list_models() -> ModelListResponse:
    models = _build_model_list()
    return ModelListResponse(
        data=models,
        has_more=False,
        first_id=models[0].id if models else None,
        last_id=models[-1].id if models else None,
    )


@router.get("/{model_id}", response_model=ModelInfo)
async def retrieve_model(model_id: str) -> ModelInfo:
    for m in _build_model_list():
        if m.id == model_id:
            return m
    raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
