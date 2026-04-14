"""OpenAI Responses API compatible endpoint.

Translates OpenAI's ``POST /v1/responses`` wire format into
castor-server's internal Anthropic-compatible agent loop, then
translates the results back. This lets ``openai.OpenAI(base_url=...)``
talk to castor-server with zero code changes.

Mapping:
  POST /v1/responses   → create/reuse agent + session, run agent, return Response
  GET  /v1/responses/X → list events for session mapped to response X
  DELETE /v1/responses/X → delete session mapped to response X
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from castor_server.core.session_manager import session_manager
from castor_server.models.openai_compat import (
    OpenAIResponse,
    OpenAIResponseRequest,
    OpenAIUsage,
)
from castor_server.store import repository as repo
from castor_server.store.database import get_session

logger = logging.getLogger("castor_server.openai_compat")

router = APIRouter(tags=["openai-compat"])

# In-memory caches (process-lifetime). Production: use Redis.
# agent_hash → agent_id
_agent_cache: dict[str, str] = {}
# response_id → session_id
_response_to_session: dict[str, str] = {}
# session_id → last response_id
_session_to_response: dict[str, str] = {}


# ---------------------------------------------------------------------------
# POST /v1/responses — create a response (run the agent synchronously)
# ---------------------------------------------------------------------------


@router.post("/v1/responses", response_model=OpenAIResponse)
async def create_response(
    body: OpenAIResponseRequest,
    db: AsyncSession = Depends(get_session),
) -> OpenAIResponse:
    # 1. Resolve or create agent
    agent_id = await _resolve_agent(db, body)

    # 2. Resolve or create session
    session_id = await _resolve_session(db, agent_id, body.previous_response_id)

    # 3. Build user message content
    if isinstance(body.input, str):
        content = [{"type": "text", "text": body.input}]
    else:
        content = body.input

    # 4. Run agent loop synchronously and collect events
    events = await session_manager.run_and_wait(db, session_id, content)

    # 5. Translate events → OpenAI Response
    response = _build_response(body, events, session_id)

    # 6. Track response_id ↔ session_id mapping
    _response_to_session[response.id] = session_id
    _session_to_response[session_id] = response.id

    return response


# ---------------------------------------------------------------------------
# GET /v1/responses/{response_id}
# ---------------------------------------------------------------------------


@router.get("/v1/responses/{response_id}", response_model=OpenAIResponse)
async def get_response(
    response_id: str,
    db: AsyncSession = Depends(get_session),
) -> OpenAIResponse:
    session_id = _response_to_session.get(response_id)
    if not session_id:
        raise HTTPException(status_code=404, detail="Response not found")

    events = await repo.list_events(db, session_id, limit=1000, order="asc")
    return _build_response_from_stored_events(response_id, events, session_id)


# ---------------------------------------------------------------------------
# DELETE /v1/responses/{response_id}
# ---------------------------------------------------------------------------


@router.delete("/v1/responses/{response_id}")
async def delete_response(
    response_id: str,
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    session_id = _response_to_session.pop(response_id, None)
    if not session_id:
        raise HTTPException(status_code=404, detail="Response not found")
    _session_to_response.pop(session_id, None)
    await repo.delete_session(db, session_id)
    return {"id": response_id, "object": "response.deleted", "deleted": True}


# ---------------------------------------------------------------------------
# Helpers — agent resolution
# ---------------------------------------------------------------------------


def _agent_hash(model: str, tools: list, instructions: str | None) -> str:
    """Deterministic hash of agent config so we reuse existing agents."""
    key = json.dumps(
        {"model": model, "tools": tools, "instructions": instructions},
        sort_keys=True,
    )
    return hashlib.sha256(key.encode()).hexdigest()[:16]


async def _resolve_agent(db: AsyncSession, body: OpenAIResponseRequest) -> str:
    """Find or create an agent matching the request's model + tools."""
    h = _agent_hash(body.model, body.tools, body.instructions)

    if h in _agent_cache:
        agent = await repo.get_agent(db, _agent_cache[h])
        if agent:
            return agent.id

    # Translate OpenAI tools to Anthropic format
    anthropic_tools: list[dict[str, Any]] = [{"type": "agent_toolset_20260401"}]
    for tool in body.tools:
        tool_type = tool.get("type", "")
        if tool_type == "function":
            fn = tool.get("function", tool)
            anthropic_tools.append(
                {
                    "type": "custom",
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters", {}),
                }
            )

    agent = await repo.create_agent(
        db,
        name=f"openai-compat-{h}",
        model=body.model,
        system=body.instructions,
        tools=anthropic_tools,
    )
    _agent_cache[h] = agent.id
    return agent.id


# ---------------------------------------------------------------------------
# Helpers — session resolution
# ---------------------------------------------------------------------------


async def _resolve_session(
    db: AsyncSession,
    agent_id: str,
    previous_response_id: str | None,
) -> str:
    """Find existing session (from previous_response_id) or create new."""
    if previous_response_id:
        session_id = _response_to_session.get(previous_response_id)
        if session_id:
            session = await repo.get_session(db, session_id)
            if session:
                return session_id

    # Create new session
    agent = await repo.get_agent(db, agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    session = await repo.create_session(db, agent=agent)
    return session.id


# ---------------------------------------------------------------------------
# Helpers — event translation
# ---------------------------------------------------------------------------


def _build_response(
    body: OpenAIResponseRequest,
    events: list[dict[str, Any]],
    session_id: str,
) -> OpenAIResponse:
    """Translate Anthropic SSE events to OpenAI Response format."""
    output: list[dict[str, Any]] = []
    total_in = 0
    total_out = 0
    status = "completed"

    for evt in events:
        evt_type = evt.get("type", "")

        if evt_type == "agent.message":
            content_blocks = evt.get("content", [])
            text = " ".join(
                b.get("text", "") for b in content_blocks if b.get("type") == "text"
            )
            if text:
                output.append(
                    {
                        "type": "message",
                        "id": evt.get("id", ""),
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": text}],
                    }
                )

        elif evt_type == "agent.tool_use":
            output.append(
                {
                    "type": "function_call",
                    "id": evt.get("id", ""),
                    "name": evt.get("name", ""),
                    "arguments": json.dumps(evt.get("input", {})),
                    "call_id": evt.get("id", ""),
                }
            )

        elif evt_type == "agent.tool_result":
            content_blocks = evt.get("content", [])
            result_text = " ".join(
                b.get("text", "")
                for b in (content_blocks or [])
                if b.get("type") == "text"
            )
            output.append(
                {
                    "type": "function_call_output",
                    "call_id": evt.get("tool_use_id", ""),
                    "output": result_text,
                }
            )

        elif evt_type == "span.model_request_end":
            usage = evt.get("model_usage", {})
            total_in += usage.get("input_tokens", 0)
            total_out += usage.get("output_tokens", 0)

        elif evt_type == "session.error":
            status = "failed"

        elif evt_type == "session.status_idle":
            sr = evt.get("stop_reason", {})
            if sr.get("type") == "requires_action":
                status = "incomplete"

    resp_id = f"resp_{session_id[-16:]}"

    return OpenAIResponse(
        id=resp_id,
        created_at=int(time.time()),
        status=status,
        model=body.model,
        output=output,
        usage=OpenAIUsage(
            input_tokens=total_in,
            output_tokens=total_out,
            total_tokens=total_in + total_out,
        ),
        metadata=body.metadata,
        previous_response_id=body.previous_response_id,
    )


def _build_response_from_stored_events(
    response_id: str,
    events: list[dict[str, Any]],
    session_id: str,
) -> OpenAIResponse:
    """Build an OpenAI Response from stored event dicts (for GET)."""

    class FakeBody:
        model = ""
        metadata: dict[str, str] = {}
        previous_response_id = None

    resp = _build_response(FakeBody(), events, session_id)  # type: ignore[arg-type]
    resp.id = response_id
    return resp
