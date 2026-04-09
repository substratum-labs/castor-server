"""Phase 2 — Castor extension endpoints (budget, scan, fork).

These endpoints expose Castor kernel capabilities that go beyond
the Anthropic Managed Agents API: speculative review, time-travel
fork, and real-time budget tracking.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from castor_server.core.agent_fn import build_agent_fn
from castor_server.core.event_bus import EventBus
from castor_server.core.kernel_adapter import build_kernel_for_agent
from castor_server.core.session_manager import session_manager
from castor_server.models.sessions import (
    BudgetItem,
    BudgetResponse,
    FlaggedStep,
    ForkRequest,
    ScanResponse,
    SessionResponse,
)
from castor_server.store import repository as repo
from castor_server.store.database import get_session

router = APIRouter(prefix="/v1/sessions/{session_id}", tags=["extensions"])


@router.get("/budget", response_model=BudgetResponse)
async def get_budget(
    session_id: str,
    db: AsyncSession = Depends(get_session),
) -> BudgetResponse:
    """Return real-time budget usage from kernel checkpoint capabilities."""
    session = await repo.get_session(db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    kernel_cp, _messages = await session_manager._load_checkpoint(db, session_id)
    if kernel_cp is None:
        return BudgetResponse(budgets=[])

    items = []
    for resource, cap in kernel_cp.capabilities.items():
        items.append(
            BudgetItem(
                resource=resource,
                max_budget=cap.max_budget,
                current_usage=cap.current_usage,
                remaining=cap.max_budget - cap.current_usage,
            )
        )

    return BudgetResponse(budgets=items)


@router.post("/scan", response_model=ScanResponse)
async def scan_session(
    session_id: str,
    db: AsyncSession = Depends(get_session),
) -> ScanResponse:
    """Run speculative execution and return flagged steps for review."""
    session = await repo.get_session(db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    kernel_cp, messages = await session_manager._load_checkpoint(db, session_id)
    if kernel_cp is None:
        return ScanResponse(
            total_steps=0,
            auto_verified=0,
            flagged_count=0,
            flagged=[],
            tools_used={},
        )

    kernel = build_kernel_for_agent(session.agent)

    # Create a disposable bus (scan events are not streamed to clients)
    scan_bus = EventBus(f"scan-{session_id}")

    agent_fn = build_agent_fn(
        agent=session.agent,
        messages=list(messages),
        bus=scan_bus,
        db=db,
        session_id=session_id,
    )

    # Run speculatively — HITL tools execute without suspension
    spec_cp = await kernel.run(agent_fn, checkpoint=kernel_cp, speculative=True)

    # Scan the journal for flagged steps
    summary = kernel.scan(spec_cp)

    return ScanResponse(
        total_steps=summary.total_steps,
        auto_verified=summary.auto_verified,
        flagged_count=summary.flagged_count,
        flagged=[
            FlaggedStep(
                index=f.index,
                tool_name=f.tool_name,
                arguments=f.arguments,
                response=f.response,
                reason=f.reason,
            )
            for f in summary.flagged
        ],
        tools_used=summary.tools_used,
    )


@router.post("/fork", response_model=SessionResponse)
async def fork_session(
    session_id: str,
    body: ForkRequest,
    db: AsyncSession = Depends(get_session),
) -> SessionResponse:
    """Fork a session at a specific step, creating a new session."""
    session = await repo.get_session(db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    kernel_cp, messages = await session_manager._load_checkpoint(db, session_id)
    if kernel_cp is None:
        raise HTTPException(status_code=400, detail="Session has no checkpoint to fork")

    # Fork the kernel checkpoint
    try:
        forked_cp = kernel_cp.fork(at_step=body.at_step)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    # Create new session
    new_session = await repo.create_session(
        db,
        agent=session.agent,
        title=f"Fork of {session_id} at step {body.at_step}",
    )

    # Save forked checkpoint to new session
    # Truncate messages proportionally (rough heuristic)
    if body.at_step < len(messages):
        forked_messages = messages[: body.at_step]
    else:
        forked_messages = list(messages)
    checkpoint_data = {
        "version": 2,
        "kernel_checkpoint": forked_cp.model_dump(),
        "messages": forked_messages,
    }
    await repo.update_session_checkpoint(db, new_session.id, checkpoint_data)

    return new_session
