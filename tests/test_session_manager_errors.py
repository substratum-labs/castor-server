"""Regression tests for session_manager error paths.

These exist primarily to prevent regressions of bugs found during the
wire-format validation pass:

- Duplicate ``session.status_idle`` events when the kernel raises
- Null sub-fields (``retry_status``, ``mcp_server_name``) leaking into
  ``session.error`` payloads
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from castor_server.core.session_manager import SessionManager
from castor_server.store.repository import (
    create_agent,
    create_session,
    list_events,
)


async def _setup(db: AsyncSession):
    agent = await create_agent(
        db,
        name="error-test",
        model="claude-sonnet-4-6",
        tools=[],
    )
    session = await create_session(db, agent=agent)
    return agent, session


@pytest.mark.asyncio
async def test_kernel_failure_emits_single_idle(db_session: AsyncSession):
    """When the kernel raises, exactly ONE session.status_idle is emitted.

    Regression: previously _run_kernel emitted idle in its except block,
    and then _handle_kernel_result emitted another idle when it saw the
    FAILED status. The bus + DB ended up with two idle events.
    """
    _, session = await _setup(db_session)
    sm = SessionManager()
    bus = sm.get_bus(session.id)

    received: list[dict] = []
    queue = bus.subscribe()

    async def drain() -> None:
        # Drain anything pending after handle_user_message returns.
        while not queue.empty():
            received.append(queue.get_nowait())

    # Force the kernel run to raise — patch the symbol the way it's
    # imported into session_manager (kernel.run on the built kernel).
    with patch(
        "castor_server.core.session_manager.build_kernel_for_agent"
    ) as build_kernel_mock:
        kernel_mock = AsyncMock()
        kernel_mock.run = AsyncMock(side_effect=RuntimeError("LLM 401"))
        build_kernel_mock.return_value = kernel_mock

        await sm.handle_user_message(
            db_session,
            session.id,
            [{"type": "text", "text": "hi"}],
        )

    await drain()

    idle_events = [e for e in received if e.get("type") == "session.status_idle"]
    error_events = [e for e in received if e.get("type") == "session.error"]

    assert len(idle_events) == 1, (
        f"expected exactly 1 session.status_idle, got {len(idle_events)}: "
        f"{[e.get('id') for e in idle_events]}"
    )
    assert len(error_events) == 1, (
        f"expected exactly 1 session.error, got {len(error_events)}"
    )

    # Also verify the DB stored exactly one idle event
    stored = await list_events(db_session, session.id, limit=100)
    stored_idle = [e for e in stored if e.get("type") == "session.status_idle"]
    assert len(stored_idle) == 1, f"DB has {len(stored_idle)} idle events, expected 1"


@pytest.mark.asyncio
async def test_session_error_omits_null_subfields(db_session: AsyncSession):
    """session.error payload omits retry_status and mcp_server_name when None.

    Regression: SessionErrorDetail was emitting ``"retry_status": null`` which
    doesn't match the Anthropic wire format (the field should be absent).
    """
    _, session = await _setup(db_session)
    sm = SessionManager()
    bus = sm.get_bus(session.id)

    received: list[dict] = []
    queue = bus.subscribe()

    with patch(
        "castor_server.core.session_manager.build_kernel_for_agent"
    ) as build_kernel_mock:
        kernel_mock = AsyncMock()
        kernel_mock.run = AsyncMock(side_effect=RuntimeError("boom"))
        build_kernel_mock.return_value = kernel_mock

        await sm.handle_user_message(
            db_session,
            session.id,
            [{"type": "text", "text": "hi"}],
        )

    while not queue.empty():
        received.append(queue.get_nowait())

    error_events = [e for e in received if e.get("type") == "session.error"]
    assert len(error_events) == 1
    err = error_events[0]
    detail = err["error"]

    assert "retry_status" not in detail, (
        f"retry_status should be omitted when None, got: {detail}"
    )
    assert "mcp_server_name" not in detail, (
        f"mcp_server_name should be omitted when None, got: {detail}"
    )
    assert detail["type"] == "unknown_error"
    assert "boom" in detail["message"]


@pytest.mark.asyncio
async def test_agent_toolset_omits_null_config_fields(db_session: AsyncSession):
    """AgentToolset serialization omits default_config/configs when None.

    Regression: tools array in agent responses included
    ``{"default_config": null, "configs": null}`` which leaks into the wire.
    """
    from castor_server.models.agents import AgentToolset

    toolset = AgentToolset()
    dumped = toolset.model_dump()

    assert "default_config" not in dumped, (
        f"default_config should be omitted when None, got: {dumped}"
    )
    assert "configs" not in dumped, (
        f"configs should be omitted when None, got: {dumped}"
    )
    assert dumped == {"type": "agent_toolset_20260401"}
