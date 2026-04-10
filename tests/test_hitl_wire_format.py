"""Regression tests for the HITL wire-format fixes.

These bugs were found by running ``scripts/tool_use_check.py`` against a
real LLM and validating the SDK could correlate
``session.status_idle.event_ids`` with ``agent.tool_use`` events.

Bug 1: ``agent.tool_use.evaluated_permission`` was always ``"allow"``
       even for HITL tools. SDK clients couldn't tell which calls needed
       a confirmation.

Bug 2: ``session.status_idle.stop_reason.event_ids`` contained a
       synthetic ``"hitl_<tool_name>"`` instead of the real
       ``agent.tool_use.id``. SDK clients couldn't send a matching
       ``user.tool_confirmation``.

Bug 3: ``handle_tool_confirmation`` ignored ``tool_use_id`` entirely.
       Mismatched ids would still approve whatever was pending.
"""

from __future__ import annotations

import pytest

from castor_server.core.session_manager import SessionManager

# ---------------------------------------------------------------------------
# Bug 1: agent.tool_use.evaluated_permission
# ---------------------------------------------------------------------------


def test_agent_fn_marks_destructive_tools_as_ask():
    """``bash`` is in DESTRUCTIVE_TOOL_NAMES → evaluated_permission='ask'."""
    from castor_server.core.kernel_adapter import resolve_hitl_tools
    from castor_server.models.agents import AgentResponse, ModelConfig

    agent = AgentResponse(
        id="agent_test",
        name="t",
        model=ModelConfig(id="mock"),
        tools=[{"type": "agent_toolset_20260401"}],
        created_at="2026-04-10T00:00:00.000Z",
        updated_at="2026-04-10T00:00:00.000Z",
    )
    hitl = set(resolve_hitl_tools(agent))
    # Defaults — destructive tools are HITL
    assert "bash" in hitl
    assert "write" in hitl
    assert "edit" in hitl
    # Non-destructive tools are not HITL by default
    assert "read" not in hitl
    assert "glob" not in hitl
    assert "grep" not in hitl


def test_agent_fn_marks_always_ask_tools_as_ask():
    """always_ask permission_policy → HITL even for non-destructive tools."""
    from castor_server.core.kernel_adapter import resolve_hitl_tools
    from castor_server.models.agents import AgentResponse, ModelConfig

    agent = AgentResponse(
        id="agent_test",
        name="t",
        model=ModelConfig(id="mock"),
        tools=[
            {
                "type": "agent_toolset_20260401",
                "configs": [
                    {
                        "name": "read",
                        "permission_policy": {"type": "always_ask"},
                    }
                ],
            }
        ],
        created_at="2026-04-10T00:00:00.000Z",
        updated_at="2026-04-10T00:00:00.000Z",
    )
    hitl = set(resolve_hitl_tools(agent))
    assert "read" in hitl  # non-destructive but always_ask


# ---------------------------------------------------------------------------
# Bug 2: _find_pending_tool_use_id finds the real id
# ---------------------------------------------------------------------------


def test_find_pending_tool_use_id_returns_real_id():
    """The helper finds the LLM's tool_call.id by tool name in the last
    assistant message."""
    messages = [
        {"role": "user", "content": "run echo"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "toolu_bdrk_01ABC",
                    "type": "function",
                    "function": {"name": "bash", "arguments": '{"command":"echo hi"}'},
                }
            ],
        },
    ]
    found = SessionManager._find_pending_tool_use_id(messages, "bash")
    assert found == "toolu_bdrk_01ABC"


def test_find_pending_tool_use_id_handles_multiple_calls():
    """Multiple tool_calls in the same assistant message — pick by name."""
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "toolu_1",
                    "type": "function",
                    "function": {"name": "read", "arguments": "{}"},
                },
                {
                    "id": "toolu_2",
                    "type": "function",
                    "function": {"name": "bash", "arguments": "{}"},
                },
            ],
        },
    ]
    assert SessionManager._find_pending_tool_use_id(messages, "bash") == "toolu_2"
    assert SessionManager._find_pending_tool_use_id(messages, "read") == "toolu_1"


def test_find_pending_tool_use_id_returns_none_when_no_match():
    """No matching tool_call → None."""
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "toolu_x",
                    "type": "function",
                    "function": {"name": "read", "arguments": "{}"},
                }
            ],
        }
    ]
    assert SessionManager._find_pending_tool_use_id(messages, "bash") is None


def test_find_pending_tool_use_id_handles_empty_messages():
    """Empty / None messages → None, no exception."""
    assert SessionManager._find_pending_tool_use_id(None, "bash") is None
    assert SessionManager._find_pending_tool_use_id([], "bash") is None


def test_find_pending_tool_use_id_skips_old_assistant_messages():
    """Only the most recent assistant message is checked.

    If an older message had a tool_call for ``bash`` and the latest one
    doesn't, we should NOT return the old id — that tool call is already
    resolved.
    """
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "toolu_old",
                    "type": "function",
                    "function": {"name": "bash", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "toolu_old", "content": "done"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "toolu_new",
                    "type": "function",
                    "function": {"name": "read", "arguments": "{}"},
                }
            ],
        },
    ]
    # Looking for bash in latest assistant message → not present → None
    assert SessionManager._find_pending_tool_use_id(messages, "bash") is None
    # read IS in latest assistant message → toolu_new
    assert SessionManager._find_pending_tool_use_id(messages, "read") == "toolu_new"


# ---------------------------------------------------------------------------
# Bug 3: handle_tool_confirmation honors tool_use_id
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_tool_confirmation_rejects_mismatched_id(db_session):
    """Sending a confirmation with a tool_use_id that doesn't match the
    pending HITL should be a no-op (logged warning)."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from castor.models.checkpoint import AgentCheckpoint

    from castor_server.store.repository import create_agent, create_session

    agent = await create_agent(db_session, name="hitl-test", model="mock", tools=[])
    session = await create_session(db_session, agent=agent)

    # Build a fake checkpoint that's suspended for HITL on bash
    fake_cp = AgentCheckpoint(
        pid=session.id,
        status="SUSPENDED_FOR_HITL",
        agent_function_name="anthropic_agent_loop",
        capabilities={},
        pending_hitl={
            "tool_name": "bash",
            "arguments": {"command": "rm -rf /"},
        },
    )
    fake_snapshot = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "toolu_REAL",
                    "type": "function",
                    "function": {"name": "bash", "arguments": "{}"},
                }
            ],
        }
    ]

    sm = SessionManager()
    # Populate the in-memory snapshot — normally written by _run_kernel
    # via the agent_fn side channel.
    sm._latest_conversation_by_session[session.id] = fake_snapshot

    # Patch _load_checkpoint to return our fake state
    sm._load_checkpoint = AsyncMock(return_value=(fake_cp, []))
    sm._save_checkpoint = AsyncMock()
    sm._emit_running = AsyncMock()
    sm._run_kernel = AsyncMock(return_value=fake_cp)

    with patch(
        "castor_server.core.session_manager.build_kernel_for_agent"
    ) as build_kernel_mock:
        kernel_mock = MagicMock()
        kernel_mock.approve = AsyncMock()
        build_kernel_mock.return_value = kernel_mock

        # Send a confirmation with the WRONG tool_use_id
        await sm.handle_tool_confirmation(
            db_session,
            session.id,
            tool_use_id="toolu_WRONG_ID",
            result="allow",
        )

        # Kernel approve should NOT have been called
        kernel_mock.approve.assert_not_called()


@pytest.mark.asyncio
async def test_handle_tool_confirmation_accepts_real_id(db_session):
    """Confirmation with the real LLM tool_use id should approve."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from castor.models.checkpoint import AgentCheckpoint

    from castor_server.store.repository import create_agent, create_session

    agent = await create_agent(db_session, name="hitl-test", model="mock", tools=[])
    session = await create_session(db_session, agent=agent)

    fake_cp = AgentCheckpoint(
        pid=session.id,
        status="SUSPENDED_FOR_HITL",
        agent_function_name="anthropic_agent_loop",
        capabilities={},
        pending_hitl={"tool_name": "bash", "arguments": {}},
    )
    fake_snapshot = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "toolu_REAL",
                    "type": "function",
                    "function": {"name": "bash", "arguments": "{}"},
                }
            ],
        }
    ]

    sm = SessionManager()
    sm._latest_conversation_by_session[session.id] = fake_snapshot
    sm._load_checkpoint = AsyncMock(return_value=(fake_cp, []))
    sm._save_checkpoint = AsyncMock()
    sm._emit_running = AsyncMock()
    sm._run_kernel = AsyncMock(return_value=fake_cp)
    sm._handle_kernel_result = AsyncMock()

    with patch(
        "castor_server.core.session_manager.build_kernel_for_agent"
    ) as build_kernel_mock:
        kernel_mock = MagicMock()
        kernel_mock.approve = AsyncMock()
        build_kernel_mock.return_value = kernel_mock

        await sm.handle_tool_confirmation(
            db_session,
            session.id,
            tool_use_id="toolu_REAL",
            result="allow",
        )

        kernel_mock.approve.assert_called_once()


@pytest.mark.asyncio
async def test_handle_tool_confirmation_accepts_synthetic_fallback(db_session):
    """Backwards compat: a synthetic 'hitl_<tool_name>' id should still
    work for clients that read older event_ids."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from castor.models.checkpoint import AgentCheckpoint

    from castor_server.store.repository import create_agent, create_session

    agent = await create_agent(db_session, name="hitl-test", model="mock", tools=[])
    session = await create_session(db_session, agent=agent)

    fake_cp = AgentCheckpoint(
        pid=session.id,
        status="SUSPENDED_FOR_HITL",
        agent_function_name="anthropic_agent_loop",
        capabilities={},
        pending_hitl={"tool_name": "bash", "arguments": {}},
    )

    sm = SessionManager()
    sm._load_checkpoint = AsyncMock(return_value=(fake_cp, []))
    sm._save_checkpoint = AsyncMock()
    sm._emit_running = AsyncMock()
    sm._run_kernel = AsyncMock(return_value=fake_cp)
    sm._handle_kernel_result = AsyncMock()

    with patch(
        "castor_server.core.session_manager.build_kernel_for_agent"
    ) as build_kernel_mock:
        kernel_mock = MagicMock()
        kernel_mock.approve = AsyncMock()
        build_kernel_mock.return_value = kernel_mock

        await sm.handle_tool_confirmation(
            db_session,
            session.id,
            tool_use_id="hitl_bash",  # synthetic fallback
            result="allow",
        )

        kernel_mock.approve.assert_called_once()
