"""Tests for skill runtime injection into LLM context."""

from __future__ import annotations

from castor_server.core.agent_fn import _build_llm_messages
from castor_server.models.agents import AgentResponse, ModelConfig


def _make_agent(**overrides) -> AgentResponse:
    defaults = {
        "id": "agent_test",
        "name": "Test Agent",
        "model": ModelConfig(id="claude-sonnet-4-6"),
        "tools": [],
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
    }
    defaults.update(overrides)
    return AgentResponse(**defaults)


def test_no_skills_no_system():
    agent = _make_agent()
    msgs = _build_llm_messages(agent, [{"role": "user", "content": "hi"}])
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"


def test_system_only():
    agent = _make_agent(system="You are helpful.")
    msgs = _build_llm_messages(agent, [{"role": "user", "content": "hi"}])
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == "You are helpful."
    assert msgs[1]["role"] == "user"


def test_skills_only():
    agent = _make_agent()
    skills = ["# Search Skill\nUse web_search to find info."]
    msgs = _build_llm_messages(
        agent, [{"role": "user", "content": "hi"}], skill_contents=skills
    )
    assert msgs[0]["role"] == "system"
    assert "Search Skill" in msgs[0]["content"]


def test_system_plus_skills():
    agent = _make_agent(system="You are helpful.")
    skills = [
        "# Skill A\nDo thing A.",
        "# Skill B\nDo thing B.",
    ]
    msgs = _build_llm_messages(
        agent, [{"role": "user", "content": "hi"}], skill_contents=skills
    )
    system = msgs[0]["content"]
    assert system.startswith("You are helpful.")
    assert "Skill A" in system
    assert "Skill B" in system
    # Skills joined with double newline
    assert "\n\n# Skill A" in system
    assert "\n\n# Skill B" in system


def test_empty_skills_list():
    agent = _make_agent(system="Base prompt.")
    msgs = _build_llm_messages(
        agent, [{"role": "user", "content": "hi"}], skill_contents=[]
    )
    assert msgs[0]["content"] == "Base prompt."


def test_conversation_preserved():
    agent = _make_agent(system="sys")
    conv = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
    ]
    skills = ["# MySkill\nInstructions."]
    msgs = _build_llm_messages(agent, conv, skill_contents=skills)
    assert len(msgs) == 4  # system + 3 conversation
    assert msgs[0]["role"] == "system"
    assert "MySkill" in msgs[0]["content"]
    assert msgs[1]["content"] == "q1"
    assert msgs[3]["content"] == "q2"
