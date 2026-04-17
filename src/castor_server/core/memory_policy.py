"""Production-grade memory policy for castor-server sessions.

Implements ``castor.MemoryPolicyProtocol`` with:
- FIFO + age-weighted eviction (oldest first, never evict last 2 user turns)
- LLM-based summarization of evicted messages (≤ 200 tokens)
- Heuristic recall trigger (past-reference cues or short context)

This is a server-level "page replacement algorithm" — the kernel's
``DefaultMemoryPolicy`` (trivial no-op) remains as test/fallback.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("castor_server.memory_policy")

# Cues that suggest the user is referencing past context
_RECALL_CUES = [
    "earlier",
    "before",
    "previous",
    "last time",
    "you said",
    "we discussed",
    "remember",
    "recall",
    "as mentioned",
]

# Minimum context length below which we always try recall
# (short context = likely a new session that could benefit from past memory)
_SHORT_CONTEXT_THRESHOLD = 5


class DefaultMemoryPolicy:
    """Server-grade memory policy implementing MemoryPolicyProtocol."""

    def __init__(
        self,
        *,
        summarizer_model: str = "claude-haiku-4-5",
        recall_score_threshold: float = 0.7,
        anchor_user_turns: int = 2,
    ) -> None:
        self._summarizer_model = summarizer_model
        self._recall_threshold = recall_score_threshold
        self._anchor_turns = anchor_user_turns

    async def should_evict(
        self,
        context_history: list[Any],
        token_budget: int,
    ) -> list[int] | None:
        """FIFO eviction: drop oldest messages until below token_budget.

        Never evict the last ``anchor_user_turns`` user messages to
        preserve conversational anchoring.
        """
        if not context_history:
            return None

        # Estimate tokens (rough: 4 chars ≈ 1 token)
        total = sum(_estimate_tokens(m) for m in context_history)
        if total <= token_budget:
            return None

        # Find protected indices (last N user turns)
        protected: set[int] = set()
        user_count = 0
        for i in range(len(context_history) - 1, -1, -1):
            msg = context_history[i]
            role = msg.role if hasattr(msg, "role") else msg.get("role", "")
            if role == "user":
                user_count += 1
                if user_count <= self._anchor_turns:
                    protected.add(i)

        # Evict from oldest, skip protected and pinned
        evict_indices: list[int] = []
        running_total = total
        for i, msg in enumerate(context_history):
            if running_total <= token_budget:
                break
            if i in protected:
                continue
            pinned = msg.pinned if hasattr(msg, "pinned") else msg.get("pinned", False)
            if pinned:
                continue
            evict_indices.append(i)
            running_total -= _estimate_tokens(msg)

        return evict_indices if evict_indices else None

    async def generate_summary(
        self,
        evicted_messages: list[Any],
    ) -> str | None:
        """Summarize evicted messages via LLM syscall.

        Returns a ≤200 token summary capturing goals and actions.
        Returns None on failure (eviction still proceeds).
        """
        if not evicted_messages:
            return None

        # Build a condensed representation of evicted messages
        parts: list[str] = []
        for msg in evicted_messages:
            role = msg.role if hasattr(msg, "role") else msg.get("role", "")
            content = msg.content if hasattr(msg, "content") else msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    b.get("text", "") for b in content if isinstance(b, dict)
                )
            text = str(content)[:500]
            parts.append(f"{role}: {text}")

        evicted_text = "\n".join(parts)

        # Use the LLM adapter directly (not through kernel syscall —
        # summarization is policy-level, not agent-level). The kernel
        # MMU will wrap this in a MEMORY_MANAGEMENT purpose syscall.
        try:
            from castor_server.core.llm_adapter import litellm_chat

            resp = await litellm_chat(
                model=self._summarizer_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Summarize the following conversation excerpt in "
                            "under 200 tokens. Focus on goals, decisions, and "
                            "actions taken. Omit greetings and filler."
                        ),
                    },
                    {"role": "user", "content": evicted_text},
                ],
                max_tokens=200,
            )
            text_blocks = resp.get("content", [])
            summary = " ".join(
                b.get("text", "") for b in text_blocks if b.get("type") == "text"
            )
            return summary.strip() if summary.strip() else None
        except Exception:
            logger.warning("summary_generation_failed", exc_info=True)
            return None

    async def should_recall(
        self,
        context_history: list[Any],
        current_query: str,
    ) -> str | None:
        """Heuristic recall trigger.

        Returns a recall query if:
        - The query contains past-reference cues, OR
        - The context is very short (suggesting a fresh session)
        """
        query_lower = current_query.lower()

        # Check for past-reference cues
        for cue in _RECALL_CUES:
            if cue in query_lower:
                return current_query

        # Short context → fresh session, try recall
        if len(context_history) < _SHORT_CONTEXT_THRESHOLD:
            return current_query

        return None

    async def on_session_end(
        self,
        context_history: list[Any],
        syscall_log: list[Any],
    ) -> None:
        """Server default: log metrics, no consolidation.

        Consolidation is Tiphys' concern — server should not impose.
        """
        logger.info(
            "session_end context_len=%d syscall_count=%d",
            len(context_history),
            len(syscall_log),
        )


def _estimate_tokens(msg: Any) -> int:
    """Rough token estimate: ~4 chars per token."""
    content = msg.content if hasattr(msg, "content") else msg.get("content", "")
    if isinstance(content, list):
        content = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
    return max(1, len(str(content)) // 4)
