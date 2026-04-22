"""Event models matching Anthropic Managed Agents API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_serializer
from pydantic.functional_serializers import SerializerFunctionWrapHandler

from .common import TextBlock, gen_id, now_rfc3339


class _OmitNoneMixin(BaseModel):
    """Pydantic models that drop None fields when serialized.

    Used for event sub-objects where Anthropic omits optional fields
    rather than emitting them as ``null``.
    """

    @model_serializer(mode="wrap")
    def _strip_none(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        result = handler(self)
        return {k: v for k, v in result.items() if v is not None}


# ---------------------------------------------------------------------------
# User events (inbound)
# ---------------------------------------------------------------------------


class UserMessage(BaseModel):
    type: Literal["user.message"] = "user.message"
    content: list[TextBlock]


class UserInterrupt(BaseModel):
    type: Literal["user.interrupt"] = "user.interrupt"


class UserToolConfirmation(BaseModel):
    type: Literal["user.tool_confirmation"] = "user.tool_confirmation"
    tool_use_id: str
    result: Literal["allow", "deny", "modify"]
    deny_message: str | None = None
    modify_feedback: str | None = None


class UserCustomToolResult(BaseModel):
    type: Literal["user.custom_tool_result"] = "user.custom_tool_result"
    custom_tool_use_id: str
    content: list[TextBlock] | None = None
    is_error: bool = False


UserEvent = UserMessage | UserInterrupt | UserToolConfirmation | UserCustomToolResult


class SendEventsRequest(BaseModel):
    events: list[UserEvent]


class SendEventsResponse(BaseModel):
    data: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Agent events (outbound)
# ---------------------------------------------------------------------------


class AgentMessageEvent(BaseModel):
    id: str = Field(default_factory=lambda: gen_id("evt"))
    type: Literal["agent.message"] = "agent.message"
    content: list[TextBlock]
    processed_at: str = Field(default_factory=now_rfc3339)


class AgentThinkingEvent(BaseModel):
    id: str = Field(default_factory=lambda: gen_id("evt"))
    type: Literal["agent.thinking"] = "agent.thinking"
    processed_at: str = Field(default_factory=now_rfc3339)


class AgentToolUseEvent(BaseModel):
    id: str = Field(default_factory=lambda: gen_id("evt"))
    type: Literal["agent.tool_use"] = "agent.tool_use"
    name: str
    input: dict[str, Any] = Field(default_factory=dict)
    evaluated_permission: Literal["allow", "ask", "deny"] | None = None
    processed_at: str = Field(default_factory=now_rfc3339)


class AgentToolResultEvent(BaseModel):
    id: str = Field(default_factory=lambda: gen_id("evt"))
    type: Literal["agent.tool_result"] = "agent.tool_result"
    tool_use_id: str
    content: list[TextBlock] | None = None
    is_error: bool = False
    processed_at: str = Field(default_factory=now_rfc3339)


class AgentCustomToolUseEvent(BaseModel):
    id: str = Field(default_factory=lambda: gen_id("evt"))
    type: Literal["agent.custom_tool_use"] = "agent.custom_tool_use"
    name: str
    input: dict[str, Any] = Field(default_factory=dict)
    processed_at: str = Field(default_factory=now_rfc3339)


class AgentMCPToolUseEvent(BaseModel):
    id: str = Field(default_factory=lambda: gen_id("evt"))
    type: Literal["agent.mcp_tool_use"] = "agent.mcp_tool_use"
    name: str
    mcp_server_name: str
    input: dict[str, Any] = Field(default_factory=dict)
    evaluated_permission: Literal["allow", "ask", "deny"] | None = None
    processed_at: str = Field(default_factory=now_rfc3339)


class AgentMCPToolResultEvent(_OmitNoneMixin):
    id: str = Field(default_factory=lambda: gen_id("evt"))
    type: Literal["agent.mcp_tool_result"] = "agent.mcp_tool_result"
    mcp_tool_use_id: str
    content: list[TextBlock] | None = None
    is_error: bool = False
    processed_at: str = Field(default_factory=now_rfc3339)


# ---------------------------------------------------------------------------
# Session events
# ---------------------------------------------------------------------------


class SessionStatusRunning(BaseModel):
    id: str = Field(default_factory=lambda: gen_id("evt"))
    type: Literal["session.status_running"] = "session.status_running"
    processed_at: str = Field(default_factory=now_rfc3339)


class StopReasonEndTurn(BaseModel):
    type: Literal["end_turn"] = "end_turn"


class StopReasonRequiresAction(BaseModel):
    type: Literal["requires_action"] = "requires_action"
    event_ids: list[str]


class StopReasonRetriesExhausted(BaseModel):
    type: Literal["retries_exhausted"] = "retries_exhausted"


StopReason = StopReasonEndTurn | StopReasonRequiresAction | StopReasonRetriesExhausted


class SessionStatusIdle(BaseModel):
    id: str = Field(default_factory=lambda: gen_id("evt"))
    type: Literal["session.status_idle"] = "session.status_idle"
    stop_reason: StopReason
    processed_at: str = Field(default_factory=now_rfc3339)


class SessionStatusTerminated(BaseModel):
    id: str = Field(default_factory=lambda: gen_id("evt"))
    type: Literal["session.status_terminated"] = "session.status_terminated"
    processed_at: str = Field(default_factory=now_rfc3339)


class RetryStatus(BaseModel):
    type: Literal["retrying", "exhausted", "terminal"]


class SessionErrorDetail(_OmitNoneMixin):
    type: str
    message: str
    retry_status: RetryStatus | None = None
    mcp_server_name: str | None = None


class SessionError(BaseModel):
    id: str = Field(default_factory=lambda: gen_id("evt"))
    type: Literal["session.error"] = "session.error"
    error: SessionErrorDetail
    processed_at: str = Field(default_factory=now_rfc3339)


# ---------------------------------------------------------------------------
# Span events
# ---------------------------------------------------------------------------


class SpanModelRequestStart(BaseModel):
    id: str = Field(default_factory=lambda: gen_id("evt"))
    type: Literal["span.model_request_start"] = "span.model_request_start"
    processed_at: str = Field(default_factory=now_rfc3339)


class ModelUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    speed: Literal["standard", "fast"] | None = None


class SpanModelRequestEnd(BaseModel):
    id: str = Field(default_factory=lambda: gen_id("evt"))
    type: Literal["span.model_request_end"] = "span.model_request_end"
    model_request_start_id: str
    is_error: bool = False
    model_usage: ModelUsage = Field(default_factory=ModelUsage)
    processed_at: str = Field(default_factory=now_rfc3339)


# ---------------------------------------------------------------------------
# Memory events
# ---------------------------------------------------------------------------


class MemoryEvictEvent(BaseModel):
    id: str = Field(default_factory=lambda: gen_id("evt"))
    type: Literal["memory.evict"] = "memory.evict"
    memory_id: str = ""
    token_count: int = 0
    summary: str | None = None
    source: str = "auto"
    processed_at: str = Field(default_factory=now_rfc3339)


class MemoryPromoteEvent(BaseModel):
    id: str = Field(default_factory=lambda: gen_id("evt"))
    type: Literal["memory.promote"] = "memory.promote"
    memory_id: str = ""
    processed_at: str = Field(default_factory=now_rfc3339)


class MemoryProtectEvent(BaseModel):
    id: str = Field(default_factory=lambda: gen_id("evt"))
    type: Literal["memory.protect"] = "memory.protect"
    memory_id: str = ""
    protect: bool = True
    processed_at: str = Field(default_factory=now_rfc3339)


class MemoryWriteEvent(BaseModel):
    id: str = Field(default_factory=lambda: gen_id("evt"))
    type: Literal["memory.write"] = "memory.write"
    memory_id: str = ""
    content_preview: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    processed_at: str = Field(default_factory=now_rfc3339)


class MemoryReadEvent(BaseModel):
    id: str = Field(default_factory=lambda: gen_id("evt"))
    type: Literal["memory.read"] = "memory.read"
    memory_id: str = ""
    found: bool = True
    processed_at: str = Field(default_factory=now_rfc3339)


class MemorySearchEvent(BaseModel):
    id: str = Field(default_factory=lambda: gen_id("evt"))
    type: Literal["memory.search"] = "memory.search"
    query: str = ""
    result_count: int = 0
    processed_at: str = Field(default_factory=now_rfc3339)


class MemoryDeleteEvent(BaseModel):
    id: str = Field(default_factory=lambda: gen_id("evt"))
    type: Literal["memory.delete"] = "memory.delete"
    memory_id: str = ""
    deleted: bool = True
    processed_at: str = Field(default_factory=now_rfc3339)


class MemoryWatermarkEvent(BaseModel):
    id: str = Field(default_factory=lambda: gen_id("evt"))
    type: Literal["memory.watermark_approached"] = "memory.watermark_approached"
    token_count: int = 0
    watermark: int = 0
    processed_at: str = Field(default_factory=now_rfc3339)


# Union of all outbound event types
ServerEvent = (
    AgentMessageEvent
    | AgentThinkingEvent
    | AgentToolUseEvent
    | AgentToolResultEvent
    | AgentCustomToolUseEvent
    | SessionStatusRunning
    | SessionStatusIdle
    | SessionStatusTerminated
    | SessionError
    | SpanModelRequestStart
    | SpanModelRequestEnd
)


# ---------------------------------------------------------------------------
# Event list response
# ---------------------------------------------------------------------------


class EventListResponse(BaseModel):
    data: list[dict[str, Any]]
    next_page: str | None = None
