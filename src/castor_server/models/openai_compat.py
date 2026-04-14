"""OpenAI Responses API compatible models.

Translates between OpenAI's /v1/responses wire format and
castor-server's internal Anthropic-compatible representation.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from castor_server.models.common import gen_id

# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class OpenAIToolFunction(BaseModel):
    type: Literal["function"] = "function"
    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)


class OpenAIResponseRequest(BaseModel):
    model: str
    input: str | list[dict[str, Any]]
    instructions: str | None = None
    tools: list[dict[str, Any]] = Field(default_factory=list)
    tool_choice: str | None = None
    previous_response_id: str | None = None
    stream: bool = False
    max_output_tokens: int | None = None
    temperature: float | None = None
    store: bool = False
    metadata: dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------


class OpenAIOutputMessage(BaseModel):
    type: Literal["message"] = "message"
    id: str = Field(default_factory=lambda: gen_id("msg"))
    role: str = "assistant"
    content: list[dict[str, Any]] = Field(default_factory=list)


class OpenAIFunctionCall(BaseModel):
    type: Literal["function_call"] = "function_call"
    id: str = Field(default_factory=lambda: gen_id("fc"))
    name: str
    arguments: str
    call_id: str = Field(default_factory=lambda: gen_id("call"))


class OpenAIFunctionCallOutput(BaseModel):
    type: Literal["function_call_output"] = "function_call_output"
    call_id: str
    output: str


class OpenAIUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class OpenAIResponse(BaseModel):
    id: str = Field(default_factory=lambda: gen_id("resp"))
    object: Literal["response"] = "response"
    created_at: int = 0
    status: Literal["completed", "failed", "in_progress", "incomplete"] = "completed"
    model: str = ""
    output: list[dict[str, Any]] = Field(default_factory=list)
    usage: OpenAIUsage = Field(default_factory=OpenAIUsage)
    metadata: dict[str, str] = Field(default_factory=dict)
    previous_response_id: str | None = None
