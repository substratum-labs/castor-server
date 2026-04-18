"""Agent models matching Anthropic Managed Agents API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_serializer
from pydantic.functional_serializers import SerializerFunctionWrapHandler

from .common import Metadata, ModelConfig

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------


class _OmitNoneMixin(BaseModel):
    """Pydantic models that should serialize with their None fields omitted.

    Used for tool definition sub-objects where Anthropic's wire format omits
    optional fields rather than emitting them as ``null``. Top-level nullable
    fields like ``archived_at`` keep using the default behavior.
    """

    @model_serializer(mode="wrap")
    def _strip_none(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        result = handler(self)
        return {k: v for k, v in result.items() if v is not None}


class PermissionPolicy(BaseModel):
    type: Literal["always_allow", "always_ask"] = "always_allow"


class ToolConfig(_OmitNoneMixin):
    name: str
    enabled: bool = True
    permission_policy: PermissionPolicy | None = None


class DefaultToolConfig(_OmitNoneMixin):
    enabled: bool = True
    permission_policy: PermissionPolicy | None = None


class AgentToolset(_OmitNoneMixin):
    type: Literal["agent_toolset_20260401"] = "agent_toolset_20260401"
    default_config: DefaultToolConfig | None = None
    configs: list[ToolConfig] | None = None


class MCPToolset(_OmitNoneMixin):
    type: Literal["mcp_toolset"] = "mcp_toolset"
    mcp_server_name: str
    default_config: DefaultToolConfig | None = None
    configs: list[ToolConfig] | None = None


class CustomToolInputSchema(BaseModel):
    type: Literal["object"] = "object"
    properties: dict[str, Any] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)


class CustomTool(BaseModel):
    type: Literal["custom"] = "custom"
    name: str = Field(..., min_length=1, max_length=128)
    description: str = Field(..., min_length=1, max_length=1024)
    input_schema: CustomToolInputSchema


ToolDefinition = AgentToolset | MCPToolset | CustomTool


# ---------------------------------------------------------------------------
# MCP servers
# ---------------------------------------------------------------------------


class MCPServer(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    type: Literal["url"] = "url"
    url: str


# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------


class Skill(BaseModel):
    skill_id: str
    type: Literal["anthropic", "custom"] = "anthropic"
    version: str | None = None


# ---------------------------------------------------------------------------
# Create / Update requests
# ---------------------------------------------------------------------------


class CreateAgentRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=256)
    model: str | ModelConfig
    system: str | None = Field(default=None, max_length=100_000)
    description: str | None = Field(default=None, max_length=2048)
    tools: list[ToolDefinition] = Field(default_factory=list, max_length=128)
    mcp_servers: list[MCPServer] = Field(default_factory=list, max_length=20)
    skills: list[Skill] = Field(default_factory=list, max_length=20)
    metadata: Metadata = Field(default_factory=dict)
    agent_fn_factory: str | None = Field(
        default=None,
        description=(
            "Optional importable module path (e.g. 'myapp.agent:run') for "
            "a custom agent_fn callable. When set, this callable is used "
            "instead of the default ReAct loop. Requires trusted deployment."
        ),
    )


class UpdateAgentRequest(BaseModel):
    version: int
    name: str | None = None
    model: str | ModelConfig | None = None
    system: str | None = None
    description: str | None = None
    tools: list[ToolDefinition] | None = None
    mcp_servers: list[MCPServer] | None = None
    skills: list[Skill] | None = None
    metadata: Metadata | None = None
    agent_fn_factory: str | None = None


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------


class AgentResponse(BaseModel):
    id: str
    type: Literal["agent"] = "agent"
    name: str
    description: str | None = None
    model: ModelConfig
    system: str | None = None
    tools: list[ToolDefinition] = Field(default_factory=list)
    mcp_servers: list[MCPServer] = Field(default_factory=list)
    skills: list[Skill] = Field(default_factory=list)
    agent_fn_factory: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
    version: int = 1
    created_at: str
    updated_at: str
    archived_at: str | None = None


class AgentListResponse(BaseModel):
    data: list[AgentResponse]
    next_page: str | None = None


class AgentVersionsResponse(BaseModel):
    data: list[AgentResponse]
    next_page: str | None = None
