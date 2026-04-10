"""Server configuration."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_prefix": "CASTOR_"}

    host: str = "0.0.0.0"
    port: int = 8080
    database_url: str = Field(
        default="sqlite+aiosqlite:///castor_server.db",
        description="SQLAlchemy async database URL",
    )
    default_model: str = "claude-sonnet-4-6"
    litellm_model_map: dict[str, str] = Field(
        default_factory=lambda: {
            # Default to native Anthropic provider — most users targeting
            # the Anthropic SDK already have ANTHROPIC_API_KEY set, so this
            # gives them a zero-config working path. Override via env if you
            # want to route through OpenRouter or another provider.
            "claude-sonnet-4-6": "anthropic/claude-sonnet-4-5",
            "claude-opus-4-6": "anthropic/claude-opus-4-5",
            "claude-haiku-4-5": "anthropic/claude-haiku-4-5-20251001",
            # "mock" — built-in echo model for offline / first-run / CI demos.
            # Handled specially in llm_adapter.litellm_chat (no LiteLLM call).
            "mock": "mock",
        },
        description="Map from Anthropic model IDs to LiteLLM model strings",
    )
    debug: bool = False

    # Phase 2 — Castor extensions
    default_llm_cost: float = Field(
        default=0.03,
        description="Default cost per LLM call for budget tracking",
    )
    default_budgets: dict[str, float] = Field(
        default_factory=lambda: {"api_usd": 10.0},
        description="Default budgets for new sessions",
    )
    enable_budgets: bool = Field(
        default=False,
        description="Enable budget enforcement for sessions",
    )

    # Roche sandbox settings
    roche_provider: str = Field(
        default="docker",
        description="Roche sandbox provider (docker, wasm, firecracker)",
    )
    roche_daemon_port: int | None = Field(
        default=None,
        description="Roche daemon gRPC port (auto-detect if None)",
    )

    # Authentication
    api_key: str | None = Field(
        default=None,
        description=(
            "Global API key for authentication. If set, all requests must "
            "include 'Authorization: Bearer <key>'. If unset, no auth (dev mode)."
        ),
    )


settings = Settings()
