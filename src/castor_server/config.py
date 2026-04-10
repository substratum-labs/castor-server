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
            "claude-sonnet-4-6": "anthropic/claude-sonnet-4-6",
            "claude-opus-4-6": "anthropic/claude-opus-4-6",
            "claude-haiku-4-5": "anthropic/claude-haiku-4-5-20251001",
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


settings = Settings()
