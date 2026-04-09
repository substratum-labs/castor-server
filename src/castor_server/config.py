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


settings = Settings()
