"""CLI entry point for castor-server."""

from __future__ import annotations

import os

import click
import uvicorn

from castor_server.config import settings

# Env vars LiteLLM looks at for the providers most users will hit. Order
# matters: the first one set is what we tell the user to use as default.
_LLM_KEY_PROVIDERS: list[tuple[str, str]] = [
    ("ANTHROPIC_API_KEY", "Anthropic (claude-* models, default model_map)"),
    ("OPENROUTER_API_KEY", "OpenRouter"),
    ("OPENAI_API_KEY", "OpenAI"),
]


def _print_llm_key_status() -> None:
    """Print which LLM provider keys are detected at startup.

    Helps users diagnose 'why does my agent return session.error' before
    they even send a message.
    """
    detected = [
        (name, label) for name, label in _LLM_KEY_PROVIDERS if os.environ.get(name)
    ]

    if detected:
        click.secho("✓ LLM provider keys detected:", fg="green")
        for name, label in detected:
            click.echo(f"    • {name}  → {label}")
    else:
        click.secho("⚠  No LLM provider key detected.", fg="yellow")
        click.echo(
            "   Set one of the following before running an agent with a real model:"
        )
        for name, label in _LLM_KEY_PROVIDERS:
            click.echo(f"     export {name}=...   # {label}")
        click.echo()
        click.secho(
            '   Or use model="mock" in your agent config for an offline demo '
            "(no API key required).",
            fg="cyan",
        )


@click.group()
def main():
    """castor-server: Self-hosted Anthropic Managed Agents API."""
    pass


@main.command()
@click.option("--host", default=None, help="Bind host")
@click.option("--port", default=None, type=int, help="Bind port")
@click.option("--reload", is_flag=True, help="Enable auto-reload")
@click.option("--debug", is_flag=True, help="Enable debug mode")
def run(host: str | None, port: int | None, reload: bool, debug: bool):
    """Start the castor-server."""
    if debug:
        settings.debug = True

    _print_llm_key_status()

    uvicorn.run(
        "castor_server.app:create_app",
        factory=True,
        host=host or settings.host,
        port=port or settings.port,
        reload=reload,
        log_level="debug" if settings.debug else "info",
    )


if __name__ == "__main__":
    main()
