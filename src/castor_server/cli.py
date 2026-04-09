"""CLI entry point for castor-server."""

from __future__ import annotations

import click
import uvicorn

from castor_server.config import settings


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
