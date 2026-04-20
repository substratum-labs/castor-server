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


def _check_port(host: str, port: int) -> None:
    """Verify the port is free before starting uvicorn.

    If the port is already in use, print a helpful message with the PID
    of the process holding it and exit. Without this check, uvicorn
    silently fails to bind and requests hit whatever else is on that port.
    """
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((host if host != "0.0.0.0" else "127.0.0.1", port))
    except OSError:
        # Try to find which process holds the port
        pid_info = ""
        try:
            import subprocess

            result = subprocess.run(
                ["lsof", "-ti", f":{port}"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            pids = result.stdout.strip()
            if pids:
                pid_info = f" (held by PID {pids.splitlines()[0]})"
        except Exception:
            pass

        click.secho(
            f"✗ Port {port} is already in use{pid_info}.",
            fg="red",
            err=True,
        )
        click.echo(
            "  Use --port or CASTOR_PORT to pick a different port:",
            err=True,
        )
        click.echo(
            f"    castor-server run --port {port + 1}",
            err=True,
        )
        raise SystemExit(1)
    finally:
        sock.close()


@click.group()
def main():
    """castor-server: Self-hosted Agent OS."""
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

    bind_host = host or settings.host
    bind_port = port or settings.port

    # Check port availability before starting uvicorn (Bug 1 fix)
    _check_port(bind_host, bind_port)

    _print_llm_key_status()

    uvicorn.run(
        "castor_server.app:create_app",
        factory=True,
        host=bind_host,
        port=bind_port,
        reload=reload,
        log_level="debug" if settings.debug else "info",
    )


@main.command()
@click.option(
    "--write",
    is_flag=True,
    help="Write docker-compose.yml to the current directory instead of stdout.",
)
def compose(write: bool):
    """Print a docker-compose.yml for self-hosting castor-server.

    Use this to bootstrap a deployment on your own machine or VPC. The
    generated compose file expects ``../roche`` to be checked out as a
    sibling of the castor-server directory because roche-sandbox is a
    local editable dependency.

    Quickstart:

        \b
        # In a parent directory containing both repos:
        git clone https://github.com/substratum-labs/castor-server
        git clone https://github.com/substratum-labs/roche
        cd castor-server
        export ANTHROPIC_API_KEY=sk-ant-...
        docker compose up -d
    """
    from pathlib import Path

    template = (Path(__file__).parent.parent.parent / "docker-compose.yml").read_text()

    if write:
        target = Path.cwd() / "docker-compose.yml"
        if target.exists():
            click.secho(f"Refusing to overwrite existing {target}", fg="red", err=True)
            raise SystemExit(1)
        target.write_text(template)
        click.secho(f"✓ Wrote {target}", fg="green")
        click.echo()
        click.echo("Next steps:")
        click.echo("  1. Make sure ../roche is checked out as a sibling")
        click.echo("  2. export ANTHROPIC_API_KEY=...   (or another LLM provider key)")
        click.echo("  3. docker compose up -d")
        return

    click.echo(template)


@main.command()
def deploy():
    """Print a deployment cheatsheet for castor-server.

    Walks the user through the simplest path to a running instance:
    local docker compose for VPC deployments, or systemd for bare metal.
    """
    click.secho("castor-server deployment cheatsheet", fg="cyan", bold=True)
    click.echo()
    click.secho("Option A — Docker Compose (recommended for VPC):", fg="green")
    click.echo("  1. git clone https://github.com/substratum-labs/castor-server")
    click.echo("  2. git clone https://github.com/substratum-labs/roche")
    click.echo("  3. cd castor-server")
    click.echo("  4. export ANTHROPIC_API_KEY=sk-ant-...")
    click.echo("  5. docker compose up -d")
    click.echo("  6. curl http://localhost:8080/health")
    click.echo()
    click.secho("Option B — Local source install:", fg="green")
    click.echo("  uv sync && castor-server run --port 8080")
    click.echo()
    click.secho("Option C — Generate compose file only:", fg="green")
    click.echo("  castor-server compose --write")
    click.echo()
    click.secho("Once running, point any anthropic-python client at it:", fg="cyan")
    click.echo(
        '  client = Anthropic(base_url="http://your-host:8080", api_key="local")'
    )


if __name__ == "__main__":
    main()
