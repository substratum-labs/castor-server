"""Tests for the `castor-server compose` and `deploy` commands."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from castor_server.cli import main


def test_compose_prints_yaml():
    runner = CliRunner()
    result = runner.invoke(main, ["compose"])
    assert result.exit_code == 0
    # Spot-check the output looks like a docker-compose file
    assert "services:" in result.output
    assert "castor-server:" in result.output
    assert "image: castor/server:latest" in result.output
    assert "8080" in result.output


def test_compose_write_to_cwd():
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(main, ["compose", "--write"])
        assert result.exit_code == 0
        assert "✓ Wrote" in result.output
        target = Path("docker-compose.yml")
        assert target.exists()
        assert "services:" in target.read_text()


def test_compose_write_refuses_to_overwrite():
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("docker-compose.yml").write_text("existing: true\n")
        result = runner.invoke(main, ["compose", "--write"])
        assert result.exit_code == 1
        assert "Refusing to overwrite" in result.output
        # Original file is preserved
        assert Path("docker-compose.yml").read_text() == "existing: true\n"


def test_deploy_prints_cheatsheet():
    runner = CliRunner()
    result = runner.invoke(main, ["deploy"])
    assert result.exit_code == 0
    assert "Docker Compose" in result.output
    assert "ANTHROPIC_API_KEY" in result.output
    assert "anthropic-python" in result.output or "Anthropic(" in result.output
