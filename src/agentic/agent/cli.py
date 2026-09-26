#!/usr/bin/env python3
"""Command-line interface for config-driven agents.

Installed as both ``simple-agent`` and ``orchestrator-agent`` (the same runtime
serves both kinds of agent; `agent.yaml` decides what an agent can do).

    simple-agent serve                 # OpenAI-compatible API + A2A + health
    simple-agent chat                  # interactive terminal chat
    simple-agent web --port 8080       # browser chat UI
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import typer
import uvicorn

import agentic.logging

from .config import CONFIG_DIR, SECRETS_DIR, ConfigError, Secrets

app = typer.Typer(
    no_args_is_help=True,
    help="Serve or chat with a config-driven Pydantic AI agent.",
)

ConfigDirOption = typer.Option(
    CONFIG_DIR, help="Directory containing agent.yaml and server.yaml."
)
SecretsDirOption = typer.Option(SECRETS_DIR, help="Directory of secret files.")


def _fail(exc: ConfigError) -> None:
    click.echo(f"Configuration error: {exc}", err=True)
    raise typer.Exit(code=2)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Address to bind."),
    port: int = typer.Option(8000, help="Port to bind."),
    config_dir: Path = ConfigDirOption,
    secrets_dir: Path = SecretsDirOption,
    public_url: str = typer.Option(
        "http://localhost:8000",
        "--public-url",
        "--agent-url",
        envvar="AGENT_PUBLIC_URL",
        help="Externally reachable base URL, advertised in the A2A agent card.",
    ),
    watch: bool = typer.Option(
        True, help="Reload automatically when the config or secrets change."
    ),
    log_level: str = typer.Option("info", help="Log level."),
) -> None:
    """Serve the OpenAI-compatible API, the A2A interface, and health probes."""
    agentic.logging.auto(log_level.upper())
    from .app import create_app

    try:
        application = create_app(config_dir, secrets_dir, public_url, watch=watch)
    except ConfigError as exc:
        _fail(exc)
    uvicorn.run(application, host=host, port=port, log_level=log_level.lower())


@app.command()
def chat(
    config_dir: Path = ConfigDirOption,
    secrets_dir: Path = SecretsDirOption,
    log_level: str = typer.Option("warning", help="Log level."),
) -> None:
    """Chat with the agent in the terminal."""
    agentic.logging.auto(log_level.upper())
    from .spec import load_agent

    try:
        agent = load_agent(config_dir / "agent.yaml", Secrets(secrets_dir))
    except ConfigError as exc:
        _fail(exc)
    agent.to_cli_sync(prog_name=Path(sys.argv[0]).name or "agent")


@app.command("web-chat", hidden=True)
@app.command()
def web(
    host: str = typer.Option("127.0.0.1", help="Address to bind."),
    port: int = typer.Option(8080, help="Port to bind."),
    config_dir: Path = ConfigDirOption,
    secrets_dir: Path = SecretsDirOption,
    log_level: str = typer.Option("info", help="Log level."),
) -> None:
    """Serve Pydantic AI's browser chat UI for the agent (for local use)."""
    agentic.logging.auto(log_level.upper())
    from .spec import load_agent

    try:
        agent = load_agent(config_dir / "agent.yaml", Secrets(secrets_dir))
    except ConfigError as exc:
        _fail(exc)
    uvicorn.run(agent.to_web(), host=host, port=port, log_level=log_level.lower())


if __name__ == "__main__":
    app()
