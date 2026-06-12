#!/usr/bin/env python3
"""Simple Agent CLI — serve or chat with a config-driven Pydantic AI agent."""

from __future__ import annotations

from pathlib import Path

import typer
import uvicorn

import agentic.logging

from .config import CONFIG_DIR, SECRETS_DIR

app = typer.Typer(no_args_is_help=True, help="Simple Agent")


@app.command(short_help="Serve all configured interfaces (default)")
def serve(
    host: str = typer.Option("0.0.0.0", help="Bind host"),  # noqa: B008
    port: int = typer.Option(8000, help="Bind port"),  # noqa: B008
    config_dir: Path = typer.Option(CONFIG_DIR, help="Config directory"),  # noqa: B008
    secrets_dir: Path = typer.Option(SECRETS_DIR, help="Secrets directory"),  # noqa: B008
    agent_url: str = typer.Option(  # noqa: B008
        "http://localhost:8000", help="Public URL for A2A agent card"
    ),
    log_level: str = typer.Option("info", help="Log level"),  # noqa: B008
) -> None:
    """Start the FastAPI server with all configured interfaces.

    Reads agent.yaml and server.yaml from config_dir. Secrets are read from
    secrets_dir on every reload cycle — no restart required for key rotation.
    """
    agentic.logging.fancy(log_level.upper())

    from .main import create_app

    application = create_app(
        config_dir=config_dir,
        secrets_dir=secrets_dir,
        agent_url=agent_url,
    )

    uvicorn.run(application, host=host, port=port, log_level=log_level)


@app.command(short_help="Serve the web chat UI")
def web_chat(
    host: str = typer.Option("0.0.0.0", help="Bind host"),  # noqa: B008
    port: int = typer.Option(8000, help="Bind port"),  # noqa: B008
    config_dir: Path = typer.Option(CONFIG_DIR, help="Config directory"),  # noqa: B008
    secrets_dir: Path = typer.Option(SECRETS_DIR, help="Secrets directory"),  # noqa: B008
    log_level: str = typer.Option("info", help="Log level"),  # noqa: B008
) -> None:
    """Serve the Pydantic AI web chat UI for the configured agent.

    Runs a standalone ASGI server for the browser-based chat interface.
    The web-chat server is separate from the serve process — run them on
    different ports if you need both simultaneously.
    """
    agentic.logging.fancy(log_level.upper())

    from .config.agent_spec import load_agent
    from .config.server_spec import AgentSecrets, load_server_spec

    secrets = AgentSecrets(secrets_dir)
    load_server_spec(config_dir / "server.yaml")  # validate config exists
    agent = load_agent(config_dir / "agent.yaml", secrets)

    uvicorn.run(agent.to_web(), host=host, port=port, log_level=log_level)


@app.command(short_help="Interactive terminal chat")
def chat(
    config_dir: Path = typer.Option(CONFIG_DIR, help="Config directory"),  # noqa: B008
    secrets_dir: Path = typer.Option(SECRETS_DIR, help="Secrets directory"),  # noqa: B008
    log_level: str = typer.Option("warning", help="Log level"),  # noqa: B008
) -> None:
    """Run an interactive terminal chat session with the configured agent."""
    agentic.logging.fancy(log_level.upper())

    from .config.agent_spec import load_agent
    from .config.server_spec import AgentSecrets, load_server_spec

    secrets = AgentSecrets(secrets_dir)
    load_server_spec(config_dir / "server.yaml")  # validate config exists
    agent = load_agent(config_dir / "agent.yaml", secrets)

    agent.to_cli_sync(prog_name="simple-agent")
