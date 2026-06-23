#!/usr/bin/env python3
"""Orchestrator Agent CLI — serve a config-driven LangGraph supervisor over A2A."""

from __future__ import annotations

from pathlib import Path

import typer
import uvicorn

import agentic.logging

from .config import CONFIG_DIR, SECRETS_DIR

app = typer.Typer(no_args_is_help=True, help="Orchestrator Agent")


@app.callback()
def main() -> None:
    """Orchestrator Agent — a config-driven LangGraph supervisor over A2A agents."""


@app.command(short_help="Serve all configured interfaces (default)")
def serve(
    host: str = typer.Option("0.0.0.0", help="Bind host"),  # noqa: B008
    port: int = typer.Option(8000, help="Bind port"),  # noqa: B008
    config_dir: Path = typer.Option(CONFIG_DIR, help="Config directory"),  # noqa: B008
    secrets_dir: Path = typer.Option(SECRETS_DIR, help="Secrets directory"),  # noqa: B008
    agent_url: str = typer.Option(  # noqa: B008
        "http://localhost:8000", help="Public URL for the A2A agent card"
    ),
    log_level: str = typer.Option("info", help="Log level"),  # noqa: B008
) -> None:
    """Start the FastAPI server with the OpenAI-compatible and A2A interfaces.

    Reads agent.yaml and server.yaml from config_dir. Secrets are read from
    secrets_dir on every reload cycle — no restart required for key rotation.
    The supervisor graph and its downstream A2A agent set are rebuilt on config
    changes without dropping the process.
    """
    agentic.logging.fancy(log_level.upper())

    from .main import create_app

    application = create_app(
        config_dir=config_dir,
        secrets_dir=secrets_dir,
        agent_url=agent_url,
    )

    uvicorn.run(application, host=host, port=port, log_level=log_level)
