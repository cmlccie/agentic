#!/usr/bin/env python3
"""MCP AIOps Server entry point.

Deterministic validation and benchmarking tools for LLM model-serving
deployments (SGLang, vLLM, NIM). See aiops/server.py for the tool definitions.
"""

import logging
import os
from typing import Annotated, Literal

import typer
from aiops.server import mcp

import agentic.logging

logger = logging.getLogger("aiops_server")

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))


def main(
    transport: Annotated[Literal["stdio", "http"], typer.Argument()] = "stdio",
) -> None:
    """Model Context Protocol (MCP) AIOps Server."""
    agentic.logging.fancy()
    logger.info("Starting %s MCP AIOps Server", transport)

    match transport:
        case "stdio":
            mcp.run(transport=transport)
        case "http":
            mcp.run(transport=transport, host=HOST, port=PORT)
        case _:
            raise typer.BadParameter(
                "Transport must be one of: stdio, http.",
                param_hint="transport",
            )


if __name__ == "__main__":
    typer.run(main)
