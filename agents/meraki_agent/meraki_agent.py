#!/usr/bin/env python3
"""Meraki Agent using Pydantic AI and MCP tools."""

import os
from datetime import datetime, timezone
from pathlib import Path

import typer
import uvicorn
from fasta2a import Skill
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStreamableHTTP
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.settings import ModelSettings

from agentic.openai import OpenAICompatibleAPI

## Global Variables

# Path to the current directory
here = Path(__file__).parent

# Required environment variables
required_env_vars = [
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
    "MODEL_NAME",
    "TOOLS_MCP_MERAKI_SERVER_URL",
]

# Check for required environment variables
for var in required_env_vars:
    if var not in os.environ:
        raise EnvironmentError(f"Environment variable {var} is not set.")


# -------------------------------------------------------------------------------------------------
# Meraki Agent
# -------------------------------------------------------------------------------------------------

MODEL_NAME = os.environ["MODEL_NAME"]
SYSTEM_PROMPT_PATH = "system_prompt.md"
TOOLS_MCP_MERAKI_SERVER_URL = os.environ["TOOLS_MCP_MERAKI_SERVER_URL"]

model = OpenAIChatModel(model_name=MODEL_NAME)
system_prompt = (here / SYSTEM_PROMPT_PATH).read_text()
agent_tools = [MCPServerStreamableHTTP(url=TOOLS_MCP_MERAKI_SERVER_URL)]

model_settings = ModelSettings(
    temperature=0.2,
)

agent = Agent(
    model=model,
    system_prompt=system_prompt,
    toolsets=agent_tools,
    model_settings=model_settings,
    output_type=str,
)


@agent.system_prompt
def add_datetime_info() -> str:
    """Dynamic system prompt - adds current UTC datetime."""
    now = datetime.now(timezone.utc)
    return f"The current UTC date and time is {now.strftime('%Y-%m-%dT%H:%M:%SZ')}."


# --------------------------------------------------------------------------------------
# Agent Skills
# --------------------------------------------------------------------------------------

agent_skills = [
    Skill(
        id="troubleshoot-client-connectivity",
        name="Troubleshoot Client Connectivity",
        description="Troubleshoot wireless client connectivity issues on a Meraki network.",
        tags=["network", "wireless", "meraki", "troubleshooting"],
        examples=[
            "Show me all clients on the network.",
            "Check the health of client DESKTOP-ABC.",
            "Why is the client at 00:11:22:33:44:55 having connectivity issues?",
            "Show me connection stats for client k74272e over the last 7 days.",
            "What connectivity events has this client experienced?",
        ],
        input_modes=["text/plain"],
        output_modes=["text/plain"],
    ),
]


# -------------------------------------------------------------------------------------------------
# Agent Interfaces
# -------------------------------------------------------------------------------------------------

app = typer.Typer(no_args_is_help=True, help="Meraki Agent")


@app.command(short_help="Command Line Interface (CLI)")
def cli():
    """Command Line Interface (CLI)."""
    agent.to_cli_sync(prog_name="meraki-agent")


@app.command(short_help="Agent2Agent (A2A) Interface")
def a2a(agent_url: str, host: str | None = None, port: int | None = None):
    """Agent2Agent interface."""

    host = host or os.environ.get("HOST", "0.0.0.0")
    port = port or int(os.environ.get("PORT", 8000))

    a2a_app = agent.to_a2a(
        name="Meraki Agent",
        description="Troubleshoots wireless client connectivity issues on a Meraki network.",
        url=agent_url,
        skills=agent_skills,
        debug=True,
    )

    uvicorn.run(a2a_app, host=host, port=port)


@app.command(short_help="OpenAI Compatible API Interface")
def openai_api(host: str | None = None, port: int | None = None):
    """OpenAI Compatible API interface."""

    host = host or os.environ.get("HOST", "0.0.0.0")
    port = port or int(os.environ.get("PORT", 8000))

    openai_api = OpenAICompatibleAPI(
        agent=agent,
        title="Meraki Agent OpenAI Compatible API",
        description="OpenAI-compatible API for the Meraki Agent.",
        model_name=MODEL_NAME,
    )

    openai_api.run(host=host, port=port)


if __name__ == "__main__":
    app()
