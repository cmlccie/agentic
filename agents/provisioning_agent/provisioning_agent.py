#!/usr/bin/env python3
"""Provisioning Agent using Pydantic AI and MCP tools."""

import os
import time
from pathlib import Path

import typer
import uvicorn
from fasta2a import Skill
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStreamableHTTP
from pydantic_ai.models.openai import OpenAIChatModel

from agentic.openai import OpenAICompatibleAPI

## Global Variables

# Path to the current directory
here = Path(__file__).parent

# Required environment variables
required_env_vars = [
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
    "MODEL_NAME",
    "TOOLS_MCP_PROVISIONING_SERVER_URL",
]

# Check for required environment variables
for var in required_env_vars:
    if var not in os.environ:
        raise EnvironmentError(f"Environment variable {var} is not set.")


# -------------------------------------------------------------------------------------------------
# Provisioning Agent
# -------------------------------------------------------------------------------------------------

MODEL_NAME = os.environ["MODEL_NAME"]
SYSTEM_PROMPT_PATH = "system_prompt.md"
TOOLS_MCP_PROVISIONING_SERVER_URL = os.environ["TOOLS_MCP_PROVISIONING_SERVER_URL"]


model = OpenAIChatModel(model_name=MODEL_NAME)
system_prompt = (here / SYSTEM_PROMPT_PATH).read_text()
agent_tools = [MCPServerStreamableHTTP(url=TOOLS_MCP_PROVISIONING_SERVER_URL)]

agent = Agent(
    model=model,
    output_type=str,
    system_prompt=system_prompt,
    toolsets=agent_tools,
)


@agent.system_prompt
def add_date_info() -> str:
    """Dynamic system prompt - adds current date information to the system prompt."""
    # Example: Thursday, December 25, 2025 (2025-12-25)
    current_date = time.strftime("%A, %B %d, %Y (%Y-%m-%d)")
    return f"Today is {current_date}."


# --------------------------------------------------------------------------------------
# Agent Skills
# --------------------------------------------------------------------------------------

agent_skills = [
    Skill(
        id="provision-server",
        name="Provision Server",
        description="Provision a new server with specified resources.",
        tags=["provisioning", "server", "infrastructure"],
        examples=[
            "Provision a server named web-server-01 with 4 CPU cores, 16GB RAM, and 100GB storage on VLAN 100.",
            "Create a new server called db-server with 8 cores and 32GB memory.",
            "Set up a server with 2 CPUs and 8GB RAM on VLAN 200.",
        ],
        input_modes=["text/plain"],
        output_modes=["text/plain"],
    ),
    Skill(
        id="provision-vlan",
        name="Provision VLAN",
        description="Provision a new VLAN with specified configuration.",
        tags=["provisioning", "vlan", "network", "infrastructure"],
        examples=[
            "Create VLAN 100 named 'production' with CIDR 10.0.100.0/24.",
            "Provision a new VLAN with ID 200 for the development environment.",
            "Set up VLAN 300 called 'dmz' with network 192.168.1.0/24.",
        ],
        input_modes=["text/plain"],
        output_modes=["text/plain"],
    ),
    Skill(
        id="check-vlan",
        name="Check VLAN",
        description="Check if a VLAN exists.",
        tags=["provisioning", "vlan", "network", "infrastructure"],
        examples=[
            "Does VLAN 100 exist?",
            "Check if VLAN 200 is available.",
            "Is there a VLAN with ID 300?",
        ],
        input_modes=["text/plain"],
        output_modes=["text/plain"],
    ),
]


# -------------------------------------------------------------------------------------------------
# Agent Interfaces
# -------------------------------------------------------------------------------------------------

app = typer.Typer(no_args_is_help=True, help="Provisioning Agent")


@app.command(short_help="Command Line Interface (CLI)")
def cli():
    """Command Line Interface (CLI)."""
    agent.to_cli_sync(prog_name="provisioning-agent")


@app.command(short_help="Agent2Agent (A2A) Interface")
def a2a(agent_url: str, host: str | None = None, port: int | None = None):
    """Agent2Agent interface."""

    host = host or os.environ.get("HOST", "0.0.0.0")
    port = port or int(os.environ.get("PORT", 8000))

    a2a_app = agent.to_a2a(
        name="Provisioning Agent",
        description="IT infrastructure provisioning agent.",
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
        title="Provisioning Agent OpenAI Compatible API",
        description="OpenAI-compatible API for the Provisioning Agent.",
        model_name=MODEL_NAME,
    )

    openai_api.run(host=host, port=port)


if __name__ == "__main__":
    app()
