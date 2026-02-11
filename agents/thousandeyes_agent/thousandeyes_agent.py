#!/usr/bin/env python3
"""ThousandEyes Agent using Pydantic AI and ThousandEyes MCP tools."""

import asyncio
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
from pydantic_ai.toolsets import FilteredToolset

from agentic.openai import OpenAICompatibleAPI

## Global Variables

# Path to the current directory
here = Path(__file__).parent

# Required environment variables
required_env_vars = [
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
    "MODEL_NAME",
    "THOUSANDEYES_TOKEN",
]

# Check for required environment variables
for var in required_env_vars:
    if var not in os.environ:
        raise EnvironmentError(f"Environment variable {var} is not set.")


# -------------------------------------------------------------------------------------------------
# ThousandEyes MCP Tool Subset
# -------------------------------------------------------------------------------------------------

# Only expose the tools needed for connectivity troubleshooting.
# The full MCP server provides ~30 tools; this subset keeps the LLM context concise.
ALLOWED_TOOLS = {
    "list_cloud_enterprise_agents",
    "list_network_app_synthetics_tests",
    "run_http_server_instant_test",
    "run_agent_to_server_instant_test",
    "run_dns_server_instant_test",
    "get_instant_test_metrics",
    "get_network_app_synthetics_metrics",
    "get_full_path_visualization",
    "list_alerts",
    "search_outages",
}


# -------------------------------------------------------------------------------------------------
# ThousandEyes Agent
# -------------------------------------------------------------------------------------------------

MODEL_NAME = os.environ["MODEL_NAME"]
SYSTEM_PROMPT_PATH = "system_prompt.md"
THOUSANDEYES_MCP_URL = os.environ.get(
    "THOUSANDEYES_MCP_URL", "https://api.thousandeyes.com/mcp"
)
THOUSANDEYES_TOKEN = os.environ["THOUSANDEYES_TOKEN"]

model = OpenAIChatModel(model_name=MODEL_NAME)
system_prompt = (here / SYSTEM_PROMPT_PATH).read_text()

mcp_server = MCPServerStreamableHTTP(
    url=THOUSANDEYES_MCP_URL,
    headers={"Authorization": f"Bearer {THOUSANDEYES_TOKEN}"},
)

filtered_tools = FilteredToolset(
    mcp_server,
    filter_func=lambda ctx, tool_def: tool_def.name in ALLOWED_TOOLS,
)

model_settings = ModelSettings(
    temperature=0.2,
)

agent = Agent(
    model=model,
    instructions=system_prompt,
    toolsets=[filtered_tools],
    model_settings=model_settings,
    output_type=str,
)


@agent.instructions
def add_datetime_info() -> str:
    """Dynamic system prompt - adds current UTC datetime for use in API time parameters."""
    now = datetime.now(timezone.utc)
    return f"The current UTC date and time is {now.strftime('%Y-%m-%dT%H:%M:%SZ')}."


@agent.tool_plain
async def wait(seconds: int) -> str:
    """Wait for the specified number of seconds before continuing.

    Use this tool to pause between polling attempts when waiting for instant test
    results to become available.
    """
    seconds = min(max(seconds, 1), 60)
    await asyncio.sleep(seconds)
    return f"Waited {seconds} seconds. You may now check for results."


# --------------------------------------------------------------------------------------
# Agent Skills
# --------------------------------------------------------------------------------------

agent_skills = [
    Skill(
        id="troubleshoot-connectivity",
        name="Troubleshoot Connectivity",
        description="Troubleshoot application connectivity issues using ThousandEyes.",
        tags=["network", "connectivity", "troubleshooting", "thousandeyes"],
        examples=[
            "Why can't users reach https://app.example.com?",
            "Check connectivity to api.example.com from US agents.",
            "Are there any outages affecting our services?",
            "Run a network test to 10.0.0.1 from our enterprise agents.",
            "What does the network path look like to cdn.example.com?",
        ],
        input_modes=["text/plain"],
        output_modes=["text/plain"],
    ),
]


# -------------------------------------------------------------------------------------------------
# Agent Interfaces
# -------------------------------------------------------------------------------------------------

app = typer.Typer(no_args_is_help=True, help="ThousandEyes Agent")


@app.command(short_help="Command Line Interface (CLI)")
def cli():
    """Command Line Interface (CLI)."""
    agent.to_cli_sync(prog_name="thousandeyes-agent")


@app.command(short_help="Agent2Agent (A2A) Interface")
def a2a(agent_url: str, host: str | None = None, port: int | None = None):
    """Agent2Agent interface."""

    host = host or os.environ.get("HOST", "0.0.0.0")
    port = port or int(os.environ.get("PORT", 8000))

    a2a_app = agent.to_a2a(
        name="ThousandEyes Agent",
        description="Troubleshoots application connectivity issues using ThousandEyes.",
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
        title="ThousandEyes Agent OpenAI Compatible API",
        description="OpenAI-compatible API for the ThousandEyes Agent.",
        model_name=MODEL_NAME,
    )

    openai_api.run(host=host, port=port)


if __name__ == "__main__":
    app()
