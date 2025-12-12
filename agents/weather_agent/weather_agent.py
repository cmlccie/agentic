#!/usr/bin/env python3
"""Weather Agent using Pydantic AI and MCP tools."""

import os
from datetime import date, timedelta
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
    "TOOLS_MCP_WEATHER_SERVER_URL",
]

# Check for required environment variables
for var in required_env_vars:
    if var not in os.environ:
        raise EnvironmentError(f"Environment variable {var} is not set.")


# -------------------------------------------------------------------------------------------------
# Weather Agent
# -------------------------------------------------------------------------------------------------

MODEL_NAME = os.environ["MODEL_NAME"]
SYSTEM_PROMPT_PATH = "system_prompt.md"
TOOLS_MCP_WEATHER_SERVER_URL = os.environ["TOOLS_MCP_WEATHER_SERVER_URL"]


model = OpenAIChatModel(model_name=MODEL_NAME)
system_prompt = (here / SYSTEM_PROMPT_PATH).read_text()
agent_tools = [MCPServerStreamableHTTP(url=TOOLS_MCP_WEATHER_SERVER_URL)]

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
def add_date_info() -> str:
    """Dynamic system prompt - adds current date information to the system prompt.

    Weather forecasts are available for 16 days, including today. Add dates and days
    of the week to the system prompt to help the agent provide accurate forecasts.
    """
    # Date Info Format: Thursday, December 25, 2025 (2025-12-25)

    today = date.today()
    date_info = [
        (today + timedelta(days=i)).strftime("%A, %B %d, %Y (%Y-%m-%d)")
        for i in range(16)
    ]

    return f"""## Date Information

    - Today is {today.strftime("%A, %B %d, %Y (%Y-%m-%d)")}.
    - Weather forecasts are available for the following dates:
      - {"\n  - ".join(date_info)}
    """


# --------------------------------------------------------------------------------------
# Agent Skills
# --------------------------------------------------------------------------------------

agent_skills = [
    Skill(
        id="get-weather-forecast",
        name="Get Weather Forecast",
        description="Get the weather forecast for the provided location.",
        tags=["weather", "forecast", "location"],
        examples=[
            "What's the weather in Knoxville, TN?",
            "Forecast for San Francisco, CA.",
            "Tell me the weather in Tokyo, Japan for the next week.",
            "Get the 3-day weather forecast for London, UK.",
            "What's the weather like in Sydney, Australia this weekend?",
        ],
        input_modes=["text/plain"],
        output_modes=["text/plain"],
    ),
]


# -------------------------------------------------------------------------------------------------
# Agent Interfaces
# -------------------------------------------------------------------------------------------------

app = typer.Typer(no_args_is_help=True, help="Weather Agent")


@app.command(short_help="Command Line Interface (CLI)")
def cli():
    """Command Line Interface (CLI)."""
    agent.to_cli_sync(prog_name="weather-agent")


@app.command(short_help="Agent2Agent (A2A) Interface")
def a2a(agent_url: str, host: str | None = None, port: int | None = None):
    """Agent2Agent interface."""

    host = host or os.environ.get("HOST", "0.0.0.0")
    port = port or int(os.environ.get("PORT", 8000))

    a2a_app = agent.to_a2a(
        name="Weather Agent",
        description="Weather forecast agent.",
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
        title="Weather Agent OpenAI Compatible API",
        description="OpenAI-compatible API for the Weather Agent.",
        model_name=MODEL_NAME,
    )

    openai_api.run(host=host, port=port)


if __name__ == "__main__":
    app()
