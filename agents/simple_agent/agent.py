#!/usr/bin/env python3
"""Simple Agent — a config-driven Pydantic AI agent."""

import os
from datetime import date
from pathlib import Path

import typer
import uvicorn
import yaml
from fasta2a import Skill
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from agentic.openai import OpenAICompatibleAPI
from agentic.openai.utils import normalize_openai_base_url

# -------------------------------------------------------------------------------------------------
# A2A Config Models
# -------------------------------------------------------------------------------------------------


class SkillConfig(BaseModel):
    id: str
    name: str
    description: str
    tags: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)
    input_modes: list[str] = Field(default=["text/plain"])
    output_modes: list[str] = Field(default=["text/plain"])


class A2AConfig(BaseModel):
    name: str
    description: str
    skills: list[SkillConfig] = Field(default_factory=list)


# -------------------------------------------------------------------------------------------------
# Agent Configuration
# -------------------------------------------------------------------------------------------------

# Environment variables
for _var in ["OPENAI_BASE_URL", "OPENAI_API_KEY", "MODEL_NAME"]:
    if _var not in os.environ:
        raise EnvironmentError(f"Environment variable {_var} is not set.")
OPENAI_BASE_URL = normalize_openai_base_url(os.environ["OPENAI_BASE_URL"])
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
MODEL_NAME = os.environ["MODEL_NAME"]

# Configuration files
AGENT_CONFIG = Path(os.environ.get("AGENT_CONFIG", "/etc/agent/agent.yml"))
A2A_CONFIG = Path(os.environ.get("A2A_CONFIG", "/etc/agent/a2a.yml"))
SYSTEM_PROMPT_PATH = Path(
    os.environ.get("SYSTEM_PROMPT_PATH", "/etc/agent/system_prompt.md")
)

agent_config = yaml.safe_load(os.path.expandvars(AGENT_CONFIG.read_text()))
a2a_config = A2AConfig.model_validate(yaml.safe_load(A2A_CONFIG.read_text()))
system_prompt = SYSTEM_PROMPT_PATH.read_text()


# -------------------------------------------------------------------------------------------------
# Simple Agent
# -------------------------------------------------------------------------------------------------

agent = Agent.from_spec(
    agent_config,
    model=OpenAIChatModel(
        model_name=MODEL_NAME,
        provider=OpenAIProvider(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY),
    ),
    instructions=system_prompt,
    output_type=str,
)


@agent.instructions
def add_date_info() -> str:
    return f"Today is {date.today().strftime('%A, %B %d, %Y (%Y-%m-%d)')}."


# --------------------------------------------------------------------------------------
# Agent Skills
# --------------------------------------------------------------------------------------

agent_skills = [
    Skill(
        id=s.id,
        name=s.name,
        description=s.description,
        tags=s.tags,
        examples=s.examples,
        input_modes=s.input_modes,
        output_modes=s.output_modes,
    )
    for s in a2a_config.skills
]


# -------------------------------------------------------------------------------------------------
# Agent Interfaces
# -------------------------------------------------------------------------------------------------

app = typer.Typer(no_args_is_help=True, help=a2a_config.description)


@app.command(short_help="Command Line Interface (CLI)")
def cli():
    """Command Line Interface (CLI)."""
    agent.to_cli_sync(prog_name="simple-agent")


@app.command(short_help="A2A + OpenAI-compatible APIs (combined)")
def serve(agent_url: str, host: str | None = None, port: int | None = None):
    """Serve A2A and OpenAI-compatible API interfaces on a single port."""
    host = host or os.environ.get("HOST", "0.0.0.0")
    port = port or int(os.environ.get("PORT", 8000))

    openai_api = OpenAICompatibleAPI(
        agent=agent,
        title=f"{a2a_config.name} OpenAI Compatible API",
        description=f"OpenAI-compatible API for the {a2a_config.name}.",
        model_name=MODEL_NAME,
    )

    a2a_asgi = agent.to_a2a(
        name=a2a_config.name,
        description=a2a_config.description,
        url=agent_url,
        skills=agent_skills,
        debug=True,
    )

    # OpenAI routes (/status, /health, /v1/…) match first; A2A catches the rest
    openai_api.app.mount("/", a2a_asgi)
    uvicorn.run(openai_api.app, host=host, port=port)


if __name__ == "__main__":
    app()
