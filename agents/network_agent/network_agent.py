#!/usr/bin/env python3
"""Network Troubleshooting Agent — orchestrates Meraki and ThousandEyes sub-agents."""

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import httpx
import typer
import uvicorn
from fasta2a import Skill
from fasta2a.client import A2AClient
from fasta2a.schema import Message, TextPart
from pydantic_ai import Agent
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
    "MERAKI_AGENT_URL",
    "THOUSANDEYES_AGENT_URL",
]

# Check for required environment variables
for var in required_env_vars:
    if var not in os.environ:
        raise EnvironmentError(f"Environment variable {var} is not set.")


# -------------------------------------------------------------------------------------------------
# Module-level state for agent discovery
# -------------------------------------------------------------------------------------------------

agent_cards: dict[str, dict] = {}  # agent_name -> agent card JSON
agent_urls: dict[str, str] = {}  # agent_name -> base URL

# Sub-agent URLs from environment
SUB_AGENT_URLS = {
    os.environ["MERAKI_AGENT_URL"],
    os.environ["THOUSANDEYES_AGENT_URL"],
}

# A2A polling configuration
A2A_REQUEST_TIMEOUT = 30  # seconds per HTTP request
A2A_POLL_INTERVAL = 5  # seconds between polls
A2A_MAX_WAIT = 300  # max seconds to wait for task completion (5 minutes)


# -------------------------------------------------------------------------------------------------
# Network Troubleshooting Agent
# -------------------------------------------------------------------------------------------------

MODEL_NAME = os.environ["MODEL_NAME"]
SYSTEM_PROMPT_PATH = "system_prompt.md"

model = OpenAIChatModel(model_name=MODEL_NAME)
system_prompt = (here / SYSTEM_PROMPT_PATH).read_text()

model_settings = ModelSettings(
    temperature=0.2,
)

agent = Agent(
    model=model,
    instructions=system_prompt,
    model_settings=model_settings,
    output_type=str,
)


@agent.instructions
def add_datetime_info() -> str:
    """Dynamic system prompt - adds current UTC datetime."""
    now = datetime.now(timezone.utc)
    return f"The current UTC date and time is {now.strftime('%Y-%m-%dT%H:%M:%SZ')}."


# -------------------------------------------------------------------------------------------------
# A2A Client Tools
# -------------------------------------------------------------------------------------------------


def _extract_text(response: dict) -> str:
    """Extract text content from an A2A SendMessageResponse."""
    if "result" in response:
        task = response["result"]
        text_parts: list[str] = []
        for artifact in task.get("artifacts", []):
            for part in artifact.get("parts", []):
                if part.get("kind") == "text":
                    text_parts.append(part["text"])
        if text_parts:
            return "\n\n".join(text_parts)

        # Fall back to status message if no artifacts
        status = task.get("status", {})
        msg = status.get("message")
        if msg:
            for part in msg.get("parts", []):
                if part.get("kind") == "text":
                    text_parts.append(part["text"])
            if text_parts:
                return "\n\n".join(text_parts)

        return f"Task completed with state: {status.get('state', 'unknown')} (no text output)"

    if "error" in response:
        error = response["error"]
        return f"Error: {error.get('message', 'Unknown error')}"

    return "Unexpected response format."


def _build_message(content: str) -> Message:
    """Build an A2A Message with a text part."""
    text_part: TextPart = {"kind": "text", "text": content}
    return {
        "role": "user",
        "parts": [text_part],
        "kind": "message",
        "message_id": str(uuid4()),
    }


TERMINAL_STATES = {"completed", "failed", "canceled", "rejected"}


async def _send_and_poll(url: str, content: str) -> str:
    """Send a message to a sub-agent via A2A and poll until completion."""
    async with httpx.AsyncClient(timeout=A2A_REQUEST_TIMEOUT) as http_client:
        client = A2AClient(base_url=url, http_client=http_client)
        msg = _build_message(content)

        response = await client.send_message(msg)
        task = response["result"]
        task_id = task["id"]
        state = task["status"]["state"]

        if state in TERMINAL_STATES:
            return _extract_text(response)

        elapsed = 0
        while elapsed < A2A_MAX_WAIT:
            await asyncio.sleep(A2A_POLL_INTERVAL)
            elapsed += A2A_POLL_INTERVAL
            task_response = await client.get_task(task_id)
            state = task_response["result"]["status"]["state"]
            if state in TERMINAL_STATES:
                return _extract_text(task_response)

        return (
            f"Task timed out after {A2A_MAX_WAIT}s "
            f"(task: {task_id}, last state: {state})"
        )


@agent.tool_plain
async def discover_agents() -> str:
    """Discover available sub-agents and their capabilities.

    Fetches agent cards from all configured sub-agent URLs. Call this at the
    start of each session to learn what agents are available.
    """
    results: list[str] = []

    async with httpx.AsyncClient(timeout=30) as http_client:
        for url in SUB_AGENT_URLS:
            card_url = f"{url.rstrip('/')}/.well-known/agent-card.json"
            try:
                resp = await http_client.get(card_url)
                resp.raise_for_status()
                card = resp.json()

                name = card.get("name", "Unknown Agent")
                agent_cards[name] = card
                agent_urls[name] = url

                description = card.get("description", "No description")
                skills = card.get("skills", [])
                skill_summaries = []
                for skill in skills:
                    skill_name = skill.get("name", "unnamed")
                    skill_desc = skill.get("description", "")
                    examples = skill.get("examples", [])
                    summary = f"  - {skill_name}: {skill_desc}"
                    if examples:
                        summary += f"\n    Examples: {'; '.join(examples[:3])}"
                    skill_summaries.append(summary)

                agent_summary = f"**{name}**: {description}"
                if skill_summaries:
                    agent_summary += "\nSkills:\n" + "\n".join(skill_summaries)
                results.append(agent_summary)

            except httpx.HTTPError as e:
                results.append(f"Failed to reach agent at {url}: {e}")

    return "\n\n".join(results)


@agent.tool_plain
async def send_task(agent_name: str, message: str) -> str:
    """Send a message to a single sub-agent and wait for its response.

    Use this for follow-up questions or when you only need to consult one agent.

    Args:
        agent_name: The name of the agent (as returned by discover_agents).
        message: The task or question to send to the agent.
    """
    url = agent_urls.get(agent_name)
    if not url:
        available = ", ".join(agent_urls.keys()) or "none (run discover_agents first)"
        return f"Unknown agent '{agent_name}'. Available agents: {available}"

    try:
        text = await _send_and_poll(url, message)
        return f"**{agent_name}** responded:\n\n{text}"
    except Exception as e:
        return f"Error communicating with {agent_name}: {e}"


@agent.tool_plain
async def send_tasks_parallel(tasks: str) -> str:
    """Send messages to multiple sub-agents in parallel and wait for all responses.

    This is the primary dispatch mechanism. All agents work simultaneously.

    Args:
        tasks: A JSON string of tasks, e.g.:
            [{"agent_name": "Meraki Agent", "message": "Check client health"},
             {"agent_name": "ThousandEyes Agent", "message": "Test connectivity"}]
    """
    try:
        task_list = json.loads(tasks)
    except json.JSONDecodeError as e:
        return f"Invalid JSON: {e}"

    if not isinstance(task_list, list) or not task_list:
        return "Expected a non-empty JSON array of task objects."

    async def _dispatch(task_spec: dict) -> str:
        name = task_spec.get("agent_name", "")
        content = task_spec.get("message", "")
        url = agent_urls.get(name)

        if not url:
            available = (
                ", ".join(agent_urls.keys()) or "none (run discover_agents first)"
            )
            return f"**{name}**: Unknown agent. Available agents: {available}"

        try:
            text = await _send_and_poll(url, content)
            return f"**{name}** responded:\n\n{text}"
        except Exception as e:
            return f"**{name}**: Error — {e}"

    results = await asyncio.gather(*[_dispatch(t) for t in task_list])
    return "\n\n---\n\n".join(results)


@agent.tool_plain
async def wait(seconds: int) -> str:
    """Wait for the specified number of seconds before continuing.

    Use this to pause between dispatch attempts if a sub-agent needs time.
    """
    seconds = min(max(seconds, 1), 60)
    await asyncio.sleep(seconds)
    return f"Waited {seconds} seconds. You may now continue."


# --------------------------------------------------------------------------------------
# Agent Skills
# --------------------------------------------------------------------------------------

agent_skills = [
    Skill(
        id="troubleshoot-network",
        name="Troubleshoot Network",
        description="Comprehensive network troubleshooting coordinating Meraki and ThousandEyes agents.",
        tags=["network", "troubleshooting", "meraki", "thousandeyes", "supervisor"],
        examples=[
            "Users can't reach app.example.com — investigate.",
            "Client DESKTOP-XYZ has intermittent connectivity. Run a full diagnostic.",
        ],
        input_modes=["text/plain"],
        output_modes=["text/plain"],
    ),
]


# -------------------------------------------------------------------------------------------------
# Agent Interfaces
# -------------------------------------------------------------------------------------------------

app = typer.Typer(no_args_is_help=True, help="Network Troubleshooting Agent")


@app.command(short_help="Command Line Interface (CLI)")
def cli():
    """Command Line Interface (CLI)."""
    agent.to_cli_sync(prog_name="network-agent")


@app.command(short_help="Agent2Agent (A2A) Interface")
def a2a(agent_url: str, host: str | None = None, port: int | None = None):
    """Agent2Agent interface."""

    host = host or os.environ.get("HOST", "0.0.0.0")
    port = port or int(os.environ.get("PORT", 8000))

    a2a_app = agent.to_a2a(
        name="Network Troubleshooting Agent",
        description="Coordinates Meraki and ThousandEyes agents for comprehensive network troubleshooting.",
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
        title="Network Troubleshooting Agent OpenAI Compatible API",
        description="OpenAI-compatible API for the Network Troubleshooting Agent.",
        model_name=MODEL_NAME,
    )

    openai_api.run(host=host, port=port)


if __name__ == "__main__":
    app()
