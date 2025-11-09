#!/usr/bin/env python3
"""Weather Agent using Pydantic AI and MCP tools."""

from pathlib import Path
from pprint import pprint
from typing import Optional

from pydantic_ai import Agent, AgentRunResult
from pydantic_ai.mcp import MCPServerStreamableHTTP
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

# -------------------------------------------------------------------------------------------------
# Weather Agent
# -------------------------------------------------------------------------------------------------

MODEL = "openai/gpt-oss-20b"
SYSTEM_PROMPT_FILE = "SYSTEM_PROMPT.md"

# --------------------------------------------------------------------------------------
# System Prompt
# --------------------------------------------------------------------------------------


here = Path(__file__).parent
with open(here / SYSTEM_PROMPT_FILE, "r") as f:
    system_prompt = f.read()


# --------------------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------------------

weather_tools = MCPServerStreamableHTTP(
    "http://tools.home.lunsford.io/weather-server/mcp"
)

# --------------------------------------------------------------------------------------
# Model Configuration
# --------------------------------------------------------------------------------------


model = OpenAIChatModel(
    model_name="openai/gpt-oss-20b",
    provider=OpenAIProvider(
        base_url="http://models.home.lunsford.io/openai/gpt-oss-20b/v1",
        api_key="",
    ),
)


# --------------------------------------------------------------------------------------
# Agent
# --------------------------------------------------------------------------------------


agent = Agent(
    model=model, output_type=str, system_prompt=system_prompt, toolsets=[weather_tools]
)


# --------------------------------------------------------------------------------------
# Main Chat Loop
# --------------------------------------------------------------------------------------


def chat_with_weather_agent():
    """Function to chat with the weather agent."""
    result: Optional[AgentRunResult] = None
    while True:
        try:
            user_input = input("❯ ")
            if user_input.lower() in {"exit", "quit"}:
                print("Exiting the weather agent. Goodbye!")
                break

            result = agent.run_sync(
                user_input,
                message_history=result.all_messages() if result else None,
            )

            pprint(result)

            print(result.output)
        except KeyboardInterrupt:
            print("\nExiting the weather agent. Goodbye!")
            break


if __name__ == "__main__":
    chat_with_weather_agent()
