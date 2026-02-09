#!/usr/bin/env python3
"""Diagnostic: inspect pydantic-ai Agent message flow during tool calling."""

import asyncio
import os

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.settings import ModelSettings

MODEL_NAME = os.environ["MODEL_NAME"]

model = OpenAIChatModel(model_name=MODEL_NAME)

agent = Agent(
    model=model,
    system_prompt="You are a helpful assistant. When asked to check alerts, use the list_alerts tool.",
    model_settings=ModelSettings(temperature=0.2),
    output_type=str,
)


@agent.tool_plain
async def list_alerts() -> str:
    """List active alerts."""
    return '{"alerts": [], "message": "No active alerts found."}'


@agent.tool_plain
async def search_outages() -> str:
    """Search for known outages."""
    return '{"outages": [], "message": "No known outages."}'


async def main():
    print(f"Model: {MODEL_NAME}")
    print(f"Base URL: {os.environ.get('OPENAI_BASE_URL')}")
    print()

    result = await agent.run("Check if there are any active alerts.")

    print(f"Output: {result.output}")
    print()

    print("=== All Messages ===")
    for i, msg in enumerate(result.all_messages()):
        print(f"\n--- Message {i} ({type(msg).__name__}) ---")
        if hasattr(msg, "parts"):
            for j, part in enumerate(msg.parts):
                part_type = type(part).__name__
                if hasattr(part, "content"):
                    content = str(part.content)[:200]
                    print(f"  Part {j} ({part_type}): {content}")
                elif hasattr(part, "tool_name"):
                    print(f"  Part {j} ({part_type}): tool={part.tool_name}, args={getattr(part, 'args', None)}")
                else:
                    print(f"  Part {j} ({part_type}): {part}")


asyncio.run(main())
