#!/usr/bin/env python3
"""Weather Agent using Pydantic AI and MCP tools."""

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import AsyncGenerator, Literal, Optional

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStreamableHTTP
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.result import StreamedRunResult
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

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
SYSTEM_PROMPT_PATH = "SYSTEM_PROMPT.md"
TOOLS_MCP_WEATHER_SERVER_URL = os.environ["TOOLS_MCP_WEATHER_SERVER_URL"]


model = OpenAIChatModel(model_name=MODEL_NAME)
system_prompt = (here / SYSTEM_PROMPT_PATH).read_text()

# Note: We don't create a global agent anymore because MCP tools
# don't work well with concurrent requests. Instead, we create
# an agent per request in the endpoint handlers.


def create_agent() -> Agent:
    """Create a new agent instance with MCP tools.

    This is needed because MCP tools use context managers that don't
    work well with concurrent requests when shared across a global agent.
    """
    agent_tools = [MCPServerStreamableHTTP(url=TOOLS_MCP_WEATHER_SERVER_URL)]
    return Agent(
        model=model,
        output_type=str,
        system_prompt=system_prompt,
        toolsets=agent_tools,
    )


# -------------------------------------------------------------------------------------------------
# Developer Console
# -------------------------------------------------------------------------------------------------


async def dev_console():
    """Function to chat with the weather agent."""
    agent = create_agent()
    result: Optional[StreamedRunResult] = None
    while True:
        try:
            prompt = input("❯ ")
            if prompt.lower() in {"exit", "quit"}:
                print("Exiting the weather agent. Goodbye!")
                break

            console = Console()

            with Live("", console=console, vertical_overflow="visible") as live:
                async with agent.run_stream(
                    prompt, message_history=result.all_messages() if result else None
                ) as result:
                    async for message in result.stream_output():
                        live.update(Markdown(message))

            console.log(result.usage())

        except KeyboardInterrupt:
            print("\nExiting the weather agent. Goodbye!")
            break


# -------------------------------------------------------------------------------------------------
# Agent API
# -------------------------------------------------------------------------------------------------


agent_api = FastAPI(
    title="Weather Agent API",
    description="Weather Agent REST APIs including OpenAI-compatible endpoints.",
    version="0.1.0",
)

# --------------------------------------------------------------------------------------
# Service Endpoints
# --------------------------------------------------------------------------------------


@agent_api.get("/", tags=["Service Endpoints"])
async def root():
    """Root endpoint."""
    return {"message": "Weather Agent API is running"}


@agent_api.get("/health", tags=["Service Endpoints"])
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


# --------------------------------------------------------------------------------------
# OpenAI Compatible Endpoints
# --------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Data Models
# -----------------------------------------------------------------------------


class Message(BaseModel):
    """Chat message."""

    role: Literal["system", "user", "assistant"]
    content: str

    def to_model_message(self) -> ModelMessage:
        """Convert to Pydantic AI ModelMessage."""
        match self.role:
            case "system":
                return ModelRequest(parts=[SystemPromptPart(content=self.content)])
            case "user":
                return ModelRequest(parts=[UserPromptPart(content=self.content)])
            case "assistant":
                return ModelResponse(parts=[TextPart(content=self.content)])
            case _:
                raise ValueError(f"Unknown role: {self.role}")


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str = Field(default=MODEL_NAME, description="Model name")
    messages: list[Message]
    stream: bool = Field(default=False, description="Enable streaming responses")
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)


class ChatCompletionChoice(BaseModel):
    """A single chat completion choice."""

    index: int
    message: Message
    finish_reason: str


class Usage(BaseModel):
    """Token usage information."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage


class ChatCompletionStreamChoice(BaseModel):
    """A single streaming chat completion choice."""

    index: int
    delta: dict[str, str]
    finish_reason: Optional[str] = None


class ChatCompletionStreamResponse(BaseModel):
    """OpenAI-compatible streaming chat completion response."""

    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionStreamChoice]


# -----------------------------------------------------------------------------
# Streaming Response Generator
# -----------------------------------------------------------------------------


async def generate_stream(
    request: ChatCompletionRequest,
    completion_id: str,
    created: int,
) -> AsyncGenerator[str, None]:
    """Generate streaming responses in OpenAI format."""
    # Create a new agent instance for this request
    agent = create_agent()

    # Convert request messages to Pydantic AI format
    user_message = request.messages[-1].content if request.messages else ""

    # Get message history (all messages except the last user message)
    message_history = (
        [msg.to_model_message() for msg in request.messages[:-1]]
        if len(request.messages) > 1
        else None
    )

    # Stream the response
    first_chunk = True
    async with agent.run_stream(
        user_message, message_history=message_history
    ) as result:
        async for message_chunk in result.stream_text(delta=True):
            if first_chunk:
                # First chunk with role
                chunk = ChatCompletionStreamResponse(
                    id=completion_id,
                    created=created,
                    model=request.model,
                    choices=[
                        ChatCompletionStreamChoice(
                            index=0,
                            delta={"role": "assistant", "content": message_chunk},
                            finish_reason=None,
                        )
                    ],
                )
                first_chunk = False
            else:
                # Subsequent chunks with content only
                chunk = ChatCompletionStreamResponse(
                    id=completion_id,
                    created=created,
                    model=request.model,
                    choices=[
                        ChatCompletionStreamChoice(
                            index=0,
                            delta={"content": message_chunk},
                            finish_reason=None,
                        )
                    ],
                )

            yield f"data: {chunk.model_dump_json()}\n\n"

    # Send final chunk with finish_reason
    final_chunk = ChatCompletionStreamResponse(
        id=completion_id,
        created=created,
        model=request.model,
        choices=[
            ChatCompletionStreamChoice(
                index=0,
                delta={},
                finish_reason="stop",
            )
        ],
    )
    yield f"data: {final_chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


# -----------------------------------------------------------------------------
# OpenAI-compatible Endpoints
# -----------------------------------------------------------------------------


@agent_api.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint.

    Supports both streaming and non-streaming responses.
    """
    assert request.messages, "Messages are required"

    completion_id = f"chatcmpl-{int(time.time() * 1000)}"
    created = int(time.time())

    if request.stream:
        # Return streaming response
        return StreamingResponse(
            generate_stream(request, completion_id, created),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    else:
        # Non-streaming response
        agent = create_agent()
        user_message = request.messages[-1].content
        message_history = (
            [msg.to_model_message() for msg in request.messages[:-1]]
            if len(request.messages) > 1
            else None
        )
        result = await agent.run(user_message, message_history=message_history)
        usage_info = result.usage()

        response = ChatCompletionResponse(
            id=completion_id,
            created=created,
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=Message(role="assistant", content=result.output),
                    finish_reason="stop",
                )
            ],
            usage=Usage(
                prompt_tokens=usage_info.input_tokens or 0,
                completion_tokens=usage_info.output_tokens or 0,
                total_tokens=usage_info.total_tokens or 0,
            ),
        )

        return response


# -------------------------------------------------------------------------------------------------
# Main Script Endpoints
# -------------------------------------------------------------------------------------------------


def run_api_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the FastAPI server."""
    uvicorn.run(agent_api, host=host, port=port)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--developer-console":
        asyncio.run(dev_console())
    else:
        host = os.environ.get("API_HOST", "0.0.0.0")
        port = int(os.environ.get("API_PORT", "8000"))
        run_api_server(host=host, port=port)
