"""OpenAI compatible API wrapper for Pydantic AI agents.

This module provides a class that wraps a Pydantic AI agent and exposes it via
OpenAI-compatible REST API endpoints using FastAPI and uvicorn.

Example:
    from pydantic_ai import Agent
    from agentic.openai.compatible_api import OpenAICompatibleAPI

    agent = Agent(model=model, system_prompt="You are a helpful assistant.")
    api = OpenAICompatibleAPI(
        agent=agent,
        title="My Agent API",
        description="My agent exposed via OpenAI-compatible API.",
    )
    api.run(host="0.0.0.0", port=8000)
"""

import logging
import time
from typing import Any, AsyncGenerator, Literal, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------------------------------
# Request Models
# -------------------------------------------------------------------------------------------------


class Message(BaseModel):
    """Chat message in OpenAI request format."""

    role: Literal["system", "user", "assistant"]
    content: str

    def to_model_message(self) -> ModelMessage:
        """Convert to Pydantic AI ModelMessage.

        Raises:
            ValueError: If the role is unknown.
        """
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

    model: str = Field(default="default", description="Model name")
    messages: list[Message]
    stream: bool = Field(default=False, description="Enable streaming responses")
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)


# -------------------------------------------------------------------------------------------------
# OpenAI Compatible API
# -------------------------------------------------------------------------------------------------


class OpenAICompatibleAPI:
    """OpenAI-compatible REST API wrapper for Pydantic AI agents.

    Wraps a Pydantic AI agent and exposes it via OpenAI-compatible REST API
    endpoints using FastAPI. Supports both streaming and non-streaming chat
    completion requests.

    Attributes:
        agent: The Pydantic AI agent to wrap.
        app: The FastAPI application instance.
        model_name: The model name to report in API responses.
    """

    def __init__(
        self,
        agent: Agent[Any, str],
        title: str = "Agent API",
        description: str = "OpenAI-compatible API for a Pydantic AI agent.",
        version: str = "0.1.0",
        model_name: str = "default",
    ) -> None:
        """Initialize the OpenAI-compatible API wrapper.

        Args:
            agent: The Pydantic AI agent to wrap. Must have str output type.
            title: The title for the FastAPI application.
            description: The description for the FastAPI application.
            version: The version string for the FastAPI application.
            model_name: The model name to report in API responses.
        """
        self.agent = agent
        self.model_name = model_name

        self.app = FastAPI(
            title=title,
            description=description,
            version=version,
        )

        self._register_routes()

    def _register_routes(self) -> None:
        """Register API routes on the FastAPI application."""

        @self.app.get("/status", tags=["Service Endpoints"])
        async def status() -> dict[str, str]:
            return {"message": f"{self.app.title} is running"}

        @self.app.get("/health", tags=["Service Endpoints"])
        async def health() -> dict[str, str]:
            return {"status": "healthy"}

        @self.app.post(
            "/v1/chat/completions",
            tags=["OpenAI Compatible"],
            response_model=None,
        )
        async def chat_completions(
            request: ChatCompletionRequest,
        ) -> ChatCompletion | StreamingResponse:
            return await self._handle_chat_completions(request)

    async def _handle_chat_completions(
        self, request: ChatCompletionRequest
    ) -> ChatCompletion | StreamingResponse:
        """Handle chat completion requests.

        Raises:
            AssertionError: If no messages are provided in the request.
        """
        if not request.messages:
            raise HTTPException(status_code=422, detail="Messages are required")

        completion_id = f"chatcmpl-{int(time.time() * 1000)}"
        created = int(time.time())

        if request.stream:
            return StreamingResponse(
                self._generate_stream(request, completion_id, created),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        user_message = request.messages[-1].content
        message_history = (
            [msg.to_model_message() for msg in request.messages[:-1]]
            if len(request.messages) > 1
            else None
        )

        result = await self.agent.run(user_message, message_history=message_history)
        usage_info = result.usage()

        return ChatCompletion(
            id=completion_id,
            created=created,
            model=request.model or self.model_name,
            object="chat.completion",
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content=result.output,
                        refusal=None,
                    ),
                    finish_reason="stop",
                    logprobs=None,
                )
            ],
            service_tier=None,
            system_fingerprint=None,
            usage=CompletionUsage(
                prompt_tokens=usage_info.input_tokens or 0,
                completion_tokens=usage_info.output_tokens or 0,
                total_tokens=usage_info.total_tokens or 0,
            ),
        )

    async def _generate_stream(
        self,
        request: ChatCompletionRequest,
        completion_id: str,
        created: int,
    ) -> AsyncGenerator[str, None]:
        """Generate streaming responses in OpenAI SSE format.

        Yields:
            Server-sent event formatted strings containing response chunks.
        """
        user_message = request.messages[-1].content if request.messages else ""
        message_history = (
            [msg.to_model_message() for msg in request.messages[:-1]]
            if len(request.messages) > 1
            else None
        )

        model = request.model or self.model_name
        first_chunk = True

        async with self.agent.run_stream(
            user_message, message_history=message_history
        ) as result:
            async for message_chunk in result.stream_text(delta=True):
                if first_chunk:
                    chunk = ChatCompletionChunk(
                        id=completion_id,
                        created=created,
                        model=model,
                        object="chat.completion.chunk",
                        choices=[
                            ChunkChoice(
                                index=0,
                                delta=ChoiceDelta(
                                    role="assistant",
                                    content=message_chunk,
                                    refusal=None,
                                ),
                                finish_reason=None,
                                logprobs=None,
                            )
                        ],
                        service_tier=None,
                        system_fingerprint=None,
                    )
                    first_chunk = False
                else:
                    chunk = ChatCompletionChunk(
                        id=completion_id,
                        created=created,
                        model=model,
                        object="chat.completion.chunk",
                        choices=[
                            ChunkChoice(
                                index=0,
                                delta=ChoiceDelta(
                                    content=message_chunk,
                                    refusal=None,
                                ),
                                finish_reason=None,
                                logprobs=None,
                            )
                        ],
                        service_tier=None,
                        system_fingerprint=None,
                    )

                yield f"data: {chunk.model_dump_json()}\n\n"

        final_chunk = ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=model,
            object="chat.completion.chunk",
            choices=[
                ChunkChoice(
                    index=0,
                    delta=ChoiceDelta(content=None, refusal=None),
                    finish_reason="stop",
                    logprobs=None,
                )
            ],
            service_tier=None,
            system_fingerprint=None,
        )
        yield f"data: {final_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        """Run the FastAPI server."""
        logger.info(f"Starting {self.app.title} on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)
