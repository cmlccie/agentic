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
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
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
# OpenAI Compatible Data Models
# -------------------------------------------------------------------------------------------------


class Message(BaseModel):
    """Chat message in OpenAI format."""

    role: Literal["system", "user", "assistant"]
    content: str

    def to_model_message(self) -> ModelMessage:
        """Convert to Pydantic AI ModelMessage.

        Returns:
            ModelMessage: The converted Pydantic AI message.

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


# -------------------------------------------------------------------------------------------------
# OpenAI Compatible API
# -------------------------------------------------------------------------------------------------


class OpenAICompatibleAPI:
    """OpenAI-compatible REST API wrapper for Pydantic AI agents.

    This class wraps a Pydantic AI agent and exposes it via OpenAI-compatible
    REST API endpoints using FastAPI. It supports both streaming and non-streaming
    chat completion requests.

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

        @self.app.get("/", tags=["Service Endpoints"])
        async def root() -> dict[str, str]:
            """Root endpoint."""
            return {"message": f"{self.app.title} is running"}

        @self.app.get("/health", tags=["Service Endpoints"])
        async def health() -> dict[str, str]:
            """Health check endpoint."""
            return {"status": "healthy"}

        @self.app.post(
            "/v1/chat/completions",
            tags=["OpenAI Compatible"],
            response_model=None,
        )
        async def chat_completions(
            request: ChatCompletionRequest,
        ) -> ChatCompletionResponse | StreamingResponse:
            """OpenAI-compatible chat completions endpoint.

            Supports both streaming and non-streaming responses.

            Args:
                request: The chat completion request.

            Returns:
                ChatCompletionResponse for non-streaming requests, or
                StreamingResponse for streaming requests.
            """
            return await self._handle_chat_completions(request)

    async def _handle_chat_completions(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse | StreamingResponse:
        """Handle chat completion requests.

        Args:
            request: The chat completion request.

        Returns:
            ChatCompletionResponse for non-streaming requests, or
            StreamingResponse for streaming requests.

        Raises:
            AssertionError: If no messages are provided in the request.
        """
        assert request.messages, "Messages are required"

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

        # Non-streaming response
        user_message = request.messages[-1].content
        message_history = (
            [msg.to_model_message() for msg in request.messages[:-1]]
            if len(request.messages) > 1
            else None
        )

        result = await self.agent.run(user_message, message_history=message_history)
        usage_info = result.usage()

        return ChatCompletionResponse(
            id=completion_id,
            created=created,
            model=request.model or self.model_name,
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

    async def _generate_stream(
        self,
        request: ChatCompletionRequest,
        completion_id: str,
        created: int,
    ) -> AsyncGenerator[str, None]:
        """Generate streaming responses in OpenAI format.

        Args:
            request: The chat completion request.
            completion_id: Unique identifier for the completion.
            created: Unix timestamp of when the completion was created.

        Yields:
            Server-sent event formatted strings containing response chunks.
        """
        user_message = request.messages[-1].content if request.messages else ""

        message_history = (
            [msg.to_model_message() for msg in request.messages[:-1]]
            if len(request.messages) > 1
            else None
        )

        first_chunk = True
        async with self.agent.run_stream(
            user_message, message_history=message_history
        ) as result:
            async for message_chunk in result.stream_text(delta=True):
                if first_chunk:
                    chunk = ChatCompletionStreamResponse(
                        id=completion_id,
                        created=created,
                        model=request.model or self.model_name,
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
                    chunk = ChatCompletionStreamResponse(
                        id=completion_id,
                        created=created,
                        model=request.model or self.model_name,
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
            model=request.model or self.model_name,
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

    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        """Run the FastAPI server.

        Args:
            host: The host address to bind to.
            port: The port to listen on.
        """
        logger.info(f"Starting {self.app.title} on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)
