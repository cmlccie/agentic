from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, AsyncGenerator

from fastapi import APIRouter
from fastapi_openai_compat import create_chat_completion_router
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)

if TYPE_CHECKING:
    from ..lifespan import AppState

_KEEPALIVE_INTERVAL = 30.0


def _messages_to_history(messages: list[dict[str, Any]]) -> list:
    """Convert OpenAI-format message dicts to pydantic_ai ModelMessage history."""
    history = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content") or ""
        if isinstance(content, list):
            content = " ".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        match role:
            case "system":
                history.append(ModelRequest(parts=[SystemPromptPart(content=content)]))
            case "user":
                history.append(ModelRequest(parts=[UserPromptPart(content=content)]))
            case "assistant":
                history.append(ModelResponse(parts=[TextPart(content=content)]))
    return history


def build_openai_router(app_state: AppState, model_name: str) -> APIRouter:
    """Build an OpenAI-compatible router.

    Closures read agent from app_state at call time, not from capture, making
    the handler reload-safe — after a reload, new requests use the new agent.
    """

    async def run_completion(
        model: str,
        messages: list[dict[str, Any]],
        body: dict[str, Any],
    ) -> str | AsyncGenerator[str, None]:
        agent = app_state.agent
        stream: bool = body.get("stream", False)

        if not messages:
            return ""

        history = _messages_to_history(messages[:-1])
        last = messages[-1]
        user_prompt = last.get("content") or ""
        if isinstance(user_prompt, list):
            user_prompt = " ".join(
                part.get("text", "") for part in user_prompt if isinstance(part, dict)
            )

        if stream:
            return _stream_response(agent, user_prompt, history or None)
        else:
            result = await agent.run(user_prompt, message_history=history or None)
            return result.output

    async def _stream_response(
        agent, user_prompt: str, history
    ) -> AsyncGenerator[str, None]:  # type: ignore[return]
        # pydantic_ai 1.97.0 emits FinalResultEvent on the first TextPart in streaming
        # mode, which causes thinking-text-first models (qwen3, etc.) to short-circuit
        # before tool calls are executed. Use agent.run() (non-streaming) to ensure the
        # full tool-calling pipeline completes, then yield the result as a single chunk.
        # Empty keepalive chunks are sent every _KEEPALIVE_INTERVAL seconds to prevent
        # Envoy's upstream idle timeout from closing the connection mid-run.
        queue: asyncio.Queue[str | None] = asyncio.Queue()

        async def _produce() -> None:
            try:
                result = await agent.run(user_prompt, message_history=history)
                await queue.put(result.output)
            except Exception:
                pass
            finally:
                await queue.put(None)

        task = asyncio.create_task(_produce())
        try:
            while True:
                try:
                    item = await asyncio.wait_for(
                        queue.get(), timeout=_KEEPALIVE_INTERVAL
                    )
                    if item is None:
                        break
                    yield item
                except asyncio.TimeoutError:
                    yield ""  # keep connection alive during tool calls / LLM thinking
        finally:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    return create_chat_completion_router(
        list_models=lambda: [model_name],
        run_completion=run_completion,
        tags=["OpenAI Compatible"],
    )
