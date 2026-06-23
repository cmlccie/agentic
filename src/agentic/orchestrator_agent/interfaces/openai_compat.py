from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING, Any, AsyncGenerator

from fastapi import APIRouter
from fastapi_openai_compat import create_chat_completion_router
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ..graph import final_ai_text

if TYPE_CHECKING:
    from ..lifespan import AppState

_KEEPALIVE_INTERVAL = 30.0


def _content_text(content: Any) -> str:
    if isinstance(content, list):
        return " ".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return content or ""


def _messages_to_langchain(messages: list[dict[str, Any]]) -> list:
    """Convert OpenAI-format message dicts to LangChain messages."""
    converted = []
    for msg in messages:
        role = msg.get("role", "user")
        content = _content_text(msg.get("content"))
        match role:
            case "system":
                converted.append(SystemMessage(content=content))
            case "assistant":
                converted.append(AIMessage(content=content))
            case _:
                converted.append(HumanMessage(content=content))
    return converted


def build_openai_router(app_state: AppState, model_name: str) -> APIRouter:
    """Build an OpenAI-compatible router backed by the LangGraph supervisor.

    Closures read the graph from app_state at call time, not from capture,
    making the handler reload-safe — after a reload, new requests use the
    rebuilt graph. Each request runs in its own checkpointer thread so the full
    OpenAI message history is the sole source of conversation state.
    """

    async def _run(messages: list[dict[str, Any]]) -> str:
        graph = app_state.graph
        if graph is None:
            return "(orchestrator is reloading, please retry)"
        lc_messages = _messages_to_langchain(messages)
        result = await graph.ainvoke(
            {"messages": lc_messages},
            config={"configurable": {"thread_id": uuid.uuid4().hex}},
        )
        return final_ai_text(result)

    async def run_completion(
        model: str,
        messages: list[dict[str, Any]],
        body: dict[str, Any],
    ) -> str | AsyncGenerator[str, None]:
        if not messages:
            return ""
        if body.get("stream", False):
            return _stream_response(messages)
        return await _run(messages)

    async def _stream_response(
        messages: list[dict[str, Any]],
    ) -> AsyncGenerator[str, None]:
        # The supervisor may fan out to several long-running downstream agents
        # before producing its answer. Run to completion and emit the result as
        # a single chunk, sending empty keepalive chunks every
        # _KEEPALIVE_INTERVAL seconds to hold the connection open through the
        # multi-agent orchestration (prevents proxy idle timeouts).
        queue: asyncio.Queue[str | None] = asyncio.Queue()

        async def _produce() -> None:
            try:
                await queue.put(await _run(messages))
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
                    yield ""  # keepalive during orchestration
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
