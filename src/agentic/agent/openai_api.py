"""OpenAI-compatible API: ``GET /v1/models`` and ``POST /v1/chat/completions``.

This is the interface most clients (Open WebUI, LibreChat, the ``openai`` SDK,
LangChain, curl) use, so it follows the OpenAI Chat Completions contract closely:

- **Non-streaming** responses are a ``chat.completion`` with the answer in
  ``message.content``, the agent's activity in ``message.reasoning_content``,
  and token ``usage``.
- **Streaming** responses are Server-Sent Events: a role chunk, then
  ``delta.reasoning_content`` chunks while the agent works (thinking, tool calls
  and results, delegated agents' activity), then the answer in
  ``delta.content``, a ``finish_reason: "stop"`` chunk, an optional usage chunk
  (``stream_options.include_usage``), and ``data: [DONE]``. SSE comment lines
  keep idle connections open while tools run.
- **Errors** use the OpenAI error shape ``{"error": {"message", "type",
  "code"}}``. A failure after streaming has started is sent as an ``error``
  event (the ``openai`` SDK raises ``APIError``) — a failed run never looks
  like an empty answer. Agent failures carry ``x-should-retry: false`` so SDK
  retries don't re-run tools with side effects.

The conversation is stateless: clients send the full history each time.
``reasoning_content`` (and Open WebUI's ``<think>`` blocks) in assistant
messages is dropped so the agent's activity isn't fed back to the model.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic_ai.messages import (
    BinaryContent,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserContent,
    UserPromptPart,
)
from pydantic_ai.settings import ModelSettings

from .runtime import Snapshot
from .streaming import (
    HEARTBEAT,
    Activity,
    Answer,
    activities,
    render,
    visible,
    with_heartbeat,
)

log = logging.getLogger(__name__)

_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_NO_RETRY = {"x-should-retry": "false"}


def failure_message(exc: BaseException) -> str:
    """Client-facing description of a failed run (details stay in the server log)."""
    return f"The agent failed ({type(exc).__name__}); see the agent's logs for details."


class OpenAIError(Exception):
    """An error returned to the client in the OpenAI error format."""

    def __init__(
        self,
        message: str,
        status_code: int = 400,
        type: str = "invalid_request_error",
        code: str | None = None,
        param: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.type = type
        self.code = code
        self.param = param
        self.headers = headers or {}

    def body(self) -> dict[str, Any]:
        return {
            "error": {
                "message": self.message,
                "type": self.type,
                "param": self.param,
                "code": self.code,
            }
        }

    def response(self) -> JSONResponse:
        return JSONResponse(
            self.body(), status_code=self.status_code, headers=self.headers
        )


# --------------------------------------------------------------------------------------
# Request model
# --------------------------------------------------------------------------------------


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra="allow")
    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    """The subset of the Chat Completions request this server acts on.

    Unknown fields are accepted and ignored, as the OpenAI API does for fields a
    model doesn't support. Client-side ``tools`` are ignored: the agent uses its
    own configured tools.
    """

    model_config = ConfigDict(extra="allow")

    model: str = ""
    messages: list[dict[str, Any]] = Field(min_length=1)
    stream: bool = False
    stream_options: StreamOptions | None = None
    n: int = 1
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    seed: int | None = None
    stop: str | list[str] | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None

    def model_settings(self) -> ModelSettings | None:
        """Sampling parameters to apply on top of the agent's own settings."""
        settings: dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_completion_tokens or self.max_tokens,
            "seed": self.seed,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "stop_sequences": [self.stop] if isinstance(self.stop, str) else self.stop,
        }
        settings = {k: v for k, v in settings.items() if v is not None}
        return ModelSettings(**settings) if settings else None


# --------------------------------------------------------------------------------------
# OpenAI messages → Pydantic AI prompt + history (pure)
# --------------------------------------------------------------------------------------


def _text_of(content: Any) -> str:
    """Plain text of a message's content (string or list of content parts)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            p.get("text", "")
            for p in content
            if isinstance(p, dict) and p.get("type") in ("text", "input_text")
        )
    return str(content)


def _user_content(content: Any) -> str | list[UserContent]:
    """Convert user message content, keeping images (URLs and data URIs)."""
    if not isinstance(content, list):
        return _text_of(content)
    parts: list[UserContent] = []
    for p in content:
        if not isinstance(p, dict):
            continue
        kind = p.get("type")
        if kind in ("text", "input_text"):
            parts.append(p.get("text", ""))
        elif kind in ("image_url", "input_image"):
            image = p.get("image_url")
            url = image.get("url") if isinstance(image, dict) else image
            if not isinstance(url, str) or not url:
                raise OpenAIError("image_url content part has no url", param="messages")
            parts.append(
                BinaryContent.from_data_uri(url)
                if url.startswith("data:")
                else ImageUrl(url)
            )
        else:
            raise OpenAIError(
                f"unsupported content part type '{kind}'", param="messages"
            )
    if all(isinstance(p, str) for p in parts):
        return "\n".join(parts)  # type: ignore[arg-type]
    return parts


def _strip_reasoning(text: str) -> str:
    return _THINK_BLOCK.sub("", text).strip()


def to_pydantic_ai(
    messages: list[dict[str, Any]],
) -> tuple[str | list[UserContent], list[ModelMessage]]:
    """Split OpenAI chat messages into the new prompt and the prior history.

    The last message must come from the user; everything before it becomes
    Pydantic AI message history (system/developer → system prompt, assistant
    text and tool calls → model responses, tool results → tool returns).

    Raises:
        OpenAIError: For malformed or unsupported messages.
    """
    *earlier, last = messages
    if last.get("role") != "user":
        raise OpenAIError("the last message must have role 'user'", param="messages")

    history: list[ModelMessage] = []
    tool_names: dict[str, str] = {}

    def add_request(part: ModelRequestPart) -> None:
        if history and isinstance(history[-1], ModelRequest):
            history[-1] = ModelRequest(parts=[*history[-1].parts, part])
        else:
            history.append(ModelRequest(parts=[part]))

    def add_response(parts: list[ModelResponsePart]) -> None:
        if not parts:
            return
        if history and isinstance(history[-1], ModelResponse):
            history[-1] = ModelResponse(parts=[*history[-1].parts, *parts])
        else:
            history.append(ModelResponse(parts=parts))

    for index, message in enumerate(earlier):
        role = message.get("role")
        content = message.get("content")
        match role:
            case "system" | "developer":
                add_request(SystemPromptPart(content=_text_of(content)))
            case "user":
                add_request(UserPromptPart(content=_user_content(content)))
            case "assistant":
                parts: list[ModelResponsePart] = []
                if text := _strip_reasoning(_text_of(content)):
                    parts.append(TextPart(content=text))
                for call in message.get("tool_calls") or []:
                    function = call.get("function") or {}
                    call_id = call.get("id") or f"call_{index}_{len(parts)}"
                    tool_names[call_id] = function.get("name", "tool")
                    parts.append(
                        ToolCallPart(
                            tool_name=function.get("name", "tool"),
                            args=function.get("arguments") or "{}",
                            tool_call_id=call_id,
                        )
                    )
                add_response(parts)
            case "tool" | "function":
                call_id = message.get("tool_call_id") or ""
                add_request(
                    ToolReturnPart(
                        tool_name=message.get("name")
                        or tool_names.get(call_id, "tool"),
                        content=_text_of(content),
                        tool_call_id=call_id,
                    )
                )
            case _:
                raise OpenAIError(
                    f"unsupported message role '{role}'", param="messages"
                )

    return _user_content(last.get("content")), history


# --------------------------------------------------------------------------------------
# Responses
# --------------------------------------------------------------------------------------


def _usage(answer: Answer) -> dict[str, Any]:
    usage = answer.usage
    body: dict[str, Any] = {
        "prompt_tokens": usage.input_tokens,
        "completion_tokens": usage.output_tokens,
        "total_tokens": usage.input_tokens + usage.output_tokens,
    }
    if usage.cache_read_tokens:
        body["prompt_tokens_details"] = {"cached_tokens": usage.cache_read_tokens}
    return body


class _Reasoning:
    """Renders activity as reasoning-channel text, attributing delegated thinking."""

    def __init__(self) -> None:
        self._thinking_source: tuple[str, ...] | None = None

    def text(self, item: Activity) -> str:
        if item.kind == "thinking":
            prefix = ""
            if item.source and item.source != self._thinking_source:
                prefix = f"\n[{'/'.join(item.source)}] 💭 "
            elif not item.source and self._thinking_source not in (None, ()):
                prefix = "\n"
            self._thinking_source = item.source
            return prefix + item.text
        self._thinking_source = None
        return f"\n{render(item)}\n"


def _chunk(
    completion_id: str,
    created: int,
    model: str,
    delta: dict[str, Any],
    finish_reason: str | None = None,
) -> str:
    body = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(body, ensure_ascii=False)}\n\n"


async def _events(
    snapshot: Snapshot, request: ChatCompletionRequest
) -> AsyncIterator[Activity | Answer]:
    prompt, history = to_pydantic_ai(request.messages)
    async with snapshot.agent.run_stream_events(
        prompt,
        message_history=history or None,
        model_settings=request.model_settings(),
    ) as events:
        async for item in activities(events, snapshot.server.streaming):
            yield item


async def complete(
    snapshot: Snapshot, request: ChatCompletionRequest
) -> dict[str, Any]:
    """Run the agent to completion and build a ``chat.completion`` body."""
    reasoning = _Reasoning()
    trace: list[str] = []
    answer: Answer | None = None
    async for item in _events(snapshot, request):
        if isinstance(item, Answer):
            answer = item
        elif visible(item, snapshot.server.streaming):
            trace.append(reasoning.text(item))
    if answer is None:
        raise OpenAIError(
            "the agent finished without an answer",
            500,
            "server_error",
            "agent_error",
            headers=_NO_RETRY,
        )
    message: dict[str, Any] = {"role": "assistant", "content": answer.text}
    if reasoning_text := "".join(trace).strip():
        message["reasoning_content"] = reasoning_text
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": snapshot.model_name,
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": _usage(answer),
    }


async def stream(
    snapshot: Snapshot, request: ChatCompletionRequest
) -> AsyncIterator[str]:
    """Run the agent and yield the SSE body of a streamed chat completion."""
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    model = snapshot.model_name
    cfg = snapshot.server.streaming
    reasoning = _Reasoning()
    answer: Answer | None = None

    yield _chunk(completion_id, created, model, {"role": "assistant", "content": ""})
    try:
        async for item in with_heartbeat(
            _events(snapshot, request), cfg.heartbeat_seconds
        ):
            if item is HEARTBEAT:
                yield ": keep-alive\n\n"
            elif isinstance(item, Answer):
                answer = item
            elif visible(item, cfg):
                yield _chunk(
                    completion_id,
                    created,
                    model,
                    {"reasoning_content": reasoning.text(item)},
                )
    except OpenAIError as exc:
        yield f"data: {json.dumps(exc.body())}\n\n"
        return
    except Exception as exc:
        log.exception("chat completion stream failed")
        error = OpenAIError(
            failure_message(exc),
            500,
            "server_error",
            "agent_error",
        )
        yield f"data: {json.dumps(error.body())}\n\n"
        return

    if answer is None:
        error = OpenAIError(
            "the agent finished without an answer", 500, "server_error", "agent_error"
        )
        yield f"data: {json.dumps(error.body())}\n\n"
        return
    if answer.text:
        yield _chunk(completion_id, created, model, {"content": answer.text})
    yield _chunk(completion_id, created, model, {}, finish_reason="stop")
    if request.stream_options and request.stream_options.include_usage:
        usage = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [],
            "usage": _usage(answer),
        }
        yield f"data: {json.dumps(usage)}\n\n"
    yield "data: [DONE]\n\n"


# --------------------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------------------


def build_openai_router(current: Callable[[], Snapshot]) -> APIRouter:
    """Build the OpenAI-compatible routes; each request uses the current snapshot."""
    router = APIRouter(tags=["OpenAI compatible"])

    @router.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        snapshot = current()
        return {
            "object": "list",
            "data": [
                {
                    "id": snapshot.model_name,
                    "object": "model",
                    "created": int(snapshot.loaded_at),
                    "owned_by": "agentic",
                }
            ],
        }

    @router.get("/v1/models/{model_id:path}", response_model=None)
    async def get_model(model_id: str) -> dict[str, Any] | JSONResponse:
        snapshot = current()
        if model_id != snapshot.model_name:
            return OpenAIError(
                f"The model '{model_id}' does not exist", 404, code="model_not_found"
            ).response()
        return {
            "id": snapshot.model_name,
            "object": "model",
            "created": int(snapshot.loaded_at),
            "owned_by": "agentic",
        }

    @router.post("/v1/chat/completions", response_model=None)
    async def chat_completions(request: Request) -> JSONResponse | StreamingResponse:
        try:
            body = await request.json()
        except ValueError:
            return OpenAIError("the request body is not valid JSON").response()
        try:
            chat = ChatCompletionRequest.model_validate(body)
        except ValidationError as exc:
            first = exc.errors()[0]
            param = ".".join(str(p) for p in first.get("loc", ()))
            return OpenAIError(
                f"invalid '{param}': {first.get('msg')}", param=param
            ).response()
        if chat.n != 1:
            return OpenAIError("only n=1 is supported", param="n").response()

        snapshot = current()
        try:
            to_pydantic_ai(chat.messages)  # validate before committing to a stream
        except OpenAIError as exc:
            return exc.response()

        if chat.stream:
            return StreamingResponse(
                stream(snapshot, chat),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        try:
            return JSONResponse(await complete(snapshot, chat))
        except OpenAIError as exc:
            return exc.response()
        except Exception as exc:
            log.exception("chat completion failed")
            return OpenAIError(
                failure_message(exc),
                500,
                "server_error",
                "agent_error",
                headers=_NO_RETRY,
            ).response()

    return router
