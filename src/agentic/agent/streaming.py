"""Agent activity: one event model shared by every interface.

While an agent works it produces a stream of Pydantic AI events: thinking
deltas, text, tool calls and results, and (for orchestrators) activity relayed
from remote A2A agents. `activities` reduces that stream to a small, stable
vocabulary:

    pydantic-ai events ──► activities() ──► Activity … Activity, Answer
                                             │                  │
            OpenAI API:        reasoning_content deltas      content
            A2A server:        WORKING status updates        Message / artifact

- `Activity` is progress the caller may display (thinking, a tool call, a tool
  result, a note the model wrote before calling a tool, a worker's activity).
- `Answer` is the final response, emitted exactly once at the end of a run.

The adapter always yields every activity; `visible` applies the configured
verbosity at render time, so interface logic (e.g. "promote an A2A exchange to a
task on the first tool call") works regardless of what is shown to the caller.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
from collections.abc import AsyncIterable, AsyncIterator, Iterable
from dataclasses import dataclass, field, replace
from typing import Any, Literal, TypeVar

from pydantic_ai import (
    AgentRunResultEvent,
    CapabilityEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
)
from pydantic_ai.messages import ModelMessage, RetryPromptPart
from pydantic_ai.usage import RunUsage

from .config import StreamingConfig

ActivityKind = Literal[
    "thinking", "note", "tool_call", "tool_result", "status", "error"
]

#: URI of the A2A extension that marks status-update messages as agent activity
#: (declared on the agent card; see `agentic.agent.a2a_server`).
ACTIVITY_EXTENSION = "urn:agentic:a2a:activity:v1"


@dataclass(frozen=True)
class Activity:
    """One unit of agent progress.

    Attributes:
        kind: What happened (see `ActivityKind`).
        text: Human-readable content (thinking text, formatted arguments, ...).
        source: Attribution path for delegated work, e.g. ``("weather",)`` for
            activity relayed from the ``weather`` agent; empty for this agent.
        tool: Tool name, for ``tool_call`` / ``tool_result``.
        call_id: Tool call id, pairing a call with its result.
    """

    kind: ActivityKind
    text: str = ""
    source: tuple[str, ...] = ()
    tool: str | None = None
    call_id: str | None = None


@dataclass(frozen=True)
class Answer:
    """The final result of a run.

    Attributes:
        text: The agent's final answer.
        usage: Token usage for the whole run.
        messages: The complete message history after the run (used by the A2A
            server to continue the conversation on the next turn).
    """

    text: str
    usage: RunUsage = field(default_factory=RunUsage)
    messages: list[ModelMessage] = field(default_factory=list)


@dataclass(kw_only=True, repr=False)
class ActivityEvent(CapabilityEvent, namespace="agentic"):
    """A Pydantic AI event carrying an `Activity` from inside a tool call.

    Emitted by the `A2AAgent` capability to relay a remote agent's activity
    into this agent's event stream while the delegation tool runs.
    """

    activity: Activity


# --------------------------------------------------------------------------------------
# Formatting helpers (pure)
# --------------------------------------------------------------------------------------


def truncate(text: str, limit: int | None) -> str:
    """Shorten ``text`` to ``limit`` characters, noting how much was cut."""
    if limit is None or len(text) <= limit:
        return text
    return f"{text[:limit]}… (+{len(text) - limit} chars)"


def redact(value: Any, keys: Iterable[str]) -> Any:
    """Replace values whose key contains any of ``keys`` (case-insensitive) with ``***``."""
    words = [k for k in keys if k]
    if not words:
        return value
    pattern = re.compile("|".join(map(re.escape, words)), re.IGNORECASE)

    def walk(v: Any) -> Any:
        if isinstance(v, dict):
            return {
                k: "***" if isinstance(k, str) and pattern.search(k) else walk(x)
                for k, x in v.items()
            }
        if isinstance(v, list | tuple):
            return [walk(x) for x in v]
        return v

    return walk(value)


def _limit(cfg: StreamingConfig, chars: int) -> int | None:
    return None if cfg.activity == "trace" else chars


def format_args(args: str | dict[str, Any] | None, cfg: StreamingConfig) -> str:
    """Render tool-call arguments as compact JSON, redacted and (in summary) truncated."""
    if isinstance(args, str):
        try:
            args = json.loads(args or "{}")
        except ValueError:
            return truncate(args, _limit(cfg, cfg.max_args_chars))
    text = json.dumps(redact(args or {}, cfg.redact_keys), default=str)
    return truncate(text, _limit(cfg, cfg.max_args_chars))


def format_result(value: Any, cfg: StreamingConfig) -> str:
    """Render a tool result, redacted and (in summary) truncated."""
    if not isinstance(value, str):
        value = json.dumps(redact(value, cfg.redact_keys), default=str)
    return truncate(value, _limit(cfg, cfg.max_result_chars))


def visible(item: Activity, cfg: StreamingConfig) -> bool:
    """Whether an activity should be shown to the caller under ``cfg``."""
    if cfg.activity == "off":
        return item.kind == "error"
    if item.kind == "thinking":
        return cfg.thinking
    return True


_ICONS = {
    "thinking": "💭",
    "note": "💬",
    "tool_call": "→",
    "tool_result": "←",
    "status": "·",
    "error": "✖",
}


def render(item: Activity) -> str:
    """Render an activity as one line of human-readable text."""
    who = f"[{'/'.join(item.source)}] " if item.source else ""
    body = {
        "tool_call": f"{item.text}",
        "tool_result": f"{item.tool}: {item.text}" if item.tool else item.text,
    }.get(item.kind, item.text)
    return f"{who}{_ICONS.get(item.kind, '·')} {body}"


# --------------------------------------------------------------------------------------
# Pydantic AI events → Activity / Answer
# --------------------------------------------------------------------------------------


def _output_text(output: Any) -> str:
    return output if isinstance(output, str) else json.dumps(output, default=str)


async def activities(
    events: AsyncIterable[Any], cfg: StreamingConfig
) -> AsyncIterator[Activity | Answer]:
    """Reduce a Pydantic AI event stream to `Activity` items and a final `Answer`.

    Text is buffered per model response: if the model goes on to call a tool,
    the text was a preamble ("Let me check the forecast") and is yielded as a
    ``note``. The final answer comes from the run result, which is
    authoritative, so streamed text is never duplicated.
    """
    pending_text: list[str] = []
    async for event in events:
        match event:
            case PartStartEvent(part=ThinkingPart(content=text)) if text:
                yield Activity("thinking", text)
            case PartDeltaEvent(delta=ThinkingPartDelta(content_delta=text)) if text:
                yield Activity("thinking", text)
            case PartStartEvent(part=TextPart(content=text)):
                pending_text.append(text)
            case PartDeltaEvent(delta=TextPartDelta(content_delta=text)):
                pending_text.append(text)
            case FunctionToolCallEvent(part=part):
                if note := "".join(pending_text).strip():
                    yield Activity("note", note)
                pending_text = []
                yield Activity(
                    "tool_call",
                    f"{part.tool_name}({format_args(part.args, cfg)})",
                    tool=part.tool_name,
                    call_id=part.tool_call_id,
                )
            case FunctionToolResultEvent(part=RetryPromptPart() as part):
                yield Activity(
                    "tool_result",
                    format_result(f"retry: {part.model_response()}", cfg),
                    tool=part.tool_name,
                    call_id=part.tool_call_id,
                )
            case FunctionToolResultEvent(part=part):
                yield Activity(
                    "tool_result",
                    format_result(part.content, cfg),
                    tool=part.tool_name,
                    call_id=part.tool_call_id,
                )
            case ActivityEvent(activity=item):
                yield item
            case AgentRunResultEvent(result=result):
                yield Answer(
                    text=_output_text(result.output),
                    usage=result.usage,
                    messages=result.all_messages(),
                )


def prefixed(item: Activity, name: str) -> Activity:
    """Attribute an activity relayed from the remote agent ``name``."""
    return replace(item, source=(name, *item.source))


# --------------------------------------------------------------------------------------
# Heartbeats
# --------------------------------------------------------------------------------------

T = TypeVar("T")


class Heartbeat:
    """Sentinel yielded by `with_heartbeat` when the source has been quiet."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "HEARTBEAT"


HEARTBEAT = Heartbeat()


async def with_heartbeat(
    source: AsyncIterator[T], interval: float
) -> AsyncIterator[T | Heartbeat]:
    """Yield items from ``source``, plus `HEARTBEAT` after each ``interval`` of silence.

    Long tool calls or slow models can leave a streaming response idle for
    minutes; proxies (Envoy, nginx, cloud load balancers) close idle
    connections. Heartbeats keep the connection alive and give the server a
    chance to notice a disconnected client.
    """
    iterator = aiter(source)
    pending: asyncio.Future[T] | None = None
    try:
        while True:
            if pending is None:
                pending = asyncio.ensure_future(anext(iterator))
            done, _ = await asyncio.wait({pending}, timeout=interval)
            if not done:
                yield HEARTBEAT
                continue
            finished, pending = pending, None
            try:
                item = finished.result()
            except StopAsyncIteration:
                return
            yield item
    finally:
        if pending is not None:
            pending.cancel()
            with contextlib.suppress(BaseException):
                await pending
        aclose = getattr(iterator, "aclose", None)
        if aclose is not None:
            await aclose()
