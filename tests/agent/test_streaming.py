"""Tests for the activity model and its helpers."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest
from pydantic_ai import Agent

from agentic.agent.a2a_wire import activity_from_message, activity_metadata, parts_text
from agentic.agent.config import StreamingConfig
from agentic.agent.streaming import (
    HEARTBEAT,
    Activity,
    Answer,
    activities,
    format_args,
    format_result,
    prefixed,
    redact,
    render,
    truncate,
    visible,
    with_heartbeat,
)

from .conftest import LocalTools, scripted_model

pytestmark = pytest.mark.anyio


def test_truncate() -> None:
    assert truncate("abc", 5) == "abc"
    assert truncate("abcdef", 3) == "abc… (+3 chars)"
    assert truncate("abcdef", None) == "abcdef"


def test_redact_nested_and_case_insensitive() -> None:
    data = {"API_KEY": "x", "nested": [{"password": "p", "city": "Oslo"}], "n": 1}
    assert redact(data, ["api_key", "password"]) == {
        "API_KEY": "***",
        "nested": [{"password": "***", "city": "Oslo"}],
        "n": 1,
    }
    assert redact(data, []) is data


def test_format_args_and_results_follow_the_verbosity() -> None:
    summary = StreamingConfig(max_args_chars=10, max_result_chars=5)
    trace = StreamingConfig(activity="trace", max_args_chars=10, max_result_chars=5)
    args = '{"city": "Paris", "token": "t"}'
    assert format_args(args, summary).startswith('{"city": "')
    assert "(+" in format_args(args, summary)
    assert format_args(args, trace) == '{"city": "Paris", "token": "***"}'
    assert format_args("not json", trace) == "not json"
    assert format_result({"a": 123456}, summary) == '{"a":… (+8 chars)'
    assert format_result("x" * 10, trace) == "x" * 10


def test_visibility() -> None:
    thinking, call, error = (
        Activity("thinking", "t"),
        Activity("tool_call", "c"),
        Activity("error", "e"),
    )
    assert all(visible(a, StreamingConfig()) for a in (thinking, call, error))
    assert not visible(thinking, StreamingConfig(thinking=False))
    off = StreamingConfig(activity="off")
    assert [visible(a, off) for a in (thinking, call, error)] == [False, False, True]


def test_render_and_attribution() -> None:
    item = prefixed(
        prefixed(Activity("tool_call", "f({})", tool="f"), "inner"), "outer"
    )
    assert item.source == ("outer", "inner")
    assert render(item) == "[outer/inner] → f({})"
    assert render(Activity("tool_result", "ok", tool="f")) == "← f: ok"
    assert render(Activity("note", "hmm")) == "💬 hmm"


def test_activity_round_trips_through_a2a_metadata() -> None:
    from a2a.helpers import new_text_message

    item = Activity("tool_result", "raw result", ("w",), tool="f", call_id="c1")
    message = new_text_message(render(item))
    message.metadata.update(activity_metadata(item))
    assert activity_from_message(message) == item
    assert activity_from_message(new_text_message("plain status")) is None
    assert parts_text(message.parts) == render(item)


async def test_activities_from_a_real_run() -> None:
    model = scripted_model(
        thinking="plan",
        note="Let me look.",
        tool="forecast",
        args={"city": "Oslo"},
        answer="Sunny",
    )
    agent = Agent(model, capabilities=[LocalTools()])
    async with agent.run_stream_events("weather?") as events:
        items = [item async for item in activities(events, StreamingConfig())]
    assert [i.kind if isinstance(i, Activity) else "answer" for i in items] == [
        "thinking",
        "note",
        "tool_call",
        "tool_result",
        "answer",
    ]
    answer = items[-1]
    assert isinstance(answer, Answer)
    assert answer.text == "Sunny"
    assert answer.usage.requests == 2
    assert len(answer.messages) == 4
    assert items[2].tool == "forecast" and items[2].call_id == items[3].call_id


async def test_heartbeat_fills_silences_and_preserves_order() -> None:
    async def source() -> AsyncIterator[int]:
        yield 1
        await asyncio.sleep(0.25)
        yield 2

    items = [item async for item in with_heartbeat(source(), 0.1)]
    assert items[0] == 1 and items[-1] == 2
    assert items.count(HEARTBEAT) >= 1
    assert set(items[1:-1]) == {HEARTBEAT}


async def test_heartbeat_propagates_errors_and_closes_the_source() -> None:
    closed: list[bool] = []

    async def source() -> AsyncIterator[int]:
        try:
            yield 1
            raise ValueError("boom")
        finally:
            closed.append(True)

    with pytest.raises(ValueError, match="boom"):
        async for _ in with_heartbeat(source(), 1):
            pass
    assert closed == [True]


async def test_heartbeat_consumer_can_stop_early() -> None:
    cancelled: list[bool] = []

    async def source() -> AsyncIterator[Any]:
        yield 1
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.append(True)
            raise
        yield 2  # pragma: no cover

    stream = with_heartbeat(source(), 0.05)
    assert await anext(stream) == 1
    assert await anext(stream) is HEARTBEAT
    await stream.aclose()
    assert cancelled == [True]
