"""Tests for the OpenAI-compatible API (the primary client interface).

Most tests drive the API through the official ``openai`` SDK so they verify
what real clients see, not just the raw JSON.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import httpx2
import openai
import pytest
from pydantic_ai.messages import (
    BinaryContent,
    ImageUrl,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from agentic.agent.openai_api import OpenAIError, to_pydantic_ai

from .conftest import RunningApp, make_app, scripted_model, use_model

pytestmark = pytest.mark.anyio

AGENT = {"model": "test:main", "name": "test-agent", "capabilities": ["LocalTools"]}


def sdk_client(app: Any) -> openai.AsyncOpenAI:
    """An openai SDK client talking to ``app`` in-process."""
    return openai.AsyncOpenAI(
        base_url="http://test/v1",
        api_key="unused",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app)),
    )


def sse_payloads(text: str) -> list[Any]:
    """Parse an SSE body into its data payloads (``[DONE]`` kept as a string)."""
    payloads = []
    for line in text.splitlines():
        if line.startswith("data: "):
            data = line.removeprefix("data: ")
            payloads.append(data if data == "[DONE]" else json.loads(data))
    return payloads


# --------------------------------------------------------------------------------------
# Message conversion
# --------------------------------------------------------------------------------------


class TestToPydanticAI:
    def test_single_user_message(self) -> None:
        prompt, history = to_pydantic_ai([{"role": "user", "content": "hi"}])
        assert prompt == "hi"
        assert history == []

    def test_full_conversation(self) -> None:
        prompt, history = to_pydantic_ai(
            [
                {"role": "system", "content": "be brief"},
                {"role": "developer", "content": "use metric"},
                {"role": "user", "content": "weather?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {
                                "name": "forecast",
                                "arguments": '{"city":"Oslo"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "4C"},
                {"role": "assistant", "content": "It is 4C."},
                {"role": "user", "content": "thanks"},
            ]
        )
        assert prompt == "thanks"
        assert [type(m) for m in history] == [
            ModelRequest,
            ModelResponse,
            ModelRequest,
            ModelResponse,
        ]
        first = history[0].parts
        assert isinstance(first[0], SystemPromptPart) and first[0].content == "be brief"
        assert (
            isinstance(first[1], SystemPromptPart) and first[1].content == "use metric"
        )
        assert isinstance(first[2], UserPromptPart)
        call = history[1].parts[0]
        assert isinstance(call, ToolCallPart)
        assert (call.tool_name, call.tool_call_id) == ("forecast", "c1")
        result = history[2].parts[0]
        assert isinstance(result, ToolReturnPart)
        assert (result.tool_name, result.content) == ("forecast", "4C")
        assert history[3].parts == [TextPart("It is 4C.")]

    def test_reasoning_blocks_are_stripped_from_assistant_messages(self) -> None:
        _, history = to_pydantic_ai(
            [
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "content": "<think>\n→ tool(...)\n</think>\nAnswer",
                },
                {"role": "user", "content": "q2"},
            ]
        )
        assert history[1].parts == [TextPart("Answer")]

    def test_images_and_text_parts(self) -> None:
        prompt, _ = to_pydantic_ai(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://x/cat.png"},
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
                        },
                    ],
                }
            ]
        )
        assert isinstance(prompt, list)
        assert prompt[0] == "what is this?"
        assert isinstance(prompt[1], ImageUrl) and prompt[1].url == "https://x/cat.png"
        assert (
            isinstance(prompt[2], BinaryContent) and prompt[2].media_type == "image/png"
        )

    def test_text_only_parts_become_a_string(self) -> None:
        prompt, _ = to_pydantic_ai(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "a"},
                        {"type": "text", "text": "b"},
                    ],
                }
            ]
        )
        assert prompt == "a\nb"

    @pytest.mark.parametrize(
        "messages, fragment",
        [
            ([{"role": "assistant", "content": "x"}], "last message"),
            (
                [{"role": "wizard", "content": "x"}, {"role": "user", "content": "y"}],
                "role 'wizard'",
            ),
            (
                [{"role": "user", "content": [{"type": "audio"}]}],
                "content part type 'audio'",
            ),
        ],
    )
    def test_invalid_messages(
        self, messages: list[dict[str, Any]], fragment: str
    ) -> None:
        with pytest.raises(OpenAIError, match=fragment):
            to_pydantic_ai(messages)


# --------------------------------------------------------------------------------------
# Through the openai SDK
# --------------------------------------------------------------------------------------


async def test_list_and_get_models(tmp_path: Path) -> None:
    use_model("main", scripted_model())
    async with RunningApp(make_app(tmp_path, AGENT)) as running:
        client = sdk_client(running.app)
        models = await client.models.list()
        assert [m.id for m in models.data] == ["test-agent"]
        assert (await client.models.retrieve("test-agent")).id == "test-agent"
        with pytest.raises(openai.NotFoundError):
            await client.models.retrieve("nope")


async def test_non_streaming_answer_with_reasoning_and_usage(tmp_path: Path) -> None:
    use_model(
        "main",
        scripted_model(
            thinking="need the forecast",
            note="Let me check.",
            tool="forecast",
            args={"city": "Paris", "api_key": "s3cret"},
            answer=lambda messages, _: (
                f"Answer: {messages[-1].parts[-1].content['sky']}"
            ),
        ),
    )
    async with RunningApp(make_app(tmp_path, AGENT)) as running:
        completion = await sdk_client(running.app).chat.completions.create(
            model="test-agent", messages=[{"role": "user", "content": "weather?"}]
        )
    choice = completion.choices[0]
    assert choice.message.content == "Answer: sunny"
    assert choice.finish_reason == "stop"
    reasoning = choice.message.model_extra["reasoning_content"]
    assert "need the forecast" in reasoning
    assert "💬 Let me check." in reasoning
    assert '→ forecast({"city": "Paris", "api_key": "***"})' in reasoning
    assert "s3cret" not in reasoning
    assert '← forecast: {"city": "Paris"' in reasoning
    assert (
        completion.usage.total_tokens
        == completion.usage.prompt_tokens + completion.usage.completion_tokens
    )
    assert completion.usage.total_tokens > 0


async def test_streaming_reasoning_then_content(tmp_path: Path) -> None:
    use_model(
        "main",
        scripted_model(
            thinking="hmm", tool="forecast", args={"city": "Oslo"}, answer="Sunny."
        ),
    )
    async with RunningApp(make_app(tmp_path, AGENT)) as running:
        stream = await sdk_client(running.app).chat.completions.create(
            model="test-agent",
            messages=[{"role": "user", "content": "weather?"}],
            stream=True,
            stream_options={"include_usage": True},
        )
        chunks = [chunk async for chunk in stream]

    reasoning = "".join(
        (c.choices[0].delta.model_extra or {}).get("reasoning_content") or ""
        for c in chunks
        if c.choices
    )
    content = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    assert (
        "hmm" in reasoning and "→ forecast" in reasoning and "← forecast" in reasoning
    )
    assert content == "Sunny."
    # reasoning arrives before any content
    first_content = next(
        i for i, c in enumerate(chunks) if c.choices and c.choices[0].delta.content
    )
    last_reasoning = max(
        i
        for i, c in enumerate(chunks)
        if c.choices and (c.choices[0].delta.model_extra or {}).get("reasoning_content")
    )
    assert last_reasoning < first_content
    finishes = [
        c.choices[0].finish_reason
        for c in chunks
        if c.choices and c.choices[0].finish_reason
    ]
    assert finishes == ["stop"]
    assert chunks[-1].choices == [] and chunks[-1].usage.total_tokens > 0
    assert len({c.id for c in chunks}) == 1


async def test_stream_wire_format(tmp_path: Path) -> None:
    use_model("main", scripted_model(answer="hello"))
    async with RunningApp(make_app(tmp_path, AGENT)) as running:
        response = await running.client.post(
            "/v1/chat/completions",
            json={
                "model": "x",
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    assert response.headers["content-type"].startswith("text/event-stream")
    payloads = sse_payloads(response.text)
    assert payloads[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    assert payloads[-1] == "[DONE]"
    assert payloads[-2]["choices"][0]["finish_reason"] == "stop"
    assert all(p["object"] == "chat.completion.chunk" for p in payloads[:-1])


async def test_activity_off_sends_only_the_answer(tmp_path: Path) -> None:
    use_model("main", scripted_model(thinking="t", tool="forecast", args={"city": "a"}))
    server = {
        "agent_card": {"display_name": "A", "description": "B"},
        "streaming": {"activity": "off"},
    }
    async with RunningApp(make_app(tmp_path, AGENT, server)) as running:
        completion = await sdk_client(running.app).chat.completions.create(
            model="x", messages=[{"role": "user", "content": "hi"}]
        )
    assert completion.choices[0].message.content == "done"
    assert "reasoning_content" not in (completion.choices[0].message.model_extra or {})


async def test_thinking_can_be_hidden(tmp_path: Path) -> None:
    use_model(
        "main",
        scripted_model(thinking="secret thoughts", tool="forecast", args={"city": "a"}),
    )
    server = {
        "agent_card": {"display_name": "A", "description": "B"},
        "streaming": {"thinking": False},
    }
    async with RunningApp(make_app(tmp_path, AGENT, server)) as running:
        completion = await sdk_client(running.app).chat.completions.create(
            model="x", messages=[{"role": "user", "content": "hi"}]
        )
    reasoning = completion.choices[0].message.model_extra["reasoning_content"]
    assert "secret thoughts" not in reasoning
    assert "→ forecast" in reasoning


async def test_conversation_history_reaches_the_model(tmp_path: Path) -> None:
    def answer(messages: Any, info: AgentInfo) -> str:
        texts = [
            p.content
            for m in messages
            for p in m.parts
            if isinstance(p, (UserPromptPart, SystemPromptPart, TextPart))
        ]
        return " | ".join(str(t) for t in texts)

    use_model("main", scripted_model(answer=answer))
    async with RunningApp(make_app(tmp_path, AGENT)) as running:
        completion = await sdk_client(running.app).chat.completions.create(
            model="x",
            messages=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "reply"},
                {"role": "user", "content": "two"},
            ],
        )
    content = completion.choices[0].message.content
    assert (
        content.index("sys")
        < content.index("one")
        < content.index("reply")
        < content.index("two")
    )


async def test_sampling_parameters_reach_the_model(tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    async def stream(messages: Any, info: AgentInfo) -> AsyncIterator[str]:
        seen.update(info.model_settings or {})
        yield "ok"

    use_model("main", FunctionModel(stream_function=stream))
    async with RunningApp(make_app(tmp_path, AGENT)) as running:
        await sdk_client(running.app).chat.completions.create(
            model="x",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.1,
            max_tokens=42,
            stop="END",
            seed=7,
        )
    assert seen["temperature"] == 0.1
    assert seen["max_tokens"] == 42
    assert seen["stop_sequences"] == ["END"]
    assert seen["seed"] == 7


# --------------------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------------------


def failing_model() -> FunctionModel:
    async def stream(messages: Any, info: AgentInfo) -> AsyncIterator[str]:
        raise RuntimeError("upstream model exploded with secret details")
        yield ""  # pragma: no cover

    def request(messages: Any, info: AgentInfo) -> ModelResponse:
        raise RuntimeError("upstream model exploded with secret details")

    return FunctionModel(request, stream_function=stream)


async def test_failure_before_streaming_is_an_openai_error(tmp_path: Path) -> None:
    use_model("main", failing_model())
    async with RunningApp(make_app(tmp_path, AGENT)) as running:
        response = await running.client.post(
            "/v1/chat/completions",
            json={"model": "x", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 500
    assert response.headers["x-should-retry"] == "false"
    error = response.json()["error"]
    assert error["type"] == "server_error" and error["code"] == "agent_error"
    assert "RuntimeError" in error["message"]
    assert "secret details" not in error["message"]


async def test_failure_mid_stream_raises_in_the_sdk(tmp_path: Path) -> None:
    use_model("main", failing_model())
    async with RunningApp(make_app(tmp_path, AGENT)) as running:
        stream = await sdk_client(running.app).chat.completions.create(
            model="x", messages=[{"role": "user", "content": "hi"}], stream=True
        )
        with pytest.raises(openai.APIError, match="The agent failed"):
            async for _ in stream:
                pass


@pytest.mark.parametrize(
    "body, fragment",
    [
        ({"model": "x", "messages": []}, "messages"),
        (
            {"model": "x", "messages": [{"role": "user", "content": "hi"}], "n": 2},
            "n=1",
        ),
        (
            {"model": "x", "messages": [{"role": "assistant", "content": "hi"}]},
            "last message",
        ),
    ],
)
async def test_bad_requests_get_400_in_openai_shape(
    tmp_path: Path, body: dict[str, Any], fragment: str
) -> None:
    use_model("main", scripted_model())
    async with RunningApp(make_app(tmp_path, AGENT)) as running:
        response = await running.client.post("/v1/chat/completions", json=body)
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert fragment in error["message"]


async def test_invalid_json_body(tmp_path: Path) -> None:
    use_model("main", scripted_model())
    async with RunningApp(make_app(tmp_path, AGENT)) as running:
        response = await running.client.post(
            "/v1/chat/completions",
            content=b"{not json",
            headers={"content-type": "application/json"},
        )
    assert response.status_code == 400
    assert "not valid JSON" in response.json()["error"]["message"]


async def test_heartbeats_keep_idle_streams_alive(tmp_path: Path) -> None:
    use_model("main", scripted_model(tool="slow", args={"seconds": 0.5}, answer="late"))
    server = {
        "agent_card": {"display_name": "A", "description": "B"},
        "streaming": {"heartbeat_seconds": 0.1},
    }
    async with RunningApp(make_app(tmp_path, AGENT, server)) as running:
        response = await running.client.post(
            "/v1/chat/completions",
            json={
                "model": "x",
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    assert ": keep-alive" in response.text
    assert sse_payloads(response.text)[-1] == "[DONE]"


async def test_unknown_request_fields_are_ignored(tmp_path: Path) -> None:
    use_model("main", scripted_model(answer="fine"))
    async with RunningApp(make_app(tmp_path, AGENT)) as running:
        response = await running.client.post(
            "/v1/chat/completions",
            json={
                "model": "x",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [
                    {"type": "function", "function": {"name": "t", "parameters": {}}}
                ],
                "response_format": {"type": "text"},
                "user": "someone",
            },
        )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "fine"


async def test_http_client_disconnect_cancels_the_run(tmp_path: Path) -> None:
    """Closing a streaming response early must cancel the agent run (and its tools)."""
    import asyncio

    from .conftest import SLOW_STATE, Server, free_port

    use_model("main", scripted_model(tool="slow", args={"seconds": 30}))
    app = make_app(tmp_path, AGENT)
    with Server(app, free_port()) as server:
        async with httpx.AsyncClient(base_url=server.url, timeout=10) as client:
            async with client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "x",
                    "stream": True,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            ) as response:
                async for line in response.aiter_lines():
                    if "→ slow" in line:
                        break
        for _ in range(100):
            if SLOW_STATE["cancelled"]:
                break
            await asyncio.sleep(0.05)
    assert SLOW_STATE == {"started": 1, "cancelled": 1, "finished": 0}


# --------------------------------------------------------------------------------------
# Hardening (from review)
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "messages, fragment",
    [
        (
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png,abc"},
                        }
                    ],
                }
            ],
            "invalid messages",
        ),
        (
            [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "", "tool_calls": {"id": "x"}},
                {"role": "user", "content": "q2"},
            ],
            "invalid messages",
        ),
        (["not a dict", {"role": "user", "content": "q"}], "'messages.0'"),
        (
            [
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "c9", "function": {"name": "t", "arguments": "{}"}}
                    ],
                },
                {"role": "user", "content": "q2"},
            ],
            "missing: c9",
        ),
    ],
)
async def test_malformed_messages_are_400s_not_500s(
    tmp_path: Path, messages: list[Any], fragment: str
) -> None:
    use_model("main", scripted_model())
    async with RunningApp(make_app(tmp_path, AGENT)) as running:
        for stream in (False, True):
            response = await running.client.post(
                "/v1/chat/completions",
                json={"model": "x", "stream": stream, "messages": messages},
            )
            assert response.status_code == 400, response.text
            assert fragment in response.json()["error"]["message"]


def test_null_text_parts_are_treated_as_empty() -> None:
    prompt, _ = to_pydantic_ai(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": None},
                    {"type": "text", "text": "hi"},
                ],
            }
        ]
    )
    assert prompt == "\nhi"


def test_open_webui_reasoning_details_are_stripped() -> None:
    _, history = to_pydantic_ai(
        [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": '<details type="reasoning" done="true" duration="2">\n'
                "<summary>Thought for 2 seconds</summary>\n→ tool(...)\n</details>\nAnswer",
            },
            {"role": "user", "content": "q2"},
        ]
    )
    assert history[1].parts == [TextPart("Answer")]


async def test_non_streaming_disconnect_cancels_the_run(tmp_path: Path) -> None:
    import asyncio

    from .conftest import SLOW_STATE, Server, free_port

    use_model("main", scripted_model(tool="slow", args={"seconds": 30}))
    app = make_app(tmp_path, AGENT)
    with Server(app, free_port()) as server:
        async with httpx.AsyncClient(base_url=server.url, timeout=10) as client:
            request = asyncio.create_task(
                client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "x",
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                )
            )
            for _ in range(100):
                if SLOW_STATE["started"]:
                    break
                await asyncio.sleep(0.05)
            request.cancel()
            await asyncio.gather(request, return_exceptions=True)
        for _ in range(100):
            if SLOW_STATE["cancelled"]:
                break
            await asyncio.sleep(0.05)
    assert SLOW_STATE == {"started": 1, "cancelled": 1, "finished": 0}
