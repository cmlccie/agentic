"""End-to-end check behavior against a mocked OpenAI-compatible endpoint."""

import asyncio
import json

import httpx
from aiops.checks import (
    check_inference,
    check_reasoning,
    check_tool_calling,
    get_endpoint_info,
    measure_tps,
)
from aiops.checks.tps import estimate_budget_s
from aiops.engines import get_adapter
from aiops.fixtures import get_fixture
from aiops.http_client import build_async_client, build_client

BASE_URL = "http://model.test:8000/v1"

MODELS_PAYLOAD = {"data": [{"id": "test-model"}]}


def make_transport(chat_payload=None, chat_status=200, sse_body=None, health_ok=True):
    """A MockTransport emulating /models, /chat/completions, and health paths."""

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/models"):
            return httpx.Response(200, json=MODELS_PAYLOAD)
        if path in ("/health", "/v1/health/ready"):
            return httpx.Response(200 if health_ok else 503, json={"status": "ok"})
        if path.endswith("/chat/completions"):
            if sse_body is not None:
                return httpx.Response(
                    200, content=sse_body, headers={"Content-Type": "text/event-stream"}
                )
            return httpx.Response(chat_status, json=chat_payload)
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def completion_payload(
    content="PONG", finish_reason="stop", completion_tokens=4, **message_extra
):
    message = {"role": "assistant", "content": content, **message_extra}
    return {
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": completion_tokens,
            "total_tokens": 12 + completion_tokens,
        },
    }


class TestEndpointInfo:
    def test_reachable_with_health(self):
        transport = make_transport(chat_payload=completion_payload())
        with build_client(BASE_URL, transport=transport) as client:
            info = get_endpoint_info(client, get_adapter("nim"), BASE_URL)
        assert info.reachable
        assert info.served_models == ["test-model"]
        assert info.engine_health_ok is True

    def test_unreachable(self):
        def handler(request):
            raise httpx.ConnectError("refused", request=request)

        with build_client(BASE_URL, transport=httpx.MockTransport(handler)) as client:
            info = get_endpoint_info(client, get_adapter("sglang"), BASE_URL)
        assert not info.reachable
        assert info.error


class TestCheckInference:
    def test_pass(self):
        transport = make_transport(chat_payload=completion_payload())
        with build_client(BASE_URL, transport=transport) as client:
            result = check_inference(
                client, get_adapter("sglang"), get_fixture("inference-basic-v1")
            )
        assert result.passed
        assert result.model_resolved == "test-model"
        assert result.finish_reason == "stop"
        assert result.completion_tokens == 4

    def test_missing_expected_substring_fails(self):
        transport = make_transport(
            chat_payload=completion_payload(content="I cannot help with that.")
        )
        with build_client(BASE_URL, transport=transport) as client:
            result = check_inference(
                client, get_adapter("vllm"), get_fixture("inference-basic-v1")
            )
        assert not result.passed
        failed = {check.name for check in result.checks if not check.passed}
        assert failed == {"content_contains"}

    def test_http_error_fails_with_error(self):
        transport = make_transport(chat_payload={"error": "boom"}, chat_status=500)
        with build_client(BASE_URL, transport=transport) as client:
            result = check_inference(
                client, get_adapter("vllm"), get_fixture("inference-basic-v1")
            )
        assert not result.passed
        assert result.error


class TestCheckReasoning:
    def test_structured_reasoning_passes(self):
        payload = completion_payload(
            content="The farmer has 9 sheep left.",
            reasoning_content="All but 9 run away means 9 remain.",
        )
        transport = make_transport(chat_payload=payload)
        with build_client(BASE_URL, transport=transport) as client:
            result = check_reasoning(
                client, get_adapter("sglang"), get_fixture("reasoning-basic-v1")
            )
        assert result.passed
        assert result.reasoning_field == "reasoning_content"
        assert not result.reasoning_in_content
        assert "reasoning_content" in result.fields_observed

    def test_inline_think_leak_fails(self):
        payload = completion_payload(
            content="<think>All but 9 means 9.</think> The answer is 9."
        )
        transport = make_transport(chat_payload=payload)
        with build_client(BASE_URL, transport=transport) as client:
            result = check_reasoning(
                client, get_adapter("vllm"), get_fixture("reasoning-basic-v1")
            )
        assert not result.passed
        assert result.reasoning_in_content
        assert not result.reasoning_present


class TestCheckToolCalling:
    def _tool_call_payload(self, arguments: str):
        return completion_payload(
            content="",
            finish_reason="tool_calls",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_current_weather",
                        "arguments": arguments,
                    },
                }
            ],
        )

    def test_structured_call_passes_both_modes(self):
        payload = self._tool_call_payload(
            json.dumps({"location": "Paris, France", "unit": "celsius"})
        )
        transport = make_transport(chat_payload=payload)
        with build_client(BASE_URL, transport=transport) as client:
            result = check_tool_calling(
                client, get_adapter("vllm"), get_fixture("toolcall-weather-v1")
            )
        assert result.passed
        assert set(result.results) == {"auto", "forced"}
        assert result.results["auto"].arguments_schema_valid

    def test_schema_violation_fails(self):
        payload = self._tool_call_payload(json.dumps({"city": "Paris"}))
        transport = make_transport(chat_payload=payload)
        with build_client(BASE_URL, transport=transport) as client:
            result = check_tool_calling(
                client,
                get_adapter("vllm"),
                get_fixture("toolcall-weather-v1"),
                tool_choice_mode="forced",
            )
        assert not result.passed
        forced = result.results["forced"]
        assert forced.arguments_valid_json
        assert not forced.arguments_schema_valid
        assert forced.schema_errors

    def test_raw_text_leak_fails(self):
        payload = completion_payload(
            content='<tool_call>{"name": "get_current_weather"}</tool_call>',
            finish_reason="stop",
        )
        transport = make_transport(chat_payload=payload)
        with build_client(BASE_URL, transport=transport) as client:
            result = check_tool_calling(
                client,
                get_adapter("sglang"),
                get_fixture("toolcall-weather-v1"),
                tool_choice_mode="auto",
            )
        assert not result.passed
        auto = result.results["auto"]
        assert auto.raw_text_leak
        assert not auto.tool_calls_present


class TestMeasureTps:
    def _sse_body(self, chunks=6, completion_tokens=512, finish_reason="length"):
        lines = []
        for _ in range(chunks):
            lines.append(
                "data: "
                + json.dumps({"choices": [{"index": 0, "delta": {"content": "tok"}}]})
            )
        lines.append(
            "data: "
            + json.dumps(
                {"choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}]}
            )
        )
        lines.append(
            "data: "
            + json.dumps(
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 30,
                        "completion_tokens": completion_tokens,
                        "total_tokens": 30 + completion_tokens,
                    },
                }
            )
        )
        lines.append("data: [DONE]")
        return ("\n\n".join(lines) + "\n\n").encode()

    def _run(self, sse_body, **kwargs):
        async def go():
            transport = make_transport(sse_body=sse_body)
            async with build_async_client(BASE_URL, transport=transport) as client:
                return await measure_tps(
                    client,
                    get_adapter("vllm"),
                    get_fixture("tps-decode-v1"),
                    **kwargs,
                )

        return asyncio.run(go())

    def test_single_stream_uses_usage_tokens(self):
        result = self._run(self._sse_body(), repetitions=2, warmup=0)
        assert result.passed
        assert len(result.runs) == 2
        assert all(run.token_count_source == "usage" for run in result.runs)
        assert all(run.completion_tokens == 512 for run in result.runs)
        assert result.decode_tps_mean and result.decode_tps_mean > 0
        assert result.decode_tps_stddev is not None
        assert result.aggregate_tps is None

    def test_concurrent_reports_aggregate(self):
        result = self._run(self._sse_body(), concurrency=3)
        assert result.passed
        assert len(result.runs) == 3
        assert result.aggregate_tps and result.aggregate_tps > 0

    def test_early_eos_fails_minimum_tokens(self):
        body = self._sse_body(chunks=3, completion_tokens=12, finish_reason="stop")
        result = self._run(body, repetitions=1, warmup=0)
        assert not result.passed
        assert result.runs[0].completion_tokens == 12

    def test_chunk_count_fallback(self):
        lines = []
        for _ in range(5):
            lines.append(
                "data: "
                + json.dumps({"choices": [{"index": 0, "delta": {"content": "t"}}]})
            )
        lines.append(
            "data: "
            + json.dumps(
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]}
            )
        )
        lines.append("data: [DONE]")
        body = ("\n\n".join(lines) + "\n\n").encode()
        result = self._run(body, repetitions=1, warmup=0)
        assert result.runs[0].token_count_source == "chunk_count"
        assert result.runs[0].completion_tokens == 5

    def test_budget_estimate(self):
        streams, estimate = estimate_budget_s(512, 3, 1, 1, 20.0)
        assert streams == 4
        assert estimate == 4 * 512 / 20.0


class TestServerRegistration:
    def test_all_tools_registered(self):
        from aiops.server import mcp

        tools = asyncio.run(mcp.list_tools())
        assert {tool.name for tool in tools} >= {
            "get_endpoint_info",
            "check_inference",
            "check_reasoning",
            "check_tool_calling",
            "measure_tps",
            "list_fixtures",
        }
