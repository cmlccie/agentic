"""MCP AIOps Server: deterministic validation tools for model-serving endpoints.

All tools are pure HTTP validators: they take an OpenAI-compatible `base_url`
(a pod IP or Service URL, including `/v1`) and never touch the Kubernetes API.
Orchestration — scaling, isolation, pod-IP resolution — belongs to the caller.
"""

import logging
from typing import Any, Dict, Optional

from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

import agentic.logging
from aiops import __version__
from aiops.checks import endpoint, inference, reasoning, tool_calling, tps
from aiops.engines import get_adapter
from aiops.fixtures import get_fixture, load_fixtures
from aiops.http_client import TIMEOUT_CAP_S, build_async_client, build_client
from aiops.models import (
    EndpointInfo,
    Engine,
    FixtureCatalog,
    FixtureInfo,
    InferenceResult,
    ReasoningResult,
    ToolCallingResult,
    ToolChoiceMode,
    TpsResult,
)

logger = logging.getLogger("aiops_server")

mcp = FastMCP("MCP AIOps Server")

# Pessimistic decode rate used only to reject measure_tps calls that cannot
# possibly finish under the server timeout cap; real models on this hardware
# decode well above this floor.
_BUDGET_FLOOR_TPS = 20.0


@mcp.custom_route("/health", methods=["GET"])
async def health(request: Request) -> JSONResponse:
    """Liveness/readiness endpoint for Kubernetes probes."""
    return JSONResponse({"status": "ok", "version": __version__})


# --------------------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------------------


@mcp.tool()
@agentic.logging.log_call(logger)
def get_endpoint_info(
    base_url: str,
    inferencing_engine: Engine,
    api_key: Optional[str] = None,
    timeout_s: float = 15.0,
) -> EndpointInfo:
    """Probe an OpenAI-compatible endpoint for readiness.

    Issues GET /models and, where the engine has one, its health endpoint
    (SGLang/vLLM: /health; NIM: /v1/health/ready). Use this after a deployment
    scales up to confirm the engine actually answers before running checks.

    Args:
        base_url: OpenAI-compatible base URL including /v1, e.g.
            http://10.0.2.34:8000/v1 or the model Service URL.
        inferencing_engine: One of sglang, vllm, nim.
        api_key: Optional bearer token.
        timeout_s: Request timeout in seconds.

    Returns:
        EndpointInfo: reachability, served model ids, engine health verdict.
    """
    adapter = get_adapter(inferencing_engine)
    with build_client(base_url, api_key, timeout_s) as client:
        return endpoint.get_endpoint_info(client, adapter, base_url)


@mcp.tool()
@agentic.logging.log_call(logger)
def check_inference(
    base_url: str,
    inferencing_engine: Engine,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    fixture_id: str = "inference-basic-v1",
    timeout_s: float = 120.0,
) -> InferenceResult:
    """Verify basic inferencing: send a fixture prompt, validate the completion.

    Deterministic checks: HTTP 200, non-empty content, valid finish_reason,
    positive completion token count, and the fixture's expected substring.

    Args:
        base_url: OpenAI-compatible base URL including /v1.
        inferencing_engine: One of sglang, vllm, nim.
        model: Served model name; resolved from GET /models when omitted.
        api_key: Optional bearer token.
        fixture_id: Inference fixture to run (see list_fixtures).
        timeout_s: Request timeout in seconds.

    Returns:
        InferenceResult: verdict plus itemized checks and completion evidence.
    """
    adapter = get_adapter(inferencing_engine)
    fixture = get_fixture(fixture_id, kind="inference")
    with build_client(base_url, api_key, timeout_s) as client:
        return inference.check_inference(client, adapter, fixture, model)


@mcp.tool()
@agentic.logging.log_call(logger)
def check_reasoning(
    base_url: str,
    inferencing_engine: Engine,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    fixture_id: str = "reasoning-basic-v1",
    reasoning_params: Optional[Dict[str, Any]] = None,
    timeout_s: float = 180.0,
) -> ReasoningResult:
    """Verify reasoning arrives as a structured field, not inline in content.

    Fails when no reasoning field is present, or when reasoning markup (for
    example <think> tags) leaks into message.content — both indicate a missing
    or wrong --reasoning-parser. Only run this against models that emit
    reasoning traces.

    Args:
        base_url: OpenAI-compatible base URL including /v1.
        inferencing_engine: One of sglang, vllm, nim.
        model: Served model name; resolved from GET /models when omitted.
        api_key: Optional bearer token.
        fixture_id: Reasoning fixture to run (see list_fixtures; family-specific
            fixtures carry the right trigger params, e.g. reasoning-qwen3-v1).
        reasoning_params: Extra request-body params merged last, e.g.
            {"chat_template_kwargs": {"enable_thinking": true}} or
            {"reasoning_effort": "high"}.
        timeout_s: Request timeout in seconds.

    Returns:
        ReasoningResult: verdict, which field held reasoning, leak detection,
        and the observed message fields for diagnostics.
    """
    adapter = get_adapter(inferencing_engine)
    fixture = get_fixture(fixture_id, kind="reasoning")
    with build_client(base_url, api_key, timeout_s) as client:
        return reasoning.check_reasoning(
            client, adapter, fixture, model, reasoning_params
        )


@mcp.tool()
@agentic.logging.log_call(logger)
def check_tool_calling(
    base_url: str,
    inferencing_engine: Engine,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    fixture_id: str = "toolcall-weather-v1",
    tool_choice_mode: ToolChoiceMode = "both",
    timeout_s: float = 180.0,
) -> ToolCallingResult:
    """Verify tool calls arrive structured and schema-valid, not as raw text.

    Runs the fixture's single-function request with tool_choice "auto" and/or a
    forced function choice, and validates: tool_calls present, correct function
    name, acceptable finish_reason, arguments parse as JSON and validate
    against the fixture's JSON Schema, and no tool-call markup leaked into
    content. Both modes are expected to pass on a correctly configured stack
    (vLLM/NIM additionally require --enable-auto-tool-choice for "auto").

    Args:
        base_url: OpenAI-compatible base URL including /v1.
        inferencing_engine: One of sglang, vllm, nim.
        model: Served model name; resolved from GET /models when omitted.
        api_key: Optional bearer token.
        fixture_id: Tool-calling fixture to run (see list_fixtures).
        tool_choice_mode: auto, forced, or both (default).
        timeout_s: Request timeout in seconds.

    Returns:
        ToolCallingResult: overall verdict plus per-mode evidence.
    """
    adapter = get_adapter(inferencing_engine)
    fixture = get_fixture(fixture_id, kind="tool_calling")
    with build_client(base_url, api_key, timeout_s) as client:
        return tool_calling.check_tool_calling(
            client, adapter, fixture, model, tool_choice_mode
        )


@mcp.tool()
@agentic.logging.log_call(logger)
async def measure_tps(
    base_url: str,
    inferencing_engine: Engine,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    fixture_id: str = "tps-decode-v1",
    max_tokens: int = 512,
    repetitions: int = 3,
    warmup: int = 1,
    concurrency: int = 1,
    timeout_s: float = 540.0,
) -> TpsResult:
    """Measure decode tokens/sec over streaming completions.

    Single-stream mode (concurrency=1): warmup discarded runs, then measured
    sequential repetitions; reports decode TPS mean/stddev and TTFT. Decode TPS
    is first-to-last token, excluding prefill/TTFT. Concurrent mode
    (concurrency>1, recommended 4): one warmup then one round of N parallel
    streams; additionally reports aggregate_tps over the round wall-clock.

    Keep a single call within the server's 570s budget — lower repetitions or
    max_tokens, or split into multiple calls, rather than raising timeout_s.

    Args:
        base_url: OpenAI-compatible base URL including /v1.
        inferencing_engine: One of sglang, vllm, nim.
        model: Served model name; resolved from GET /models when omitted.
        api_key: Optional bearer token.
        fixture_id: TPS fixture to run (see list_fixtures).
        max_tokens: Tokens generated per stream.
        repetitions: Measured runs (single-stream mode).
        warmup: Discarded warmup runs (single-stream mode).
        concurrency: Parallel streams; 1 = single-stream methodology.
        timeout_s: Per-request timeout in seconds.

    Returns:
        TpsResult: per-run measurements plus summary statistics.
    """
    adapter = get_adapter(inferencing_engine)
    fixture = get_fixture(fixture_id, kind="tps")

    streams, estimate_s = tps.estimate_budget_s(
        max_tokens, repetitions, warmup, concurrency, _BUDGET_FLOOR_TPS
    )
    if estimate_s > TIMEOUT_CAP_S:
        raise ValueError(
            f"Requested {streams} streams x {max_tokens} tokens could take "
            f"~{estimate_s:.0f}s at {_BUDGET_FLOOR_TPS:.0f} TPS, exceeding the "
            f"{TIMEOUT_CAP_S:.0f}s server budget. Reduce repetitions/max_tokens/"
            "concurrency or split into multiple calls."
        )

    async with build_async_client(base_url, api_key, timeout_s) as client:
        return await tps.measure_tps(
            client,
            adapter,
            fixture,
            model,
            max_tokens=max_tokens,
            repetitions=repetitions,
            warmup=warmup,
            concurrency=concurrency,
        )


@mcp.tool()
@agentic.logging.log_call(logger)
def list_fixtures(
    kind: str = "all",
    model_family: Optional[str] = None,
) -> FixtureCatalog:
    """List available test fixtures and the server version.

    Args:
        kind: Filter by fixture kind (inference, reasoning, tool_calling, tps)
            or "all".
        model_family: When given, return fixtures targeting that family plus
            the generic ("*") fixtures.

    Returns:
        FixtureCatalog: server version and matching fixture descriptors.
    """
    fixtures = []
    for fixture in load_fixtures().values():
        if kind != "all" and fixture.kind != kind:
            continue
        if model_family is not None and not (
            "*" in fixture.model_families or model_family in fixture.model_families
        ):
            continue
        fixtures.append(
            FixtureInfo(
                id=fixture.id,
                kind=fixture.kind,
                description=fixture.description,
                model_families=fixture.model_families,
                default_params=fixture.params,
                expected_summary=fixture.expected_summary(),
            )
        )
    return FixtureCatalog(server_version=__version__, fixtures=fixtures)
