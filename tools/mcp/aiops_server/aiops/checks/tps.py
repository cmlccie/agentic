"""Decode-throughput measurement over streaming completions.

Methodology:

- Streaming chat completion with `stream_options.include_usage`, temperature 0.
- TTFT = first content-bearing delta minus request send.
- decode_tps = (completion_tokens - 1) / (last delta - first delta) — prefill
  excluded.
- Token counts prefer the final `usage` chunk; the fallback counts
  content-bearing delta chunks and is flagged via `token_count_source`.
- `concurrency == 1`: `warmup` discarded runs, then `repetitions` measured
  sequential runs. `concurrency > 1`: one discarded warmup, then a single round
  of N parallel streams; `aggregate_tps` = total completion tokens over the
  round's wall-clock.
"""

import asyncio
import statistics
import time
from typing import List, Optional, Tuple

import httpx

from aiops.checks.common import (
    VALID_STOP_FINISH_REASONS,
    add_check,
    build_chat_body,
)
from aiops.engines import EngineAdapter
from aiops.fixtures import FixtureSpec
from aiops.http_client import aiter_sse_json
from aiops.models import CheckItem, TpsResult, TpsRun

MIN_DECODE_WINDOW_S = 1e-6


async def _resolve_model(client: httpx.AsyncClient, model: Optional[str]) -> str:
    if model:
        return model
    response = await client.get("/models")
    response.raise_for_status()
    data = response.json().get("data", [])
    if not data:
        raise ValueError("GET /models returned no models.")
    return data[0]["id"]


async def _run_stream(client: httpx.AsyncClient, body: dict) -> TpsRun:
    t_send = time.perf_counter()
    t_first: Optional[float] = None
    t_last: Optional[float] = None
    chunk_tokens = 0
    usage_completion: Optional[int] = None
    finish_reason: Optional[str] = None

    async with client.stream("POST", "/chat/completions", json=body) as response:
        response.raise_for_status()
        async for payload in aiter_sse_json(response):
            usage = payload.get("usage")
            if isinstance(usage, dict) and usage.get("completion_tokens") is not None:
                usage_completion = usage["completion_tokens"]
            for choice in payload.get("choices") or []:
                delta = choice.get("delta") or {}
                if (
                    delta.get("content")
                    or delta.get("reasoning_content")
                    or delta.get("reasoning")
                ):
                    now = time.perf_counter()
                    if t_first is None:
                        t_first = now
                    t_last = now
                    chunk_tokens += 1
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]

    t_end = time.perf_counter()
    if usage_completion is not None:
        completion_tokens, source = usage_completion, "usage"
    else:
        completion_tokens, source = chunk_tokens, "chunk_count"

    if t_first is None or t_last is None or completion_tokens < 2:
        decode_tps = 0.0
    else:
        decode_tps = (completion_tokens - 1) / max(
            t_last - t_first, MIN_DECODE_WINDOW_S
        )

    return TpsRun(
        decode_tps=decode_tps,
        ttft_ms=((t_first or t_end) - t_send) * 1000,
        total_ms=(t_end - t_send) * 1000,
        completion_tokens=completion_tokens,
        finish_reason=finish_reason,
        token_count_source=source,
    )


async def measure_tps(
    client: httpx.AsyncClient,
    engine: EngineAdapter,
    fixture: FixtureSpec,
    model: Optional[str] = None,
    max_tokens: int = 512,
    repetitions: int = 3,
    warmup: int = 1,
    concurrency: int = 1,
) -> TpsResult:
    started = time.perf_counter()
    checks: List[CheckItem] = []

    def _result(**kwargs) -> TpsResult:
        return TpsResult(
            passed=bool(checks) and all(check.passed for check in checks),
            engine=engine.name,
            fixture_id=fixture.id,
            duration_ms=(time.perf_counter() - started) * 1000,
            checks=checks,
            concurrency=concurrency,
            max_tokens=max_tokens,
            **kwargs,
        )

    try:
        resolved = await _resolve_model(client, model)
        body = build_chat_body(
            fixture,
            engine,
            resolved,
            {
                "max_tokens": max_tokens,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )

        runs: List[TpsRun] = []
        aggregate_tps: Optional[float] = None
        if concurrency <= 1:
            for _ in range(max(warmup, 0)):
                await _run_stream(client, body)
            for _ in range(max(repetitions, 1)):
                runs.append(await _run_stream(client, body))
        else:
            await _run_stream(client, body)  # single warmup
            round_start = time.perf_counter()
            runs = list(
                await asyncio.gather(
                    *(_run_stream(client, body) for _ in range(concurrency))
                )
            )
            round_elapsed = max(time.perf_counter() - round_start, MIN_DECODE_WINDOW_S)
            aggregate_tps = sum(run.completion_tokens for run in runs) / round_elapsed
    except (httpx.HTTPError, ValueError) as exc:
        return _result(error=f"{type(exc).__name__}: {exc}")

    minimum_tokens = fixture.expected.min_completion_tokens or 0
    for index, run in enumerate(runs):
        run_ok = (
            run.finish_reason in VALID_STOP_FINISH_REASONS
            and run.completion_tokens >= minimum_tokens
            and run.decode_tps > 0
        )
        add_check(
            checks,
            f"run_{index}_valid",
            run_ok,
            f"finish_reason={run.finish_reason!r}, "
            f"completion_tokens={run.completion_tokens} (min {minimum_tokens}), "
            f"decode_tps={run.decode_tps:.1f} ({run.token_count_source})",
        )

    decode_values = [run.decode_tps for run in runs]
    ttft_values = [run.ttft_ms for run in runs]
    return _result(
        model_resolved=resolved,
        runs=runs,
        decode_tps_mean=statistics.mean(decode_values) if decode_values else None,
        decode_tps_stddev=(
            statistics.stdev(decode_values) if len(decode_values) > 1 else None
        ),
        ttft_ms_mean=statistics.mean(ttft_values) if ttft_values else None,
        aggregate_tps=aggregate_tps,
    )


def estimate_budget_s(
    max_tokens: int, repetitions: int, warmup: int, concurrency: int, floor_tps: float
) -> Tuple[int, float]:
    """Worst-case stream count and rough runtime at a pessimistic decode rate."""
    streams = (warmup + repetitions) if concurrency <= 1 else (1 + concurrency)
    return streams, streams * (max_tokens / max(floor_tps, 1.0))
