"""Basic inference check: prompt in, valid completion out."""

import time
from typing import List, Optional

import httpx

from aiops.checks.common import (
    VALID_STOP_FINISH_REASONS,
    add_check,
    build_chat_body,
    first_finish_reason,
    first_message,
    resolve_model,
    snippet,
)
from aiops.engines import EngineAdapter
from aiops.fixtures import FixtureSpec
from aiops.models import CheckItem, InferenceResult


def check_inference(
    client: httpx.Client,
    engine: EngineAdapter,
    fixture: FixtureSpec,
    model: Optional[str] = None,
) -> InferenceResult:
    started = time.perf_counter()
    checks: List[CheckItem] = []

    def _result(**kwargs) -> InferenceResult:
        return InferenceResult(
            passed=bool(checks) and all(check.passed for check in checks),
            engine=engine.name,
            fixture_id=fixture.id,
            duration_ms=(time.perf_counter() - started) * 1000,
            checks=checks,
            **kwargs,
        )

    try:
        resolved = resolve_model(client, model)
        body = build_chat_body(fixture, engine, resolved)
        request_started = time.perf_counter()
        response = client.post("/chat/completions", json=body)
        latency_ms = (time.perf_counter() - request_started) * 1000
    except httpx.HTTPError as exc:
        return _result(error=f"{type(exc).__name__}: {exc}")

    add_check(
        checks,
        "http_ok",
        response.status_code == 200,
        f"HTTP {response.status_code}",
    )
    if response.status_code != 200:
        return _result(
            model_resolved=resolved,
            latency_ms=latency_ms,
            error=snippet(response.text),
        )

    payload = response.json()
    message = first_message(payload)
    content = message.get("content") or ""
    finish_reason = first_finish_reason(payload)
    usage = engine.normalize_usage(payload) or {}
    completion_tokens = usage.get("completion_tokens")

    add_check(
        checks,
        "content_present",
        bool(content.strip()),
        f"content length {len(content)}",
    )
    add_check(
        checks,
        "finish_reason_valid",
        finish_reason in VALID_STOP_FINISH_REASONS,
        f"finish_reason={finish_reason!r}",
    )
    add_check(
        checks,
        "completion_tokens_positive",
        bool(completion_tokens),
        f"completion_tokens={completion_tokens}",
    )
    if fixture.expected.content_contains:
        expected = fixture.expected.content_contains
        add_check(
            checks,
            "content_contains",
            expected.lower() in content.lower(),
            f"expected substring {expected!r}",
        )

    return _result(
        model_resolved=resolved,
        latency_ms=latency_ms,
        completion_snippet=snippet(content),
        finish_reason=finish_reason,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=completion_tokens,
        total_tokens=usage.get("total_tokens"),
    )
