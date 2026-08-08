"""Reasoning-parsing check: reasoning must arrive as a structured field."""

import time
from typing import Any, Dict, List, Optional

import httpx

from aiops.checks.common import (
    add_check,
    build_chat_body,
    first_finish_reason,
    first_message,
    resolve_model,
    snippet,
)
from aiops.engines import EngineAdapter
from aiops.fixtures import FixtureSpec
from aiops.models import CheckItem, ReasoningResult


def check_reasoning(
    client: httpx.Client,
    engine: EngineAdapter,
    fixture: FixtureSpec,
    model: Optional[str] = None,
    reasoning_params: Optional[Dict[str, Any]] = None,
) -> ReasoningResult:
    started = time.perf_counter()
    checks: List[CheckItem] = []

    def _result(**kwargs) -> ReasoningResult:
        return ReasoningResult(
            passed=bool(checks) and all(check.passed for check in checks),
            engine=engine.name,
            fixture_id=fixture.id,
            duration_ms=(time.perf_counter() - started) * 1000,
            checks=checks,
            **kwargs,
        )

    try:
        resolved = resolve_model(client, model)
        body = build_chat_body(fixture, engine, resolved, reasoning_params)
        response = client.post("/chat/completions", json=body)
    except httpx.HTTPError as exc:
        return _result(error=f"{type(exc).__name__}: {exc}")

    add_check(
        checks, "http_ok", response.status_code == 200, f"HTTP {response.status_code}"
    )
    if response.status_code != 200:
        return _result(model_resolved=resolved, error=snippet(response.text))

    payload = response.json()
    message = first_message(payload)
    content = message.get("content") or ""
    finish_reason = first_finish_reason(payload)
    fields_observed = sorted(message.keys())

    reasoning_field = None
    reasoning_text = None
    for candidate in engine.reasoning_field_candidates():
        value = message.get(candidate)
        if isinstance(value, str) and value.strip():
            reasoning_field = candidate
            reasoning_text = value
            break

    reasoning_in_content = any(
        pattern.search(content) for pattern in engine.reasoning_leak_patterns()
    )

    add_check(
        checks,
        "reasoning_present",
        reasoning_field is not None,
        f"reasoning field {reasoning_field!r}; message fields: {fields_observed}",
    )
    add_check(
        checks,
        "no_inline_leak",
        not reasoning_in_content,
        "reasoning markup leaked into content"
        if reasoning_in_content
        else "content free of reasoning markup",
    )
    add_check(
        checks,
        "final_content_present",
        bool(content.strip()),
        f"content length {len(content)}",
    )

    return _result(
        model_resolved=resolved,
        reasoning_present=reasoning_field is not None,
        reasoning_field=reasoning_field,
        reasoning_in_content=reasoning_in_content,
        fields_observed=fields_observed,
        reasoning_snippet=snippet(reasoning_text),
        content_snippet=snippet(content),
        finish_reason=finish_reason,
    )
