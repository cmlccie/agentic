"""Tool-call-parsing check: calls must arrive structured, not as raw text."""

import json
import time
from typing import List, Optional

import httpx
import jsonschema

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
from aiops.models import (
    CheckItem,
    ToolCallingResult,
    ToolCallModeResult,
    ToolChoiceMode,
)


def _run_mode(
    client: httpx.Client,
    engine: EngineAdapter,
    fixture: FixtureSpec,
    model: Optional[str],
    mode: str,
) -> ToolCallModeResult:
    expected_tool = fixture.expected.tool_name
    if mode == "forced":
        tool_choice = {"type": "function", "function": {"name": expected_tool}}
    else:
        tool_choice = "auto"

    result = ToolCallModeResult(mode=mode, passed=False)

    try:
        body = build_chat_body(fixture, engine, model, {"tool_choice": tool_choice})
        response = client.post("/chat/completions", json=body)
    except httpx.HTTPError as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        return result

    if response.status_code != 200:
        result.error = f"HTTP {response.status_code}: {snippet(response.text)}"
        return result

    payload = response.json()
    message = first_message(payload)
    content = message.get("content") or ""
    tool_calls = message.get("tool_calls") or []

    result.finish_reason = first_finish_reason(payload)
    result.content_snippet = snippet(content)
    result.tool_calls_present = bool(tool_calls)
    result.raw_text_leak = any(
        pattern.search(content) for pattern in engine.tool_call_leak_patterns()
    )
    # A forced tool call is allowed to report the engine's acceptable finish
    # reasons; "stop" after a forced call is tolerated by some engines, so the
    # adapter owns the accepted set.
    result.finish_reason_ok = (
        result.finish_reason in engine.acceptable_tool_finish_reasons()
    )

    if tool_calls:
        function = (tool_calls[0] or {}).get("function") or {}
        result.call_name = function.get("name")
        result.call_name_ok = result.call_name == expected_tool
        result.arguments_raw = function.get("arguments")
        try:
            arguments = json.loads(result.arguments_raw or "")
            result.arguments_valid_json = isinstance(arguments, dict)
        except (json.JSONDecodeError, TypeError):
            arguments = None
            result.arguments_valid_json = False

        schema = fixture.expected.arguments_schema
        if result.arguments_valid_json and schema is not None:
            validator = jsonschema.Draft202012Validator(schema)
            errors = sorted(validator.iter_errors(arguments), key=str)
            result.schema_errors = [error.message for error in errors]
            result.arguments_schema_valid = not errors
        elif result.arguments_valid_json:
            result.arguments_schema_valid = True

    result.passed = all(
        (
            result.tool_calls_present,
            result.call_name_ok,
            result.finish_reason_ok,
            result.arguments_valid_json,
            result.arguments_schema_valid,
            not result.raw_text_leak,
        )
    )
    return result


def check_tool_calling(
    client: httpx.Client,
    engine: EngineAdapter,
    fixture: FixtureSpec,
    model: Optional[str] = None,
    tool_choice_mode: ToolChoiceMode = "both",
) -> ToolCallingResult:
    started = time.perf_counter()
    checks: List[CheckItem] = []

    if not fixture.tools or not fixture.expected.tool_name:
        raise ValueError(
            f"Fixture {fixture.id!r} does not define tools and an expected tool name."
        )

    try:
        resolved = resolve_model(client, model)
    except httpx.HTTPError as exc:
        return ToolCallingResult(
            passed=False,
            engine=engine.name,
            fixture_id=fixture.id,
            duration_ms=(time.perf_counter() - started) * 1000,
            error=f"{type(exc).__name__}: {exc}",
        )

    modes = ["auto", "forced"] if tool_choice_mode == "both" else [tool_choice_mode]
    results = {}
    for mode in modes:
        mode_result = _run_mode(client, engine, fixture, resolved, mode)
        results[mode] = mode_result
        detail = (
            f"tool_calls_present={mode_result.tool_calls_present}, "
            f"call_name={mode_result.call_name!r}, "
            f"finish_reason={mode_result.finish_reason!r}, "
            f"arguments_schema_valid={mode_result.arguments_schema_valid}, "
            f"raw_text_leak={mode_result.raw_text_leak}"
        )
        if mode_result.error:
            detail = f"{detail}, error={mode_result.error}"
        add_check(checks, f"tool_choice_{mode}", mode_result.passed, detail)

    return ToolCallingResult(
        passed=bool(checks) and all(check.passed for check in checks),
        engine=engine.name,
        model_resolved=resolved,
        fixture_id=fixture.id,
        duration_ms=(time.perf_counter() - started) * 1000,
        checks=checks,
        results=results,
    )
