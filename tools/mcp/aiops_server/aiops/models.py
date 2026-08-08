"""Pydantic result models for the AIOps MCP tools.

Every check tool returns a subclass of `BaseCheckResult`: `passed` is the single
deterministic verdict, and `checks` itemizes the evidence that produced it.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

Engine = Literal["sglang", "vllm", "nim"]

ToolChoiceMode = Literal["auto", "forced", "both"]


class CheckItem(BaseModel):
    """One deterministic check and its outcome."""

    name: str = Field(..., description="Short identifier of the check.")
    passed: bool = Field(..., description="Whether this check passed.")
    detail: str = Field(..., description="Evidence for the outcome.")


class BaseCheckResult(BaseModel):
    """Fields shared by every check tool result."""

    passed: bool = Field(
        ..., description="Overall verdict: true only if every check passed."
    )
    engine: Engine = Field(..., description="Inferencing engine adapter used.")
    model_resolved: Optional[str] = Field(
        None, description="Served model name used for the request(s)."
    )
    fixture_id: str = Field(..., description="Fixture the check ran with.")
    duration_ms: float = Field(..., description="Total tool wall-clock in ms.")
    error: Optional[str] = Field(
        None, description="Transport/HTTP failure, if any. Checks may be partial."
    )
    checks: List[CheckItem] = Field(
        default_factory=list, description="Itemized deterministic checks."
    )


class EndpointInfo(BaseModel):
    """Result of the `get_endpoint_info` readiness probe."""

    reachable: bool = Field(..., description="GET /models succeeded.")
    engine: Engine = Field(..., description="Inferencing engine adapter used.")
    served_models: List[str] = Field(
        default_factory=list, description="Model ids reported by GET /models."
    )
    engine_health_ok: Optional[bool] = Field(
        None, description="Engine health endpoint verdict; null if not probed."
    )
    latency_ms: float = Field(..., description="GET /models latency in ms.")
    error: Optional[str] = Field(None, description="Failure detail, if any.")


class InferenceResult(BaseCheckResult):
    """Result of `check_inference`."""

    latency_ms: Optional[float] = Field(
        None, description="Completion request latency in ms."
    )
    completion_snippet: Optional[str] = Field(
        None, description="First characters of the completion content."
    )
    finish_reason: Optional[str] = Field(None, description="Reported finish_reason.")
    prompt_tokens: Optional[int] = Field(None, description="usage.prompt_tokens.")
    completion_tokens: Optional[int] = Field(
        None, description="usage.completion_tokens."
    )
    total_tokens: Optional[int] = Field(None, description="usage.total_tokens.")


class ReasoningResult(BaseCheckResult):
    """Result of `check_reasoning`."""

    reasoning_present: bool = Field(
        False, description="A structured reasoning field was present and non-empty."
    )
    reasoning_field: Optional[str] = Field(
        None, description="Which message field held the reasoning content."
    )
    reasoning_in_content: bool = Field(
        False, description="Reasoning markup leaked inline into message.content."
    )
    fields_observed: List[str] = Field(
        default_factory=list,
        description="Keys observed on the response message object (the failure diagnostic).",
    )
    reasoning_snippet: Optional[str] = Field(
        None, description="First characters of the reasoning content."
    )
    content_snippet: Optional[str] = Field(
        None, description="First characters of the final content."
    )
    finish_reason: Optional[str] = Field(None, description="Reported finish_reason.")


class ToolCallModeResult(BaseModel):
    """Per-`tool_choice` mode result within `check_tool_calling`."""

    mode: Literal["auto", "forced"] = Field(..., description="tool_choice mode.")
    passed: bool = Field(..., description="All checks for this mode passed.")
    tool_calls_present: bool = Field(
        False, description="message.tool_calls was present and non-empty."
    )
    call_name: Optional[str] = Field(None, description="Function name of the call.")
    call_name_ok: bool = Field(
        False, description="Call name matched the fixture's expected tool."
    )
    finish_reason: Optional[str] = Field(None, description="Reported finish_reason.")
    finish_reason_ok: bool = Field(
        False, description="finish_reason was acceptable for a tool call."
    )
    arguments_raw: Optional[str] = Field(
        None, description="Raw function.arguments string."
    )
    arguments_valid_json: bool = Field(
        False, description="function.arguments parsed as JSON."
    )
    arguments_schema_valid: bool = Field(
        False, description="Arguments validated against the fixture's JSON Schema."
    )
    schema_errors: List[str] = Field(
        default_factory=list, description="JSON Schema validation errors."
    )
    raw_text_leak: bool = Field(
        False, description="Tool-call markup leaked as raw text into content."
    )
    content_snippet: Optional[str] = Field(
        None, description="First characters of message.content."
    )
    error: Optional[str] = Field(None, description="Transport/HTTP failure, if any.")


class ToolCallingResult(BaseCheckResult):
    """Result of `check_tool_calling`."""

    results: Dict[str, ToolCallModeResult] = Field(
        default_factory=dict, description="Per-mode results keyed 'auto'/'forced'."
    )


class TpsRun(BaseModel):
    """One measured streaming run within `measure_tps`."""

    decode_tps: float = Field(
        ..., description="Decode tokens/sec, first-to-last token (excludes TTFT)."
    )
    ttft_ms: float = Field(..., description="Time to first token in ms.")
    total_ms: float = Field(..., description="Total stream wall-clock in ms.")
    completion_tokens: int = Field(..., description="Completion tokens counted.")
    finish_reason: Optional[str] = Field(None, description="Reported finish_reason.")
    token_count_source: Literal["usage", "chunk_count"] = Field(
        ..., description="Whether tokens came from usage or counted stream chunks."
    )


class TpsResult(BaseCheckResult):
    """Result of `measure_tps`."""

    concurrency: int = Field(..., description="Parallel streams in the measured round.")
    max_tokens: int = Field(..., description="max_tokens per stream.")
    runs: List[TpsRun] = Field(default_factory=list, description="Measured runs.")
    decode_tps_mean: Optional[float] = Field(
        None, description="Mean decode TPS across runs."
    )
    decode_tps_stddev: Optional[float] = Field(
        None, description="Sample stddev of decode TPS across runs (null if <2 runs)."
    )
    ttft_ms_mean: Optional[float] = Field(None, description="Mean TTFT in ms.")
    aggregate_tps: Optional[float] = Field(
        None,
        description="Concurrent mode only: total completion tokens / round wall-clock.",
    )


class FixtureInfo(BaseModel):
    """Catalog entry for one test fixture."""

    id: str = Field(..., description="Fixture id (versioned, ends in -vN).")
    kind: Literal["inference", "reasoning", "tool_calling", "tps"] = Field(
        ..., description="Which check the fixture drives."
    )
    description: str = Field(..., description="What the fixture exercises.")
    model_families: List[str] = Field(
        ..., description="Model families the fixture targets ('*' = generic)."
    )
    default_params: Dict[str, Any] = Field(
        default_factory=dict, description="Request params the fixture applies."
    )
    expected_summary: str = Field(
        ..., description="Summary of the fixture's expectations."
    )


class FixtureCatalog(BaseModel):
    """Result of `list_fixtures`."""

    server_version: str = Field(..., description="AIOps server package version.")
    fixtures: List[FixtureInfo] = Field(
        default_factory=list, description="Matching fixtures."
    )
