"""Versioned test-fixture library.

Fixtures are YAML files in this package, validated against `FixtureSpec` at load
time. Rules:

- Fixture ids are versioned (`-vN` suffix). A published fixture's semantics
  never change — add `-v(N+1)` instead, and keep the old one until nothing
  references it.
- `engine_overrides.<engine>` is a dict of request-body params merged over
  `params` for that engine only.
"""

import re
from functools import cache
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator

FixtureKind = Literal["inference", "reasoning", "tool_calling", "tps"]

_FIXTURE_DIR = Path(__file__).parent

_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*-v\d+$")


class FixtureExpected(BaseModel):
    """Deterministic expectations a fixture asserts."""

    content_contains: Optional[str] = Field(
        None, description="Case-insensitive substring the content must contain."
    )
    tool_name: Optional[str] = Field(
        None, description="Function name the model is expected to call."
    )
    arguments_schema: Optional[Dict[str, Any]] = Field(
        None, description="JSON Schema the call arguments must validate against."
    )
    min_completion_tokens: Optional[int] = Field(
        None, description="Minimum completion tokens for a run to count as valid."
    )


class FixtureSpec(BaseModel):
    """One test fixture."""

    id: str
    kind: FixtureKind
    description: str
    model_families: List[str] = Field(default_factory=lambda: ["*"])
    messages: List[Dict[str, Any]]
    params: Dict[str, Any] = Field(default_factory=dict)
    tools: Optional[List[Dict[str, Any]]] = None
    expected: FixtureExpected = Field(default_factory=FixtureExpected)
    engine_overrides: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def _versioned_id(cls, value: str) -> str:
        if not _ID_PATTERN.match(value):
            raise ValueError(f"Fixture id {value!r} must be kebab-case ending in -vN.")
        return value

    def expected_summary(self) -> str:
        parts = []
        if self.expected.content_contains:
            parts.append(f"content contains {self.expected.content_contains!r}")
        if self.expected.tool_name:
            parts.append(f"calls {self.expected.tool_name!r}")
        if self.expected.arguments_schema:
            parts.append("arguments validate against schema")
        if self.expected.min_completion_tokens:
            parts.append(f">= {self.expected.min_completion_tokens} completion tokens")
        return "; ".join(parts) or "valid completion"


@cache
def load_fixtures() -> Dict[str, FixtureSpec]:
    """Load and validate every fixture YAML in this package, keyed by id."""
    fixtures: Dict[str, FixtureSpec] = {}
    for path in sorted(_FIXTURE_DIR.glob("*.yaml")):
        with path.open() as handle:
            document = yaml.safe_load(handle) or {}
        for entry in document.get("fixtures", []):
            fixture = FixtureSpec.model_validate(entry)
            if fixture.id in fixtures:
                raise ValueError(f"Duplicate fixture id {fixture.id!r} in {path.name}.")
            fixtures[fixture.id] = fixture
    return fixtures


def get_fixture(fixture_id: str, kind: Optional[FixtureKind] = None) -> FixtureSpec:
    """Return a fixture by id, optionally asserting its kind."""
    fixtures = load_fixtures()
    if fixture_id not in fixtures:
        raise ValueError(f"Unknown fixture {fixture_id!r}; known: {sorted(fixtures)}.")
    fixture = fixtures[fixture_id]
    if kind is not None and fixture.kind != kind:
        raise ValueError(
            f"Fixture {fixture_id!r} is kind {fixture.kind!r}, expected {kind!r}."
        )
    return fixture
