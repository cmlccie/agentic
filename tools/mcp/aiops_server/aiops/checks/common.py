"""Shared helpers for all checks: request assembly, model resolution, snippets."""

from typing import Any, Dict, List, Optional

import httpx

from aiops.engines import EngineAdapter
from aiops.fixtures import FixtureSpec
from aiops.models import CheckItem

SNIPPET_LENGTH = 240

VALID_STOP_FINISH_REASONS = {"stop", "length"}


def snippet(text: Optional[str], length: int = SNIPPET_LENGTH) -> Optional[str]:
    if text is None:
        return None
    return text[:length]


def build_chat_body(
    fixture: FixtureSpec,
    engine: EngineAdapter,
    model: Optional[str],
    extra_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble a chat-completions body from fixture tiers.

    Merge order (later wins): fixture.params -> fixture.engine_overrides[engine]
    -> extra_params (explicit per-call overrides).
    """
    body: Dict[str, Any] = {"messages": fixture.messages}
    if model:
        body["model"] = model
    if fixture.tools:
        body["tools"] = fixture.tools
    body.update(fixture.params)
    body.update(fixture.engine_overrides.get(engine.name, {}))
    body.update(extra_params or {})
    return body


def resolve_model(client: httpx.Client, model: Optional[str]) -> Optional[str]:
    """Return the model name to use; query GET /models when not supplied."""
    if model:
        return model
    response = client.get("/models")
    response.raise_for_status()
    data = response.json().get("data", [])
    return data[0]["id"] if data else None


def first_message(payload: Dict[str, Any]) -> Dict[str, Any]:
    choices = payload.get("choices") or []
    if not choices:
        return {}
    return choices[0].get("message") or {}


def first_finish_reason(payload: Dict[str, Any]) -> Optional[str]:
    choices = payload.get("choices") or []
    if not choices:
        return None
    return choices[0].get("finish_reason")


def add_check(checks: List[CheckItem], name: str, passed: bool, detail: str) -> bool:
    checks.append(CheckItem(name=name, passed=passed, detail=detail))
    return passed
