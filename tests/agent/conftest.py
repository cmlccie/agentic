"""Shared fixtures for the agent runtime tests.

Agents under test are built from real `agent.yaml` / `server.yaml` files, with
two test hooks:

- ``resolve_model`` is patched so the spec's model name selects a scripted
  `FunctionModel` from `MODELS` (registered per test with `use_model`).
- A ``LocalTools`` capability (declarable in `agent.yaml`) gives agents local
  tools to call: ``forecast`` (quick) and ``slow`` (sleeps; for cancellation and
  task-promotion tests).
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest
import uvicorn
import yaml
from fastapi import FastAPI
from pydantic_ai import FunctionToolset
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models import Model
from pydantic_ai.models.function import (
    AgentInfo,
    DeltaThinkingPart,
    DeltaToolCall,
    FunctionModel,
)

import agentic.agent.spec as spec_module
from agentic.agent.app import create_app

MODELS: dict[str, Model] = {}


# --------------------------------------------------------------------------------------
# Test tools
# --------------------------------------------------------------------------------------

tools = FunctionToolset()
SLOW_STATE: dict[str, int] = {"started": 0, "cancelled": 0, "finished": 0}


@tools.tool_plain
async def forecast(city: str, api_key: str = "") -> dict[str, Any]:
    """Return the forecast for a city."""
    return {"city": city, "temp_c": 18, "sky": "sunny"}


@tools.tool_plain
async def slow(seconds: float) -> str:
    """Wait for a while, then return."""
    SLOW_STATE["started"] += 1
    try:
        await asyncio.sleep(seconds)
    except asyncio.CancelledError:
        SLOW_STATE["cancelled"] += 1
        raise
    SLOW_STATE["finished"] += 1
    return f"waited {seconds}s"


@dataclass
class LocalTools(AbstractCapability[Any]):
    """Test-only capability exposing the tools above."""

    def get_toolset(self) -> FunctionToolset:
        return tools


# --------------------------------------------------------------------------------------
# Scripted models
# --------------------------------------------------------------------------------------


def last_part(messages: list[ModelMessage]) -> Any:
    return messages[-1].parts[-1]


def scripted_model(
    *,
    answer: str | Callable[[list[ModelMessage], AgentInfo], str] = "done",
    thinking: str | None = None,
    note: str | None = None,
    tool: str | None = None,
    args: dict[str, Any] | None = None,
) -> FunctionModel:
    """A model that (optionally) thinks, writes a note, calls one tool, then answers.

    The answer may be a function of the messages (e.g. to echo the tool result
    or count history).
    """

    def final(messages: list[ModelMessage], info: AgentInfo) -> str:
        return answer(messages, info) if callable(answer) else answer

    def should_call_tool(messages: list[ModelMessage]) -> bool:
        return tool is not None and not isinstance(last_part(messages), ToolReturnPart)

    def request(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if not should_call_tool(messages):
            return ModelResponse(parts=[TextPart(final(messages, info))])
        parts: list[Any] = []
        if thinking:
            parts.append(ThinkingPart(thinking))
        if note:
            parts.append(TextPart(note))
        parts.append(ToolCallPart(tool, args or {}, tool_call_id="call_1"))
        return ModelResponse(parts=parts)

    async def stream(
        messages: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[Any]:
        if not should_call_tool(messages):
            if thinking and tool is None:
                yield {0: DeltaThinkingPart(content=thinking)}
            yield final(messages, info)
            return
        index = 0
        if thinking:
            yield {index: DeltaThinkingPart(content=thinking)}
            index += 1
        if note:
            yield note
            index += 1
        yield {
            index: DeltaToolCall(
                name=tool, json_args=json.dumps(args or {}), tool_call_id="call_1"
            )
        }

    return FunctionModel(request, stream_function=stream)


def use_model(name: str, model: Model) -> None:
    """Register the model used by agents whose spec says ``model: test:<name>``."""
    MODELS[name] = model


@pytest.fixture(autouse=True)
def _patch_models(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Route ``test:<name>`` model strings to `MODELS` and register LocalTools."""

    real_resolve = spec_module.resolve_model

    def resolve(model: str | None, secrets: Any) -> Any:
        if model and model.startswith("test:"):
            return MODELS[model.removeprefix("test:")]
        return real_resolve(model, secrets)

    monkeypatch.setattr(spec_module, "resolve_model", resolve)
    monkeypatch.setattr(
        spec_module,
        "CUSTOM_CAPABILITIES",
        (*spec_module.CUSTOM_CAPABILITIES, LocalTools),
    )
    SLOW_STATE.update(started=0, cancelled=0, finished=0)
    yield
    MODELS.clear()


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# --------------------------------------------------------------------------------------
# Config and apps
# --------------------------------------------------------------------------------------

DEFAULT_SERVER = {
    "agent_card": {"display_name": "Test Agent", "description": "An agent under test."}
}


def write_config(
    directory: Path,
    agent: dict[str, Any],
    server: dict[str, Any] | None = None,
) -> Path:
    """Write agent.yaml and server.yaml into ``directory`` and return it."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "agent.yaml").write_text(yaml.safe_dump(agent))
    (directory / "server.yaml").write_text(yaml.safe_dump(server or DEFAULT_SERVER))
    return directory


def make_app(
    tmp_path: Path,
    agent: dict[str, Any],
    server: dict[str, Any] | None = None,
    secrets: dict[str, str] | None = None,
    public_url: str = "http://test",
) -> FastAPI:
    """Build an app from config written under ``tmp_path`` (no file watching)."""
    config_dir = write_config(tmp_path / "config", agent, server)
    secrets_dir = tmp_path / "secrets"
    secrets_dir.mkdir(exist_ok=True)
    for key, value in (secrets or {}).items():
        (secrets_dir / key).write_text(value)
    return create_app(config_dir, secrets_dir, public_url, watch=False)


class RunningApp:
    """An app served in-process through httpx's ASGI transport."""

    def __init__(self, app: FastAPI) -> None:
        self.app = app
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test", timeout=30
        )

    async def __aenter__(self) -> RunningApp:
        self._lifespan = self.app.router.lifespan_context(self.app)
        await self._lifespan.__aenter__()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.client.aclose()
        await self._lifespan.__aexit__(None, None, None)


# --------------------------------------------------------------------------------------
# Real HTTP servers (for streaming timing and agent-to-agent tests)
# --------------------------------------------------------------------------------------


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Server:
    """Run an ASGI app with uvicorn on a background thread."""

    def __init__(self, app: FastAPI, port: int) -> None:
        self.port = port
        self.url = f"http://127.0.0.1:{port}"
        self._server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
        )
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    def __enter__(self) -> Server:
        self._thread.start()
        deadline = time.monotonic() + 10
        while not self._server.started:
            if time.monotonic() > deadline:
                raise RuntimeError("server did not start")
            time.sleep(0.02)
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=10)
