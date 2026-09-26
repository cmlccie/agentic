"""Tests for hot reload, the interface gate (switches + auth), health, and the CLI."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from agentic.agent.app import bearer_token_matches
from agentic.agent.cli import app as cli
from agentic.agent.config import Secrets
from agentic.agent.runtime import AgentRuntime, load_snapshot

from .conftest import (
    DEFAULT_SERVER,
    RunningApp,
    make_app,
    scripted_model,
    use_model,
    write_config,
)

pytestmark = pytest.mark.anyio


# --------------------------------------------------------------------------------------
# Reload
# --------------------------------------------------------------------------------------


def runtime_for(tmp_path: Path, agent: dict[str, Any]) -> AgentRuntime:
    config_dir = write_config(tmp_path / "config", agent)
    secrets = Secrets(tmp_path / "secrets")
    return AgentRuntime(config_dir, secrets, load_snapshot(config_dir, secrets))


async def test_reload_swaps_in_the_new_configuration(tmp_path: Path) -> None:
    runtime = runtime_for(tmp_path, {"model": "test", "name": "one"})
    seen: list[tuple[str, str]] = []
    runtime.on_reload(lambda old, new: seen.append((old.model_name, new.model_name)))
    write_config(tmp_path / "config", {"model": "test", "name": "two"})
    assert runtime.reload() is True
    assert runtime.current.model_name == "two"
    assert runtime.current.generation == 2
    assert seen == [("one", "two")]


async def test_invalid_configuration_keeps_the_last_good_one(tmp_path: Path) -> None:
    runtime = runtime_for(tmp_path, {"model": "test", "name": "good"})
    (tmp_path / "config" / "agent.yaml").write_text("model: test\nbogus_key: 1\n")
    assert runtime.reload() is False
    assert runtime.current.model_name == "good"
    assert runtime.current.generation == 1
    write_config(tmp_path / "config", {"model": "test", "name": "fixed"})
    assert runtime.reload() is True
    assert runtime.current.model_name == "fixed"


async def test_file_changes_trigger_a_reload(tmp_path: Path) -> None:
    runtime = runtime_for(tmp_path, {"model": "test", "name": "before"})
    stop = asyncio.Event()
    task = asyncio.create_task(runtime.run(stop))
    await asyncio.sleep(0.3)  # let the watcher start
    write_config(tmp_path / "config", {"model": "test", "name": "after"})
    for _ in range(100):
        if runtime.current.model_name == "after":
            break
        await asyncio.sleep(0.1)
    stop.set()
    await asyncio.wait_for(task, 5)
    assert runtime.current.model_name == "after"


async def test_in_flight_requests_finish_on_the_old_agent(tmp_path: Path) -> None:
    """A reload mid-request must not disturb that request."""
    use_model(
        "main", scripted_model(tool="slow", args={"seconds": 0.3}, answer="old agent")
    )
    app = make_app(
        tmp_path, {"model": "test:main", "name": "a", "capabilities": ["LocalTools"]}
    )
    async with RunningApp(app) as running:
        request = asyncio.create_task(
            running.client.post(
                "/v1/chat/completions",
                json={"model": "a", "messages": [{"role": "user", "content": "hi"}]},
            )
        )
        await asyncio.sleep(0.1)
        write_config(tmp_path / "config", {"model": "test", "name": "b"})
        assert app.state.runtime.reload()
        response = await request
        assert response.json()["choices"][0]["message"]["content"] == "old agent"
        models = (await running.client.get("/v1/models")).json()
        assert models["data"][0]["id"] == "b"


# --------------------------------------------------------------------------------------
# Health, interface switches, auth
# --------------------------------------------------------------------------------------


async def test_health_probes(tmp_path: Path) -> None:
    use_model("main", scripted_model())
    async with RunningApp(
        make_app(tmp_path, {"model": "test:main", "name": "h"})
    ) as running:
        assert (await running.client.get("/health/live")).json() == {"status": "alive"}
        ready = (await running.client.get("/health/ready")).json()
        assert ready == {"status": "ready", "agent": "h", "generation": 1}


@pytest.mark.parametrize(
    "interfaces, openai_status, a2a_status",
    [({"openai": False}, 404, 200), ({"a2a": False}, 200, 404)],
)
async def test_interfaces_can_be_disabled(
    tmp_path: Path, interfaces: dict[str, bool], openai_status: int, a2a_status: int
) -> None:
    use_model("main", scripted_model())
    server = {**DEFAULT_SERVER, "interfaces": interfaces}
    async with RunningApp(
        make_app(tmp_path, {"model": "test:main"}, server)
    ) as running:
        assert (await running.client.get("/v1/models")).status_code == openai_status
        assert (
            await running.client.get("/.well-known/agent-card.json")
        ).status_code == a2a_status


AUTH_SERVER = {**DEFAULT_SERVER, "auth": {"bearer_token_secret": "api_token"}}


async def test_bearer_auth(tmp_path: Path) -> None:
    use_model("main", scripted_model())
    app = make_app(
        tmp_path, {"model": "test:main"}, AUTH_SERVER, secrets={"api_token": "t0k"}
    )
    body = {"model": "x", "messages": [{"role": "user", "content": "hi"}]}
    async with RunningApp(app) as running:
        client = running.client
        missing = await client.post("/v1/chat/completions", json=body)
        assert missing.status_code == 401
        assert missing.json()["error"]["code"] == "invalid_api_key"
        assert missing.headers["www-authenticate"] == "Bearer"
        wrong = await client.get("/v1/models", headers={"Authorization": "Bearer nope"})
        assert wrong.status_code == 401
        ok = await client.post(
            "/v1/chat/completions", json=body, headers={"Authorization": "Bearer t0k"}
        )
        assert ok.status_code == 200
        assert (await client.post("/a2a", json={})).status_code == 401
        # public endpoints
        assert (await client.get("/health/ready")).status_code == 200
        assert (await client.get("/.well-known/agent-card.json")).status_code == 200


async def test_auth_fails_closed_when_the_secret_is_missing(tmp_path: Path) -> None:
    use_model("main", scripted_model())
    app = make_app(tmp_path, {"model": "test:main"}, AUTH_SERVER)
    async with RunningApp(app) as running:
        response = await running.client.get(
            "/v1/models", headers={"Authorization": "Bearer anything"}
        )
    assert response.status_code == 503


@pytest.mark.parametrize(
    "header, expected",
    [
        ("Bearer t0k", True),
        ("bearer t0k", True),
        ("Bearer  t0k ", True),
        ("Bearer t0k2", False),
        ("Basic t0k", False),
        ("", False),
        (None, False),
    ],
)
def test_bearer_token_matching(header: str | None, expected: bool) -> None:
    assert bearer_token_matches(header, "t0k") is expected


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


runner = CliRunner()


@pytest.mark.parametrize(
    "command", [[], ["serve", "--help"], ["chat", "--help"], ["web", "--help"]]
)
def test_cli_help(command: list[str]) -> None:
    result = runner.invoke(cli, [*command] or ["--help"])
    assert result.exit_code == 0
    assert "Usage" in result.output


def test_serve_with_invalid_config_exits_with_a_clear_error(tmp_path: Path) -> None:
    config = tmp_path / "config"
    config.mkdir()
    (config / "agent.yaml").write_text(yaml.safe_dump({"model": "test"}))
    (config / "server.yaml").write_text(
        yaml.safe_dump({"agent_card": {"display_name": "x"}})
    )
    result = runner.invoke(
        cli, ["serve", "--config-dir", str(config), "--secrets-dir", str(tmp_path)]
    )
    assert result.exit_code == 2
    assert "Configuration error" in result.output
    assert "description" in result.output
