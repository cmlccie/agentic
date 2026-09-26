"""Tests for the A2A server: spec compliance, message vs task, activity, storage."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.helpers import new_text_message
from a2a.server.context import ServerCallContext
from a2a.types import (
    CancelTaskRequest,
    GetTaskRequest,
    Role,
    SendMessageConfiguration,
    SendMessageRequest,
    Task,
    TaskState,
    TaskStatus,
)
from google.protobuf.json_format import MessageToDict
from pydantic_ai.messages import ModelRequest, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from agentic.agent.a2a_server import push_url_validator
from agentic.agent.config import PushNotificationsConfig
from agentic.agent.stores import BoundedMemoryTaskStore, trim_history
from agentic.agent.streaming import ACTIVITY_EXTENSION

from .conftest import SLOW_STATE, RunningApp, make_app, scripted_model, use_model

pytestmark = pytest.mark.anyio

AGENT = {"model": "test:main", "name": "test-agent", "capabilities": ["LocalTools"]}


def server(**a2a: Any) -> dict[str, Any]:
    return {
        "agent_card": {
            "display_name": "Test Agent",
            "description": "An agent under test.",
            "skills": [
                {
                    "id": "s1",
                    "name": "Weather",
                    "description": "Forecasts",
                    "tags": ["w"],
                }
            ],
        },
        "a2a": a2a,
    }


async def a2a_client(running: RunningApp, streaming: bool = True) -> Any:
    card = await A2ACardResolver(running.client, "http://test/a2a").get_agent_card()
    return ClientFactory(
        ClientConfig(httpx_client=running.client, streaming=streaming)
    ).create(card)


async def send(
    client: Any,
    text: str,
    *,
    context_id: str | None = None,
    return_immediately: bool = False,
) -> list[Any]:
    request = SendMessageRequest(
        message=new_text_message(text, context_id=context_id, role=Role.ROLE_USER),
        configuration=SendMessageConfiguration(return_immediately=return_immediately),
    )
    return [event async for event in client.send_message(request)]


def kinds(events: list[Any]) -> list[str]:
    return [e.WhichOneof("payload") for e in events]


def states(events: list[Any]) -> list[str]:
    names = []
    for e in events:
        match e.WhichOneof("payload"):
            case "task":
                names.append(TaskState.Name(e.task.status.state))
            case "status_update":
                names.append(TaskState.Name(e.status_update.status.state))
    return [n.removeprefix("TASK_STATE_") for n in names]


def activity_kinds(events: list[Any]) -> list[str]:
    result = []
    for e in events:
        if e.WhichOneof(
            "payload"
        ) == "status_update" and e.status_update.status.HasField("message"):
            meta = MessageToDict(e.status_update.status.message.metadata)
            if ACTIVITY_EXTENSION in meta:
                result.append(meta[ACTIVITY_EXTENSION]["kind"])
    return result


def registry_size(running: RunningApp) -> int:
    return len(running.app.state.a2a.handler._active_task_registry._active_tasks)


# --------------------------------------------------------------------------------------
# Agent card
# --------------------------------------------------------------------------------------


async def test_agent_card_is_spec_compliant(tmp_path: Path) -> None:
    use_model("main", scripted_model())
    async with RunningApp(make_app(tmp_path, AGENT, server())) as running:
        for path in (
            "/.well-known/agent-card.json",
            "/a2a/.well-known/agent-card.json",
            "/.well-known/agent.json",
        ):
            card = (await running.client.get(path)).json()
            assert card["name"] == "Test Agent"
        interfaces = card["supportedInterfaces"]
        assert {(i["protocolBinding"], i["protocolVersion"]) for i in interfaces} == {
            ("JSONRPC", "1.0"),
            ("JSONRPC", "0.3"),
        }
        assert all(i["url"] == "http://test/a2a" for i in interfaces)
        assert card["capabilities"]["streaming"] is True
        assert card["capabilities"].get("pushNotifications", False) is False
        assert card["capabilities"]["extensions"][0]["uri"] == ACTIVITY_EXTENSION
        assert card["skills"][0]["id"] == "s1"
        for field in (
            "version",
            "defaultInputModes",
            "defaultOutputModes",
            "description",
        ):
            assert field in card
        assert "securitySchemes" not in card
        # legacy (0.3) clients get the flattened fields too
        assert card["url"] == "http://test/a2a"


async def test_card_declares_bearer_auth_when_enabled(tmp_path: Path) -> None:
    use_model("main", scripted_model())
    config = server()
    config["auth"] = {"bearer_token_secret": "api_token"}
    app = make_app(tmp_path, AGENT, config, secrets={"api_token": "t0k"})
    async with RunningApp(app) as running:
        card = (await running.client.get("/.well-known/agent-card.json")).json()
    assert (
        card["securitySchemes"]["bearer"]["httpAuthSecurityScheme"]["scheme"]
        == "Bearer"
    )
    assert card["securityRequirements"] == [{"schemes": {"bearer": {}}}]


# --------------------------------------------------------------------------------------
# Message vs task
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("streaming", [False, True])
async def test_quick_answer_is_a_direct_message(
    tmp_path: Path, streaming: bool
) -> None:
    use_model("main", scripted_model(answer="quick"))
    async with RunningApp(make_app(tmp_path, AGENT, server())) as running:
        events = await send(await a2a_client(running, streaming), "hi")
        assert kinds(events) == ["message"]
        message = events[0].message
        assert message.role == Role.ROLE_AGENT
        assert message.parts[0].text == "quick"
        assert message.context_id
        assert not message.task_id
        await asyncio.sleep(0.05)
        assert registry_size(running) == 0


async def test_tool_use_promotes_to_a_task_with_activity(tmp_path: Path) -> None:
    use_model(
        "main",
        scripted_model(
            thinking="planning",
            note="Checking.",
            tool="forecast",
            args={"city": "Paris", "api_key": "s3cret"},
            answer="Sunny in Paris.",
        ),
    )
    async with RunningApp(make_app(tmp_path, AGENT, server())) as running:
        events = await send(await a2a_client(running), "weather?")
        await asyncio.sleep(0.05)
        assert registry_size(running) == 0

    assert kinds(events)[0] == "task"
    assert states(events)[0] == "SUBMITTED"
    assert states(events)[-1] == "COMPLETED"
    assert set(states(events)[1:-1]) == {"WORKING"}
    assert activity_kinds(events) == ["thinking", "note", "tool_call", "tool_result"]
    text = str(events)
    assert "s3cret" not in text
    artifacts = [
        e.artifact_update
        for e in events
        if e.WhichOneof("payload") == "artifact_update"
    ]
    assert len(artifacts) == 1
    assert artifacts[0].artifact.name == "response"
    assert artifacts[0].artifact.parts[0].text == "Sunny in Paris."
    assert artifacts[0].last_chunk


async def test_blocking_send_returns_the_completed_task(tmp_path: Path) -> None:
    use_model(
        "main", scripted_model(tool="forecast", args={"city": "a"}, answer="done!")
    )
    async with RunningApp(make_app(tmp_path, AGENT, server())) as running:
        events = await send(await a2a_client(running, streaming=False), "hi")
    assert kinds(events) == ["task"]
    task = events[0].task
    assert task.status.state == TaskState.TASK_STATE_COMPLETED
    assert task.artifacts[0].parts[0].text == "done!"


async def test_slow_answer_promotes_after_the_threshold(tmp_path: Path) -> None:
    async def stream(messages: Any, info: AgentInfo) -> AsyncIterator[str]:
        await asyncio.sleep(0.4)
        yield "slow but no tools"

    use_model("main", FunctionModel(stream_function=stream))
    config = server(promote_after_seconds=0.1)
    async with RunningApp(make_app(tmp_path, AGENT, config)) as running:
        events = await send(await a2a_client(running), "hi")
    assert kinds(events)[0] == "task"
    assert states(events)[-1] == "COMPLETED"


@pytest.mark.parametrize(
    "mode, expected",
    [("task", "task"), ("message", "message")],
)
async def test_response_mode_forces_a_shape(
    tmp_path: Path, mode: str, expected: str
) -> None:
    use_model("main", scripted_model(tool="forecast", args={"city": "a"}, answer="x"))
    async with RunningApp(
        make_app(tmp_path, AGENT, server(response_mode=mode))
    ) as running:
        events = await send(await a2a_client(running), "hi")
    assert kinds(events)[0] == expected
    if expected == "message":
        assert kinds(events) == ["message"]
        assert events[0].message.parts[0].text == "x"


async def test_return_immediately_gets_a_task_to_poll(tmp_path: Path) -> None:
    use_model(
        "main", scripted_model(tool="slow", args={"seconds": 0.3}, answer="finally")
    )
    async with RunningApp(make_app(tmp_path, AGENT, server())) as running:
        client = await a2a_client(running, streaming=False)
        events = await send(client, "hi", return_immediately=True)
        assert kinds(events) == ["task"]
        task_id = events[0].task.id
        assert events[0].task.status.state in (
            TaskState.TASK_STATE_SUBMITTED,
            TaskState.TASK_STATE_WORKING,
        )
        for _ in range(100):
            task = await client.get_task(GetTaskRequest(id=task_id))
            if task.status.state == TaskState.TASK_STATE_COMPLETED:
                break
            await asyncio.sleep(0.05)
        assert task.status.state == TaskState.TASK_STATE_COMPLETED
        assert task.artifacts[0].parts[0].text == "finally"


async def test_follow_ups_continue_the_conversation(tmp_path: Path) -> None:
    def answer(messages: Any, info: AgentInfo) -> str:
        prompts = [
            p.content
            for m in messages
            if isinstance(m, ModelRequest)
            for p in m.parts
            if isinstance(p, UserPromptPart)
        ]
        return f"turn {len(prompts)}: {' / '.join(prompts)}"

    use_model("main", scripted_model(answer=answer))
    async with RunningApp(make_app(tmp_path, AGENT, server())) as running:
        client = await a2a_client(running)
        first = await send(client, "one")
        context_id = first[0].message.context_id
        second = await send(client, "two", context_id=context_id)
        other = await send(client, "fresh")
    assert first[0].message.parts[0].text == "turn 1: one"
    assert second[0].message.parts[0].text == "turn 2: one / two"
    assert other[0].message.parts[0].text == "turn 1: fresh"


async def test_failures_become_failed_tasks_without_leaking_details(
    tmp_path: Path,
) -> None:
    async def stream(messages: Any, info: AgentInfo) -> AsyncIterator[str]:
        raise RuntimeError("database password is hunter2")
        yield ""  # pragma: no cover

    use_model("main", FunctionModel(stream_function=stream))
    async with RunningApp(make_app(tmp_path, AGENT, server())) as running:
        events = await send(await a2a_client(running), "hi")
    assert states(events)[-1] == "FAILED"
    assert "RuntimeError" in str(events)
    assert "hunter2" not in str(events)


async def test_cancel_stops_the_running_agent(tmp_path: Path) -> None:
    use_model("main", scripted_model(tool="slow", args={"seconds": 30}))
    async with RunningApp(make_app(tmp_path, AGENT, server())) as running:
        client = await a2a_client(running, streaming=False)
        events = await send(client, "hi", return_immediately=True)
        task_id = events[0].task.id
        for _ in range(100):
            if SLOW_STATE["started"]:
                break
            await asyncio.sleep(0.02)
        task = await client.cancel_task(CancelTaskRequest(id=task_id))
        assert task.status.state == TaskState.TASK_STATE_CANCELED
        for _ in range(100):
            if SLOW_STATE["cancelled"]:
                break
            await asyncio.sleep(0.02)
        assert SLOW_STATE == {"started": 1, "cancelled": 1, "finished": 0}
        task = await client.get_task(GetTaskRequest(id=task_id))
        assert task.status.state == TaskState.TASK_STATE_CANCELED


async def test_a2a_0_3_clients_are_supported(tmp_path: Path) -> None:
    use_model("main", scripted_model(answer="legacy ok"))
    async with RunningApp(make_app(tmp_path, AGENT, server())) as running:
        response = await running.client.post(
            "/a2a",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "message/send",
                "params": {
                    "message": {
                        "kind": "message",
                        "messageId": "m1",
                        "role": "user",
                        "parts": [{"kind": "text", "text": "hi"}],
                    }
                },
            },
        )
    result = response.json()["result"]
    assert result["kind"] == "message"
    assert result["parts"][0]["text"] == "legacy ok"


async def test_quick_replies_do_not_leak_active_tasks(tmp_path: Path) -> None:
    """Guards the a2a-sdk workaround in AgentRequestHandler (see its docstring)."""
    use_model("main", scripted_model(answer="q"))
    async with RunningApp(make_app(tmp_path, AGENT, server())) as running:
        for streaming in (False, True, False, True):
            await send(await a2a_client(running, streaming), "hi")
        await asyncio.sleep(0.05)
        assert registry_size(running) == 0


# --------------------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------------------


async def test_sql_store_persists_tasks_and_history_across_restarts(
    tmp_path: Path,
) -> None:
    def answer(messages: Any, info: AgentInfo) -> str:
        return f"{sum(isinstance(m, ModelRequest) for m in messages)} requests"

    use_model(
        "main", scripted_model(tool="forecast", args={"city": "a"}, answer=answer)
    )
    db = tmp_path / "agent.db"
    secrets = {"a2a.database_url": f"sqlite+aiosqlite:///{db}"}
    config = server(store={"backend": "sql"})

    async with RunningApp(
        make_app(tmp_path / "one", AGENT, config, secrets)
    ) as running:
        events = await send(await a2a_client(running), "first")
        task_id = events[0].task.id
        context_id = events[0].task.context_id

    async with RunningApp(
        make_app(tmp_path / "two", AGENT, config, secrets)
    ) as running:
        client = await a2a_client(running)
        task = await client.get_task(GetTaskRequest(id=task_id))
        assert task.status.state == TaskState.TASK_STATE_COMPLETED
        events = await send(client, "second", context_id=context_id)
    artifact = [e for e in events if e.WhichOneof("payload") == "artifact_update"][0]
    # first turn: prompt, tool return; second turn adds prompt, tool return
    assert artifact.artifact_update.artifact.parts[0].text == "4 requests"


async def test_sql_store_requires_the_database_secret(tmp_path: Path) -> None:
    from agentic.agent.config import ConfigError

    use_model("main", scripted_model())
    with pytest.raises(ConfigError, match="a2a.database_url"):
        make_app(tmp_path, AGENT, server(store={"backend": "sql"}))


async def test_bounded_task_store_evicts_oldest_finished_tasks() -> None:
    store = BoundedMemoryTaskStore(max_tasks=2)
    ctx = ServerCallContext()

    def task(task_id: str, state: int) -> Task:
        return Task(id=task_id, context_id="c", status=TaskStatus(state=state))

    await store.save(task("running", TaskState.TASK_STATE_WORKING), ctx)
    await store.save(task("done-1", TaskState.TASK_STATE_COMPLETED), ctx)
    await store.save(task("done-2", TaskState.TASK_STATE_COMPLETED), ctx)
    assert await store.get("running", ctx) is not None
    assert await store.get("done-1", ctx) is None
    assert await store.get("done-2", ctx) is not None


def test_trim_history_cuts_only_at_turn_boundaries() -> None:
    from pydantic_ai.messages import ModelResponse, ToolCallPart, ToolReturnPart

    def turn(n: int) -> list[Any]:
        return [
            ModelRequest(parts=[UserPromptPart(f"q{n}")]),
            ModelResponse(parts=[ToolCallPart("t", {}, tool_call_id=f"c{n}")]),
            ModelRequest(parts=[ToolReturnPart("t", "r", tool_call_id=f"c{n}")]),
            ModelResponse(parts=[TextPart(f"a{n}")]),
        ]

    history = turn(1) + turn(2) + turn(3)
    assert trim_history(history, 100) == history
    assert trim_history(history, 5) == history[8:]
    assert trim_history(history, 8) == history[4:]
    assert trim_history(history, 2) == history[8:]  # never splits a turn


@pytest.mark.parametrize(
    "enabled, hosts, url, allowed",
    [
        (False, [], "https://hooks.example.com/x", False),
        (True, [], "https://hooks.example.com/x", True),
        (True, [], "file:///etc/passwd", False),
        (True, [], "ftp://example.com", False),
        (True, [], "http://127.0.0.1:8080/x", False),
        (True, [], "http://10.1.2.3/x", False),
        (True, [], "http://169.254.169.254/latest/meta-data", False),
        (True, [], "http://[::1]/x", False),
        (True, [], "http://localhost/x", False),
        (True, ["hooks.example.com"], "https://hooks.example.com/x", True),
        (True, ["hooks.example.com"], "https://evil.com/x", False),
        (True, [".example.com"], "https://a.b.example.com/x", True),
        (True, [".example.com"], "https://example.com.evil.com/x", False),
        (True, ["10.1.2.3"], "http://10.1.2.3/x", True),
    ],
)
async def test_push_url_policy(
    enabled: bool, hosts: list[str], url: str, allowed: bool
) -> None:
    config = PushNotificationsConfig(enabled=enabled, allowed_hosts=hosts)
    assert await push_url_validator(config)(url) is allowed


async def test_inline_push_configs_are_refused_when_disabled(tmp_path: Path) -> None:
    """The a2a-sdk stores inline push configs even when the card disables push."""
    from a2a.types import TaskPushNotificationConfig

    use_model("main", scripted_model(tool="forecast", args={"city": "a"}))
    async with RunningApp(make_app(tmp_path, AGENT, server())) as running:
        client = await a2a_client(running, streaming=False)
        request = SendMessageRequest(
            message=new_text_message("hi", role=Role.ROLE_USER),
            configuration=SendMessageConfiguration(
                task_push_notification_config=TaskPushNotificationConfig(
                    url="https://hooks.example.com/x"
                )
            ),
        )
        with pytest.raises(Exception, match="(?i)push notification url"):
            async for _ in client.send_message(request):
                pass


def test_agent_card_fields_are_plain_strings_under_pure_python_protobuf() -> None:
    """Alpine images use the pure-Python protobuf runtime, which stores str(enum)."""
    import os
    import subprocess
    import sys

    script = (
        "from agentic.agent.a2a_server import build_agent_card\n"
        "from agentic.agent.config import ServerSpec\n"
        "spec = ServerSpec.model_validate("
        "{'agent_card': {'display_name': 'A', 'description': 'B'}})\n"
        "card = build_agent_card(spec, 'http://x', auth=True)\n"
        "from google.protobuf.internal import api_implementation\n"
        "assert api_implementation.Type() == 'python', api_implementation.Type()\n"
        "print(sorted({i.protocol_binding for i in card.supported_interfaces}))\n"
    )
    env = {**os.environ, "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python"}
    result = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "['JSONRPC']"


async def test_client_leaving_before_a_quick_reply_does_not_leak(
    tmp_path: Path,
) -> None:
    """A streaming consumer cancelled before the reply shape is decided."""
    from a2a.server.context import ServerCallContext

    async def stream(messages: Any, info: AgentInfo) -> AsyncIterator[str]:
        await asyncio.sleep(0.3)
        yield "late but quick"

    use_model("main", FunctionModel(stream_function=stream))
    app = make_app(tmp_path, AGENT, server(promote_after_seconds=5))
    async with RunningApp(app) as running:
        handler = running.app.state.a2a.handler
        registry = handler._active_task_registry._active_tasks

        async def consume() -> None:
            request = SendMessageRequest(
                message=new_text_message("hi", role=Role.ROLE_USER)
            )
            async for _ in handler.on_message_send_stream(request, ServerCallContext()):
                pass

        consumer = asyncio.create_task(consume())
        await asyncio.sleep(0.1)
        consumer.cancel()  # the client disconnects
        await asyncio.gather(consumer, return_exceptions=True)
        for _ in range(40):
            await asyncio.sleep(0.05)
            if not registry:
                break
        assert registry == {}
        assert getattr(handler, "_awaiting_reply", {}) == {}
