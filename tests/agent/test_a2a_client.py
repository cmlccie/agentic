"""Tests for the A2AAgent capability: orchestrators delegating to remote agents."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from a2a.helpers import new_task_from_user_message, new_text_part
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    TaskState,
)
from a2a.utils import TransportProtocol
from a2a.utils.constants import PROTOCOL_VERSION_1_0
from pydantic_ai.messages import ModelRequest, ToolReturnPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel
from starlette.applications import Starlette

from agentic.agent.a2a_client import (
    A2AAgentToolset,
    _Outcome,
    tool_description_for,
    tool_name_for,
)

from .conftest import (
    SLOW_STATE,
    RunningApp,
    Server,
    free_port,
    make_app,
    scripted_model,
    use_model,
)

pytestmark = pytest.mark.anyio

WORKER_SERVER = {
    "agent_card": {
        "display_name": "Weather Agent",
        "description": "Knows the weather.",
        "skills": [
            {
                "id": "f",
                "name": "Forecast",
                "description": "3-day forecasts",
                "tags": [],
            }
        ],
    }
}


def orchestrator(url: str, **extra: Any) -> dict[str, Any]:
    return {
        "model": "test:orchestrator",
        "name": "orchestrator",
        "capabilities": [{"A2AAgent": {"url": url, **extra}}],
    }


def delegating_model(
    request: str = "weather in Paris?", calls: int = 1
) -> FunctionModel:
    """Calls the first available tool ``calls`` times, then echoes the last result."""

    async def stream(messages: Any, info: AgentInfo) -> AsyncIterator[Any]:
        returns = [
            p for m in messages for p in m.parts if isinstance(p, ToolReturnPart)
        ]
        tools = [t.name for t in info.function_tools]
        new_turn = isinstance(messages[-1].parts[-1], UserPromptPart)
        if tools and (new_turn or len(returns) % calls) and len(returns) < calls * 10:
            if new_turn or len(returns) % calls:
                yield {
                    0: DeltaToolCall(
                        name=tools[0],
                        json_args=json.dumps({"request": request}),
                        tool_call_id=f"o{len(returns)}",
                    )
                }
                return
        yield f"tools={tools} result={returns[-1].content if returns else None}"

    return FunctionModel(stream_function=stream)


async def chat(running: RunningApp, content: str = "weather?") -> dict[str, Any]:
    response = await running.client.post(
        "/v1/chat/completions",
        json={"model": "x", "messages": [{"role": "user", "content": content}]},
    )
    assert response.status_code == 200, response.text
    return response.json()["choices"][0]["message"]


def serve_worker(tmp_path: Path, agent: dict[str, Any]) -> Server:
    """Serve a worker agent over HTTP, advertising its real URL in its card."""
    port = free_port()
    app = make_app(
        tmp_path / "w", agent, WORKER_SERVER, public_url=f"http://127.0.0.1:{port}"
    )
    return Server(app, port)


# --------------------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------------------


def test_cards_with_enum_repr_bindings_are_repaired() -> None:
    from a2a.types import AgentInterface

    from agentic.agent.a2a_client import normalize_card

    card = AgentCard(
        supported_interfaces=[
            AgentInterface(url="u", protocol_binding="TransportProtocol.JSONRPC"),
            AgentInterface(url="u", protocol_binding="HTTP+JSON"),
        ]
    )
    bindings = [i.protocol_binding for i in normalize_card(card).supported_interfaces]
    assert bindings == ["JSONRPC", "HTTP+JSON"]


def test_tool_names_are_safe_identifiers() -> None:
    assert tool_name_for("Weather Agent") == "weather_agent"
    assert tool_name_for("  --  ") == "agent"
    assert tool_name_for("Ünïcode/Agent v2!") == "n_code_agent_v2"
    assert len(tool_name_for("x" * 200)) == 64


def test_tool_description_uses_card_and_skills() -> None:
    card = AgentCard(
        name="Weather",
        description="Knows the weather.",
        skills=[
            AgentSkill(id="f", name="Forecast", description="3-day forecasts", tags=[])
        ],
    )
    description = tool_description_for(card)
    assert "'Weather' agent" in description
    assert "Knows the weather." in description
    assert "Forecast: 3-day forecasts" in description


@pytest.mark.parametrize(
    "outcome, fragment",
    [
        (_Outcome(state=TaskState.TASK_STATE_COMPLETED, reply="hi"), "hi"),
        (
            _Outcome(state=TaskState.TASK_STATE_COMPLETED, artifacts={"a": ["x", "y"]}),
            "xy",
        ),
        (_Outcome(state=TaskState.TASK_STATE_COMPLETED), "returned no text"),
        (
            _Outcome(state=TaskState.TASK_STATE_FAILED, status_text="boom"),
            "could not complete the request (failed): boom",
        ),
        (_Outcome(state=TaskState.TASK_STATE_REJECTED), "(rejected)"),
        (_Outcome(state=TaskState.TASK_STATE_CANCELED), "(canceled)"),
        (
            _Outcome(
                state=TaskState.TASK_STATE_INPUT_REQUIRED, status_text="which city?"
            ),
            "needs more information: which city?",
        ),
        (_Outcome(state=TaskState.TASK_STATE_AUTH_REQUIRED), "requires authorization"),
        (
            _Outcome(state=TaskState.TASK_STATE_WORKING, artifacts={"a": ["part"]}),
            "stopped before finishing (working); partial result: part",
        ),
    ],
)
def test_result_text_for_every_outcome(outcome: _Outcome, fragment: str) -> None:
    assert fragment in A2AAgentToolset._result_text("w", outcome)


# --------------------------------------------------------------------------------------
# Orchestrator ↔ worker over HTTP
# --------------------------------------------------------------------------------------


async def test_delegation_relays_attributed_activity(tmp_path: Path) -> None:
    use_model(
        "worker",
        scripted_model(
            thinking="need the forecast",
            note="Checking.",
            tool="forecast",
            args={"city": "Paris", "api_key": "s3cret"},
            answer="Paris: 18C sunny",
        ),
    )
    use_model("orchestrator", delegating_model())
    with serve_worker(
        tmp_path,
        {"model": "test:worker", "name": "weather", "capabilities": ["LocalTools"]},
    ) as worker:
        app = make_app(tmp_path / "o", orchestrator(f"{worker.url}/a2a"))
        async with RunningApp(app) as running:
            message = await chat(running)

    assert message["content"] == "tools=['weather_agent'] result=Paris: 18C sunny"
    reasoning = message["reasoning_content"]
    assert '→ weather_agent({"request": "weather in Paris?"})' in reasoning
    assert "[weather_agent] 💭 need the forecast" in reasoning
    assert "[weather_agent] 💬 Checking." in reasoning
    assert (
        '[weather_agent] → forecast({"city": "Paris", "api_key": "***"})' in reasoning
    )
    assert "[weather_agent] ← forecast:" in reasoning
    assert "← weather_agent: Paris: 18C sunny" in reasoning
    assert "s3cret" not in reasoning


async def test_tool_name_and_description_overrides(tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    async def stream(messages: Any, info: AgentInfo) -> AsyncIterator[str]:
        seen.update({t.name: t.description for t in info.function_tools})
        yield "ok"

    use_model("worker", scripted_model())
    use_model("orchestrator", FunctionModel(stream_function=stream))
    with serve_worker(tmp_path, {"model": "test:worker"}) as worker:
        app = make_app(
            tmp_path / "o",
            orchestrator(f"{worker.url}/a2a", name="Forecaster", description="Ask me."),
        )
        async with RunningApp(app) as running:
            await chat(running)
    assert seen == {"forecaster": "Ask me."}


async def test_unreachable_agent_offers_no_tool(tmp_path: Path) -> None:
    use_model("orchestrator", delegating_model())
    app = make_app(tmp_path, orchestrator(f"http://127.0.0.1:{free_port()}/a2a"))
    async with RunningApp(app) as running:
        message = await chat(running)
    assert message["content"] == "tools=[] result=None"


async def test_worker_failure_is_reported_to_the_model(tmp_path: Path) -> None:
    async def broken(messages: Any, info: AgentInfo) -> AsyncIterator[str]:
        raise RuntimeError("worker model down")
        yield ""  # pragma: no cover

    use_model("worker", FunctionModel(stream_function=broken))
    use_model("orchestrator", delegating_model())
    with serve_worker(tmp_path, {"model": "test:worker"}) as worker:
        app = make_app(tmp_path / "o", orchestrator(f"{worker.url}/a2a"))
        async with RunningApp(app) as running:
            message = await chat(running)
    assert "could not complete the request (failed)" in message["content"]
    assert "RuntimeError" in message["content"]


async def test_worker_context_is_reused_across_orchestrator_turns(
    tmp_path: Path,
) -> None:
    def worker_answer(messages: Any, info: AgentInfo) -> str:
        prompts = sum(
            isinstance(p, UserPromptPart)
            for m in messages
            if isinstance(m, ModelRequest)
            for p in m.parts
        )
        return f"worker turn {prompts}"

    use_model("worker", scripted_model(answer=worker_answer))
    use_model("orchestrator", delegating_model())
    with serve_worker(tmp_path, {"model": "test:worker"}) as worker:
        app = make_app(tmp_path / "o", orchestrator(f"{worker.url}/a2a"))
        async with RunningApp(app) as running:
            from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
            from a2a.helpers import new_text_message
            from a2a.types import Role, SendMessageRequest

            card = await A2ACardResolver(
                running.client, "http://test/a2a"
            ).get_agent_card()
            client = ClientFactory(ClientConfig(httpx_client=running.client)).create(
                card
            )

            async def ask(context_id: str | None) -> Any:
                request = SendMessageRequest(
                    message=new_text_message(
                        "q", context_id=context_id, role=Role.ROLE_USER
                    )
                )
                return [e async for e in client.send_message(request)]

            first = await ask(None)
            context_id = first[0].task.context_id
            second = await ask(context_id)

    def answer(events: list[Any]) -> str:
        update = [e for e in events if e.WhichOneof("payload") == "artifact_update"][0]
        return update.artifact_update.artifact.parts[0].text

    assert answer(first).endswith("result=worker turn 1")
    assert answer(second).endswith("result=worker turn 2")


async def test_cancelling_the_orchestrator_cancels_the_worker(tmp_path: Path) -> None:
    use_model("worker", scripted_model(tool="slow", args={"seconds": 30}))
    use_model("orchestrator", delegating_model())
    with serve_worker(
        tmp_path, {"model": "test:worker", "capabilities": ["LocalTools"]}
    ) as worker:
        app = make_app(tmp_path / "o", orchestrator(f"{worker.url}/a2a"))
        with Server(app, free_port()) as orch:
            async with httpx.AsyncClient(base_url=orch.url, timeout=10) as client:
                async with client.stream(
                    "POST",
                    "/v1/chat/completions",
                    json={
                        "model": "x",
                        "stream": True,
                        "messages": [{"role": "user", "content": "go"}],
                    },
                ) as response:
                    async for line in response.aiter_lines():
                        if "→ slow" in line:
                            break
            for _ in range(200):
                if SLOW_STATE["cancelled"]:
                    break
                await asyncio.sleep(0.05)
    assert SLOW_STATE == {"started": 1, "cancelled": 1, "finished": 0}


# --------------------------------------------------------------------------------------
# input-required continuation (against a minimal hand-written A2A server)
# --------------------------------------------------------------------------------------


class AskingExecutor(AgentExecutor):
    """Asks which city on the first message of a task, answers on the follow-up."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = context.current_task
        if task is None:
            task = new_task_from_user_message(context.message)
            await event_queue.enqueue_event(task)
            updater = TaskUpdater(event_queue, task.id, task.context_id)
            await updater.requires_input(
                updater.new_agent_message([new_text_part("Which city?")])
            )
            return
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        city = context.get_user_input()
        await updater.add_artifact([new_text_part(f"Sunny in {city}")], name="response")
        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        pass


def asking_app(url: str) -> Starlette:
    card = AgentCard(
        name="Asker",
        description="Asks for details.",
        version="1",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(streaming=True),
        supported_interfaces=[
            AgentInterface(
                url=url,
                protocol_binding=TransportProtocol.JSONRPC.value,
                protocol_version=PROTOCOL_VERSION_1_0,
            )
        ],
    )
    handler = DefaultRequestHandler(
        agent_executor=AskingExecutor(), task_store=InMemoryTaskStore(), agent_card=card
    )
    return Starlette(
        routes=[
            *create_agent_card_routes(
                card, card_url="/a2a/.well-known/agent-card.json"
            ),
            *create_jsonrpc_routes(handler, "/a2a"),
        ]
    )


async def test_input_required_is_continued_on_the_next_call(tmp_path: Path) -> None:
    async def stream(messages: Any, info: AgentInfo) -> AsyncIterator[Any]:
        returns = [
            p for m in messages for p in m.parts if isinstance(p, ToolReturnPart)
        ]
        if len(returns) == 0:
            yield {
                0: DeltaToolCall(
                    name="asker", json_args='{"request": "weather?"}', tool_call_id="a"
                )
            }
        elif len(returns) == 1:
            yield {
                0: DeltaToolCall(
                    name="asker", json_args='{"request": "Oslo"}', tool_call_id="b"
                )
            }
        else:
            yield " || ".join(str(r.content) for r in returns)

    use_model("orchestrator", FunctionModel(stream_function=stream))
    port = free_port()
    with Server(asking_app(f"http://127.0.0.1:{port}/a2a"), port) as asker:
        app = make_app(tmp_path, orchestrator(f"{asker.url}/a2a"))
        async with RunningApp(app) as running:
            message = await chat(running)
    first, second = message["content"].split(" || ")
    assert "needs more information: Which city?" in first
    assert second == "Sunny in Oslo"
