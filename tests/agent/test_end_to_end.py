"""End to end over real HTTP: a fake vLLM server, a real MCP server, two agents.

    openai SDK / a2a client ──► orchestrator ──A2A──► worker ──MCP──► provisioning server
                                     │                  │
                                     └──── vllm: ───────┴──► fake vLLM (Chat Completions)

Unlike the other tests, nothing here is scripted inside the agent process: the
agents use the real ``vllm:`` model client (streaming Chat Completions with
``reasoning_content`` and tool calls) and a real MCP server over HTTP.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx2
import openai
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from google.protobuf.json_format import MessageToDict

from agentic.agent.streaming import ACTIVITY_EXTENSION

from ..tools.helpers import load_server
from .conftest import RunningApp, Server, free_port, make_app

pytestmark = pytest.mark.anyio


# --------------------------------------------------------------------------------------
# A fake vLLM server
# --------------------------------------------------------------------------------------


def _script(
    model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
) -> dict[str, Any]:
    """Decide the fake model's next turn: reasoning + a tool call, or an answer."""
    last = messages[-1]
    if last["role"] == "tool":
        return {"content": f"{model} says: {last['content']}"}
    if model == "worker-model":
        return {
            "reasoning": "I should check whether VLAN 42 is free.",
            "tool": ("check_vlan", {"vlan_id": 42}),
        }
    name = tools[0]["function"]["name"] if tools else None
    if name is None:
        return {"content": "no agents available"}
    return {
        "reasoning": "The network agent can check VLANs.",
        "tool": (name, {"request": "Is VLAN 42 free?"}),
    }


def fake_vllm() -> FastAPI:
    app = FastAPI()

    @app.post("/v1/chat/completions", response_model=None)
    async def completions(request: Request) -> Any:
        body = await request.json()
        turn = _script(body["model"], body["messages"], body.get("tools") or [])
        base = {
            "id": "chatcmpl-fake",
            "created": int(time.time()),
            "model": body["model"],
        }
        usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        tool = turn.get("tool")
        tool_calls = (
            [
                {
                    "index": 0,
                    "id": f"call_{tool[0]}",
                    "type": "function",
                    "function": {"name": tool[0], "arguments": json.dumps(tool[1])},
                }
            ]
            if tool
            else None
        )
        if not body.get("stream"):
            message = {"role": "assistant", "content": turn.get("content")}
            if turn.get("reasoning"):
                message["reasoning_content"] = turn["reasoning"]
            if tool_calls:
                message["tool_calls"] = [
                    {k: v for k, v in c.items() if k != "index"} for c in tool_calls
                ]
            return JSONResponse(
                {
                    **base,
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": message,
                            "finish_reason": "tool_calls" if tool else "stop",
                        }
                    ],
                    "usage": usage,
                }
            )

        def chunk(delta: dict[str, Any], finish: str | None = None) -> str:
            payload = {
                **base,
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            return f"data: {json.dumps(payload)}\n\n"

        def events() -> Iterator[str]:
            yield chunk({"role": "assistant"})
            if reasoning := turn.get("reasoning"):
                for word in reasoning.split(" "):
                    yield chunk({"reasoning_content": word + " "})
            if tool_calls:
                yield chunk({"tool_calls": tool_calls})
                yield chunk({}, "tool_calls")
            else:
                yield chunk({"content": turn["content"]})
                yield chunk({}, "stop")
            yield f"data: {json.dumps({**base, 'object': 'chat.completion.chunk', 'choices': [], 'usage': usage})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    return app


# --------------------------------------------------------------------------------------
# The test
# --------------------------------------------------------------------------------------


async def test_orchestrator_worker_mcp_over_http(tmp_path: Path) -> None:
    provisioning = load_server("provisioning_server")
    mcp_port, vllm_port, worker_port = free_port(), free_port(), free_port()
    model_secrets = {
        "model.base_url": f"http://127.0.0.1:{vllm_port}/v1",
        "model.api_key": "k",
    }

    worker_app = make_app(
        tmp_path / "worker",
        {
            "name": "netops",
            "model": "vllm:worker-model",
            "instructions": "You manage VLANs.",
            "capabilities": [{"MCP": {"url": f"http://127.0.0.1:{mcp_port}/mcp"}}],
        },
        {
            "agent_card": {
                "display_name": "NetOps Agent",
                "description": "Manages VLANs.",
            }
        },
        secrets=model_secrets,
        public_url=f"http://127.0.0.1:{worker_port}",
    )

    with (
        Server(provisioning.mcp.http_app(), mcp_port),
        Server(fake_vllm(), vllm_port),
        Server(worker_app, worker_port) as worker,
    ):
        orchestrator = make_app(
            tmp_path / "orchestrator",
            {
                "name": "orchestrator",
                "model": "vllm:orchestrator-model",
                "capabilities": [{"A2AAgent": {"url": f"{worker.url}/a2a"}}],
            },
            secrets=model_secrets,
        )
        async with RunningApp(orchestrator) as running:
            # OpenAI API, streaming, through the official SDK
            client = openai.AsyncOpenAI(
                base_url="http://test/v1",
                api_key="unused",
                max_retries=0,
                http_client=httpx2.AsyncClient(
                    transport=httpx2.ASGITransport(app=running.app), timeout=30
                ),
            )
            stream = await client.chat.completions.create(
                model="orchestrator",
                messages=[{"role": "user", "content": "Can I use VLAN 42?"}],
                stream=True,
                stream_options={"include_usage": True},
            )
            chunks = [c async for c in stream]
            reasoning = "".join(
                (c.choices[0].delta.model_extra or {}).get("reasoning_content") or ""
                for c in chunks
                if c.choices
            )
            content = "".join(
                c.choices[0].delta.content or "" for c in chunks if c.choices
            )

            # A2A, through the a2a-sdk client
            from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
            from a2a.helpers import new_text_message
            from a2a.types import Role, SendMessageRequest

            card = await A2ACardResolver(
                running.client, "http://test/a2a"
            ).get_agent_card()
            a2a = ClientFactory(ClientConfig(httpx_client=running.client)).create(card)
            events = [
                e
                async for e in a2a.send_message(
                    SendMessageRequest(
                        message=new_text_message(
                            "Can I use VLAN 42?", role=Role.ROLE_USER
                        )
                    )
                )
            ]

    # The orchestrator's own thinking, the delegation, and the worker's MCP tool use
    assert "The network agent can check VLANs." in reasoning
    assert '→ netops_agent({"request": "Is VLAN 42 free?"})' in reasoning
    assert "[netops_agent] 💭 I should check whether VLAN 42 is free." in reasoning
    assert '[netops_agent] → check_vlan({"vlan_id": 42})' in reasoning
    assert "[netops_agent] ← check_vlan:" in reasoning
    assert "← netops_agent: worker-model says:" in reasoning
    assert content.startswith("orchestrator-model says: worker-model says:")
    assert chunks[-1].usage is not None and chunks[-1].usage.total_tokens > 0

    kinds = [e.WhichOneof("payload") for e in events]
    assert kinds[0] == "task"
    artifact = [e for e in events if e.WhichOneof("payload") == "artifact_update"][0]
    assert artifact.artifact_update.artifact.parts[0].text.startswith(
        "orchestrator-model says:"
    )
    relayed = [
        MessageToDict(e.status_update.status.message.metadata)[ACTIVITY_EXTENSION]
        for e in events
        if e.WhichOneof("payload") == "status_update"
        and e.status_update.status.HasField("message")
        and e.status_update.status.message.HasField("metadata")
    ]
    by_source = {(tuple(a["source"]), a["kind"]) for a in relayed}
    assert {
        ((), "thinking"),
        ((), "tool_call"),
        (("netops_agent",), "thinking"),
        (("netops_agent",), "tool_call"),
        (("netops_agent",), "tool_result"),
        ((), "tool_result"),
    } <= by_source


async def test_non_streaming_vllm_path(tmp_path: Path) -> None:
    """The OpenAI API's non-streaming response through the real vllm: client."""
    vllm_port = free_port()
    app = make_app(
        tmp_path,
        {"name": "solo", "model": "vllm:orchestrator-model"},
        secrets={"model.base_url": f"http://127.0.0.1:{vllm_port}/v1"},
    )
    with Server(fake_vllm(), vllm_port):
        async with RunningApp(app) as running:
            response = await running.client.post(
                "/v1/chat/completions",
                json={"model": "solo", "messages": [{"role": "user", "content": "hi"}]},
            )
    message = response.json()["choices"][0]["message"]
    assert message["content"] == "no agents available"


async def test_unreachable_model_server_is_a_clean_error(tmp_path: Path) -> None:
    app = make_app(
        tmp_path,
        {"name": "solo", "model": "vllm:m"},
        secrets={"model.base_url": f"http://127.0.0.1:{free_port()}/v1"},
    )
    async with RunningApp(app) as running:
        response = await running.client.post(
            "/v1/chat/completions",
            json={"model": "solo", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "agent_error"
