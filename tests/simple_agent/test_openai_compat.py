"""Tests for agentic.simple_agent.interfaces.openai_compat."""

from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)

from agentic.simple_agent.interfaces.openai_compat import (
    _messages_to_history,
    build_openai_router,
)

# -------------------------------------------------------------------------------------------------
# Helpers / fixtures
# -------------------------------------------------------------------------------------------------


def _make_mock_agent(output: str = "Hello!") -> MagicMock:
    agent = MagicMock()
    agent.name = "test-agent"

    mock_result = MagicMock()
    mock_result.output = output
    agent.run = AsyncMock(return_value=mock_result)

    async def _fake_stream_text(delta=False):
        for chunk in ["Hel", "lo", "!"]:
            yield chunk

    mock_stream_result = MagicMock()
    mock_stream_result.stream_text = _fake_stream_text

    mock_ctx = MagicMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_stream_result)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)
    agent.run_stream = MagicMock(return_value=mock_ctx)

    return agent


def _make_app(agent=None) -> tuple[FastAPI, MagicMock]:
    agent = agent or _make_mock_agent()
    app_state = MagicMock()
    app_state.agent = agent

    app = FastAPI()
    router = build_openai_router(app_state, model_name="test-model")
    app.include_router(router)
    return app, agent


# -------------------------------------------------------------------------------------------------
# _messages_to_history
# -------------------------------------------------------------------------------------------------


class TestMessagesToHistory:
    def test_system_message(self):
        msgs = [{"role": "system", "content": "Be helpful."}]
        history = _messages_to_history(msgs)
        assert len(history) == 1
        assert isinstance(history[0], ModelRequest)
        assert isinstance(history[0].parts[0], SystemPromptPart)
        assert history[0].parts[0].content == "Be helpful."

    def test_user_message(self):
        msgs = [{"role": "user", "content": "Hello"}]
        history = _messages_to_history(msgs)
        assert isinstance(history[0].parts[0], UserPromptPart)

    def test_assistant_message(self):
        msgs = [{"role": "assistant", "content": "Hi there"}]
        history = _messages_to_history(msgs)
        assert isinstance(history[0], ModelResponse)
        assert isinstance(history[0].parts[0], TextPart)

    def test_multipart_content_concatenated(self):
        msgs = [{"role": "user", "content": [{"text": "Hello"}, {"text": "World"}]}]
        history = _messages_to_history(msgs)
        assert history[0].parts[0].content == "Hello World"

    def test_empty_returns_empty(self):
        assert _messages_to_history([]) == []


# -------------------------------------------------------------------------------------------------
# Non-streaming endpoint
# -------------------------------------------------------------------------------------------------


class TestNonStreaming:
    def test_returns_200(self):
        app, _ = _make_app()
        client = TestClient(app)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        assert resp.status_code == 200

    def test_response_has_assistant_content(self):
        app, _ = _make_app()
        client = TestClient(app)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        data = resp.json()
        assert data["choices"][0]["message"]["content"] == "Hello!"

    def test_agent_run_called_with_last_message(self):
        app, agent = _make_app()
        client = TestClient(app)
        client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [
                    {"role": "system", "content": "Be helpful."},
                    {"role": "user", "content": "Final question"},
                ],
            },
        )
        agent.run.assert_awaited_once()
        call_args = agent.run.call_args
        assert call_args.args[0] == "Final question"

    def test_history_passed_for_multi_message(self):
        app, agent = _make_app()
        client = TestClient(app)
        client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [
                    {"role": "system", "content": "Be helpful."},
                    {"role": "user", "content": "Q"},
                ],
            },
        )
        call_args = agent.run.call_args
        assert call_args.kwargs["message_history"] is not None
        assert len(call_args.kwargs["message_history"]) == 1

    def test_no_history_for_single_message(self):
        app, agent = _make_app()
        client = TestClient(app)
        client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Q"}],
            },
        )
        call_args = agent.run.call_args
        assert call_args.kwargs["message_history"] is None


# -------------------------------------------------------------------------------------------------
# Streaming endpoint
# -------------------------------------------------------------------------------------------------


class TestStreaming:
    def _post_stream(self, client):
        return client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )

    def test_returns_200(self):
        app, _ = _make_app()
        assert self._post_stream(TestClient(app)).status_code == 200

    def test_content_type_is_event_stream(self):
        app, _ = _make_app()
        resp = self._post_stream(TestClient(app))
        assert "text/event-stream" in resp.headers["content-type"]

    def test_ends_with_finish_reason_stop(self):
        app, _ = _make_app()
        resp = self._post_stream(TestClient(app))
        import json as _json

        last_chunk = None
        for line in resp.text.split("\n"):
            line = line.strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                try:
                    last_chunk = _json.loads(line[6:])
                except Exception:
                    pass
        assert last_chunk is not None
        assert last_chunk["choices"][0]["finish_reason"] == "stop"

    def test_concatenated_content(self):
        app, _ = _make_app()
        resp = self._post_stream(TestClient(app))
        import json as _json

        content = ""
        for line in resp.text.split("\n"):
            line = line.strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = _json.loads(line[6:])
            content += chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
        assert content == "Hel lo !" or content == "Hello!"


# -------------------------------------------------------------------------------------------------
# Models endpoint
# -------------------------------------------------------------------------------------------------


class TestModelsEndpoint:
    def test_returns_model_name(self):
        app, _ = _make_app()
        client = TestClient(app)
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        models = resp.json()["data"]
        assert any(m["id"] == "test-model" for m in models)
