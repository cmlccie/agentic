"""Tests for agentic.simple_agent.interfaces.ui."""

import json
from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from agentic.simple_agent.interfaces.ui import build_ui_router

# -------------------------------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------------------------------


def _make_mock_agent() -> MagicMock:
    agent = MagicMock()

    async def _fake_stream_text(delta=False):
        for chunk in ["Hello", " world", "!"]:
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
    router = build_ui_router(app_state)
    app.include_router(router, prefix="/ui")
    return app, agent


# -------------------------------------------------------------------------------------------------
# GET /ui/ — HTML page
# -------------------------------------------------------------------------------------------------


class TestUIRoot:
    def test_returns_200(self):
        app, _ = _make_app()
        assert TestClient(app).get("/ui/").status_code == 200

    def test_returns_html(self):
        app, _ = _make_app()
        resp = TestClient(app).get("/ui/")
        assert "text/html" in resp.headers["content-type"]

    def test_html_has_chat_form(self):
        app, _ = _make_app()
        resp = TestClient(app).get("/ui/")
        assert "<textarea" in resp.text
        assert "<button" in resp.text


# -------------------------------------------------------------------------------------------------
# POST /ui/chat — SSE stream
# -------------------------------------------------------------------------------------------------


class TestUIChat:
    def _post(self, app, messages):
        return TestClient(app).post("/ui/chat", json={"messages": messages})

    def test_returns_200(self):
        app, _ = _make_app()
        resp = self._post(app, [{"role": "user", "content": "Hi"}])
        assert resp.status_code == 200

    def test_content_type_is_event_stream(self):
        app, _ = _make_app()
        resp = self._post(app, [{"role": "user", "content": "Hi"}])
        assert "text/event-stream" in resp.headers["content-type"]

    def test_ends_with_done(self):
        app, _ = _make_app()
        resp = self._post(app, [{"role": "user", "content": "Hi"}])
        assert "data: [DONE]" in resp.text

    def test_chunks_contain_content(self):
        app, _ = _make_app()
        resp = self._post(app, [{"role": "user", "content": "Hi"}])
        content = ""
        for line in resp.text.split("\n"):
            line = line.strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            content += chunk["choices"][0]["delta"]["content"]
        assert content == "Hello world!"

    def test_run_stream_called_with_last_message(self):
        app, agent = _make_app()
        self._post(
            app,
            [
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "What's up?"},
            ],
        )
        agent.run_stream.assert_called_once()
        call_args = agent.run_stream.call_args
        assert call_args.args[0] == "What's up?"

    def test_history_passed_for_multi_message(self):
        app, agent = _make_app()
        self._post(
            app,
            [
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "Q"},
            ],
        )
        call_args = agent.run_stream.call_args
        assert call_args.kwargs["message_history"] is not None

    def test_no_messages_returns_error_sse(self):
        app, _ = _make_app()
        resp = TestClient(app).post("/ui/chat", json={"messages": []})
        assert resp.status_code == 200
        assert "error" in resp.text
