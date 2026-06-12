"""Tests for agentic.simple_agent.interfaces.ui."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from fastapi import FastAPI
from starlette.responses import HTMLResponse, JSONResponse
from starlette.testclient import TestClient

from agentic.simple_agent.interfaces.ui import build_ui_app


# -------------------------------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------------------------------


def _html_asgi(content: str) -> Any:
    """Minimal ASGI app that serves an HTML response."""

    async def app(scope, receive, send):
        await HTMLResponse(content)(scope, receive, send)

    return app


def _json_asgi(data: dict) -> Any:
    """Minimal ASGI app that serves a JSON response."""

    async def app(scope, receive, send):
        await JSONResponse(data)(scope, receive, send)

    return app


def _make_app(inner_app: Any, prefix: str = '/ui') -> FastAPI:
    """Mount build_ui_app at *prefix* and return the outer FastAPI app."""
    state = MagicMock()
    state.agent = MagicMock()
    state.agent.to_web.return_value = inner_app
    app = FastAPI()
    app.mount(prefix, build_ui_app(state))
    return app


# -------------------------------------------------------------------------------------------------
# Rebase script injection
# -------------------------------------------------------------------------------------------------


class TestRebaseInjection:
    """HTML responses served under a mount prefix get the rebase shim injected."""

    _HTML = '<html><head><title>T</title></head><body>body</body></html>'

    def test_rebase_flag_present(self):
        app = _make_app(_html_asgi(self._HTML))
        resp = TestClient(app).get('/ui/')
        assert 'window.__aiUiRebased' in resp.text

    def test_base_path_is_mount_prefix(self):
        app = _make_app(_html_asgi(self._HTML))
        resp = TestClient(app).get('/ui/')
        # The JS variable must be set to the mount prefix as a JSON string.
        assert 'var b="/ui"' in resp.text

    def test_script_inserted_before_head_close(self):
        app = _make_app(_html_asgi(self._HTML))
        resp = TestClient(app).get('/ui/')
        # Script tag is injected and </head> still closes the head element.
        script_pos = resp.text.index('<script>')
        head_close_pos = resp.text.index('</head>')
        assert script_pos < head_close_pos

    def test_different_prefix(self):
        app = _make_app(_html_asgi(self._HTML), prefix='/agents/demo')
        resp = TestClient(app).get('/agents/demo/')
        assert 'var b="/agents/demo"' in resp.text

    def test_content_length_updated(self):
        app = _make_app(_html_asgi(self._HTML))
        resp = TestClient(app).get('/ui/')
        cl = int(resp.headers['content-length'])
        assert cl == len(resp.content)

    def test_status_code_preserved(self):
        app = _make_app(_html_asgi(self._HTML))
        assert TestClient(app).get('/ui/').status_code == 200


# -------------------------------------------------------------------------------------------------
# Non-HTML responses pass through unmodified
# -------------------------------------------------------------------------------------------------


class TestPassThrough:
    def test_json_not_modified(self):
        payload = {'key': 'value'}
        app = _make_app(_json_asgi(payload))
        resp = TestClient(app).get('/ui/')
        assert resp.json() == payload
        assert 'aiUiRebased' not in resp.text

    def test_json_content_length_unchanged(self):
        payload = {'key': 'value'}
        app = _make_app(_json_asgi(payload))
        resp = TestClient(app).get('/ui/')
        cl = int(resp.headers['content-length'])
        assert cl == len(resp.content)


# -------------------------------------------------------------------------------------------------
# No injection when mounted at root (root_path is empty)
# -------------------------------------------------------------------------------------------------


class TestNoInjectionAtRoot:
    """When the UI sub-app is at the domain root its root_path is '' — no injection."""

    _HTML = '<html><head></head><body>ok</body></html>'

    def test_no_rebase_flag_at_root(self):
        # Mount at '/' means no stripping, root_path stays ''.
        state = MagicMock()
        state.agent = MagicMock()
        state.agent.to_web.return_value = _html_asgi(self._HTML)

        # Call _DynamicUIApp directly via a bare ASGI scope without root_path.
        from agentic.simple_agent.interfaces.ui import _DynamicUIApp

        ui = _DynamicUIApp(state)

        # Build a minimal HTTP scope with no root_path.
        collected: list[dict] = []

        async def fake_send(msg):
            collected.append(msg)

        async def fake_receive():
            return {'type': 'http.disconnect'}

        scope = {
            'type': 'http',
            'method': 'GET',
            'path': '/',
            'query_string': b'',
            'headers': [],
        }

        import asyncio

        asyncio.run(ui(scope, fake_receive, fake_send))

        body = b''.join(m.get('body', b'') for m in collected if m['type'] == 'http.response.body')
        assert b'aiUiRebased' not in body
