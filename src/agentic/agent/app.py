"""The agent's HTTP service: OpenAI-compatible API, A2A server, and health probes.

`create_app` loads the configuration (failing fast if it is invalid), then
serves:

- ``/v1/models``, ``/v1/chat/completions`` — see `agentic.agent.openai_api`
- ``/a2a`` and the agent card — see `agentic.agent.a2a_server`
- ``/health/live``, ``/health/ready`` — Kubernetes probes

Everything reads the runtime's current snapshot per request, so edits to
`agent.yaml`, `server.yaml`, or the secrets take effect without a restart (see
`agentic.agent.runtime`). A small ASGI gate in front of the routes applies the
per-interface switches and optional bearer-token authentication.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from starlette.types import ASGIApp, Receive, Scope, Send

from .a2a_server import A2A_PATH, CARD_PATHS, build_a2a_server
from .config import CONFIG_DIR, SECRETS_DIR, Secrets
from .openai_api import build_openai_router
from .runtime import AgentRuntime, Snapshot, load_snapshot
from .telemetry import setup_telemetry

log = logging.getLogger(__name__)


def _interface_of(path: str) -> str | None:
    """Which interface a request path belongs to (None for health and other routes)."""
    if path.startswith("/v1/") or path == "/v1":
        return "openai"
    if path == A2A_PATH or path.startswith(f"{A2A_PATH}/") or path in CARD_PATHS:
        return "a2a"
    return None


def _is_public(path: str, method: str) -> bool:
    """Requests that never require authentication: health probes and the agent card."""
    return path.startswith("/health/") or (
        path in CARD_PATHS and method in ("GET", "HEAD")
    )


def bearer_token_matches(header: str | None, token: str) -> bool:
    """Constant-time check of an ``Authorization: Bearer <token>`` header."""
    if not header or not header.lower().startswith("bearer "):
        return False
    return hmac.compare_digest(header[7:].strip().encode(), token.encode())


class InterfaceGate:
    """Pure-ASGI middleware: interface switches and bearer-token auth.

    Implemented at the ASGI level (not `BaseHTTPMiddleware`) so streaming
    responses pass straight through untouched.
    """

    def __init__(self, app: ASGIApp, runtime: AgentRuntime) -> None:
        self.app = app
        self.runtime = runtime

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        server = self.runtime.current.server
        path, method = scope["path"], scope["method"]

        interface = _interface_of(path)
        if interface == "openai" and not server.interfaces.openai:
            await _json(
                send,
                404,
                {
                    "error": {
                        "message": "the OpenAI API is disabled",
                        "type": "invalid_request_error",
                        "code": "interface_disabled",
                        "param": None,
                    }
                },
            )
            return
        if interface == "a2a" and not server.interfaces.a2a:
            await _json(send, 404, {"detail": "the A2A interface is disabled"})
            return

        secret = server.auth.bearer_token_secret
        if secret and interface and not _is_public(path, method):
            token = self.runtime.secrets.get(secret)
            if token is None:
                log.error(
                    "auth: bearer token secret '%s' is missing; refusing requests",
                    secret,
                )
                await _json(send, 503, {"detail": "authentication is misconfigured"})
                return
            headers = dict(scope.get("headers") or [])
            header = headers.get(b"authorization", b"").decode("latin-1")
            if not bearer_token_matches(header, token):
                body: dict[str, Any] = (
                    {
                        "error": {
                            "message": "invalid or missing bearer token",
                            "type": "invalid_request_error",
                            "code": "invalid_api_key",
                            "param": None,
                        }
                    }
                    if interface == "openai"
                    else {"detail": "invalid or missing bearer token"}
                )
                await _json(send, 401, body, {"www-authenticate": "Bearer"})
                return

        await self.app(scope, receive, send)


async def _json(
    send: Send, status: int, body: dict[str, Any], headers: dict[str, str] | None = None
) -> None:
    payload = json.dumps(body).encode()
    raw_headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(payload)).encode()),
        *((k.encode(), v.encode()) for k, v in (headers or {}).items()),
    ]
    await send(
        {"type": "http.response.start", "status": status, "headers": raw_headers}
    )
    await send({"type": "http.response.body", "body": payload})


def create_app(
    config_dir: Path = CONFIG_DIR,
    secrets_dir: Path = SECRETS_DIR,
    public_url: str = "http://localhost:8000",
    watch: bool = True,
) -> FastAPI:
    """Load the configuration and build the application.

    Args:
        config_dir: Directory containing `agent.yaml` and `server.yaml`.
        secrets_dir: Directory of secret files.
        public_url: Externally reachable base URL, advertised in the A2A card.
        watch: Reload automatically when the config or secrets change.

    Raises:
        ConfigError: If the configuration is invalid (the service should not
            start with a broken config; later reloads keep the last good one).
    """
    secrets = Secrets(secrets_dir)
    runtime = AgentRuntime(config_dir, secrets, load_snapshot(config_dir, secrets))

    def current() -> Snapshot:
        return runtime.current

    a2a = build_a2a_server(
        current,
        secrets,
        public_url,
        auth=lambda: bool(runtime.current.server.auth.bearer_token_secret),
    )

    def warn_on_restart_only_changes(old: Snapshot, new: Snapshot) -> None:
        if old.server.a2a.store != new.server.a2a.store:
            log.warning(
                "server.yaml: a2a.store changed; storage settings apply after a restart"
            )

    runtime.on_reload(warn_on_restart_only_changes)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        stop = asyncio.Event()
        watcher = (
            asyncio.create_task(runtime.run(stop), name="config-reloader")
            if watch
            else None
        )
        runtime.install_sighup_handler()
        try:
            yield
        finally:
            stop.set()
            if watcher is not None:
                await asyncio.gather(watcher, return_exceptions=True)
            await a2a.aclose()

    card = runtime.current.server.agent_card
    app = FastAPI(
        title=card.display_name,
        description=card.description,
        version=card.version,
        lifespan=lifespan,
    )
    app.state.runtime = runtime
    app.state.a2a = a2a

    @app.get("/health/live", tags=["Health"])
    async def live() -> dict[str, str]:
        """Liveness: the process is up."""
        return {"status": "alive"}

    @app.get("/health/ready", tags=["Health"])
    async def ready() -> dict[str, Any]:
        """Readiness: a valid configuration is loaded and serving."""
        snapshot = runtime.current
        return {
            "status": "ready",
            "agent": snapshot.model_name,
            "generation": snapshot.generation,
        }

    app.include_router(build_openai_router(current))
    app.router.routes.extend(a2a.routes)
    app.add_middleware(InterfaceGate, runtime=runtime)
    setup_telemetry(app, service_name=runtime.current.model_name)
    return app
