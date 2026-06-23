from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette import status

from agentic.runtime.config import AgentSecrets, load_server_spec
from agentic.runtime.health import router as health_router
from agentic.runtime.reload import ReloadCoordinator

from .config import CONFIG_DIR, SECRETS_DIR
from .config.agent_spec import load_orchestrator_spec
from .interfaces.a2a import build_a2a_interface
from .interfaces.openai_compat import build_openai_router
from .lifespan import AppState, lifespan

log = logging.getLogger(__name__)


def create_app(
    config_dir: Path = CONFIG_DIR,
    secrets_dir: Path = SECRETS_DIR,
    agent_url: str = "http://localhost:8000",
) -> FastAPI:
    """Create and configure the FastAPI application.

    Loads config from disk and wires all interfaces. The LangGraph supervisor
    graph itself is built during the lifespan startup (it fetches downstream
    agent cards over the network), then rebuilt on each reload.
    """
    secrets = AgentSecrets(secrets_dir)
    server_spec = load_server_spec(config_dir / "server.yaml")
    spec = load_orchestrator_spec(config_dir / "agent.yaml")
    coordinator = ReloadCoordinator(drain_timeout=server_spec.reload.drain_timeout)

    _state = AppState(
        coordinator=coordinator,
        secrets=secrets,
        spec=spec,
        server_spec=server_spec,
        graph=None,
        config_dir=config_dir,
        secrets_dir=secrets_dir,
    )

    app = FastAPI(
        title=server_spec.agent_card.display_name,
        description=server_spec.agent_card.description,
        version=server_spec.agent_card.version,
        lifespan=lifespan,
    )
    app.state.app_state = _state

    # ── Reload Gate Middleware ──────────────────────────────────────────────────
    # Belt-and-suspenders: returns 503 for all non-health requests during reload.
    # Kubernetes should have already drained traffic via the readiness probe, but
    # this catches requests that arrive during the propagation window.
    @app.middleware("http")
    async def reload_gate(request: Request, call_next):
        if request.url.path.startswith("/health"):
            return await call_next(request)

        granted = await coordinator.request_slot()
        if not granted:
            return JSONResponse(
                {"detail": "agent reloading, please retry"},
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                headers={"Retry-After": "5"},
            )
        try:
            return await call_next(request)
        finally:
            await coordinator.release_slot()

    # ── Interface Mounts ────────────────────────────────────────────────────────
    app.include_router(health_router)

    if server_spec.interfaces.a2a:
        a2a_interface = build_a2a_interface(_state, server_spec, secrets, agent_url)
        _state.push_httpx_client = a2a_interface.push_httpx_client
        _state.task_store_engine = a2a_interface.task_store_engine
        app.router.routes.extend(a2a_interface.routes)

    if server_spec.interfaces.openai_compat:
        model_name = spec.name or server_spec.agent_card.display_name
        openai_router = build_openai_router(_state, model_name=model_name)
        app.include_router(openai_router)

    return app
