from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette import status

from .config import CONFIG_DIR, SECRETS_DIR
from .config.agent_spec import load_agent
from .config.server_spec import AgentSecrets, load_server_spec
from .health import router as health_router
from .interfaces.a2a import build_a2a_app
from .interfaces.openai_compat import build_openai_router
from .interfaces.ui import build_ui_router
from .lifespan import AppState, lifespan
from .reload import ReloadCoordinator

log = logging.getLogger(__name__)


def create_app(
    config_dir: Path = CONFIG_DIR,
    secrets_dir: Path = SECRETS_DIR,
    agent_url: str = "http://localhost:8000",
) -> FastAPI:
    """Create and configure the FastAPI application.

    Loads config from disk, wires all interfaces, and installs the reload gate
    middleware. Exposed as a factory so tests can inject alternate config paths.
    """
    secrets = AgentSecrets(secrets_dir)
    server_spec = load_server_spec(config_dir / "server.yaml")
    agent = load_agent(config_dir / "agent.yaml", secrets)
    coordinator = ReloadCoordinator(drain_timeout=server_spec.reload.drain_timeout)

    _state = AppState(
        coordinator=coordinator,
        secrets=secrets,
        agent=agent,
        server_spec=server_spec,
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
        a2a_app = build_a2a_app(agent, server_spec, secrets, agent_url=agent_url)
        _state.a2a_app = a2a_app
        app.mount("/a2a", a2a_app)

    if server_spec.interfaces.openai_compat:
        model_name = agent.name or server_spec.agent_card.display_name
        openai_router = build_openai_router(_state, model_name=model_name)
        app.include_router(openai_router)

    if server_spec.interfaces.ui:
        ui_router = build_ui_router(_state)
        app.include_router(ui_router, prefix="/ui")

    return app
