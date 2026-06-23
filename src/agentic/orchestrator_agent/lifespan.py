from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from langgraph.graph.state import CompiledStateGraph

from agentic.runtime.config import (
    CONFIG_DIR,
    SECRETS_DIR,
    AgentSecrets,
    ServerSpec,
    load_server_spec,
)
from agentic.runtime.reload import (
    ReloadCoordinator,
    install_sighup_handler,
    watch_config_directory,
)

from .config.agent_spec import OrchestratorSpec, load_orchestrator_spec
from .graph import build_graph

log = logging.getLogger(__name__)


@dataclass
class AppState:
    """Mutable shared state. Held on app.state.app_state.

    graph, spec, and server_spec are replaced atomically on each reload cycle.
    coordinator and secrets are long-lived singletons for the process lifetime.
    graph is None until the initial build completes (or while a reload that
    failed to rebuild is pending retry); interface handlers read it at call time
    so they always see the current graph.
    """

    coordinator: ReloadCoordinator
    secrets: AgentSecrets
    spec: OrchestratorSpec
    server_spec: ServerSpec
    graph: CompiledStateGraph | None = None
    config_dir: Path = field(default=CONFIG_DIR)
    secrets_dir: Path = field(default=SECRETS_DIR)
    push_httpx_client: Any = field(default=None)


async def _run_reload_loop(app: FastAPI) -> None:
    """Wait for reload requests, drain, then rebuild the graph from fresh config.

    Runs for the entire process lifetime. Re-fetches downstream agent cards as
    part of `build_graph`, so adding/removing A2A servers takes effect here.
    """
    state: AppState = app.state.app_state

    while True:
        await state.coordinator.wait_for_reload_request()
        log.info("reload_loop: reload requested — beginning drain")

        await state.coordinator.wait_for_drain()

        log.info("reload_loop: loading new configuration")
        try:
            new_server_spec = load_server_spec(state.config_dir / "server.yaml")
            new_spec = load_orchestrator_spec(state.config_dir / "agent.yaml")

            state.coordinator.drain_timeout = new_server_spec.reload.drain_timeout

            new_graph = await build_graph(new_spec, state.secrets)

            state.graph = new_graph
            state.spec = new_spec
            state.server_spec = new_server_spec

            state.coordinator.mark_running()
            log.info("reload_loop: reload complete — orchestrator is RUNNING")

        except Exception as exc:
            log.error(
                "reload_loop: reload failed: %s — orchestrator remains in RELOADING "
                "state; will retry on next trigger",
                exc,
            )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    state: AppState = app.state.app_state

    initial_build_failed = False
    try:
        state.graph = await build_graph(state.spec, state.secrets)
    except Exception as exc:
        log.warning(
            "lifespan: initial graph build failed (%s); starting in degraded mode — "
            "will retry automatically",
            exc,
        )
        initial_build_failed = True

    state.coordinator.mark_running()

    stop_event = asyncio.Event()

    watcher_task = asyncio.create_task(
        watch_config_directory(state.config_dir, state.coordinator, stop_event),
        name="config-watcher",
    )
    secrets_watcher_task = asyncio.create_task(
        watch_config_directory(state.secrets_dir, state.coordinator, stop_event),
        name="secrets-watcher",
    )
    reload_task = asyncio.create_task(
        _run_reload_loop(app),
        name="reload-loop",
    )

    install_sighup_handler(state.coordinator)

    if initial_build_failed:
        state.coordinator.trigger_reload()

    try:
        yield
    finally:
        log.info("lifespan: shutting down")
        stop_event.set()
        reload_task.cancel()
        watcher_task.cancel()
        secrets_watcher_task.cancel()

        if state.push_httpx_client is not None:
            try:
                await state.push_httpx_client.aclose()
            except Exception as exc:
                log.error("lifespan: error closing push httpx client: %s", exc)

        await asyncio.gather(
            reload_task,
            watcher_task,
            secrets_watcher_task,
            return_exceptions=True,
        )
