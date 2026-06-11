from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from pydantic_ai import Agent

from .config import CONFIG_DIR, SECRETS_DIR
from .config.agent_spec import load_agent
from .config.server_spec import AgentSecrets, ServerSpec, load_server_spec
from .reload import (
    ReloadCoordinator,
    install_sighup_handler,
    watch_config_directory,
)

log = logging.getLogger(__name__)


@dataclass
class AppState:
    """Mutable shared state. Held on app.state.app_state.

    agent and server_spec are replaced atomically on each reload cycle.
    coordinator and secrets are long-lived singletons for the process lifetime.
    a2a_app holds a reference to the mounted FastA2A sub-app (if enabled) so
    lifespan can manage its task_manager — Starlette does not call mounted
    sub-app lifespans automatically.
    """

    coordinator: ReloadCoordinator
    secrets: AgentSecrets
    agent: Agent
    server_spec: ServerSpec
    config_dir: Path = field(default=CONFIG_DIR)
    secrets_dir: Path = field(default=SECRETS_DIR)
    a2a_app: Any = field(default=None)


async def _run_reload_loop(app: FastAPI) -> None:
    """Long-running task that waits for reload requests, drains, reloads, and
    re-enters the agent context. Runs for the entire process lifetime.
    """
    state: AppState = app.state.app_state

    while True:
        await state.coordinator.wait_for_reload_request()
        log.info("reload_loop: reload requested — beginning drain")

        await state.coordinator.wait_for_drain()

        log.info("reload_loop: closing MCP sessions")
        try:
            await state.agent.__aexit__(None, None, None)
        except Exception as exc:
            log.error("reload_loop: error closing agent MCP sessions: %s", exc)

        log.info("reload_loop: loading new configuration")
        try:
            new_spec = load_server_spec(state.config_dir / "server.yaml")
            new_agent = load_agent(state.config_dir / "agent.yaml", state.secrets)

            state.coordinator.drain_timeout = new_spec.reload.drain_timeout

            await new_agent.__aenter__()

            state.agent = new_agent
            state.server_spec = new_spec

            state.coordinator.mark_running()
            log.info("reload_loop: reload complete — agent is RUNNING")

        except Exception as exc:
            log.error(
                "reload_loop: reload failed: %s — agent remains in RELOADING state; "
                "will retry on next trigger",
                exc,
            )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    state: AppState = app.state.app_state

    await state.agent.__aenter__()
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

    # Starlette does not call mounted sub-app lifespans. Manually start the
    # FastA2A task_manager so A2A requests can be processed.
    a2a_ctx = None
    if state.a2a_app is not None:
        a2a_ctx = state.a2a_app.task_manager
        await a2a_ctx.__aenter__()

    try:
        yield
    finally:
        log.info("lifespan: shutting down")
        stop_event.set()
        reload_task.cancel()
        watcher_task.cancel()
        secrets_watcher_task.cancel()

        if a2a_ctx is not None:
            try:
                await a2a_ctx.__aexit__(None, None, None)
            except Exception as exc:
                log.error("lifespan: error stopping A2A task manager: %s", exc)

        try:
            await state.agent.__aexit__(None, None, None)
        except Exception as exc:
            log.error("lifespan: error on shutdown agent exit: %s", exc)

        await asyncio.gather(
            reload_task,
            watcher_task,
            secrets_watcher_task,
            return_exceptions=True,
        )
