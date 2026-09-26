"""The running agent and its hot reload.

A `Snapshot` pairs a constructed agent with the `server.yaml` it was loaded
with. `AgentRuntime` holds the current snapshot and replaces it when the config
or secrets directory changes (or on SIGHUP):

    load agent.yaml + server.yaml ──► build new Snapshot ──► swap ``current``

Reloads never interrupt work in progress. Interfaces read ``runtime.current``
once per request, so a request that started before a reload finishes with the
agent it started with, and every request after the swap uses the new one. There
is nothing to drain or close: MCP connections are opened per run and remote A2A
agents are called per request.

If a new configuration is invalid, the error is logged and the previous
snapshot keeps serving — a bad ConfigMap push never takes the agent down.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from pydantic_ai import Agent
from watchfiles import awatch

from .config import Secrets, ServerSpec, load_server_spec
from .spec import build_agent, load_agent_spec

log = logging.getLogger(__name__)

_WATCH_RETRY_SECONDS = 5.0


@dataclass(frozen=True)
class Snapshot:
    """An agent and the server settings it was loaded with.

    Attributes:
        agent: The Pydantic AI agent built from `agent.yaml`.
        server: The validated `server.yaml`.
        generation: Increments on every successful reload (1 = initial load).
        loaded_at: Unix time the snapshot was built.
    """

    agent: Agent[None, str]
    server: ServerSpec
    generation: int = 1
    loaded_at: float = field(default_factory=time.time)

    @property
    def model_name(self) -> str:
        """The model id this agent is published under on the OpenAI API."""
        return self.agent.name or self.server.agent_card.display_name


def load_snapshot(config_dir: Path, secrets: Secrets, generation: int = 1) -> Snapshot:
    """Load both config files and build the agent.

    Raises:
        ConfigError: If either file is missing or invalid, or the agent can't be
            constructed (unknown model, bad capability arguments, ...).
    """
    server = load_server_spec(config_dir / "server.yaml")
    agent = build_agent(load_agent_spec(config_dir / "agent.yaml", secrets), secrets)
    return Snapshot(agent=agent, server=server, generation=generation)


class AgentRuntime:
    """Holds the current `Snapshot` and reloads it when configuration changes."""

    def __init__(self, config_dir: Path, secrets: Secrets, initial: Snapshot) -> None:
        self.config_dir = config_dir
        self.secrets = secrets
        self._current = initial
        self._reload_requested = asyncio.Event()
        self._listeners: list[Callable[[Snapshot, Snapshot], None]] = []

    @property
    def current(self) -> Snapshot:
        """The snapshot to use for a new request."""
        return self._current

    def on_reload(self, listener: Callable[[Snapshot, Snapshot], None]) -> None:
        """Register ``listener(old, new)``, called after each successful swap."""
        self._listeners.append(listener)

    def reload(self) -> bool:
        """Load the configuration now and swap it in; keep the old one on error.

        Returns:
            True if the new snapshot is live, False if the reload failed.
        """
        old = self._current
        try:
            new = load_snapshot(self.config_dir, self.secrets, old.generation + 1)
        except Exception as exc:  # any failure keeps the last good configuration
            log.error(
                "reload failed; still serving configuration generation %d: %s",
                old.generation,
                exc,
            )
            return False
        self._current = new
        log.info("reload complete: configuration generation %d is live", new.generation)
        for listener in self._listeners:
            try:
                listener(old, new)
            except Exception:
                log.exception("reload listener failed")
        return True

    def request_reload(self) -> None:
        """Ask the background loop to reload soon (idempotent, signal-safe)."""
        self._reload_requested.set()

    async def run(self, stop: asyncio.Event) -> None:
        """Watch the config and secrets directories and reload on changes.

        Runs until ``stop`` is set. Each directory is watched as a whole because
        Kubernetes updates mounted ConfigMaps/Secrets by atomically swapping a
        ``..data`` symlink rather than editing files in place.
        """
        watchers = [
            asyncio.create_task(self._watch(directory, stop), name=f"watch:{directory}")
            for directory in {self.config_dir, self.secrets.directory}
        ]
        stop_waiter = asyncio.create_task(stop.wait())
        try:
            while not stop.is_set():
                reload_waiter = asyncio.create_task(self._reload_requested.wait())
                await asyncio.wait(
                    {reload_waiter, stop_waiter}, return_when=asyncio.FIRST_COMPLETED
                )
                reload_waiter.cancel()
                if stop.is_set():
                    break
                self._reload_requested.clear()
                self.reload()
        finally:
            stop_waiter.cancel()
            for task in watchers:
                task.cancel()
            await asyncio.gather(*watchers, stop_waiter, return_exceptions=True)

    async def _watch(self, directory: Path, stop: asyncio.Event) -> None:
        """Request a reload on every change under ``directory``; never gives up."""
        while not stop.is_set():
            if not directory.is_dir():
                log.info("watcher: %s does not exist; not watching it", directory)
                return
            try:
                log.info("watcher: monitoring %s", directory)
                async for _ in awatch(directory, stop_event=stop, recursive=True):
                    log.info("watcher: change detected in %s", directory)
                    self.request_reload()
            except Exception as exc:
                log.warning(
                    "watcher: error watching %s (%s); restarting in %.0fs",
                    directory,
                    exc,
                    _WATCH_RETRY_SECONDS,
                )
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(stop.wait(), _WATCH_RETRY_SECONDS)

    def install_sighup_handler(self) -> None:
        """Reload on SIGHUP (operator-triggered). No-op where unsupported."""
        try:
            asyncio.get_running_loop().add_signal_handler(
                signal.SIGHUP, self.request_reload
            )
            log.info("SIGHUP triggers a configuration reload")
        except (RuntimeError, NotImplementedError, ValueError, AttributeError) as exc:
            log.debug("SIGHUP reload unavailable: %s", exc)
