from __future__ import annotations

import asyncio
import logging
import signal
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path

from watchfiles import awatch

log = logging.getLogger(__name__)


class AgentState(Enum):
    RUNNING = auto()
    DRAINING = auto()
    RELOADING = auto()


@dataclass
class ReloadCoordinator:
    """Central state machine for hot-reload.

    Shared by:
      - lifespan.py  (owns the agent lifecycle and calls perform_reload)
      - health.py    (reads state and in_flight_count for readiness)
      - middleware    (reads state to gate 503 responses)
      - watcher task (calls trigger_reload on file change)
      - SIGHUP handler (calls trigger_reload on signal)
    """

    drain_timeout: float = 30.0

    state: AgentState = field(default=AgentState.RUNNING, init=False)
    in_flight_count: int = field(default=0, init=False)

    _reload_requested: asyncio.Event = field(default_factory=asyncio.Event, init=False)
    _drained: asyncio.Event = field(default_factory=asyncio.Event, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    @property
    def is_ready(self) -> bool:
        return self.state == AgentState.RUNNING

    @property
    def is_reloading(self) -> bool:
        return self.state in (AgentState.DRAINING, AgentState.RELOADING)

    def trigger_reload(self) -> None:
        """Signal that a reload should begin. Idempotent — safe to call multiple times."""
        if self.state == AgentState.RUNNING:
            log.info("reload.trigger: transitioning RUNNING → DRAINING")
            self.state = AgentState.DRAINING
            self._reload_requested.set()

    async def request_slot(self) -> bool:
        """Called by request handlers before starting work.

        Returns True if slot was granted (RUNNING state).
        Returns False if the agent is draining/reloading — caller should return 503.
        """
        async with self._lock:
            if self.state != AgentState.RUNNING:
                return False
            self.in_flight_count += 1
            return True

    async def release_slot(self) -> None:
        """Called by request handlers when work completes (success or error)."""
        async with self._lock:
            self.in_flight_count = max(0, self.in_flight_count - 1)
            if self.state == AgentState.DRAINING and self.in_flight_count == 0:
                log.info("reload.drain: all slots released, signaling drained")
                self._drained.set()

    async def wait_for_drain(self) -> None:
        """Block until drained (in_flight == 0) or drain_timeout expires.

        Transitions to RELOADING state regardless of outcome.
        """
        log.info(
            "reload.drain: waiting for %d in-flight requests (timeout=%ss)",
            self.in_flight_count,
            self.drain_timeout,
        )
        try:
            await asyncio.wait_for(self._drained.wait(), timeout=self.drain_timeout)
            log.info("reload.drain: clean drain complete")
        except asyncio.TimeoutError:
            log.warning(
                "reload.drain: timeout after %ss with %d requests still in flight — "
                "forcing reload",
                self.drain_timeout,
                self.in_flight_count,
            )
        finally:
            self.state = AgentState.RELOADING
            self._drained.clear()

    def mark_running(self) -> None:
        """Called after reload completes successfully."""
        self._reload_requested.clear()
        self.state = AgentState.RUNNING
        log.info("reload.complete: agent is RUNNING")

    async def wait_for_reload_request(self) -> None:
        """Awaited by the reload loop in lifespan to block until a reload is needed."""
        await self._reload_requested.wait()


async def watch_config_directory(
    config_dir: Path,
    coordinator: ReloadCoordinator,
    stop_event: asyncio.Event,
) -> None:
    """Watch a config directory for any changes and trigger reload.

    Watches the DIRECTORY, not individual files, because Kubernetes AtomicWriter
    swaps a symlink (..data) rather than modifying files in place. watchfiles.awatch()
    backed by Rust's notify crate handles the symlink-swap event correctly.
    """
    log.info("watcher: monitoring %s", config_dir)
    try:
        async for _changes in awatch(str(config_dir), stop_event=stop_event):
            log.info("watcher: detected config change in %s", config_dir)
            coordinator.trigger_reload()
            # Brief back-off to avoid duplicate events from the multi-step
            # AtomicWriter symlink chain.
            await asyncio.sleep(1.0)
    except Exception as exc:
        log.error("watcher: unexpected error: %s", exc)


def install_sighup_handler(coordinator: ReloadCoordinator) -> None:
    """Install SIGHUP handler for operator-triggered reload.

    Uses loop.add_signal_handler for async-safe delivery. Signal handlers can
    only be installed on the main thread of the main interpreter; when running
    off the main thread (e.g. under a test client) or on a platform without
    SIGHUP, this degrades to a no-op with a warning rather than failing startup.
    """
    loop = asyncio.get_event_loop()

    def _handle_sighup() -> None:
        log.info("sighup: received SIGHUP, triggering reload")
        coordinator.trigger_reload()

    try:
        loop.add_signal_handler(signal.SIGHUP, _handle_sighup)
        log.info("sighup: handler installed")
    except (RuntimeError, NotImplementedError, ValueError, AttributeError) as exc:
        log.warning(
            "sighup: could not install handler (%s); SIGHUP-triggered reload "
            "disabled (config/secret file watching still active)",
            exc,
        )
