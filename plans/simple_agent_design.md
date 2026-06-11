# Multi-Interface Pydantic AI Agent Container — Architecture Design

## Design Principles

- **Single process, single port** — one FastAPI root app, all interfaces mounted as ASGI sub-apps or routers
- **Minimum Kubernetes objects** — one ConfigMap (two keys), one Secret (file-projected), no sidecars
- **`agent.yaml` is the source of truth** for model, instructions, capabilities, and MCP servers
- **`server.yaml` is the source of truth** for serving infrastructure — A2A metadata, broker backend, interface toggles
- **Secrets are files, never env vars** — enables hot-reload without process restart
- **Lifespan owns all async lifecycle** — MCP connections, broker, watcher task, SIGHUP handler, readiness gate
- **Hot-reload is a first-class concern** — config file changes drain in-flight work, reload cleanly, restore readiness

---

## Critical Upstream Constraint

**`AgentSpec` does not yet support OpenAI-compatible custom providers in YAML** (open issue [#5471](https://github.com/pydantic/pydantic-ai/issues/5471)).
The workaround is the `model: openai-compat` sentinel in `agent.yaml`, which triggers Python-side
`OpenAIChatModel` construction using `AGENT_MODEL_BASE_URL` and `AGENT_MODEL_API_KEY` from the secret
files. Everything else stays in YAML. First-class providers (Anthropic, OpenAI, Gemini) work natively.

---

## Kubernetes inotify Gotcha

Kubernetes uses `AtomicWriter` to update ConfigMap/Secret volumes: it writes new content to a
timestamped directory, then atomically swaps the `..data` symlink. This means inotify never fires
`IN_MODIFY` on individual files — only `IN_DELETE_SELF` after the swap, which breaks the watch.

**The correct strategy is to watch the parent directory (`/etc/agent`), not individual files.**
`watchfiles.awatch()` (built on Rust's `notify` crate) watches directories and correctly surfaces
the symlink swap as a change event. It is the right tool.

**Do not use `subPath` mounts in the Deployment** — subPath-mounted files bypass `AtomicWriter`
and never receive updates at all.

---

## File Structure Layout

Create a `simple_agent` sub-package in the repository's `agentic` package in the `src/agentic` directory.

```
simple_agent/
├── config/
│   ├── agent_spec.py       # Loads agent.yaml, resolves custom model provider if needed
│   └── server_spec.py      # Loads server.yaml + secret files; typed Pydantic models
│
├── interfaces/
│   ├── a2a.py              # fasta2a ASGI sub-app factory
│   ├── openai_compat.py    # fastapi-openai-compat router factory
│   └── ui.py               # Testing web UI router (chat + SSE stream)
│
├── reload.py               # ReloadCoordinator — drain counter, state machine, watcher task
├── health.py               # /health router — liveness + readiness, reload-aware
├── lifespan.py             # Async startup/teardown; wires everything together
└── cli.py                  # simple-agent command line interface (CLI) to the main app
└── main.py                 # Root FastAPI app; mounts all interfaces
```

---

## Kubernetes Objects

### ConfigMap Example (one per agent, two keys)

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: my-agent-config
data:
  agent.yaml: |
    name: my-agent
    description: "Does X"
    model: anthropic:claude-opus-4-6    # or "openai-compat" sentinel
    instructions: |
      You are a helpful assistant that...
    model_settings:
      max_tokens: 4096
      temperature: 0.3
    capabilities:
      - MCP:
          url: http://tool-server-a/mcp
          allowed_tools: [search, summarize]
          headers:
            Authorization: "Bearer ${MCP_TOKEN_A}"   # expanded at load time
          id: tools-a
      - MCP:
          url: http://tool-server-b/mcp
          allowed_tools: [execute_query]
          id: tools-b

  server.yaml: |
    agent_card:
      display_name: "My Agent"
      description: "Does X for callers"
      version: "1.0.0"
    broker:
      backend: memory          # "memory" | "redis"
    interfaces:
      a2a: true
      openai_compat: true
      ui: true
    reload:
      drain_timeout: 30        # seconds to wait for in-flight tasks before forcing reload
```

### Agent Secret Example (one per agent, **mounted as files**)

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: my-agent-secrets
stringData:
  anthropic_api_key: "sk-ant-..." # file: /etc/agent/secrets/anthropic_api_key
  mcp_token_a: "tok-..." # file: /etc/agent/secrets/mcp_token_a
  # agent_model_base_url: "http://..."  # only for openai-compat model
  # agent_model_api_key: "..."          # only for openai-compat model
  # agent_redis_url: "redis://..."      # only when broker.backend = redis
```

> **Why files, not `envFrom`?** Environment variables are injected at pod start and never updated.
> File-projected secrets are updated atomically by kubelet when the Secret changes — the same
> `AtomicWriter` mechanism as ConfigMaps. The secret reader in `config/server_spec.py` re-reads
> files on every reload cycle, picking up rotated credentials automatically.

### Example Deployment (projected volume combining both sources)

```yaml
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      containers:
        - name: agent
          image: simple-agent:latest
          ports:
            - containerPort: 8000
          volumeMounts:
            - name: config
              mountPath: /etc/agent/config # agent.yaml, server.yaml
              readOnly: true
            - name: secrets
              mountPath: /etc/agent/secrets # one file per secret key
              readOnly: true
          livenessProbe:
            httpGet: { path: /health/live, port: 8000 }
            initialDelaySeconds: 5
            periodSeconds: 10
          readinessProbe:
            httpGet: { path: /health/ready, port: 8000 }
            initialDelaySeconds: 5
            periodSeconds: 5
            failureThreshold: 24 # 120s window for drain + reload
      volumes:
        - name: config
          configMap:
            name: my-agent-config # NO subPath — must mount whole directory
        - name: secrets
          secret:
            secretName: my-agent-secrets # NO subPath — must mount whole directory
```

> **Critical**: Never use `subPath` in volumeMounts for either volume. `subPath`-mounted files
> are bind-mounted directly and bypass `AtomicWriter` — they never receive live updates.

---

## Example `server.yaml`

```yaml
agent_card:
  display_name: string
  description: string
  version: "1.0.0"
  icon_url: ""

broker:
  backend: memory # "memory" | "redis"

interfaces:
  a2a: true
  openai_compat: true
  ui: true

reload:
  drain_timeout:
    30 # max seconds to wait for in-flight A2A tasks to complete
    # before forcing reload regardless
```

---

## Hot-Reload State Machine

```
┌───────────┐   change detected   ┌────────────────┐
│  RUNNING  │ ──────────────────► │    DRAINING    │
│ ready=True│                     │  ready=False   │
└───────────┘                     │ (503 on /ready)│
      ▲                           └────────────────┘
      │                                 │
      │                    in-flight == 0
      │                    OR drain_timeout
      │                                 ▼
      │                          ┌──────────────┐
      └──────────────────────────│  RELOADING   │
          reload complete        │ ready=False  │
                                 └──────────────┘
```

**State transitions:**

- `RUNNING → DRAINING`: file change detected (watchfiles) OR SIGHUP received
- `DRAINING → RELOADING`: `in_flight_count == 0` OR `drain_timeout` expires (whichever first)
- `RELOADING → RUNNING`: new agent constructed, MCP sessions opened, interfaces remounted

**During DRAINING and RELOADING:**

- `/health/ready` returns 503 → Kubernetes stops routing new traffic to this pod
- `/health/live` continues to return 200 → pod is not killed
- All interface endpoints return 503 via middleware (belt-and-suspenders; Kubernetes should have
  already drained traffic before any request reaches the pod in this state)

---

## Module Design

### Example `reload.py`

```python
from __future__ import annotations

import asyncio
import logging
import signal
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import AsyncIterator

from watchfiles import awatch

log = logging.getLogger(__name__)


class AgentState(Enum):
    RUNNING = auto()
    DRAINING = auto()
    RELOADING = auto()


@dataclass
class ReloadCoordinator:
    """
    Central state machine for hot-reload.

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

    # Internal coordination primitives
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
        """
        Called by request handlers before starting work.
        Returns True if the slot was granted (RUNNING state).
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
        """
        Block until drained (in_flight == 0) or drain_timeout expires.
        Transitions to RELOADING state.
        """
        log.info(
            "reload.drain: waiting for %d in-flight requests "
            "(timeout=%ss)", self.in_flight_count, self.drain_timeout
        )
        try:
            await asyncio.wait_for(self._drained.wait(), timeout=self.drain_timeout)
            log.info("reload.drain: clean drain complete")
        except asyncio.TimeoutError:
            log.warning(
                "reload.drain: timeout after %ss with %d requests still in flight — "
                "forcing reload", self.drain_timeout, self.in_flight_count
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


async def _watch_config_directory(
    config_dir: Path,
    coordinator: ReloadCoordinator,
    stop_event: asyncio.Event,
) -> None:
    """
    Watch the config directory for any changes.

    Watches the DIRECTORY, not individual files, because Kubernetes AtomicWriter
    swaps a symlink (..data) rather than modifying files in place. Watching
    individual files only yields IN_DELETE_SELF (which breaks the watch);
    watching the directory catches the symlink swap correctly.

    watchfiles.awatch() is backed by Rust's notify crate and handles the
    symlink-swap event on Linux correctly without re-establishing the watch.
    """
    log.info("watcher: monitoring %s", config_dir)
    try:
        async for _changes in awatch(str(config_dir), stop_event=stop_event):
            log.info("watcher: detected config change in %s", config_dir)
            coordinator.trigger_reload()
            # After triggering, back off briefly to avoid duplicate events
            # from the multi-step AtomicWriter symlink chain.
            await asyncio.sleep(1.0)
    except Exception as e:
        log.error("watcher: unexpected error: %s", e)


def install_sighup_handler(coordinator: ReloadCoordinator) -> None:
    """
    Install SIGHUP handler. SIGHUP forces reload even if files haven't changed
    (e.g. operator-triggered refresh, secret rotation without ConfigMap update).

    Uses loop.add_signal_handler for async-safe delivery.
    """
    loop = asyncio.get_event_loop()

    def _handle_sighup():
        log.info("sighup: received SIGHUP, triggering reload")
        coordinator.trigger_reload()

    loop.add_signal_handler(signal.SIGHUP, _handle_sighup)
    log.info("sighup: handler installed")
```

### `config/server_spec.py` — Updated for File-Based Secrets

```python
from __future__ import annotations

import logging
from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

_SECRETS_DIR = Path("/etc/agent/secrets")


class BrokerBackend(StrEnum):
    MEMORY = "memory"
    REDIS = "redis"


class AgentCardConfig(BaseModel):
    display_name: str
    description: str
    version: str = "1.0.0"
    icon_url: str = ""


class BrokerConfig(BaseModel):
    backend: BrokerBackend = BrokerBackend.MEMORY


class InterfacesConfig(BaseModel):
    a2a: bool = True
    openai_compat: bool = True
    ui: bool = True


class ReloadConfig(BaseModel):
    drain_timeout: float = 30.0


class ServerSpec(BaseModel):
    agent_card: AgentCardConfig
    broker: BrokerConfig = Field(default_factory=BrokerConfig)
    interfaces: InterfacesConfig = Field(default_factory=InterfacesConfig)
    reload: ReloadConfig = Field(default_factory=ReloadConfig)


class AgentSecrets:
    """
    Reads secrets from files at /etc/agent/secrets/<key>.
    Re-read on every call — picks up rotated values automatically.
    All secret names are lowercase filenames (Secret keys are lowercased by Kubernetes).
    """

    def __init__(self, secrets_dir: Path = _SECRETS_DIR):
        self._dir = secrets_dir

    def _read(self, key: str, default: str | None = None) -> str | None:
        p = self._dir / key
        if p.exists():
            return p.read_text().strip()
        if default is not None:
            return default
        return None

    def require(self, key: str) -> str:
        val = self._read(key)
        if val is None:
            raise RuntimeError(
                f"Required secret '{key}' not found at {self._dir / key}"
            )
        return val

    @property
    def anthropic_api_key(self) -> str | None:
        return self._read("anthropic_api_key")

    @property
    def openai_api_key(self) -> str | None:
        return self._read("openai_api_key")

    @property
    def agent_model_base_url(self) -> str | None:
        return self._read("agent_model_base_url")

    @property
    def agent_model_api_key(self) -> str | None:
        return self._read("agent_model_api_key")

    @property
    def agent_redis_url(self) -> str:
        return self._read("agent_redis_url", default="redis://localhost:6379/0")

    def get(self, key: str) -> str | None:
        """Generic accessor for arbitrary secret keys (e.g. MCP tokens)."""
        return self._read(key)


def load_server_spec(path: Path) -> ServerSpec:
    raw = yaml.safe_load(path.read_text())
    return ServerSpec.model_validate(raw)
```

### `config/agent_spec.py` — Updated for File-Based Secrets

```python
from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import yaml
from pydantic_ai import Agent, AgentSpec

from .server_spec import AgentSecrets

log = logging.getLogger(__name__)

_SECRET_REF_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _expand_secret_refs(value: str, secrets: AgentSecrets) -> str:
    """
    Expand ${SECRET_KEY} patterns using file-based secrets.
    Key is lowercased to match Kubernetes Secret key naming.
    Falls back to os.environ for local dev convenience.
    """
    def _resolve(m: re.Match) -> str:
        key = m.group(1).lower()
        val = secrets.get(key)
        if val is not None:
            return val
        # Local dev fallback — env var with same name (uppercase)
        env_val = os.environ.get(m.group(1))
        if env_val is not None:
            return env_val
        log.warning("secret ref ${%s} not resolved — leaving as-is", m.group(1))
        return m.group(0)

    return _SECRET_REF_PATTERN.sub(_resolve, value)


def _expand_headers_in_spec(spec_dict: dict, secrets: AgentSecrets) -> dict:
    """Post-process MCP capability headers to expand ${SECRET_KEY} patterns."""
    for cap in spec_dict.get("capabilities", []):
        if not isinstance(cap, dict):
            continue
        mcp_conf = cap.get("MCP", {})
        if headers := mcp_conf.get("headers"):
            mcp_conf["headers"] = {
                k: _expand_secret_refs(v, secrets)
                for k, v in headers.items()
            }
    return spec_dict


def load_agent(spec_path: Path, secrets: AgentSecrets) -> Agent:
    """
    Load and construct the Pydantic AI agent from agent.yaml.
    Called on initial startup and on every reload cycle.
    """
    raw = yaml.safe_load(spec_path.read_text())
    raw = _expand_headers_in_spec(raw, secrets)

    # Inject API keys into environment for pydantic-ai's own provider resolution.
    # pydantic-ai reads ANTHROPIC_API_KEY, OPENAI_API_KEY, etc. from os.environ.
    # We set them here from secret files so they're always current after a reload.
    if key := secrets.anthropic_api_key:
        os.environ["ANTHROPIC_API_KEY"] = key
    if key := secrets.openai_api_key:
        os.environ["OPENAI_API_KEY"] = key

    spec = AgentSpec.model_validate(raw)

    # Workaround for upstream #5471 — custom OpenAI-compatible endpoints
    model_override = None
    if raw.get("model") == "openai-compat":
        base_url = secrets.agent_model_base_url
        api_key = secrets.agent_model_api_key
        if not (base_url and api_key):
            raise RuntimeError(
                "model: openai-compat requires secret files "
                "'agent_model_base_url' and 'agent_model_api_key'"
            )
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider
        model_override = OpenAIChatModel(
            raw.get("model_id", "custom"),
            provider=OpenAIProvider(base_url=base_url, api_key=api_key),
        )

    return Agent.from_spec(spec, model=model_override)
```

### Example `lifespan.py`

```python
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator

from fastapi import FastAPI
from pydantic_ai import Agent

from config.agent_spec import load_agent
from config.server_spec import AgentSecrets, BrokerBackend, ServerSpec, load_server_spec
from reload import AgentState, ReloadCoordinator, _watch_config_directory, install_sighup_handler

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

_CONFIG_DIR = Path("/etc/agent/config")
_SECRETS_DIR = Path("/etc/agent/secrets")


@dataclass
class AppState:
    """
    Mutable shared state. Held on app.state.app_state.

    agent and server_spec are replaced atomically on each reload cycle.
    coordinator and secrets are long-lived singletons for the process lifetime.
    """
    coordinator: ReloadCoordinator
    secrets: AgentSecrets
    # Replaced on each reload:
    agent: Agent
    server_spec: ServerSpec


def _build_a2a_backends(spec: ServerSpec, secrets: AgentSecrets):
    from fasta2a import InMemoryStorage, InMemoryBroker

    if spec.broker.backend == BrokerBackend.MEMORY:
        return InMemoryStorage(), InMemoryBroker()

    try:
        from fasta2a.redis import RedisStorage, RedisBroker  # type: ignore[import]
        import redis.asyncio as aioredis
    except ImportError as e:
        raise RuntimeError(
            "broker.backend = redis requires fasta2a[redis] and redis packages"
        ) from e

    pool = aioredis.ConnectionPool.from_url(secrets.agent_redis_url)
    client = aioredis.Redis(connection_pool=pool)
    return RedisStorage(client), RedisBroker(client)


async def _run_reload_loop(app: FastAPI) -> None:
    """
    Long-running task that waits for reload requests, drains, reloads, and
    re-enters the agent context. Runs for the entire process lifetime.
    """
    state: AppState = app.state.app_state

    while True:
        # Block until a reload is triggered (file change or SIGHUP)
        await state.coordinator.wait_for_reload_request()

        log.info("reload_loop: reload requested — beginning drain")

        # Step 1: Drain — block new requests, wait for in-flight to finish
        await state.coordinator.wait_for_drain()

        # Step 2: Exit the current agent context (closes MCP sessions)
        log.info("reload_loop: closing MCP sessions")
        try:
            await state.agent.__aexit__(None, None, None)
        except Exception as e:
            log.error("reload_loop: error closing agent MCP sessions: %s", e)

        # Step 3: Build the new agent from fresh config files
        log.info("reload_loop: loading new configuration")
        try:
            new_spec = load_server_spec(_CONFIG_DIR / "server.yaml")
            new_agent = load_agent(_CONFIG_DIR / "agent.yaml", state.secrets)

            # Update coordinator drain_timeout if it changed
            state.coordinator.drain_timeout = new_spec.reload.drain_timeout

            # Step 4: Enter new agent context (opens MCP sessions)
            await new_agent.__aenter__()

            # Atomically swap state — existing requests already have the old agent ref
            state.agent = new_agent
            state.server_spec = new_spec

            state.coordinator.mark_running()
            log.info("reload_loop: reload complete — agent is RUNNING")

        except Exception as e:
            log.error(
                "reload_loop: reload failed: %s — agent remains in RELOADING state; "
                "will retry on next trigger", e
            )
            # Don't mark_running — leave in RELOADING (503) until next trigger succeeds.
            # Operator can fix the config and send another SIGHUP or update the ConfigMap.


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    state: AppState = app.state.app_state

    # Initial MCP session open
    await state.agent.__aenter__()
    state.coordinator.mark_running()

    # Start the file watcher as a background task
    stop_event = asyncio.Event()
    watcher_task = asyncio.create_task(
        _watch_config_directory(_CONFIG_DIR, state.coordinator, stop_event),
        name="config-watcher",
    )
    # Also watch the secrets directory for credential rotation
    secrets_watcher_task = asyncio.create_task(
        _watch_config_directory(_SECRETS_DIR, state.coordinator, stop_event),
        name="secrets-watcher",
    )

    # Start the reload loop
    reload_task = asyncio.create_task(
        _run_reload_loop(app),
        name="reload-loop",
    )

    # Install SIGHUP handler
    install_sighup_handler(state.coordinator)

    try:
        yield
    finally:
        log.info("lifespan: shutting down")
        # Stop background tasks cleanly
        stop_event.set()
        reload_task.cancel()
        watcher_task.cancel()
        secrets_watcher_task.cancel()

        # Close the current agent context
        try:
            await state.agent.__aexit__(None, None, None)
        except Exception as e:
            log.error("lifespan: error on shutdown agent exit: %s", e)

        # Wait briefly for tasks to acknowledge cancellation
        await asyncio.gather(
            reload_task, watcher_task, secrets_watcher_task,
            return_exceptions=True,
        )
```

### Example `health.py`

```python
from __future__ import annotations

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

from reload import AgentState

router = APIRouter()


@router.get("/health/live")
async def liveness():
    """
    Kubernetes liveness probe.
    Always 200 — process is alive even during drain/reload.
    Returning non-200 here causes pod restart, which we never want during a hot-reload.
    """
    return {"status": "alive"}


@router.get("/health/ready")
async def readiness(request: Request):
    """
    Kubernetes readiness probe.
    503 during DRAINING and RELOADING — Kubernetes stops routing new traffic.
    200 only when RUNNING.
    """
    coordinator = request.app.state.app_state.coordinator
    state = coordinator.state

    if state == AgentState.RUNNING:
        return {"status": "ready", "in_flight": coordinator.in_flight_count}

    return JSONResponse(
        {
            "status": "not_ready",
            "state": state.name,
            "in_flight": coordinator.in_flight_count,
        },
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
    )
```

### Example `main.py`

```python
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette import status

from config.agent_spec import load_agent
from config.server_spec import AgentSecrets, load_server_spec
from health import router as health_router
from interfaces.a2a import build_a2a_app
from interfaces.openai_compat import build_openai_router
from interfaces.ui import build_ui_router
from lifespan import AppState, lifespan
from reload import ReloadCoordinator

log = logging.getLogger(__name__)

_CONFIG_DIR = Path("/etc/agent/config")
_SECRETS_DIR = Path("/etc/agent/secrets")

# ── Bootstrap ──────────────────────────────────────────────────────────────────
# Load once at process start. All subsequent loads happen inside _run_reload_loop.
secrets = AgentSecrets(_SECRETS_DIR)
server_spec = load_server_spec(_CONFIG_DIR / "server.yaml")
agent = load_agent(_CONFIG_DIR / "agent.yaml", secrets)
coordinator = ReloadCoordinator(drain_timeout=server_spec.reload.drain_timeout)

# ── App State ──────────────────────────────────────────────────────────────────
_state = AppState(
    coordinator=coordinator,
    secrets=secrets,
    agent=agent,
    server_spec=server_spec,
)

# ── Root App ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title=server_spec.agent_card.display_name,
    lifespan=lifespan,
)
app.state.app_state = _state


# ── Reload Gate Middleware ─────────────────────────────────────────────────────
@app.middleware("http")
async def reload_gate(request: Request, call_next):
    """
    Belt-and-suspenders gate: returns 503 for all non-health requests during reload.
    Kubernetes should have already drained traffic via the readiness probe, but this
    catches any requests that slip through the propagation window.

    Health endpoints are always allowed through so probes continue to function.
    """
    if request.url.path.startswith("/health"):
        return await call_next(request)

    coordinator = request.app.state.app_state.coordinator
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


# ── Interface Mounts ───────────────────────────────────────────────────────────
# Interfaces are mounted at startup based on the initial server.yaml.
# NOTE: Interface mounts are NOT hot-reloaded — toggling interfaces requires a pod restart.
# Agent identity (model, instructions, MCP servers) IS hot-reloaded.
app.include_router(health_router)

if server_spec.interfaces.a2a:
    a2a_app = build_a2a_app(agent, server_spec, secrets)
    app.mount("/a2a", a2a_app)

if server_spec.interfaces.openai_compat:
    openai_router = build_openai_router(agent, model_name=agent.name or "agent")
    app.include_router(openai_router, prefix="/v1")

if server_spec.interfaces.ui:
    ui_router = build_ui_router(agent)
    app.include_router(ui_router, prefix="/ui")
```

> **Interface mount caveat**: The OpenAI and UI interface handlers close over the `agent` variable
> captured at mount time. During a reload, `AppState.agent` is replaced, but the closures in
> `openai_compat.py` and `ui.py` must be written to read the agent from `app.state.app_state.agent`
> at call time — not from a captured local. See the interface module implementations.

### Example `interfaces/openai_compat.py`

```python
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi_openai_compat import create_chat_completion_router


def build_openai_router(initial_agent, model_name: str) -> APIRouter:
    """
    Closures read agent from app state at call time, not from capture.
    This makes the handler reload-safe: after a reload, new requests use the new agent.
    """

    async def run_completion(model: str, messages: list[dict], body: dict, request: Request) -> str:
        agent = request.app.state.app_state.agent
        history = [{"role": m["role"], "content": m["content"]} for m in messages[:-1]]
        user_prompt = messages[-1]["content"] if messages else ""
        result = await agent.run(user_prompt, message_history=history)
        return result.output

    async def run_stream(model: str, messages: list[dict], body: dict, request: Request):
        agent = request.app.state.app_state.agent
        history = [{"role": m["role"], "content": m["content"]} for m in messages[:-1]]
        user_prompt = messages[-1]["content"] if messages else ""
        async with agent.run_stream(user_prompt, message_history=history) as result:
            async for chunk in result.stream_text(delta=True):
                yield chunk

    return create_chat_completion_router(
        list_models=lambda: [model_name],
        run_completion=run_completion,
        run_stream=run_stream,
    )
```

---

## URL Map

| Path                              | Protocol       | Reload behavior                          |
| --------------------------------- | -------------- | ---------------------------------------- |
| `GET /health/live`                | HTTP           | Always 200, exempt from reload gate      |
| `GET /health/ready`               | HTTP           | 503 during DRAINING/RELOADING            |
| `POST /a2a/`                      | JSON-RPC / SSE | 503 during reload; in-flight tasks drain |
| `GET /a2a/.well-known/agent.json` | HTTP           | 503 during reload                        |
| `POST /v1/chat/completions`       | HTTP / SSE     | 503 during reload                        |
| `GET /v1/models`                  | HTTP           | 503 during reload                        |
| `GET /ui/`                        | HTTP           | 503 during reload                        |
| `POST /ui/chat`                   | SSE            | 503 during reload                        |

---

## CLI

Create a typer CLI interface with the following sub-commands:

- `simple-agent serve` (default container command)
- `simple-agent chat` (calls `agent.to_cli_sync()` to allow testing the agent from the terminal)

---

## `pyproject.toml` Dependencies

Use `uv add <package>` to add dependencies. Allow `uv` to get the latest package version and manage
the versions and environment.

Required dependencies:

- pydantic-ai[a2a]
- fasta2a[pydantic-ai,redis]
- fastapi
- uvicorn[standard]
- fastapi-openai-compat
- pydantic-settings
- pyyaml
- httpx
- watchfiles
- redis[hiredis]
- typer

Add the `simple-agent` CLI as a package script so that it is accessible as `simple-agent` after the
package is installed.

---

## Design Rationale for Hot-Reload

**Why watch the directory, not the files?**
Kubernetes `AtomicWriter` never modifies files in place. It writes to a new timestamped directory
and atomically swaps the `..data` symlink. inotify watching an individual file sees `IN_DELETE_SELF`
(which breaks the watch) and never sees a write event. Watching the directory with `watchfiles.awatch()`
correctly surfaces the symlink swap as a directory change without needing to re-establish the watch.

**Why `watchfiles` over `asyncinotify` or `aionotify`?**
`watchfiles` is backed by Rust's `notify` crate, handles the symlink-swap pattern out of the box,
provides built-in debouncing to coalesce the multi-step AtomicWriter sequence into a single event,
and works identically in local dev (macOS `FSEvents`, Linux inotify) without any platform conditionals.
The async API is a clean `async for changes in awatch(path)`.

**Why watch `/etc/agent/secrets` separately?**
Secret rotation (e.g. rotating an API key without changing agent config) updates only the secrets
volume. Watching both directories means a key rotation triggers a reload that re-reads credentials,
ensuring the agent immediately uses the new key without downtime.

**Why 503 (not 429) during drain?**
503 causes Kubernetes to remove the pod from the Service endpoint slice immediately when the
readiness probe fails. The pod stops receiving new traffic at the load balancer level — which is
what you want. 429 keeps the pod in rotation but rejects at the application layer, requiring
callers to handle retry logic that Kubernetes would otherwise handle transparently through its
endpoint routing.

**Why are interface mounts not hot-reloaded?**
The FastAPI router tree is built at startup and is not mutable after that. Toggling `a2a: false`
in `server.yaml` during a running pod has no effect on the mounted routes. Changing interface
topology requires a pod restart (which is appropriate — it's an infrastructure change, not an
agent config change). Agent identity changes (model, instructions, MCP servers) are hot-reloaded.

**Why does `reload_gate` middleware return 503 even though the readiness probe does too?**
Readiness probe failover takes 5-10 seconds to propagate through kube-proxy to all clients. During
that window, in-flight requests may still arrive. The middleware is a belt-and-suspenders guard for
that propagation window, not the primary traffic control mechanism.

**Why does a failed reload leave the agent in RELOADING state (not revert to old config)?**
By the time a reload fails, the old agent's MCP sessions have already been closed. There is no
safe "rollback to old agent" — the old sessions are gone. Leaving the pod in 503 until the operator
fixes the config and triggers another reload (via ConfigMap fix or SIGHUP) is the honest behavior.
Kubernetes will keep the pod out of rotation until it recovers. If all pods fail to reload, the
Deployment's `minReadySeconds` and PodDisruptionBudget prevent a full outage.
