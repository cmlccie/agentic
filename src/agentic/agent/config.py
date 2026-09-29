"""Configuration for config-driven agents: `server.yaml`, secrets, and loaders.

An agent is described by two files in its config directory:

- `agent.yaml` — *what* the agent is: a Pydantic AI `AgentSpec` (model,
  instructions, model settings, capabilities). See `agentic.agent.spec`.
- `server.yaml` — *how* it is served: the A2A agent card, which interfaces are
  enabled, A2A storage and task behavior, activity streaming, and auth.

Secrets are read from files (a Kubernetes Secret volume) on every access, so
rotated values are picked up on the next reload without a process restart.

Every model here forbids unknown keys so typos fail loudly instead of being
silently ignored. A small set of keys from earlier versions of `server.yaml` is
migrated (with a warning) by `migrate_legacy_server_config` before validation.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

log = logging.getLogger(__name__)

CONFIG_DIR = Path(os.environ.get("AGENT_CONFIG_DIR", "/etc/agent/config"))
SECRETS_DIR = Path(os.environ.get("AGENT_SECRETS_DIR", "/etc/agent/secrets"))

_SECRET_REF = re.compile(r"\$\{([^}]+)\}")


class ConfigError(ValueError):
    """Raised when a config file is missing, malformed, or fails validation."""


# --------------------------------------------------------------------------------------
# server.yaml models
# --------------------------------------------------------------------------------------


class _Strict(BaseModel):
    """Base model that rejects unknown keys (catches typos in YAML)."""

    model_config = ConfigDict(extra="forbid")


class SkillConfig(_Strict):
    """An A2A agent-card skill: something the agent is good at."""

    id: str
    name: str
    description: str
    tags: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)
    input_modes: list[str] = Field(default_factory=lambda: ["text/plain"])
    output_modes: list[str] = Field(default_factory=lambda: ["text/plain"])


class ProviderConfig(_Strict):
    """The organization that operates the agent (shown on the agent card)."""

    organization: str
    url: str


class AgentCardConfig(_Strict):
    """Public identity of the agent, published as the A2A agent card."""

    display_name: str
    description: str
    version: str = "1.0.0"
    icon_url: str = ""
    documentation_url: str = ""
    provider: ProviderConfig | None = None
    skills: list[SkillConfig] = Field(default_factory=list)


class InterfacesConfig(_Strict):
    """Which client-facing interfaces are served."""

    openai: bool = True
    a2a: bool = True


class ResponseMode(StrEnum):
    """How the A2A server answers a new message.

    - ``auto`` — reply with a direct Message when the agent answers quickly
      without using tools; promote the exchange to a Task (with streamed status
      updates) as soon as the agent starts working (a tool call) or takes longer
      than ``promote_after_seconds``.
    - ``message`` — always reply with a direct Message (no Task).
    - ``task`` — always create a Task.
    """

    AUTO = "auto"
    MESSAGE = "message"
    TASK = "task"


class StoreBackend(StrEnum):
    """Where A2A tasks and conversation history are kept."""

    MEMORY = "memory"
    SQL = "sql"


class StoreConfig(_Strict):
    """A2A persistence.

    ``memory`` needs no external services (single replica; lost on restart).
    ``sql`` persists tasks and conversation history in any SQLAlchemy async
    database (PostgreSQL via ``postgresql+asyncpg://``, SQLite via
    ``sqlite+aiosqlite://``). The DSN is read from the secret named by
    ``database_url_secret``.
    """

    backend: StoreBackend = StoreBackend.MEMORY
    database_url_secret: str = "a2a.database_url"
    #: memory backend only: most tasks / conversations kept (oldest evicted first)
    max_tasks: int = Field(default=10_000, ge=1)
    max_contexts: int = Field(default=1000, ge=1)
    max_history_messages: int = Field(default=200, ge=2)


class PushNotificationsConfig(_Strict):
    """A2A push notifications (webhooks the server calls on task updates).

    Disabled by default: when enabled the server makes outbound HTTP requests to
    client-supplied URLs, so restrict the destinations with ``allowed_hosts``.
    """

    enabled: bool = False
    allowed_hosts: list[str] = Field(default_factory=list)


class A2AConfig(_Strict):
    """A2A server behavior."""

    response_mode: ResponseMode = ResponseMode.AUTO
    promote_after_seconds: float = Field(default=2.0, ge=0)
    store: StoreConfig = Field(default_factory=StoreConfig)
    push_notifications: PushNotificationsConfig = Field(
        default_factory=PushNotificationsConfig
    )


class StreamingConfig(_Strict):
    """What agent activity is streamed to clients while the agent works.

    Activity (thinking, tool calls and results, sub-agent activity) goes to the
    OpenAI ``reasoning_content`` channel and to A2A ``working`` status updates;
    the final answer is always sent separately.

    - ``off`` — no activity; only the final answer.
    - ``summary`` — tool names with short argument/result previews.
    - ``trace`` — full (still redacted) arguments and results.
    """

    activity: Literal["off", "summary", "trace"] = "summary"
    thinking: bool = True
    max_args_chars: int = Field(default=200, ge=0)
    max_result_chars: int = Field(default=300, ge=0)
    redact_keys: list[str] = Field(
        default_factory=lambda: [
            "password",
            "secret",
            "token",
            "api_key",
            "apikey",
            "authorization",
            "credential",
        ]
    )
    heartbeat_seconds: float = Field(default=15.0, gt=0)


class AuthConfig(_Strict):
    """Optional bearer-token authentication for `/v1/*` and `/a2a/*`.

    When ``bearer_token_secret`` names a secret file, requests must send
    ``Authorization: Bearer <token>``. Health probes and the public agent card
    stay unauthenticated.
    """

    bearer_token_secret: str | None = None


class ServerSpec(_Strict):
    """The complete, validated contents of `server.yaml`."""

    agent_card: AgentCardConfig
    interfaces: InterfacesConfig = Field(default_factory=InterfacesConfig)
    a2a: A2AConfig = Field(default_factory=A2AConfig)
    streaming: StreamingConfig = Field(default_factory=StreamingConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)


# --------------------------------------------------------------------------------------
# Secrets
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Secrets:
    """Read secrets from files in a directory (e.g. a mounted Kubernetes Secret).

    Values are re-read on every access, so rotated secrets take effect on the
    next use or reload. Keys are file names; Kubernetes Secret keys are case
    sensitive, and by convention this project uses lowercase keys.
    """

    directory: Path = SECRETS_DIR

    def get(self, key: str) -> str | None:
        """Return the secret's stripped value, or None if the file doesn't exist."""
        try:
            return (self.directory / key).read_text().strip()
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            return None

    def require(self, key: str) -> str:
        """Return the secret's value or raise a ConfigError naming the missing file."""
        value = self.get(key)
        if value is None:
            raise ConfigError(
                f"required secret '{key}' was not found at {self.directory / key}"
            )
        return value


def expand_secret_refs(value: str, secrets: Secrets) -> str:
    """Expand ``${NAME}`` references in a string.

    Each reference resolves from the secret file ``NAME`` (tried as written and
    lowercased), then from the environment variable ``NAME``. An unresolved
    reference raises ConfigError instead of sending the literal ``${NAME}`` to a
    downstream service (which would fail later with a confusing 401).
    """

    def resolve(match: re.Match[str]) -> str:
        name = match.group(1)
        for candidate in (secrets.get(name), secrets.get(name.lower())):
            if candidate is not None:
                return candidate
        if (env := os.environ.get(name)) is not None:
            return env
        raise ConfigError(
            f"unresolved reference ${{{name}}}: create the secret file "
            f"'{secrets.directory / name.lower()}' or set the environment variable "
            f"{name}"
        )

    return _SECRET_REF.sub(resolve, value)


def expand_secret_refs_deep(data: Any, secrets: Secrets) -> Any:
    """Recursively expand ``${NAME}`` references in every string of a YAML tree."""
    if isinstance(data, str):
        return expand_secret_refs(data, secrets)
    if isinstance(data, Mapping):
        return {k: expand_secret_refs_deep(v, secrets) for k, v in data.items()}
    if isinstance(data, list):
        return [expand_secret_refs_deep(v, secrets) for v in data]
    return data


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


def read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML mapping from ``path`` with friendly errors."""
    try:
        data = yaml.safe_load(path.read_text())
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path} is not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a YAML mapping at the top level")
    return data


def migrate_legacy_server_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Translate keys from earlier `server.yaml` versions, warning for each one.

    - ``broker.backend`` → ``a2a.store.backend`` (``postgres`` → ``sql``;
      ``redis`` was never functional and falls back to ``memory``)
    - ``interfaces.openai_compat`` → ``interfaces.openai``
    - ``interfaces.ui`` and ``reload`` are no longer used and are dropped
    """
    raw = dict(raw)

    if (broker := raw.pop("broker", None)) is not None:
        backend = (broker or {}).get("backend", "memory")
        mapped = {"postgres": "sql", "redis": "memory"}.get(backend, backend)
        log.warning(
            "server.yaml: 'broker.backend: %s' is deprecated; use "
            "'a2a.store.backend: %s'",
            backend,
            mapped,
        )
        a2a = dict(raw.get("a2a") or {})
        store = dict(a2a.get("store") or {})
        store.setdefault("backend", mapped)
        if backend == "postgres":
            store.setdefault("database_url_secret", "task_broker.database_url")
        a2a["store"] = store
        raw["a2a"] = a2a

    if isinstance(interfaces := raw.get("interfaces"), dict):
        interfaces = dict(interfaces)
        if "openai_compat" in interfaces:
            log.warning(
                "server.yaml: 'interfaces.openai_compat' is deprecated; use "
                "'interfaces.openai'"
            )
            interfaces.setdefault("openai", interfaces.pop("openai_compat"))
        if "ui" in interfaces:
            log.warning(
                "server.yaml: 'interfaces.ui' is not served by 'serve'; ignoring it "
                "(use the 'web' command for the web chat UI)"
            )
            interfaces.pop("ui")
        raw["interfaces"] = interfaces

    if raw.pop("reload", None) is not None:
        log.warning(
            "server.yaml: 'reload' is no longer used (reloads swap the agent "
            "without draining); ignoring it"
        )

    return raw


def load_server_spec(path: Path) -> ServerSpec:
    """Load, migrate, and validate `server.yaml`."""
    raw = migrate_legacy_server_config(read_yaml(path))
    try:
        return ServerSpec.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"{path} is invalid:\n{exc}") from exc
