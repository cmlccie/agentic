"""Shared configuration: server spec models and file-based secrets.

`server.yaml` describes the agent card, A2A task broker backend, enabled
interfaces, and reload behavior — all framework-agnostic and identical between
the simple-agent and orchestrator-agent runtimes. Secrets are read from files
(Kubernetes Secret volume) on every access so rotated values are picked up
without a process restart.
"""

from __future__ import annotations

import logging
import os
import re
from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

CONFIG_DIR = Path("/etc/agent/config")
SECRETS_DIR = Path("/etc/agent/secrets")

log = logging.getLogger(__name__)

_SECRET_REF_PATTERN = re.compile(r"\$\{([^}]+)\}")


class BrokerBackend(StrEnum):
    MEMORY = "memory"
    REDIS = "redis"
    POSTGRES = "postgres"


class SkillConfig(BaseModel):
    id: str
    name: str
    description: str
    tags: list[str] = Field(default_factory=list)
    examples: list[str] | None = None
    input_modes: list[str] = Field(default_factory=lambda: ["text/plain"])
    output_modes: list[str] = Field(default_factory=lambda: ["text/plain"])


class ProviderConfig(BaseModel):
    organization: str
    url: str


class AgentCardConfig(BaseModel):
    display_name: str
    description: str
    version: str = "1.0.0"
    icon_url: str = ""
    documentation_url: str | None = None
    provider: ProviderConfig | None = None
    skills: list[SkillConfig] = Field(default_factory=list)


class BrokerConfig(BaseModel):
    backend: BrokerBackend = BrokerBackend.MEMORY


class InterfacesConfig(BaseModel):
    a2a: bool = True
    openai_compat: bool = True


class ReloadConfig(BaseModel):
    drain_timeout: float = 30.0


class ServerSpec(BaseModel):
    agent_card: AgentCardConfig
    broker: BrokerConfig = Field(default_factory=BrokerConfig)
    interfaces: InterfacesConfig = Field(default_factory=InterfacesConfig)
    reload: ReloadConfig = Field(default_factory=ReloadConfig)


class AgentSecrets:
    """Reads secrets from files at /etc/agent/secrets/<key>.

    Re-read on every call — picks up rotated values automatically.
    All secret names are lowercase filenames (Secret keys are lowercased by Kubernetes).
    """

    def __init__(self, secrets_dir: Path = SECRETS_DIR) -> None:
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
        return self._read("agent_redis_url", default="redis://localhost:6379/0")  # type: ignore[return-value]

    @property
    def agent_database_url(self) -> str | None:
        """SQLAlchemy async DSN for the A2A task store, e.g.
        ``postgresql+asyncpg://user:pass@host:5432/dbname``.
        """
        return self._read("agent_database_url")

    def get(self, key: str) -> str | None:
        """Generic accessor for arbitrary secret keys (e.g. MCP tokens, A2A tokens)."""
        return self._read(key)


def load_server_spec(path: Path) -> ServerSpec:
    raw = yaml.safe_load(path.read_text())
    return ServerSpec.model_validate(raw)


def expand_secret_refs(value: str, secrets: AgentSecrets) -> str:
    """Expand ${SECRET_KEY} patterns using file-based secrets.

    Key is lowercased to match Kubernetes Secret key naming.
    Falls back to os.environ for local dev convenience.
    """

    def _resolve(m: re.Match) -> str:
        key = m.group(1).lower()
        val = secrets.get(key)
        if val is not None:
            return val
        env_val = os.environ.get(m.group(1))
        if env_val is not None:
            return env_val
        log.warning("secret ref ${%s} not resolved — leaving as-is", m.group(1))
        return m.group(0)

    return _SECRET_REF_PATTERN.sub(_resolve, value)
