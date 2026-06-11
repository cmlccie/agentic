from __future__ import annotations

import logging
from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from . import SECRETS_DIR

log = logging.getLogger(__name__)


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

    def get(self, key: str) -> str | None:
        """Generic accessor for arbitrary secret keys (e.g. MCP tokens)."""
        return self._read(key)


def load_server_spec(path: Path) -> ServerSpec:
    raw = yaml.safe_load(path.read_text())
    return ServerSpec.model_validate(raw)
