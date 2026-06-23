"""Orchestrator agent spec (agent.yaml) loader.

Defines the model, supervisor system prompt, and the list of downstream A2A
servers the orchestrator delegates to. Unlike the simple-agent (whose tools are
MCP servers), the orchestrator's "tools" are remote A2A agents — see
`a2a_workers.py`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from agentic.runtime.config import AgentSecrets, expand_secret_refs

log = logging.getLogger(__name__)


class A2AServerConfig(BaseModel):
    """A downstream A2A server the orchestrator can delegate to.

    `url` is the base URL of the agent's A2A endpoint (the path under which its
    `/.well-known/agent-card.json` is served). `headers` may contain
    ``${SECRET_KEY}`` references resolved from the secrets directory (e.g. for
    bearer tokens). The agent card is fetched at startup/reload to derive the
    delegation tool's name, description, and skills.
    """

    url: str
    id: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)


class OrchestratorSpec(BaseModel):
    name: str
    description: str = ""
    model: str
    model_id: str | None = None
    instructions: str = ""
    model_settings: dict[str, Any] = Field(default_factory=dict)
    a2a_servers: list[A2AServerConfig] = Field(default_factory=list)

    def resolved_headers(self, server: A2AServerConfig, secrets: AgentSecrets) -> dict:
        """Return a server's headers with ${SECRET_KEY} references expanded."""
        return {k: expand_secret_refs(v, secrets) for k, v in server.headers.items()}


def load_orchestrator_spec(path: Path) -> OrchestratorSpec:
    """Load and validate the orchestrator agent.yaml.

    Called on initial startup and on every reload cycle. Performs no network
    access — downstream agent cards are fetched separately when the graph is
    built (see `graph.build_graph`).
    """
    raw = yaml.safe_load(path.read_text())
    return OrchestratorSpec.model_validate(raw)
