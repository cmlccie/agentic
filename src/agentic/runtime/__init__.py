"""Framework-agnostic runtime shared by config-driven agents.

Provides the hot-reload state machine, Kubernetes health probes, and the
configuration loaders (server spec + file-based secrets) used by both the
simple-agent (Pydantic AI) and orchestrator-agent (LangGraph) runtimes.
"""

from .config import CONFIG_DIR, SECRETS_DIR

__all__ = ["CONFIG_DIR", "SECRETS_DIR"]
