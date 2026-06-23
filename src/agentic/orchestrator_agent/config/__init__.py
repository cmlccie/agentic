"""Orchestrator Agent configuration loaders.

Server-spec models and file-based secrets live in the shared
`agentic.runtime.config` module; re-exported here so existing imports
(``from .config import CONFIG_DIR``) keep working. The orchestrator-specific
agent spec (model, instructions, downstream A2A servers) lives in `agent_spec.py`.
"""

from agentic.runtime.config import CONFIG_DIR, SECRETS_DIR

__all__ = ["CONFIG_DIR", "SECRETS_DIR"]
