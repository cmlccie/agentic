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


def _expand_headers_in_spec(spec_dict: dict, secrets: AgentSecrets) -> dict:
    """Post-process MCP capability headers to expand ${SECRET_KEY} patterns."""
    for cap in spec_dict.get("capabilities", []):
        if not isinstance(cap, dict):
            continue
        mcp_conf = cap.get("MCP", {})
        if headers := mcp_conf.get("headers"):
            mcp_conf["headers"] = {
                k: _expand_secret_refs(v, secrets) for k, v in headers.items()
            }
    return spec_dict


def load_agent(spec_path: Path, secrets: AgentSecrets) -> Agent:
    """Load and construct the Pydantic AI agent from agent.yaml.

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

    # Workaround for upstream #5471 — custom OpenAI-compatible endpoints.
    # "openai-compat" sentinel triggers Python-side OpenAIChatModel construction
    # using secrets files, bypassing AgentSpec model resolution.
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

        model_id = raw.get("model_id", "custom")
        model_override = OpenAIChatModel(
            model_id,
            provider=OpenAIProvider(base_url=base_url, api_key=api_key),
        )
        # Replace sentinel so AgentSpec validation doesn't fail on unknown model
        raw = {**raw, "model": model_id}

    spec = AgentSpec.model_validate(raw)
    return Agent.from_spec(spec, model=model_override)
