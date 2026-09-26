"""Load `agent.yaml` into a Pydantic AI agent.

`agent.yaml` is a standard Pydantic AI agent spec (see
https://pydantic.dev/docs/ai/agent-spec/): model, instructions, model settings,
and capabilities. On top of the built-in capabilities (``MCP``, ``Thinking``,
``WebSearch``, ``ToolSearch``, ``PrefixTools``, ...), agents can declare:

- ``A2AAgent`` — delegate work to a remote A2A agent (this is what makes an
  agent an orchestrator); see `agentic.agent.a2a_client`.
- A curated set of Pydantic AI Harness capabilities that suit long-running
  server agents (see `HARNESS_CAPABILITIES`).

``${NAME}`` references anywhere in the capability arguments (e.g. MCP or A2A
``headers``) are expanded from secret files or environment variables.

Models are ordinary Pydantic AI model strings. For self-hosted OpenAI-compatible
servers (vLLM, SGLang, NVIDIA NIM) use ``vllm:<model-name>``: the endpoint comes
from the ``model.base_url`` / ``model.api_key`` secret files when present, or the
``VLLM_BASE_URL`` / ``VLLM_API_KEY`` environment variables otherwise.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from pydantic_ai import Agent, AgentSpec
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import Model, infer_model
from pydantic_ai.providers import Provider, infer_provider, infer_provider_class
from pydantic_ai_harness import (
    ClampOversizedMessages,
    ClearToolResults,
    Planning,
    SlidingWindowCompaction,
    SpendLimits,
    SummarizingCompaction,
    ToolOutputLimits,
    WarnNearLimits,
)
from pydantic_ai_harness.repair_tool_arguments import (
    RepairToolArguments as _HarnessRepairToolArguments,
)

from .a2a_client import A2AAgent
from .config import ConfigError, Secrets, expand_secret_refs_deep, read_yaml

log = logging.getLogger(__name__)


@dataclass
class RepairToolArguments(_HarnessRepairToolArguments[Any]):
    """Repair malformed JSON tool-call arguments before validation.

    Self-hosted models occasionally emit almost-JSON tool arguments (trailing
    commas, unquoted keys, truncated objects); this repairs them instead of
    spending a retry. Wraps the Harness capability, which isn't a dataclass and
    so can't be declared in a spec directly.
    """


#: Harness capabilities that can be declared in `agent.yaml`. Deliberately
#: excludes capabilities that touch the local filesystem or shell, need
#: callables, or require extra packages.
HARNESS_CAPABILITIES: tuple[type[AbstractCapability[Any]], ...] = (
    RepairToolArguments,
    ToolOutputLimits,
    ClampOversizedMessages,
    ClearToolResults,
    SlidingWindowCompaction,
    SummarizingCompaction,
    WarnNearLimits,
    Planning,
    SpendLimits,
)

#: Every custom capability type `agent.yaml` may use (in addition to the
#: Pydantic AI built-ins).
CUSTOM_CAPABILITIES: tuple[type[AbstractCapability[Any]], ...] = (
    A2AAgent,
    *HARNESS_CAPABILITIES,
)

#: Secret files holding the endpoint for OpenAI-compatible model servers.
#: The ``openai_compatible.*`` names are accepted for backward compatibility.
_BASE_URL_SECRETS = ("model.base_url", "openai_compatible.base_url")
_API_KEY_SECRETS = ("model.api_key", "openai_compatible.api_key")

#: Providers whose endpoint comes from the model secrets above. Hosted providers
#: (``openai:``, ``anthropic:``, ...) always use their own endpoints and keys.
_ENDPOINT_PROVIDERS = frozenset({"vllm"})

#: Top-level keys `agent.yaml` accepts (``$schema`` is the alias of json_schema_path).
_SPEC_KEYS = (frozenset(AgentSpec.model_fields) - {"json_schema_path"}) | {"$schema"}


# --------------------------------------------------------------------------------------
# Legacy agent.yaml formats
# --------------------------------------------------------------------------------------


def migrate_legacy_agent_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Translate earlier `agent.yaml` formats, warning for each change.

    - ``model: openai-compat`` + ``model_id: X`` (custom OpenAI-compatible
      endpoint) → ``model: vllm:X``; the endpoint still comes from the
      ``openai_compatible.*`` secrets.
    - ``a2a_servers: [{url, id, headers}]`` (the LangGraph orchestrator) →
      ``A2AAgent`` capabilities.
    """
    raw = dict(raw)

    if raw.get("model") == "openai-compat":
        model_id = raw.pop("model_id", None) or "default"
        log.warning(
            "agent.yaml: 'model: openai-compat' + 'model_id' is deprecated; use "
            "'model: vllm:%s'",
            model_id,
        )
        raw["model"] = f"vllm:{model_id}"
    elif "model_id" in raw:
        raise ConfigError(
            "agent.yaml: 'model_id' is only valid with the legacy 'model: "
            "openai-compat'; put the model name in 'model' (e.g. 'vllm:my-model')"
        )

    if (servers := raw.pop("a2a_servers", None)) is not None:
        log.warning(
            "agent.yaml: 'a2a_servers' is deprecated; declare each downstream agent "
            "as an 'A2AAgent' capability"
        )
        capabilities = list(raw.get("capabilities") or [])
        for server in servers or []:
            args = {"url": server["url"]}
            if server.get("id"):
                args["name"] = server["id"]
            if server.get("headers"):
                args["headers"] = server["headers"]
            capabilities.append({"A2AAgent": args})
        raw["capabilities"] = capabilities

    return raw


# --------------------------------------------------------------------------------------
# Model resolution
# --------------------------------------------------------------------------------------


def _first_secret(secrets: Secrets, keys: Sequence[str]) -> str | None:
    return next((v for k in keys if (v := secrets.get(k)) is not None), None)


def endpoint_provider_factory(
    secrets: Secrets,
) -> Callable[[str], Provider[Any]] | None:
    """Return a provider factory that points OpenAI-compatible providers at the
    endpoint from the model secrets, or None when no endpoint secret exists.
    """
    base_url = _first_secret(secrets, _BASE_URL_SECRETS)
    if base_url is None:
        return None
    api_key = _first_secret(secrets, _API_KEY_SECRETS)

    def factory(provider_name: str) -> Provider[Any]:
        if provider_name not in _ENDPOINT_PROVIDERS:
            return infer_provider(provider_name)
        provider_cls = infer_provider_class(provider_name)
        return provider_cls(base_url=base_url, api_key=api_key or "api-key-not-set")

    return factory


def resolve_model(model: str | None, secrets: Secrets) -> Model | str | None:
    """Resolve the spec's model string, applying the endpoint secrets if present."""
    if model is None:
        return None
    factory = endpoint_provider_factory(secrets)
    if factory is None or model.partition(":")[0] not in _ENDPOINT_PROVIDERS:
        return model
    try:
        return infer_model(model, provider_factory=factory)
    except UserError as exc:
        raise ConfigError(f"agent.yaml: invalid model '{model}': {exc}") from exc


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


def load_agent_spec(path: Path, secrets: Secrets) -> AgentSpec:
    """Read, migrate, secret-expand, and validate `agent.yaml` into an AgentSpec."""
    raw = migrate_legacy_agent_config(read_yaml(path))

    if unknown := sorted(set(raw) - _SPEC_KEYS):
        raise ConfigError(
            f"{path}: unknown key(s) {unknown}; valid keys are "
            f"{sorted(_SPEC_KEYS - {'$schema'})}"
        )
    if "capabilities" in raw:
        raw["capabilities"] = expand_secret_refs_deep(raw["capabilities"], secrets)

    try:
        return AgentSpec.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"{path} is invalid:\n{exc}") from exc


def build_agent(spec: AgentSpec, secrets: Secrets) -> Agent[None, str]:
    """Construct the agent from a validated spec.

    Construction performs no network I/O: MCP servers are connected per run and
    remote A2A agent cards are resolved lazily, so a slow or unavailable
    dependency never prevents the agent from starting or reloading.
    """
    try:
        return Agent.from_spec(
            spec,
            custom_capability_types=CUSTOM_CAPABILITIES,
            model=resolve_model(spec.model, secrets),
        )
    except (UserError, ValueError, TypeError) as exc:
        raise ConfigError(f"agent.yaml could not be loaded: {exc}") from exc


def load_agent(path: Path, secrets: Secrets) -> Agent[None, str]:
    """Load `agent.yaml` and construct the agent (convenience for the CLI)."""
    return build_agent(load_agent_spec(path, secrets), secrets)
