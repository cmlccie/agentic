"""Build the LangGraph supervisor agent from configuration.

Uses LangChain's `create_agent` (the v1.0 supervisor / subagents-as-tools
pattern): each downstream A2A server is a tool the model may call. The compiled
graph is rebuilt on every reload so model, instructions, and the downstream
agent set can change without a restart.
"""

from __future__ import annotations

import logging
import os

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph

from agentic.runtime.config import AgentSecrets

from .a2a_workers import build_a2a_tools
from .config.agent_spec import OrchestratorSpec

log = logging.getLogger(__name__)


def build_model(spec: OrchestratorSpec, secrets: AgentSecrets) -> BaseChatModel:
    """Construct the chat model from the spec, mirroring the simple-agent modes.

    - ``openai-compat`` — custom OpenAI-compatible endpoint via secret files
      ``agent_model_base_url`` + ``agent_model_api_key`` (e.g. LM Studio, vLLM).
    - ``provider:model`` — first-class provider string (e.g.
      ``anthropic:claude-sonnet-4-6``, ``openai:gpt-4o``); API keys are injected
      from secret files into the environment for the provider integration.
    """
    settings = dict(spec.model_settings)

    if spec.model == "openai-compat":
        base_url = secrets.agent_model_base_url
        api_key = secrets.agent_model_api_key
        if not (base_url and api_key):
            raise RuntimeError(
                "model: openai-compat requires secret files "
                "'agent_model_base_url' and 'agent_model_api_key'"
            )
        return ChatOpenAI(
            model=spec.model_id or "custom",
            base_url=base_url,
            api_key=api_key,
            **settings,
        )

    # First-class provider:model — inject keys from secret files so they are
    # current after a reload, then let the integration resolve credentials.
    if key := secrets.anthropic_api_key:
        os.environ["ANTHROPIC_API_KEY"] = key
    if key := secrets.openai_api_key:
        os.environ["OPENAI_API_KEY"] = key

    return init_chat_model(spec.model, **settings)


async def build_graph(
    spec: OrchestratorSpec, secrets: AgentSecrets
) -> CompiledStateGraph:
    """Build the compiled supervisor graph (model + downstream A2A tools)."""
    model = build_model(spec, secrets)
    tools = await build_a2a_tools(spec, secrets)
    log.info(
        "graph: building supervisor with model=%s and %d downstream agent tool(s)",
        spec.model,
        len(tools),
    )
    return create_agent(
        model,
        tools=tools,
        system_prompt=spec.instructions or None,
        checkpointer=InMemorySaver(),
    )


def message_to_text(message) -> str:
    """Extract plain text from a LangChain message (or raw string)."""
    if isinstance(message, str):
        return message
    text_attr = getattr(message, "text", None)
    if text_attr is not None:
        # `.text` is a str-like accessor on LangChain messages; coerce to str
        # (calling it as a method is deprecated).
        return str(text_attr)
    content = getattr(message, "content", "")
    if isinstance(content, list):
        return "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return content or ""


def final_ai_text(state) -> str:
    """Return the text of the final assistant message in a graph result state."""
    if isinstance(state, dict):
        messages = state.get("messages", [])
    else:
        messages = getattr(state, "messages", [])
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            text = message_to_text(message)
            if text.strip():
                return text
    return message_to_text(messages[-1]) if messages else ""
