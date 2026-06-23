"""Downstream A2A agents exposed to the supervisor as LangChain tools.

Implements the subagents-as-tools pattern: each configured A2A server becomes a
single LangChain tool whose name/description/skills are derived from the agent's
fetched AgentCard. When the supervisor calls the tool, it sends the request to
the remote agent and drives the full client-side A2A lifecycle — consuming the
streamed responses (or polling, as negotiated by the a2a-sdk client) until the
task reaches a terminal state — then returns the collected text + artifacts.
"""

from __future__ import annotations

import logging
import re

import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.helpers import get_artifact_text, get_message_text, new_text_message
from a2a.types import AgentCard, Role, SendMessageRequest
from langchain_core.tools import BaseTool, StructuredTool

from agentic.runtime.config import AgentSecrets

from .config.agent_spec import A2AServerConfig, OrchestratorSpec

log = logging.getLogger(__name__)

# Generous timeout: downstream agents may run long tasks. The a2a-sdk client
# streams status updates which keep the connection active.
_CLIENT_TIMEOUT = httpx.Timeout(300.0, connect=10.0)
_CARD_TIMEOUT = httpx.Timeout(10.0)


def _safe_tool_name(raw: str) -> str:
    """Coerce an agent name into a valid, readable tool identifier."""
    name = re.sub(r"[^0-9a-zA-Z_]+", "_", raw).strip("_").lower()
    return name or "agent"


def _tool_description(card: AgentCard) -> str:
    """Build a delegation-tool description from the agent card + its skills."""
    parts = [
        f"Delegate a task to the '{card.name}' agent.",
        card.description.strip() if card.description else "",
    ]
    if card.skills:
        skills = "; ".join(
            f"{s.name}: {s.description}" for s in card.skills if s.description
        )
        if skills:
            parts.append(f"Capabilities — {skills}.")
    parts.append(
        "Provide a complete, self-contained natural-language request; the agent "
        "has no memory of this conversation."
    )
    return " ".join(p for p in parts if p)


async def _fetch_card(
    server: A2AServerConfig, spec: OrchestratorSpec, secrets: AgentSecrets
) -> AgentCard:
    headers = spec.resolved_headers(server, secrets)
    async with httpx.AsyncClient(headers=headers, timeout=_CARD_TIMEOUT) as http:
        resolver = A2ACardResolver(http, base_url=server.url.rstrip("/"))
        return await resolver.get_agent_card()


def _collect_result(messages: list[str], artifacts: list[str]) -> str:
    """Pick the most complete representation of the downstream agent's output."""
    if artifacts:
        return "\n".join(a for a in artifacts if a)
    if messages:
        return messages[-1]
    return ""


def _build_delegate(
    server: A2AServerConfig,
    card: AgentCard,
    spec: OrchestratorSpec,
    secrets: AgentSecrets,
):
    base_url = server.url.rstrip("/")

    async def delegate(request: str) -> str:
        """Send `request` to the downstream agent and return its final output."""
        headers = spec.resolved_headers(server, secrets)
        messages: list[str] = []
        artifacts: list[str] = []
        task_artifacts: list[str] = []

        async with httpx.AsyncClient(headers=headers, timeout=_CLIENT_TIMEOUT) as http:
            config = ClientConfig(
                httpx_client=http,
                streaming=True,
                accepted_output_modes=["text/plain"],
            )
            client = ClientFactory(config).create(card)
            message = new_text_message(request, role=Role.ROLE_USER)
            send = SendMessageRequest(message=message)

            try:
                async for response in client.send_message(send):
                    if response.HasField("artifact_update"):
                        text = get_artifact_text(response.artifact_update.artifact)
                        if text:
                            artifacts.append(text)
                    elif response.HasField("message"):
                        text = get_message_text(response.message)
                        if text:
                            messages.append(text)
                    elif response.HasField("status_update"):
                        status = response.status_update.status
                        if status.HasField("message"):
                            text = get_message_text(status.message)
                            if text:
                                messages.append(text)
                    elif response.HasField("task"):
                        for artifact in response.task.artifacts:
                            text = get_artifact_text(artifact)
                            if text:
                                task_artifacts.append(text)
            finally:
                await client.close()

        result = _collect_result(messages, task_artifacts or artifacts)
        if not result:
            log.warning("a2a_worker: agent at %s returned no text output", base_url)
            return "(the downstream agent returned no textual output)"
        return result

    return delegate


async def build_a2a_tools(
    spec: OrchestratorSpec, secrets: AgentSecrets
) -> list[BaseTool]:
    """Build one delegation tool per reachable downstream A2A server.

    Unreachable servers are logged and skipped (degraded mode) so the
    orchestrator still starts with whatever agents are available.
    """
    tools: list[BaseTool] = []
    seen_names: set[str] = set()

    for server in spec.a2a_servers:
        try:
            card = await _fetch_card(server, spec, secrets)
        except Exception as exc:
            log.warning(
                "a2a_worker: failed to fetch agent card from %s (%s) — skipping",
                server.url,
                exc,
            )
            continue

        name = _safe_tool_name(server.id or card.name)
        # Disambiguate collisions so create_agent receives unique tool names.
        unique = name
        suffix = 2
        while unique in seen_names:
            unique = f"{name}_{suffix}"
            suffix += 1
        seen_names.add(unique)

        tool = StructuredTool.from_function(
            coroutine=_build_delegate(server, card, spec, secrets),
            name=unique,
            description=_tool_description(card),
        )
        tools.append(tool)
        log.info("a2a_worker: registered downstream agent '%s' as tool", unique)

    return tools
