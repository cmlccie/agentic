"""The `A2AAgent` capability: delegate work to a remote A2A agent.

Declaring one or more ``A2AAgent`` capabilities in `agent.yaml` turns an agent
into an orchestrator — no separate framework or code path::

    capabilities:
      - A2AAgent:
          url: http://weather-agent/a2a
          headers:
            Authorization: Bearer ${WEATHER_AGENT_TOKEN}

Each capability contributes one tool to the agent. The tool's name and
description come from the remote agent's card (fetched lazily and cached;
override them with ``name`` / ``description``). When the model calls the tool:

1. The request is sent to the remote agent with streaming enabled.
2. The remote agent's activity (thinking, tool calls, its own sub-agents) is
   relayed into this agent's event stream as `ActivityEvent`s, attributed to the
   remote agent, so callers see delegated work as it happens.
3. The final answer is returned to the model. Failures (unreachable agent,
   failed/rejected/canceled task, timeouts) are returned to the model as text so
   it can route around them — a flaky downstream agent never aborts the run.

Follow-ups within the same conversation reuse the remote agent's
``contextId``, and a remote task that asks for more input (``input-required``)
is continued on the next call.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.client.errors import A2AClientError
from a2a.helpers import new_text_message
from a2a.types import (
    AgentCard,
    CancelTaskRequest,
    Role,
    SendMessageRequest,
    TaskState,
)
from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.toolsets.abstract import ToolsetTool
from pydantic_core import SchemaValidator, core_schema

from .a2a_wire import (
    INTERRUPTED_STATES,
    TERMINAL_STATES,
    activity_from_message,
    parts_text,
    state_name,
)
from .streaming import Activity, ActivityEvent, prefixed

log = logging.getLogger(__name__)

_CARD_TIMEOUT = 10.0
_CARD_RETRY_SECONDS = 30.0
_CANCEL_TIMEOUT = 5.0
_MAX_CONVERSATIONS = 1024

_ARGS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "request": {
            "type": "string",
            "description": (
                "A complete, self-contained natural-language request for the agent."
            ),
        }
    },
    "required": ["request"],
    "additionalProperties": False,
}
_ARGS_VALIDATOR = SchemaValidator(
    core_schema.typed_dict_schema(
        {"request": core_schema.typed_dict_field(core_schema.str_schema())},
        extra_behavior="forbid",
    )
)


def tool_name_for(raw: str) -> str:
    """Coerce an agent name into a valid, readable tool name."""
    name = re.sub(r"[^0-9a-zA-Z_]+", "_", raw).strip("_").lower()
    return name[:64] or "agent"


def tool_description_for(card: AgentCard) -> str:
    """Describe the delegation tool from the remote agent's card and skills."""
    parts = [f"Delegate a task to the '{card.name}' agent."]
    if card.description:
        parts.append(card.description.strip())
    if skills := "; ".join(
        f"{s.name}: {s.description}" for s in card.skills if s.description
    ):
        parts.append(f"Skills: {skills}.")
    parts.append(
        "Send a complete, self-contained request; follow-up calls in the same "
        "conversation continue the agent's context."
    )
    return " ".join(parts)


def normalize_card(card: AgentCard) -> AgentCard:
    """Repair interface bindings published as ``TransportProtocol.JSONRPC``.

    a2a-sdk servers on the pure-Python protobuf runtime (e.g. Alpine images)
    that pass the SDK's enum members to the card publish ``str(member)`` instead
    of ``JSONRPC``; the client then finds no compatible transport.
    """
    for interface in card.supported_interfaces:
        interface.protocol_binding = interface.protocol_binding.removeprefix(
            "TransportProtocol."
        )
    return card


@dataclass
class _Outcome:
    """What a delegation produced, accumulated from the response stream."""

    state: int = TaskState.TASK_STATE_UNSPECIFIED
    task_id: str | None = None
    context_id: str | None = None
    reply: str = ""
    status_text: str = ""
    artifacts: dict[str, list[str]] = field(default_factory=dict)

    def artifact_text(self) -> str:
        return "\n".join("".join(chunks) for chunks in self.artifacts.values()).strip()


class A2AAgentToolset(AbstractToolset[Any]):
    """Toolset exposing one remote A2A agent as a single delegation tool."""

    def __init__(
        self,
        url: str,
        *,
        name: str | None,
        description: str | None,
        headers: dict[str, str],
        timeout: float,
    ) -> None:
        self.url = url.rstrip("/")
        self._name = name
        self._description = description
        self._headers = headers
        self._timeout = timeout
        self._card: AgentCard | None = None
        self._card_retry_at = 0.0
        self._card_lock = asyncio.Lock()
        # conversation id → (remote context id, remote task id awaiting input)
        self._conversations: OrderedDict[str, tuple[str | None, str | None]] = (
            OrderedDict()
        )

    @property
    def id(self) -> str:
        return f"a2a:{self.url}"

    @property
    def label(self) -> str:
        return f"A2A agent at {self.url}"

    # ── card resolution ───────────────────────────────────────────────────────────

    async def resolve_card(self) -> AgentCard | None:
        """Fetch (and cache) the remote agent card; None while it is unreachable.

        Failures are retried at most every 30 seconds, so an unavailable agent
        simply has no tool until it comes back — the rest of the agent keeps
        working.
        """
        if self._card is not None:
            return self._card
        async with self._card_lock:
            if self._card is not None or time.monotonic() < self._card_retry_at:
                return self._card
            try:
                async with httpx.AsyncClient(
                    headers=self._headers, timeout=_CARD_TIMEOUT
                ) as http:
                    self._card = normalize_card(
                        await A2ACardResolver(http, base_url=self.url).get_agent_card()
                    )
                log.info(
                    "a2a: resolved agent card for %s (%s)", self.url, self._card.name
                )
            except Exception as exc:  # any failure means "not available yet"
                self._card_retry_at = time.monotonic() + _CARD_RETRY_SECONDS
                log.warning(
                    "a2a: agent card for %s unavailable (%s); retrying in %.0fs",
                    self.url,
                    exc,
                    _CARD_RETRY_SECONDS,
                )
            return self._card

    def tool_name(self, card: AgentCard) -> str:
        return tool_name_for(self._name or card.name)

    # ── toolset protocol ──────────────────────────────────────────────────────────

    async def get_tools(self, ctx: RunContext[Any]) -> dict[str, ToolsetTool[Any]]:
        card = await self.resolve_card()
        if card is None:
            return {}
        name = self.tool_name(card)
        definition = ToolDefinition(
            name=name,
            description=self._description or tool_description_for(card),
            parameters_json_schema=_ARGS_SCHEMA,
            metadata={"a2a_agent": self.url},
        )
        return {
            name: ToolsetTool(
                toolset=self,
                tool_def=definition,
                max_retries=1,
                args_validator=_ARGS_VALIDATOR,
            )
        }

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[Any],
        tool: ToolsetTool[Any],
    ) -> str:
        return await self.delegate(name, tool_args["request"], ctx)

    # ── delegation ────────────────────────────────────────────────────────────────

    async def _emit(self, ctx: RunContext[Any], item: Activity) -> None:
        try:
            await ctx.emit(ActivityEvent(activity=item))
        except UserError:  # no event stream for this run; activity is optional
            pass

    def _conversation(self, key: str) -> tuple[str | None, str | None]:
        if key in self._conversations:
            self._conversations.move_to_end(key)
            return self._conversations[key]
        return None, None

    def _remember(self, key: str, context_id: str | None, task_id: str | None) -> None:
        self._conversations[key] = (context_id, task_id)
        self._conversations.move_to_end(key)
        while len(self._conversations) > _MAX_CONVERSATIONS:
            self._conversations.popitem(last=False)

    async def delegate(self, name: str, request: str, ctx: RunContext[Any]) -> str:
        """Send ``request`` to the remote agent and return its answer as text."""
        card = await self.resolve_card()
        if card is None:
            return f"The '{name}' agent is currently unavailable."

        conversation = ctx.conversation_id or ctx.run_id or ""
        context_id, pending_task_id = self._conversation(conversation)
        outcome = _Outcome(context_id=context_id)
        message = new_text_message(
            request, context_id=context_id, task_id=pending_task_id, role=Role.ROLE_USER
        )

        async with httpx.AsyncClient(
            headers=self._headers, timeout=httpx.Timeout(self._timeout, connect=10.0)
        ) as http:
            client = ClientFactory(
                ClientConfig(httpx_client=http, streaming=True)
            ).create(card)
            try:
                async for response in client.send_message(
                    SendMessageRequest(message=message)
                ):
                    await self._absorb(response, outcome, name, ctx)
            except asyncio.CancelledError:
                await self._cancel_remote(client, outcome.task_id)
                raise
            except (A2AClientError, httpx.HTTPError) as exc:
                log.warning("a2a: delegation to %s failed: %s", self.url, exc)
                await self._emit(ctx, Activity("error", f"{name}: {exc}"))
                return f"The '{name}' agent could not be reached: {exc}"
            finally:
                await client.close()

        waiting = outcome.state in INTERRUPTED_STATES
        self._remember(
            conversation, outcome.context_id, outcome.task_id if waiting else None
        )
        return self._result_text(name, outcome)

    async def _absorb(
        self, response: Any, outcome: _Outcome, name: str, ctx: RunContext[Any]
    ) -> None:
        """Fold one StreamResponse into ``outcome`` and relay any activity."""
        match response.WhichOneof("payload"):
            case "message":
                outcome.reply = parts_text(response.message.parts)
                outcome.context_id = response.message.context_id or outcome.context_id
                outcome.state = TaskState.TASK_STATE_COMPLETED
            case "task":
                task = response.task
                outcome.task_id, outcome.context_id = task.id, task.context_id
                outcome.state = task.status.state
                if task.status.HasField("message"):
                    outcome.status_text = parts_text(task.status.message.parts)
                for artifact in task.artifacts:
                    outcome.artifacts[artifact.artifact_id] = [
                        parts_text(artifact.parts)
                    ]
            case "status_update":
                update = response.status_update
                outcome.task_id = update.task_id or outcome.task_id
                outcome.context_id = update.context_id or outcome.context_id
                outcome.state = update.status.state
                if update.status.HasField("message"):
                    msg = update.status.message
                    if (item := activity_from_message(msg)) is not None:
                        await self._emit(ctx, prefixed(item, name))
                    elif text := parts_text(msg.parts):
                        outcome.status_text = text
                        if update.status.state == TaskState.TASK_STATE_WORKING:
                            await self._emit(ctx, Activity("status", text, (name,)))
            case "artifact_update":
                update = response.artifact_update
                chunks = outcome.artifacts.setdefault(update.artifact.artifact_id, [])
                if not update.append:
                    chunks.clear()
                chunks.append(parts_text(update.artifact.parts))

    async def _cancel_remote(self, client: Any, task_id: str | None) -> None:
        """Best-effort cancel of the remote task when this run is cancelled."""
        if not task_id:
            return
        try:
            await asyncio.wait_for(
                asyncio.shield(client.cancel_task(CancelTaskRequest(id=task_id))),
                timeout=_CANCEL_TIMEOUT,
            )
        except Exception as exc:  # the remote task may already be finished
            log.debug("a2a: could not cancel remote task %s: %s", task_id, exc)

    @staticmethod
    def _result_text(name: str, outcome: _Outcome) -> str:
        """Turn the final outcome into text for the model."""
        state = outcome.state
        detail = outcome.status_text or outcome.reply
        if state == TaskState.TASK_STATE_COMPLETED:
            return (
                outcome.artifact_text()
                or outcome.reply
                or outcome.status_text
                or f"(the '{name}' agent returned no text)"
            )
        if state == TaskState.TASK_STATE_INPUT_REQUIRED:
            return (
                f"The '{name}' agent needs more information: {detail or '(no details)'}"
                f" — call it again with the answer to continue."
            )
        if state == TaskState.TASK_STATE_AUTH_REQUIRED:
            return (
                f"The '{name}' agent requires authorization: {detail or '(no details)'}"
            )
        if state in TERMINAL_STATES:
            return (
                f"The '{name}' agent could not complete the request "
                f"({state_name(state)}): {detail or '(no details)'}"
            )
        # The stream ended without a terminal state (e.g. the server closed it).
        partial = outcome.artifact_text() or detail
        return f"The '{name}' agent stopped before finishing ({state_name(state)})" + (
            f"; partial result: {partial}" if partial else "."
        )


@dataclass(kw_only=True)
class A2AAgent(AbstractCapability[Any]):
    """Delegate to a remote A2A agent through a tool (see module docstring).

    Attributes:
        url: Base URL of the remote agent's A2A endpoint; its card is fetched
            from ``<url>/.well-known/agent-card.json``.
        name: Tool name override (defaults to the card's name).
        description: Tool description override (defaults to one built from the
            card's description and skills).
        headers: Extra HTTP headers, e.g. ``Authorization``; values may use
            ``${SECRET}`` references in `agent.yaml`.
        timeout: Seconds to wait between streamed events from the remote agent.
    """

    url: str
    name: str | None = None
    description: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    timeout: float = 300.0

    def __post_init__(self) -> None:
        self._toolset = A2AAgentToolset(
            self.url,
            name=self.name,
            description=self.description,
            headers=dict(self.headers),
            timeout=self.timeout,
        )

    def get_toolset(self) -> AbstractToolset[Any]:
        return self._toolset
