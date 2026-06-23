"""A2A server interface for the orchestrator (built on the a2a-sdk).

Exposes the orchestrator as a first-class A2A agent supporting the full task
lifecycle: short message responses, long-running tasks with streamed status
updates (SSE via `message/stream`), artifacts, `tasks/get` polling, and push
notifications. An `AgentExecutor` drives the LangGraph supervisor and reads the
current graph from `AppState` at execution time so reloads take effect without
remounting routes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx
from a2a.helpers import new_task_from_user_message, new_text_part
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import (
    BasePushNotificationSender,
    InMemoryPushNotificationConfigStore,
    InMemoryTaskStore,
    TaskUpdater,
)
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentProvider,
    AgentSkill,
    TaskState,
)
from a2a.utils import TransportProtocol
from langchain_core.messages import AIMessage, HumanMessage
from starlette.routing import Route

from agentic.runtime.config import BrokerBackend

from ..graph import final_ai_text, message_to_text

if TYPE_CHECKING:
    from agentic.runtime.config import AgentSecrets, ServerSpec

    from ..lifespan import AppState

log = logging.getLogger(__name__)

_A2A_PREFIX = "/a2a"
_PUSH_TIMEOUT = httpx.Timeout(30.0)


def build_agent_card(server_spec: ServerSpec, agent_url: str) -> AgentCard:
    """Build the orchestrator's A2A agent card from server.yaml."""
    cfg = server_spec.agent_card
    skills = [
        AgentSkill(
            id=s.id,
            name=s.name,
            description=s.description,
            tags=list(s.tags),
            examples=list(s.examples or []),
            input_modes=list(s.input_modes),
            output_modes=list(s.output_modes),
        )
        for s in cfg.skills
    ]
    a2a_url = f"{agent_url.rstrip('/')}{_A2A_PREFIX}"
    kwargs = dict(
        name=cfg.display_name,
        description=cfg.description,
        version=cfg.version,
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(streaming=True, push_notifications=True),
        skills=skills,
        supported_interfaces=[
            AgentInterface(url=a2a_url, protocol_binding=TransportProtocol.JSONRPC)
        ],
    )
    if cfg.icon_url:
        kwargs["icon_url"] = cfg.icon_url
    if cfg.documentation_url:
        kwargs["documentation_url"] = cfg.documentation_url
    if cfg.provider:
        kwargs["provider"] = AgentProvider(
            organization=cfg.provider.organization, url=cfg.provider.url
        )
    return AgentCard(**kwargs)


class OrchestratorAgentExecutor(AgentExecutor):
    """Drives the LangGraph supervisor for incoming A2A tasks.

    Reads the current graph from AppState at execution time (reload-safe).
    Streams the supervisor's progress as `working` status updates and returns
    the final answer as a task artifact.
    """

    def __init__(self, app_state: AppState) -> None:
        self.app_state = app_state

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        # The A2A protocol requires a Task to be enqueued before any status
        # update. For a brand-new request (no existing task) create and enqueue
        # one from the user's message, then drive it via the TaskUpdater.
        task = context.current_task
        if task is None:
            task = new_task_from_user_message(context.message)
            await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.start_work()

        graph = self.app_state.graph
        if graph is None:
            await updater.failed(
                updater.new_agent_message(
                    [new_text_part("orchestrator is reloading, please retry")]
                )
            )
            return

        user_text = context.get_user_input()
        thread_id = task.context_id or task.id

        try:
            final_state = None
            last_status = ""
            async for state in graph.astream(
                {"messages": [HumanMessage(content=user_text)]},
                config={"configurable": {"thread_id": thread_id}},
                stream_mode="values",
            ):
                final_state = state
                messages = state.get("messages", []) if isinstance(state, dict) else []
                if not messages:
                    continue
                latest = messages[-1]
                if isinstance(latest, AIMessage):
                    text = message_to_text(latest).strip()
                    if text and text != last_status:
                        last_status = text
                        await updater.update_status(
                            TaskState.TASK_STATE_WORKING,
                            message=updater.new_agent_message([new_text_part(text)]),
                        )

            answer = final_ai_text(final_state) if final_state else ""
            await updater.add_artifact(
                [new_text_part(answer or "(no output produced)")],
                name="response",
            )
            await updater.complete()
        except Exception as exc:
            log.exception("a2a: orchestration failed")
            await updater.failed(
                updater.new_agent_message(
                    [new_text_part(f"orchestration error: {exc}")]
                )
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await updater.cancel()


@dataclass
class A2AInterface:
    """Mountable A2A routes plus resources the lifespan must close on shutdown."""

    routes: list[Route]
    push_httpx_client: httpx.AsyncClient


def build_a2a_interface(
    app_state: AppState,
    server_spec: ServerSpec,
    secrets: AgentSecrets,
    agent_url: str,
) -> A2AInterface:
    """Build the A2A request handler and Starlette routes mounted under /a2a."""
    card = build_agent_card(server_spec, agent_url)

    if server_spec.broker.backend == BrokerBackend.REDIS:
        # a2a-sdk persistent task storage uses a SQL DatabaseTaskStore; a Redis
        # store is not provided. Fall back to in-memory and warn so single-replica
        # deployments still work and multi-replica setups know to add persistence.
        log.warning(
            "a2a: broker.backend=redis is not yet supported for the orchestrator "
            "A2A task store; falling back to in-memory storage"
        )
    task_store = InMemoryTaskStore()

    push_config_store = InMemoryPushNotificationConfigStore()
    push_httpx_client = httpx.AsyncClient(timeout=_PUSH_TIMEOUT)
    push_sender = BasePushNotificationSender(push_httpx_client, push_config_store)

    handler = DefaultRequestHandler(
        agent_executor=OrchestratorAgentExecutor(app_state),
        task_store=task_store,
        agent_card=card,
        push_config_store=push_config_store,
        push_sender=push_sender,
    )

    routes: list[Route] = []
    routes.extend(create_jsonrpc_routes(handler, _A2A_PREFIX))
    routes.extend(
        create_agent_card_routes(
            card, card_url=f"{_A2A_PREFIX}/.well-known/agent-card.json"
        )
    )
    # A2A spec standard alias path.
    routes.extend(
        create_agent_card_routes(card, card_url=f"{_A2A_PREFIX}/.well-known/agent.json")
    )

    return A2AInterface(routes=routes, push_httpx_client=push_httpx_client)
