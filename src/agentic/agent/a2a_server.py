"""A spec-compliant A2A server for the agent, built on the official a2a-sdk.

Endpoints (mounted by `agentic.agent.app`):

- ``POST /a2a`` — JSON-RPC 2.0 for A2A 1.0 (``SendMessage``,
  ``SendStreamingMessage``, ``GetTask``, ``ListTasks``, ``CancelTask``,
  ``SubscribeToTask``, push-notification config) and the A2A 0.3 method names
  (``message/send``, ``message/stream``, ``tasks/get``, ...) for older clients.
- ``GET /.well-known/agent-card.json`` and ``GET /a2a/.well-known/agent-card.json``
  — the agent card (plus the legacy ``agent.json`` paths).

**Messages vs tasks.** A2A lets an agent answer with a direct ``Message`` (a
quick reply, nothing to track) or a ``Task`` (tracked work with status updates,
artifacts, cancel, and polling). With ``a2a.response_mode: auto`` (default) the
executor decides per request:

- The agent starts working immediately, but the reply shape is held back.
- If the run finishes quickly without using tools, the answer is sent as a
  ``Message``.
- As soon as the agent calls a tool — or the run takes longer than
  ``promote_after_seconds`` — the exchange becomes a ``Task``: a ``submitted``
  task, then ``working`` status updates carrying the agent's activity, then the
  answer as an artifact and ``completed``.

Requests that continue an existing task, or ask to ``return_immediately``,
always get a task. ``response_mode: message`` / ``task`` force one shape.

**Activity.** Working status updates carry the `ACTIVITY_EXTENSION` metadata
(declared on the agent card) so A2A clients — including another agent's
``A2AAgent`` capability — can tell activity from other status messages.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import time
import weakref
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import httpx
from a2a.helpers import new_task_from_user_message, new_text_message, new_text_part
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import (
    BasePushNotificationSender,
    DatabasePushNotificationConfigStore,
    DatabaseTaskStore,
    InMemoryPushNotificationConfigStore,
    TaskStore,
    TaskUpdater,
)
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentExtension,
    AgentInterface,
    AgentProvider,
    AgentSkill,
    HTTPAuthSecurityScheme,
    Message,
    Role,
    SecurityRequirement,
    SecurityScheme,
    StringList,
    TaskState,
)
from a2a.utils import TransportProtocol
from a2a.utils.constants import PROTOCOL_VERSION_0_3, PROTOCOL_VERSION_1_0
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from starlette.routing import BaseRoute

from .a2a_wire import activity_metadata, parts_text
from .config import (
    ConfigError,
    PushNotificationsConfig,
    ResponseMode,
    Secrets,
    ServerSpec,
    StoreBackend,
)
from .runtime import Snapshot
from .stores import (
    BoundedMemoryTaskStore,
    HistoryStore,
    MemoryHistoryStore,
    SqlHistoryStore,
)
from .streaming import (
    ACTIVITY_EXTENSION,
    HEARTBEAT,
    Activity,
    Answer,
    activities,
    render,
    visible,
    with_heartbeat,
)

log = logging.getLogger(__name__)

A2A_PATH = "/a2a"
CARD_PATHS = (
    "/.well-known/agent-card.json",
    f"{A2A_PATH}/.well-known/agent-card.json",
    "/.well-known/agent.json",
    f"{A2A_PATH}/.well-known/agent.json",
)
BEARER_SCHEME = "bearer"

#: Activities that mean "the agent is doing work" and promote a reply to a task.
_WORK_KINDS = frozenset({"note", "tool_call"})
#: Thinking text is batched into one status update per burst of this size.
_THINKING_FLUSH_CHARS = 2000
#: How often the executor re-checks elapsed time and flushes thinking.
_TICK_SECONDS = 1.0
_PUSH_TIMEOUT = httpx.Timeout(30.0)
#: Bookkeeping bounds for releasing Message replies after a client disconnect.
_MAX_REPLIED = 1024
_RELEASE_WAIT_SECONDS = 3600.0


# --------------------------------------------------------------------------------------
# Agent card
# --------------------------------------------------------------------------------------


def build_agent_card(server: ServerSpec, public_url: str, auth: bool) -> AgentCard:
    """Build the A2A agent card from `server.yaml`.

    Args:
        server: The server settings.
        public_url: The externally reachable base URL of this service.
        auth: Whether bearer-token authentication is enforced.
    """
    card = server.agent_card
    url = f"{public_url.rstrip('/')}{A2A_PATH}"
    # Pass plain strings (not the SDK's str-enum members) to protobuf fields: the
    # pure-Python protobuf runtime (used on Alpine/musl) stores str(member), e.g.
    # "TransportProtocol.JSONRPC", which clients then can't match.
    kwargs: dict[str, Any] = dict(
        name=card.display_name,
        description=card.description,
        version=card.version,
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(
            streaming=True,
            push_notifications=server.a2a.push_notifications.enabled,
            extensions=[
                AgentExtension(
                    uri=ACTIVITY_EXTENSION,
                    description=(
                        "Working status-update messages whose metadata contains "
                        "this URI report agent activity (thinking, tool calls and "
                        "results, delegated work)."
                    ),
                    required=False,
                )
            ],
        ),
        skills=[
            AgentSkill(
                id=s.id,
                name=s.name,
                description=s.description,
                tags=list(s.tags),
                examples=list(s.examples),
                input_modes=list(s.input_modes),
                output_modes=list(s.output_modes),
            )
            for s in card.skills
        ],
        supported_interfaces=[
            AgentInterface(
                url=url,
                protocol_binding=TransportProtocol.JSONRPC.value,
                protocol_version=PROTOCOL_VERSION_1_0,
            ),
            AgentInterface(
                url=url,
                protocol_binding=TransportProtocol.JSONRPC.value,
                protocol_version=PROTOCOL_VERSION_0_3,
            ),
        ],
    )
    if card.icon_url:
        kwargs["icon_url"] = card.icon_url
    if card.documentation_url:
        kwargs["documentation_url"] = card.documentation_url
    if card.provider:
        kwargs["provider"] = AgentProvider(
            organization=card.provider.organization, url=card.provider.url
        )
    if auth:
        kwargs["security_schemes"] = {
            BEARER_SCHEME: SecurityScheme(
                http_auth_security_scheme=HTTPAuthSecurityScheme(scheme="Bearer")
            )
        }
        kwargs["security_requirements"] = [
            SecurityRequirement(schemes={BEARER_SCHEME: StringList(list=[])})
        ]
    return AgentCard(**kwargs)


# --------------------------------------------------------------------------------------
# Executor
# --------------------------------------------------------------------------------------


class _Exchange:
    """The reply to one A2A request: a Message until promoted to a Task."""

    def __init__(
        self, context: RequestContext, queue: EventQueue, snapshot: Snapshot
    ) -> None:
        self.context = context
        self.queue = queue
        self.streaming = snapshot.server.streaming
        self.updater: TaskUpdater | None = None
        self.buffered: list[Activity] = []
        self.thinking: list[str] = []
        self.thinking_source: tuple[str, ...] = ()
        self.last_flush = 0.0

    @property
    def promoted(self) -> bool:
        return self.updater is not None

    async def promote(self) -> None:
        """Become a task: enqueue it (if new), start work, flush buffered activity."""
        if self.updater is not None:
            return
        task = self.context.current_task
        if task is None:
            task = new_task_from_user_message(self.context.message)
            await self.queue.enqueue_event(task)
        self.updater = TaskUpdater(self.queue, task.id, task.context_id)
        await self.updater.start_work()
        buffered, self.buffered = self.buffered, []
        for item in buffered:
            await self.activity(item)

    async def activity(self, item: Activity) -> None:
        """Report one activity (buffered until promoted)."""
        if not visible(item, self.streaming):
            return
        if self.updater is None:
            self.buffered.append(item)
            return
        if item.kind == "thinking":
            if self.thinking and item.source != self.thinking_source:
                await self.flush_thinking()
            self.thinking_source = item.source
            self.thinking.append(item.text)
            if sum(map(len, self.thinking)) >= _THINKING_FLUSH_CHARS:
                await self.flush_thinking()
            return
        await self.flush_thinking()
        await self._status(render(item), item)

    async def flush_thinking(self, min_interval: float = 0.0) -> None:
        """Send accumulated thinking as one status update.

        Args:
            min_interval: Skip the flush if the previous one was more recent
                than this (periodic flushes stay batched).
        """
        if self.updater is None or not self.thinking:
            return
        now = time.monotonic()
        if now - self.last_flush < min_interval:
            return
        self.last_flush = now
        text, self.thinking = "".join(self.thinking), []
        item = Activity("thinking", text, self.thinking_source)
        await self._status(render(item) if item.source else text, item)

    async def _status(self, text: str, item: Activity) -> None:
        assert self.updater is not None
        await self.updater.update_status(
            TaskState.TASK_STATE_WORKING,
            message=self.updater.new_agent_message(
                [new_text_part(text)], metadata=activity_metadata(item)
            ),
        )

    async def finish(self, answer: Answer) -> None:
        """Deliver the answer: an artifact + completed, or a direct Message."""
        if self.updater is None:
            await self.queue.enqueue_event(
                new_text_message(
                    answer.text,
                    context_id=self.context.context_id,
                    role=Role.ROLE_AGENT,
                )
            )
            return
        await self.flush_thinking()
        await self.updater.add_artifact(
            [new_text_part(answer.text)], name="response", last_chunk=True
        )
        await self.updater.complete()

    async def fail(self, reason: str) -> None:
        """Report a failure as a failed task (a Message can't carry failure)."""
        await self.promote()
        assert self.updater is not None
        await self.flush_thinking()
        await self.updater.failed(
            self.updater.new_agent_message([new_text_part(reason)])
        )


class AgentRequestExecutor(AgentExecutor):
    """Runs the current agent for each A2A request (see module docstring)."""

    def __init__(self, current: Callable[[], Snapshot], history: HistoryStore) -> None:
        self._current = current
        self._history = history
        #: Called with the request's task id after a direct Message reply.
        self.on_message_reply: Callable[[str], None] = lambda task_id: None
        self._locks: weakref.WeakValueDictionary[str, asyncio.Lock] = (
            weakref.WeakValueDictionary()
        )

    def _lock(self, context_id: str) -> asyncio.Lock:
        lock = self._locks.get(context_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[context_id] = lock
        return lock

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        snapshot = self._current()
        a2a = snapshot.server.a2a
        exchange = _Exchange(context, event_queue, snapshot)
        context_id = context.context_id or ""
        prompt = parts_text(context.message.parts) if context.message else ""
        return_immediately = bool(
            context.configuration and context.configuration.return_immediately
        )

        if (
            a2a.response_mode == ResponseMode.TASK
            or context.current_task is not None
            or return_immediately
        ):
            await exchange.promote()
        may_promote = a2a.response_mode == ResponseMode.AUTO

        async with self._lock(context_id):
            history = await self._history.load(context_id)
            started = time.monotonic()
            answer: Answer | None = None
            try:
                async with snapshot.agent.run_stream_events(
                    prompt, message_history=history or None, conversation_id=context_id
                ) as events:
                    tick = _TICK_SECONDS
                    if may_promote and not exchange.promoted:
                        tick = min(tick, max(a2a.promote_after_seconds, 0.05))
                    stream = with_heartbeat(
                        activities(events, snapshot.server.streaming), tick
                    )
                    async for item in stream:
                        if isinstance(item, Answer):
                            answer = item
                            continue
                        if may_promote and not exchange.promoted:
                            slow = (
                                time.monotonic() - started > a2a.promote_after_seconds
                            )
                            working = item is not HEARTBEAT and item.kind in _WORK_KINDS
                            if slow or working:
                                await exchange.promote()
                        if item is HEARTBEAT:
                            await exchange.flush_thinking(min_interval=_TICK_SECONDS)
                        else:
                            await exchange.activity(item)
            except Exception as exc:
                log.exception("a2a: agent run failed (context %s)", context_id)
                await exchange.fail(
                    f"The agent failed ({type(exc).__name__}); see the agent's logs."
                )
                return

            if answer is None:  # defensive: a run always ends with a result
                await exchange.fail("The agent finished without producing an answer.")
                return
            await self._history.save(context_id, answer.messages)
        await exchange.finish(answer)
        if not exchange.promoted and context.task_id:
            self.on_message_reply(context.task_id)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Mark the task canceled; the SDK then cancels the running `execute`."""
        if context.task_id:
            await TaskUpdater(
                event_queue, context.task_id, context.context_id or ""
            ).cancel()


# --------------------------------------------------------------------------------------
# Request handler
# --------------------------------------------------------------------------------------


class AgentRequestHandler(DefaultRequestHandler):
    """The a2a-sdk request handler with two fixes for long-running servers.

    - The agent card is read from the current snapshot, so reloads that change
      the card (e.g. enabling push notifications) take effect immediately.
    - a2a-sdk 1.1.x never releases the internal task entry it creates for a
      request answered with a direct Message, which leaks memory on every quick
      reply. Those entries are released here once the Message is delivered.
      (tests/agent/test_a2a_server.py guards this workaround.)
    """

    def __init__(self, *, card: Callable[[], AgentCard], **kwargs: Any) -> None:
        self._card = card
        super().__init__(agent_card=card(), **kwargs)
        # Message replies whose streaming client left before seeing any event:
        # released once the executor reports the reply (bounded bookkeeping).
        self._awaiting_reply: dict[str, asyncio.Event] = {}
        self._replied: OrderedDict[str, None] = OrderedDict()
        executor = kwargs["agent_executor"]
        if isinstance(executor, AgentRequestExecutor):
            executor.on_message_reply = self._message_replied

    @property
    def _agent_card(self) -> AgentCard:  # read by the base class's capability checks
        return self._card()

    @_agent_card.setter
    def _agent_card(self, _: AgentCard) -> None:
        pass  # the base constructor assigns it; the current card always wins

    async def _release(self, task_id: str) -> None:
        registry = self._active_task_registry
        active = await registry.get(task_id)
        if active is not None:
            await active.aclose()
            await registry._remove_task(task_id)

    async def on_message_send(self, params: Any, context: Any) -> Any:
        result = await super().on_message_send(params, context)
        if isinstance(result, Message):
            await self._release(params.message.task_id)
        return result

    async def on_message_send_stream(self, params: Any, context: Any) -> Any:
        last = None
        try:
            async for event in super().on_message_send_stream(params, context):
                last = event
                yield event
        finally:
            task_id = params.message.task_id
            if isinstance(last, Message):
                await self._release(task_id)
            elif last is None and task_id:
                # The client left before the reply shape was decided; if the
                # executor ends up replying with a Message, release it then.
                task = asyncio.create_task(self._release_after_reply(task_id))
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)

    def _message_replied(self, task_id: str) -> None:
        if (event := self._awaiting_reply.get(task_id)) is not None:
            event.set()
            return
        self._replied[task_id] = None
        while len(self._replied) > _MAX_REPLIED:
            self._replied.popitem(last=False)

    async def _release_after_reply(self, task_id: str) -> None:
        if self._replied.pop(task_id, "missing") is None:
            await self._release(task_id)
            return
        event = self._awaiting_reply[task_id] = asyncio.Event()
        try:
            await asyncio.wait_for(event.wait(), _RELEASE_WAIT_SECONDS)
        except TimeoutError:
            return  # became a task (cleaned up by the SDK) or is still running
        finally:
            self._awaiting_reply.pop(task_id, None)
        await asyncio.sleep(0)  # let the executor return first
        await self._release(task_id)


# --------------------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------------------


def _is_internal_host(host: str) -> bool:
    """Whether ``host`` is ``localhost`` or a loopback/private/link-local IP literal."""
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        address = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        return False  # a hostname; restrict those with allowed_hosts
    return not address.is_global


def push_url_validator(config: PushNotificationsConfig) -> Callable[[str], Any]:
    """Build the push-notification URL policy.

    Push notifications make the server POST to client-supplied URLs, so:

    - every URL is refused while ``push_notifications.enabled`` is false (the
      a2a-sdk accepts inline push configs on SendMessage regardless of the
      card's capabilities);
    - only ``http``/``https`` URLs are accepted;
    - with ``allowed_hosts``, the host must match an entry exactly, or be a
      subdomain of an entry that starts with a dot (``.example.com``);
    - without ``allowed_hosts``, loopback, private, and link-local IP addresses
      and ``localhost`` are refused. Hostnames are not resolved, so use
      ``allowed_hosts`` to restrict destinations fully.
    """
    allowed = [h.lower() for h in config.allowed_hosts]

    async def validate(url: str) -> bool:
        if not config.enabled:
            return False
        parts = urlsplit(url)
        host = (parts.hostname or "").lower()
        if parts.scheme not in ("http", "https") or not host:
            return False
        if not allowed:
            return not _is_internal_host(host)
        return any(
            host == h or (h.startswith(".") and host.endswith(h)) for h in allowed
        )

    return validate


@dataclass
class A2AServer:
    """The A2A routes plus resources to close on shutdown."""

    routes: list[BaseRoute]
    handler: AgentRequestHandler
    engine: AsyncEngine | None
    push_client: httpx.AsyncClient

    async def aclose(self) -> None:
        await self.handler.aclose()
        await self.push_client.aclose()
        if self.engine is not None:
            await self.engine.dispose()


def build_a2a_server(
    current: Callable[[], Snapshot],
    secrets: Secrets,
    public_url: str,
    auth: Callable[[], bool],
) -> A2AServer:
    """Create the process-lifetime A2A handler, stores, and routes.

    Storage is chosen from the ``server.yaml`` loaded at startup; changing
    ``a2a.store`` requires a restart (everything else reloads live).
    """
    store = current().server.a2a.store
    engine: AsyncEngine | None = None
    task_store: TaskStore
    history: HistoryStore
    if store.backend == StoreBackend.SQL:
        dsn = secrets.get(store.database_url_secret)
        if not dsn:
            raise ConfigError(
                f"a2a.store.backend: sql needs the database URL in the secret file "
                f"'{secrets.directory / store.database_url_secret}' (e.g. "
                "postgresql+asyncpg://user:pass@host:5432/db or "
                "sqlite+aiosqlite:////data/agent.db)"
            )
        engine = create_async_engine(dsn)
        task_store = DatabaseTaskStore(engine, create_table=True)
        history = SqlHistoryStore(engine, store.max_history_messages)
        push_store: Any = DatabasePushNotificationConfigStore(engine, create_table=True)
        log.info("a2a: tasks and conversation history are stored in SQL")
    else:
        task_store = BoundedMemoryTaskStore(store.max_tasks)
        history = MemoryHistoryStore(store.max_contexts, store.max_history_messages)
        push_store = InMemoryPushNotificationConfigStore()
        log.info("a2a: tasks and conversation history are kept in memory")

    def card() -> AgentCard:
        return build_agent_card(current().server, public_url, auth())

    push_client = httpx.AsyncClient(timeout=_PUSH_TIMEOUT)
    handler = AgentRequestHandler(
        card=card,
        agent_executor=AgentRequestExecutor(current, history),
        task_store=task_store,
        push_config_store=push_store,
        push_sender=BasePushNotificationSender(push_client, push_store),
        push_url_validator=lambda url: push_url_validator(
            current().server.a2a.push_notifications
        )(url),
    )

    async def current_card(_: AgentCard) -> AgentCard:
        return card()

    routes: list[BaseRoute] = [
        *create_jsonrpc_routes(handler, A2A_PATH, enable_v0_3_compat=True)
    ]
    for path in CARD_PATHS:
        routes.extend(
            create_agent_card_routes(card(), card_modifier=current_card, card_url=path)
        )
    return A2AServer(
        routes=routes, handler=handler, engine=engine, push_client=push_client
    )
