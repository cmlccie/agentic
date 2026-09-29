"""Persistence: A2A tasks, conversation history, and delegation state.

Three things need storing:

- **Tasks** (A2A protocol state) — handled by the a2a-sdk `TaskStore`
  implementations. In memory by default (bounded here, since the SDK's store
  grows without limit), or in SQL via `DatabaseTaskStore`.
- **Conversation history** (the Pydantic AI message history for each A2A
  ``contextId``) — needed because quick exchanges are answered with a direct
  Message, which the SDK never persists. `MemoryHistoryStore` (bounded LRU)
  or `SqlHistoryStore` keep it so follow-up messages in the same context
  continue the conversation.
- **Delegation state** (for orchestrators) — for each remote agent and
  conversation, the remote ``contextId`` and any remote task waiting for input.
  `MemoryDelegationStore` or `SqlDelegationStore` keep it outside the agent, so
  it survives configuration reloads and (with SQL) is shared by all replicas.

`open_storage` builds all of them from ``server.yaml``'s ``a2a.store``. The
in-memory stores need no external services and are the default; the SQL stores
work with any SQLAlchemy async driver (PostgreSQL via asyncpg, SQLite via
aiosqlite) and share one engine.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from a2a.server.context import ServerCallContext
from a2a.server.tasks import (
    DatabasePushNotificationConfigStore,
    DatabaseTaskStore,
    InMemoryPushNotificationConfigStore,
    InMemoryTaskStore,
    TaskStore,
)
from a2a.types import ListTasksRequest, ListTasksResponse, Task
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    UserPromptPart,
)
from sqlalchemy import (
    Column,
    Float,
    MetaData,
    String,
    Table,
    Text,
    delete,
    insert,
    select,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from .a2a_wire import TERMINAL_STATES
from .config import ConfigError, Secrets, StoreBackend, StoreConfig

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# History trimming (pure)
# --------------------------------------------------------------------------------------


def _starts_turn(message: ModelMessage) -> bool:
    return isinstance(message, ModelRequest) and any(
        isinstance(part, UserPromptPart) for part in message.parts
    )


def trim_history(messages: Sequence[ModelMessage], limit: int) -> list[ModelMessage]:
    """Keep at most ``limit`` messages, cutting only at the start of a user turn.

    Cutting elsewhere could orphan a tool return from its tool call, which
    model APIs reject. If no turn boundary falls within the window, the most
    recent complete turn is kept even if it exceeds the limit.
    """
    if len(messages) <= limit:
        return list(messages)
    window_start = len(messages) - limit
    starts = [i for i, m in enumerate(messages) if _starts_turn(m)]
    cut = next((i for i in starts if i >= window_start), starts[-1] if starts else 0)
    return list(messages[cut:])


# --------------------------------------------------------------------------------------
# Conversation history stores
# --------------------------------------------------------------------------------------


class HistoryStore(Protocol):
    """Loads and saves the message history of an A2A context."""

    async def load(self, context_id: str) -> list[ModelMessage]: ...

    async def save(self, context_id: str, messages: Sequence[ModelMessage]) -> None: ...


class MemoryHistoryStore:
    """In-process history store: an LRU of at most ``max_contexts`` conversations."""

    def __init__(self, max_contexts: int = 1000, max_messages: int = 200) -> None:
        self._max_contexts = max_contexts
        self._max_messages = max_messages
        self._data: OrderedDict[str, list[ModelMessage]] = OrderedDict()

    async def load(self, context_id: str) -> list[ModelMessage]:
        messages = self._data.get(context_id)
        if messages is None:
            return []
        self._data.move_to_end(context_id)
        return list(messages)

    async def save(self, context_id: str, messages: Sequence[ModelMessage]) -> None:
        self._data[context_id] = trim_history(messages, self._max_messages)
        self._data.move_to_end(context_id)
        while len(self._data) > self._max_contexts:
            self._data.popitem(last=False)


_metadata = MetaData()
_contexts = Table(
    "agent_contexts",
    _metadata,
    Column("context_id", String(255), primary_key=True),
    Column("messages", Text, nullable=False),
    Column("updated_at", Float, nullable=False),
)


_delegations = Table(
    "agent_delegations",
    _metadata,
    Column("agent_url", String(512), primary_key=True),
    Column("conversation_id", String(256), primary_key=True),
    Column("context_id", String(255), nullable=True),
    Column("task_id", String(255), nullable=True),
    Column("updated_at", Float, nullable=False),
)


class _SqlTables:
    """Creates this module's tables on first use (once per engine)."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine
        self._ready = False
        self._init_lock = asyncio.Lock()

    async def _ensure_table(self) -> None:
        if self._ready:
            return
        async with self._init_lock:
            if not self._ready:
                async with self._engine.begin() as conn:
                    await conn.run_sync(_metadata.create_all)
                self._ready = True

    async def _replace(
        self, table: Table, key: dict[str, Any], row: dict[str, Any]
    ) -> None:
        """Write ``row`` as the only row matching ``key`` (portable upsert)."""
        await self._ensure_table()
        where = [table.c[column] == value for column, value in key.items()]
        for attempt in (1, 2):  # a concurrent insert of the same key can race
            try:
                async with self._engine.begin() as conn:
                    await conn.execute(delete(table).where(*where))
                    await conn.execute(insert(table).values(**key, **row))
                return
            except IntegrityError:
                if attempt == 2:
                    raise


class SqlHistoryStore(_SqlTables):
    """History store in any SQLAlchemy async database (table ``agent_contexts``)."""

    def __init__(self, engine: AsyncEngine, max_messages: int = 200) -> None:
        super().__init__(engine)
        self._max_messages = max_messages

    async def load(self, context_id: str) -> list[ModelMessage]:
        await self._ensure_table()
        async with self._engine.connect() as conn:
            row = (
                await conn.execute(
                    select(_contexts.c.messages).where(
                        _contexts.c.context_id == context_id
                    )
                )
            ).first()
        if row is None:
            return []
        return list(ModelMessagesTypeAdapter.validate_json(row[0]))

    async def save(self, context_id: str, messages: Sequence[ModelMessage]) -> None:
        payload = ModelMessagesTypeAdapter.dump_json(
            trim_history(messages, self._max_messages)
        ).decode()
        await self._replace(
            _contexts,
            {"context_id": context_id},
            {"messages": payload, "updated_at": time.time()},
        )


# --------------------------------------------------------------------------------------
# Delegation state stores
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RemoteContext:
    """Where a conversation stands with one remote agent.

    Attributes:
        context_id: The remote agent's ``contextId`` for this conversation.
        task_id: A remote task waiting for more input, continued on the next
            call; None when there is nothing to continue.
    """

    context_id: str | None = None
    task_id: str | None = None


class DelegationStore(Protocol):
    """Remembers each conversation's `RemoteContext` with each remote agent."""

    async def load(self, agent_url: str, conversation_id: str) -> RemoteContext: ...

    async def save(
        self, agent_url: str, conversation_id: str, remote: RemoteContext
    ) -> None: ...


class MemoryDelegationStore:
    """In-process delegation store: an LRU of at most ``max_entries`` entries."""

    def __init__(self, max_entries: int = 1024) -> None:
        self._max_entries = max_entries
        self._data: OrderedDict[tuple[str, str], RemoteContext] = OrderedDict()

    async def load(self, agent_url: str, conversation_id: str) -> RemoteContext:
        key = (agent_url, conversation_id)
        if key not in self._data:
            return RemoteContext()
        self._data.move_to_end(key)
        return self._data[key]

    async def save(
        self, agent_url: str, conversation_id: str, remote: RemoteContext
    ) -> None:
        key = (agent_url, conversation_id)
        self._data[key] = remote
        self._data.move_to_end(key)
        while len(self._data) > self._max_entries:
            self._data.popitem(last=False)


class SqlDelegationStore(_SqlTables):
    """Delegation store in any SQLAlchemy async database (``agent_delegations``)."""

    async def load(self, agent_url: str, conversation_id: str) -> RemoteContext:
        await self._ensure_table()
        async with self._engine.connect() as conn:
            row = (
                await conn.execute(
                    select(_delegations.c.context_id, _delegations.c.task_id).where(
                        _delegations.c.agent_url == agent_url,
                        _delegations.c.conversation_id == conversation_id,
                    )
                )
            ).first()
        return RemoteContext() if row is None else RemoteContext(*row)

    async def save(
        self, agent_url: str, conversation_id: str, remote: RemoteContext
    ) -> None:
        await self._replace(
            _delegations,
            {"agent_url": agent_url, "conversation_id": conversation_id},
            {
                "context_id": remote.context_id,
                "task_id": remote.task_id,
                "updated_at": time.time(),
            },
        )


# --------------------------------------------------------------------------------------
# Task store
# --------------------------------------------------------------------------------------


class BoundedMemoryTaskStore(TaskStore):
    """The a2a-sdk in-memory task store, capped at ``max_tasks`` tasks.

    When the cap is exceeded, the oldest *finished* tasks are deleted. Tasks
    that are still running are never evicted.
    """

    def __init__(self, max_tasks: int = 10_000) -> None:
        self._store = InMemoryTaskStore()
        self._max_tasks = max_tasks
        self._running: dict[str, ServerCallContext] = {}
        self._finished: OrderedDict[str, ServerCallContext] = OrderedDict()

    async def save(self, task: Task, context: ServerCallContext) -> None:
        await self._store.save(task, context)
        if task.status.state in TERMINAL_STATES:
            self._running.pop(task.id, None)
            self._finished[task.id] = context
            self._finished.move_to_end(task.id)
        else:
            self._finished.pop(task.id, None)
            self._running[task.id] = context
        while (
            self._finished
            and len(self._running) + len(self._finished) > self._max_tasks
        ):
            task_id, oldest = self._finished.popitem(last=False)
            await self._store.delete(task_id, oldest)

    async def get(self, task_id: str, context: ServerCallContext) -> Task | None:
        return await self._store.get(task_id, context)

    async def list(
        self, params: ListTasksRequest, context: ServerCallContext
    ) -> ListTasksResponse:
        return await self._store.list(params, context)

    async def delete(self, task_id: str, context: ServerCallContext) -> None:
        self._running.pop(task_id, None)
        self._finished.pop(task_id, None)
        await self._store.delete(task_id, context)


# --------------------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Storage:
    """Every store the service uses, built once for the life of the process.

    Attributes:
        tasks: A2A task state.
        history: Conversation history per A2A context.
        push_configs: A2A push-notification configurations.
        delegations: Orchestrator delegation state per remote agent.
        engine: The shared SQL engine (None for the memory backend).
    """

    tasks: TaskStore
    history: HistoryStore
    push_configs: Any
    delegations: DelegationStore
    engine: AsyncEngine | None = None

    async def aclose(self) -> None:
        """Release the database connection pool, if any."""
        if self.engine is not None:
            await self.engine.dispose()


def open_storage(config: StoreConfig, secrets: Secrets) -> Storage:
    """Build the stores selected by ``server.yaml``'s ``a2a.store``.

    Opens no connections: the SQL engine connects (and creates its tables) on
    first use.

    Raises:
        ConfigError: If the SQL backend is selected but its database URL secret
            is missing.
    """
    if config.backend != StoreBackend.SQL:
        log.info(
            "storage: tasks, conversation history, and delegation state are kept "
            "in memory"
        )
        return Storage(
            tasks=BoundedMemoryTaskStore(config.max_tasks),
            history=MemoryHistoryStore(
                config.max_contexts, config.max_history_messages
            ),
            push_configs=InMemoryPushNotificationConfigStore(),
            delegations=MemoryDelegationStore(config.max_contexts),
        )

    dsn = secrets.get(config.database_url_secret)
    if not dsn:
        raise ConfigError(
            f"a2a.store.backend: sql needs the database URL in the secret file "
            f"'{secrets.directory / config.database_url_secret}' (e.g. "
            "postgresql+asyncpg://user:pass@host:5432/db or "
            "sqlite+aiosqlite:////data/agent.db)"
        )
    engine = create_async_engine(dsn)
    log.info(
        "storage: tasks, conversation history, and delegation state are stored in SQL"
    )
    return Storage(
        tasks=DatabaseTaskStore(engine, create_table=True),
        history=SqlHistoryStore(engine, config.max_history_messages),
        push_configs=DatabasePushNotificationConfigStore(engine, create_table=True),
        delegations=SqlDelegationStore(engine),
        engine=engine,
    )
