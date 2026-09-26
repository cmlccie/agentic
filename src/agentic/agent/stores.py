"""Persistence for the A2A server: conversation history and tasks.

Two things need storing:

- **Tasks** (A2A protocol state) — handled by the a2a-sdk `TaskStore`
  implementations. In memory by default (bounded here, since the SDK's store
  grows without limit), or in SQL via `DatabaseTaskStore`.
- **Conversation history** (the Pydantic AI message history for each A2A
  ``contextId``) — needed because quick exchanges are answered with a direct
  Message, which the SDK never persists. `MemoryHistoryStore` (bounded LRU)
  or `SqlHistoryStore` keep it so follow-up messages in the same context
  continue the conversation.

The in-memory stores need no external services and are the default; the SQL
stores work with any SQLAlchemy async driver (PostgreSQL via asyncpg, SQLite via
aiosqlite).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from collections.abc import Sequence
from typing import Protocol

from a2a.server.context import ServerCallContext
from a2a.server.tasks import InMemoryTaskStore, TaskStore
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
from sqlalchemy.ext.asyncio import AsyncEngine

from .a2a_wire import TERMINAL_STATES

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


class SqlHistoryStore:
    """History store in any SQLAlchemy async database (table ``agent_contexts``)."""

    def __init__(self, engine: AsyncEngine, max_messages: int = 200) -> None:
        self._engine = engine
        self._max_messages = max_messages
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
        await self._ensure_table()
        payload = ModelMessagesTypeAdapter.dump_json(
            trim_history(messages, self._max_messages)
        ).decode()
        row = {"context_id": context_id, "messages": payload, "updated_at": time.time()}
        for attempt in (1, 2):  # a concurrent insert of the same context can race
            try:
                async with self._engine.begin() as conn:
                    await conn.execute(
                        delete(_contexts).where(_contexts.c.context_id == context_id)
                    )
                    await conn.execute(insert(_contexts).values(**row))
                return
            except IntegrityError:
                if attempt == 2:
                    raise


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
