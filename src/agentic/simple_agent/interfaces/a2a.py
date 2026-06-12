from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fasta2a import FastA2A
from fasta2a.broker import InMemoryBroker
from fasta2a.pydantic_ai import agent_to_a2a
from fasta2a.schema import AgentProvider, Skill
from fasta2a.storage import InMemoryStorage
from pydantic_ai import Agent

if TYPE_CHECKING:
    from ..config.server_spec import AgentSecrets, ServerSpec

log = logging.getLogger(__name__)


def build_a2a_app(
    agent: Agent,
    server_spec: ServerSpec,
    secrets: AgentSecrets,
    agent_url: str = "http://localhost:8000",
) -> FastA2A:
    """Build the fasta2a ASGI sub-app.

    The sub-app is mounted once at startup. It retains the agent reference
    captured here — hot-reload updates AppState.agent, but the A2A sub-app
    continues using the agent that was live at mount time. Toggling A2A requires
    a pod restart.
    """
    card = server_spec.agent_card

    if server_spec.broker.backend.value == "redis":
        storage, broker = _build_redis_backends(secrets)
    else:
        storage, broker = InMemoryStorage(), InMemoryBroker()

    provider: AgentProvider | None = None
    if card.provider:
        provider = AgentProvider(organization=card.provider.organization, url=card.provider.url)

    skills: list[Skill] = [
        Skill(
            id=s.id,
            name=s.name,
            description=s.description,
            tags=s.tags,
            input_modes=s.input_modes,
            output_modes=s.output_modes,
            **({"examples": s.examples} if s.examples is not None else {}),
        )
        for s in card.skills
    ]

    return agent_to_a2a(
        agent,
        storage=storage,
        broker=broker,
        name=card.display_name,
        description=card.description,
        version=card.version,
        url=agent_url,
        provider=provider,
        skills=skills or None,
    )


def _build_redis_backends(secrets: AgentSecrets):
    try:
        import redis.asyncio as aioredis
        from fasta2a.redis import RedisBroker, RedisStorage  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            "broker.backend = redis requires fasta2a[redis] and redis packages"
        ) from exc

    pool = aioredis.ConnectionPool.from_url(secrets.agent_redis_url)
    client = aioredis.Redis(connection_pool=pool)
    return RedisStorage(client), RedisBroker(client)
