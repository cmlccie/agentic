"""Tests for the orchestrator A2A task store backend selection."""

from pathlib import Path

import pytest
from a2a.server.tasks import DatabaseTaskStore, InMemoryTaskStore

from agentic.orchestrator_agent.interfaces.a2a import _build_task_store
from agentic.runtime.config import AgentSecrets, ServerSpec


def _spec(backend: str) -> ServerSpec:
    return ServerSpec.model_validate(
        {
            "agent_card": {"display_name": "O", "description": "d"},
            "broker": {"backend": backend},
        }
    )


def _secrets(tmp_path: Path, **files: str) -> AgentSecrets:
    d = tmp_path / "secrets"
    d.mkdir()
    for name, value in files.items():
        (d / name).write_text(value)
    return AgentSecrets(d)


class TestBuildTaskStore:
    def test_memory_backend(self, tmp_path):
        store, engine = _build_task_store(_spec("memory"), _secrets(tmp_path))
        assert isinstance(store, InMemoryTaskStore)
        assert engine is None

    def test_redis_falls_back_to_memory(self, tmp_path):
        store, engine = _build_task_store(_spec("redis"), _secrets(tmp_path))
        assert isinstance(store, InMemoryTaskStore)
        assert engine is None

    def test_postgres_requires_database_url(self, tmp_path):
        with pytest.raises(RuntimeError, match="agent_database_url"):
            _build_task_store(_spec("postgres"), _secrets(tmp_path))

    def test_postgres_builds_database_store(self, tmp_path):
        secrets = _secrets(
            tmp_path,
            agent_database_url="postgresql+asyncpg://u:p@localhost:5432/db",
        )
        store, engine = _build_task_store(_spec("postgres"), secrets)
        assert isinstance(store, DatabaseTaskStore)
        assert engine is not None
