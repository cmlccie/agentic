"""Tests for agentic.runtime.config."""

from pathlib import Path

import pytest
import yaml

from agentic.runtime.config import (
    AgentSecrets,
    BrokerBackend,
    ProviderConfig,
    ServerSpec,
    SkillConfig,
    load_server_spec,
)

# -------------------------------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------------------------------


def _write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.dump(data))


def _secrets_dir(tmp_path: Path, files: dict[str, str] | None = None) -> Path:
    d = tmp_path / "secrets"
    d.mkdir()
    for name, value in (files or {}).items():
        (d / name).write_text(value)
    return d


# -------------------------------------------------------------------------------------------------
# ServerSpec validation
# -------------------------------------------------------------------------------------------------


class TestServerSpec:
    def test_minimal_required_fields(self):
        raw = {"agent_card": {"display_name": "X", "description": "Y"}}
        spec = ServerSpec.model_validate(raw)
        assert spec.agent_card.display_name == "X"
        assert spec.broker.backend == BrokerBackend.MEMORY
        assert spec.interfaces.a2a is True
        assert spec.interfaces.openai_compat is True
        assert spec.reload.drain_timeout == 30.0

    def test_all_fields_parsed(self):
        raw = {
            "agent_card": {
                "display_name": "Agent",
                "description": "Desc",
                "version": "2.0.0",
                "icon_url": "http://x.com/icon.png",
                "documentation_url": "http://docs.example.com",
                "provider": {"organization": "Acme", "url": "http://acme.com"},
                "skills": [
                    {
                        "id": "chat",
                        "name": "Chat",
                        "description": "General chat",
                        "tags": ["general"],
                        "examples": ["Hello!"],
                        "input_modes": ["text/plain"],
                        "output_modes": ["text/plain"],
                    }
                ],
            },
            "broker": {"backend": "redis"},
            "interfaces": {"a2a": False, "openai_compat": True},
            "reload": {"drain_timeout": 60.0},
        }
        spec = ServerSpec.model_validate(raw)
        assert spec.broker.backend == BrokerBackend.REDIS
        assert spec.interfaces.a2a is False
        assert spec.reload.drain_timeout == 60.0
        assert spec.agent_card.version == "2.0.0"
        assert spec.agent_card.documentation_url == "http://docs.example.com"
        assert isinstance(spec.agent_card.provider, ProviderConfig)
        assert spec.agent_card.provider.organization == "Acme"
        assert len(spec.agent_card.skills) == 1
        skill = spec.agent_card.skills[0]
        assert isinstance(skill, SkillConfig)
        assert skill.id == "chat"
        assert skill.tags == ["general"]
        assert skill.examples == ["Hello!"]

    def test_skills_default_empty(self):
        raw = {"agent_card": {"display_name": "X", "description": "Y"}}
        spec = ServerSpec.model_validate(raw)
        assert spec.agent_card.skills == []
        assert spec.agent_card.provider is None
        assert spec.agent_card.documentation_url is None

    def test_skill_default_modes(self):
        raw = {
            "agent_card": {
                "display_name": "X",
                "description": "Y",
                "skills": [
                    {"id": "s1", "name": "S1", "description": "desc", "tags": []}
                ],
            }
        }
        spec = ServerSpec.model_validate(raw)
        skill = spec.agent_card.skills[0]
        assert skill.input_modes == ["text/plain"]
        assert skill.output_modes == ["text/plain"]

    def test_missing_agent_card_raises(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ServerSpec.model_validate({})


# -------------------------------------------------------------------------------------------------
# load_server_spec
# -------------------------------------------------------------------------------------------------


class TestLoadServerSpec:
    def test_loads_valid_yaml(self, tmp_path):
        data = {
            "agent_card": {"display_name": "A", "description": "B"},
            "broker": {"backend": "memory"},
        }
        p = tmp_path / "server.yaml"
        _write_yaml(p, data)
        spec = load_server_spec(p)
        assert spec.agent_card.display_name == "A"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_server_spec(tmp_path / "missing.yaml")


# -------------------------------------------------------------------------------------------------
# AgentSecrets
# -------------------------------------------------------------------------------------------------


class TestAgentSecrets:
    def test_reads_existing_secret(self, tmp_path):
        d = _secrets_dir(
            tmp_path, {"openai_compatible.base_url": "http://127.0.0.1:1234/v1"}
        )
        secrets = AgentSecrets(d)
        assert secrets.openai_compatible.base_url == "http://127.0.0.1:1234/v1"

    def test_returns_none_for_missing_secret(self, tmp_path):
        d = _secrets_dir(tmp_path)
        secrets = AgentSecrets(d)
        assert secrets.openai_compatible.base_url is None
        assert secrets.openai_compatible.api_key is None

    def test_strips_trailing_whitespace(self, tmp_path):
        d = _secrets_dir(tmp_path, {"openai_compatible.api_key": "sk-test\n"})
        secrets = AgentSecrets(d)
        assert secrets.openai_compatible.api_key == "sk-test"

    def test_task_broker_redis_url_default(self, tmp_path):
        d = _secrets_dir(tmp_path)
        secrets = AgentSecrets(d)
        assert secrets.task_broker.redis_url == "redis://localhost:6379/0"

    def test_task_broker_redis_url_from_file(self, tmp_path):
        d = _secrets_dir(
            tmp_path, {"task_broker.redis_url": "redis://redis-host:6379/1"}
        )
        secrets = AgentSecrets(d)
        assert secrets.task_broker.redis_url == "redis://redis-host:6379/1"

    def test_task_broker_database_url_default_none(self, tmp_path):
        d = _secrets_dir(tmp_path)
        secrets = AgentSecrets(d)
        assert secrets.task_broker.database_url is None

    def test_task_broker_database_url_from_file(self, tmp_path):
        d = _secrets_dir(
            tmp_path,
            {"task_broker.database_url": "postgresql+asyncpg://u:p@host:5432/db"},
        )
        secrets = AgentSecrets(d)
        assert (
            secrets.task_broker.database_url == "postgresql+asyncpg://u:p@host:5432/db"
        )

    def test_get_generic_key(self, tmp_path):
        d = _secrets_dir(tmp_path, {"mcp_token": "tok-abc"})
        secrets = AgentSecrets(d)
        assert secrets.get("mcp_token") == "tok-abc"

    def test_get_missing_returns_none(self, tmp_path):
        d = _secrets_dir(tmp_path)
        secrets = AgentSecrets(d)
        assert secrets.get("nonexistent") is None

    def test_require_raises_for_missing(self, tmp_path):
        d = _secrets_dir(tmp_path)
        secrets = AgentSecrets(d)
        with pytest.raises(RuntimeError, match="Required secret"):
            secrets.require("nonexistent_key")

    def test_require_returns_value_when_present(self, tmp_path):
        d = _secrets_dir(tmp_path, {"my_key": "secret-value"})
        secrets = AgentSecrets(d)
        assert secrets.require("my_key") == "secret-value"
