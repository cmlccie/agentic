"""Tests for agentic.simple_agent.config.server_spec."""

from pathlib import Path

import pytest
import yaml

from agentic.simple_agent.config.server_spec import (
    AgentSecrets,
    BrokerBackend,
    ServerSpec,
    load_server_spec,
)

# -------------------------------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------------------------------


def _write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.dump(data))


def _secrets_dir(tmp_path: Path, **files: str) -> Path:
    d = tmp_path / "secrets"
    d.mkdir()
    for name, value in files.items():
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
        assert spec.interfaces.ui is True
        assert spec.reload.drain_timeout == 30.0

    def test_all_fields_parsed(self):
        raw = {
            "agent_card": {
                "display_name": "Agent",
                "description": "Desc",
                "version": "2.0.0",
                "icon_url": "http://x.com/icon.png",
            },
            "broker": {"backend": "redis"},
            "interfaces": {"a2a": False, "openai_compat": True, "ui": False},
            "reload": {"drain_timeout": 60.0},
        }
        spec = ServerSpec.model_validate(raw)
        assert spec.broker.backend == BrokerBackend.REDIS
        assert spec.interfaces.a2a is False
        assert spec.reload.drain_timeout == 60.0
        assert spec.agent_card.version == "2.0.0"

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
        d = _secrets_dir(tmp_path, anthropic_api_key="sk-ant-test")
        secrets = AgentSecrets(d)
        assert secrets.anthropic_api_key == "sk-ant-test"

    def test_returns_none_for_missing_secret(self, tmp_path):
        d = _secrets_dir(tmp_path)
        secrets = AgentSecrets(d)
        assert secrets.anthropic_api_key is None

    def test_strips_trailing_whitespace(self, tmp_path):
        d = _secrets_dir(tmp_path, openai_api_key="sk-test\n")
        secrets = AgentSecrets(d)
        assert secrets.openai_api_key == "sk-test"

    def test_agent_redis_url_default(self, tmp_path):
        d = _secrets_dir(tmp_path)
        secrets = AgentSecrets(d)
        assert secrets.agent_redis_url == "redis://localhost:6379/0"

    def test_agent_redis_url_from_file(self, tmp_path):
        d = _secrets_dir(tmp_path, agent_redis_url="redis://redis-host:6379/1")
        secrets = AgentSecrets(d)
        assert secrets.agent_redis_url == "redis://redis-host:6379/1"

    def test_get_generic_key(self, tmp_path):
        d = _secrets_dir(tmp_path, mcp_token="tok-abc")
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
        d = _secrets_dir(tmp_path, my_key="secret-value")
        secrets = AgentSecrets(d)
        assert secrets.require("my_key") == "secret-value"
