"""Tests for agentic.orchestrator_agent.config.agent_spec."""

from pathlib import Path

import pytest
import yaml

from agentic.orchestrator_agent.config.agent_spec import (
    A2AServerConfig,
    OrchestratorSpec,
    load_orchestrator_spec,
)
from agentic.runtime.config import AgentSecrets

# -------------------------------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------------------------------


def _secrets_dir(tmp_path: Path, **files: str) -> Path:
    d = tmp_path / "secrets"
    d.mkdir()
    for name, value in files.items():
        (d / name).write_text(value)
    return d


# -------------------------------------------------------------------------------------------------
# OrchestratorSpec validation
# -------------------------------------------------------------------------------------------------


class TestOrchestratorSpec:
    def test_minimal_required_fields(self):
        spec = OrchestratorSpec.model_validate(
            {"name": "orch", "model": "openai-compat"}
        )
        assert spec.name == "orch"
        assert spec.model == "openai-compat"
        assert spec.a2a_servers == []
        assert spec.model_settings == {}

    def test_all_fields_parsed(self):
        raw = {
            "name": "orch",
            "description": "d",
            "model": "anthropic:claude-sonnet-4-6",
            "model_id": "ignored",
            "instructions": "be a supervisor",
            "model_settings": {"temperature": 0.2, "max_tokens": 1024},
            "a2a_servers": [
                {"url": "http://a/a2a"},
                {"url": "http://b/a2a", "id": "bee", "headers": {"X": "y"}},
            ],
        }
        spec = OrchestratorSpec.model_validate(raw)
        assert spec.instructions == "be a supervisor"
        assert spec.model_settings["temperature"] == 0.2
        assert len(spec.a2a_servers) == 2
        assert isinstance(spec.a2a_servers[1], A2AServerConfig)
        assert spec.a2a_servers[1].id == "bee"
        assert spec.a2a_servers[1].headers == {"X": "y"}

    def test_missing_model_raises(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            OrchestratorSpec.model_validate({"name": "x"})


# -------------------------------------------------------------------------------------------------
# resolved_headers (secret expansion)
# -------------------------------------------------------------------------------------------------


class TestResolvedHeaders:
    def test_expands_secret_refs(self, tmp_path):
        secrets = AgentSecrets(_secrets_dir(tmp_path, wx_token="tok-123"))
        spec = OrchestratorSpec.model_validate(
            {
                "name": "o",
                "model": "openai-compat",
                "a2a_servers": [
                    {
                        "url": "http://a/a2a",
                        "headers": {"Authorization": "Bearer ${WX_TOKEN}"},
                    }
                ],
            }
        )
        headers = spec.resolved_headers(spec.a2a_servers[0], secrets)
        assert headers == {"Authorization": "Bearer tok-123"}

    def test_empty_headers(self, tmp_path):
        secrets = AgentSecrets(_secrets_dir(tmp_path))
        server = A2AServerConfig(url="http://a/a2a")
        spec = OrchestratorSpec(name="o", model="openai-compat")
        assert spec.resolved_headers(server, secrets) == {}


# -------------------------------------------------------------------------------------------------
# load_orchestrator_spec
# -------------------------------------------------------------------------------------------------


class TestLoadOrchestratorSpec:
    def test_loads_valid_yaml(self, tmp_path):
        p = tmp_path / "agent.yaml"
        p.write_text(yaml.dump({"name": "o", "model": "openai:gpt-4o"}))
        spec = load_orchestrator_spec(p)
        assert spec.model == "openai:gpt-4o"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_orchestrator_spec(tmp_path / "missing.yaml")
