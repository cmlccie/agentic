"""Tests for server.yaml models, secrets, and agent.yaml loading."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic_ai.models.openai import OpenAIChatModel

from agentic.agent.config import (
    ConfigError,
    ResponseMode,
    Secrets,
    StoreBackend,
    expand_secret_refs,
    expand_secret_refs_deep,
    load_server_spec,
    migrate_legacy_server_config,
)
from agentic.agent.spec import (
    CUSTOM_CAPABILITIES,
    HARNESS_CAPABILITIES,
    build_agent,
    load_agent_spec,
    migrate_legacy_agent_config,
    resolve_model,
)

CARD = {"display_name": "A", "description": "B"}


def write(path: Path, data: dict[str, Any]) -> Path:
    path.write_text(yaml.safe_dump(data))
    return path


def secrets_with(tmp_path: Path, files: dict[str, str]) -> Secrets:
    directory = tmp_path / "secrets"
    directory.mkdir(exist_ok=True)
    for key, value in files.items():
        (directory / key).write_text(value + "\n")
    return Secrets(directory)


# --------------------------------------------------------------------------------------
# server.yaml
# --------------------------------------------------------------------------------------


class TestServerSpec:
    def test_defaults(self, tmp_path: Path) -> None:
        spec = load_server_spec(write(tmp_path / "s.yaml", {"agent_card": CARD}))
        assert spec.interfaces.openai and spec.interfaces.a2a
        assert spec.a2a.response_mode == ResponseMode.AUTO
        assert spec.a2a.store.backend == StoreBackend.MEMORY
        assert not spec.a2a.push_notifications.enabled
        assert spec.streaming.activity == "summary"
        assert spec.auth.bearer_token_secret is None

    def test_unknown_keys_are_rejected(self, tmp_path: Path) -> None:
        path = write(
            tmp_path / "s.yaml", {"agent_card": CARD, "streaming": {"activty": "off"}}
        )
        with pytest.raises(ConfigError, match="activty"):
            load_server_spec(path)

    def test_missing_file_and_bad_yaml(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="not found"):
            load_server_spec(tmp_path / "missing.yaml")
        bad = tmp_path / "bad.yaml"
        bad.write_text("agent_card: [unclosed")
        with pytest.raises(ConfigError, match="not valid YAML"):
            load_server_spec(bad)
        bad.write_text("- a list")
        with pytest.raises(ConfigError, match="mapping"):
            load_server_spec(bad)

    @pytest.mark.parametrize(
        "legacy, backend, secret",
        [
            ("memory", "memory", "a2a.database_url"),
            ("postgres", "sql", "task_broker.database_url"),
            ("redis", "memory", "a2a.database_url"),
        ],
    )
    def test_legacy_broker_is_migrated(
        self, tmp_path: Path, legacy: str, backend: str, secret: str, caplog: Any
    ) -> None:
        raw = {"agent_card": CARD, "broker": {"backend": legacy}}
        with caplog.at_level(logging.WARNING):
            spec = load_server_spec(write(tmp_path / "s.yaml", raw))
        assert spec.a2a.store.backend == backend
        assert spec.a2a.store.database_url_secret == secret
        assert "deprecated" in caplog.text

    def test_legacy_interfaces_and_reload_are_migrated(self, tmp_path: Path) -> None:
        raw = {
            "agent_card": CARD,
            "interfaces": {"a2a": True, "openai_compat": False, "ui": True},
            "reload": {"drain_timeout": 30},
        }
        spec = load_server_spec(write(tmp_path / "s.yaml", raw))
        assert spec.interfaces.openai is False
        assert spec.interfaces.a2a is True

    def test_migration_does_not_mutate_input(self) -> None:
        raw = {"agent_card": CARD, "broker": {"backend": "postgres"}}
        migrate_legacy_server_config(raw)
        assert "broker" in raw


# --------------------------------------------------------------------------------------
# Secrets
# --------------------------------------------------------------------------------------


class TestSecrets:
    def test_get_and_require(self, tmp_path: Path) -> None:
        secrets = secrets_with(tmp_path, {"token": "abc"})
        assert secrets.get("token") == "abc"
        assert secrets.get("missing") is None
        assert secrets.require("token") == "abc"
        with pytest.raises(ConfigError, match="missing"):
            secrets.require("missing")

    def test_values_are_reread(self, tmp_path: Path) -> None:
        secrets = secrets_with(tmp_path, {"token": "one"})
        (secrets.directory / "token").write_text("two")
        assert secrets.get("token") == "two"

    def test_missing_directory_is_empty(self, tmp_path: Path) -> None:
        assert Secrets(tmp_path / "nope").get("x") is None

    def test_expand_references(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        secrets = secrets_with(tmp_path, {"weather_token": "s3cret"})
        monkeypatch.setenv("REGION", "eu")
        assert expand_secret_refs("Bearer ${WEATHER_TOKEN}", secrets) == "Bearer s3cret"
        assert expand_secret_refs("${weather_token}-${REGION}", secrets) == "s3cret-eu"
        with pytest.raises(ConfigError, match=r"\$\{NOPE\}"):
            expand_secret_refs("${NOPE}", secrets)

    def test_expand_deep(self, tmp_path: Path) -> None:
        secrets = secrets_with(tmp_path, {"t": "x"})
        data = {"a": ["${t}", {"b": "${t}!"}], "n": 3}
        assert expand_secret_refs_deep(data, secrets) == {
            "a": ["x", {"b": "x!"}],
            "n": 3,
        }


# --------------------------------------------------------------------------------------
# agent.yaml
# --------------------------------------------------------------------------------------


class TestAgentSpec:
    def test_valid_spec(self, tmp_path: Path) -> None:
        path = write(
            tmp_path / "a.yaml",
            {
                "model": "test",
                "name": "x",
                "instructions": "be kind",
                "model_settings": {"temperature": 0.2},
                "capabilities": [{"MCP": {"url": "http://tools/mcp"}}],
            },
        )
        agent = build_agent(load_agent_spec(path, Secrets(tmp_path)), Secrets(tmp_path))
        assert agent.name == "x"

    def test_unknown_keys_are_rejected(self, tmp_path: Path) -> None:
        path = write(tmp_path / "a.yaml", {"model": "test", "instruction": "typo"})
        with pytest.raises(ConfigError, match="instruction"):
            load_agent_spec(path, Secrets(tmp_path))

    def test_secret_references_in_capabilities(self, tmp_path: Path) -> None:
        secrets = secrets_with(tmp_path, {"tools_token": "t0k"})
        path = write(
            tmp_path / "a.yaml",
            {
                "model": "test",
                "capabilities": [
                    {
                        "MCP": {
                            "url": "http://t/mcp",
                            "headers": {"Authorization": "Bearer ${TOOLS_TOKEN}"},
                        }
                    }
                ],
            },
        )
        spec = load_agent_spec(path, secrets)
        assert spec.capabilities[0].arguments["headers"] == {
            "Authorization": "Bearer t0k"
        }

    def test_unresolved_secret_reference_fails(self, tmp_path: Path) -> None:
        path = write(
            tmp_path / "a.yaml",
            {
                "model": "test",
                "capabilities": [
                    {"A2AAgent": {"url": "http://x", "headers": {"A": "${NOPE}"}}}
                ],
            },
        )
        with pytest.raises(ConfigError, match="NOPE"):
            load_agent_spec(path, Secrets(tmp_path))

    def test_bad_capability_arguments_fail(self, tmp_path: Path) -> None:
        path = write(
            tmp_path / "a.yaml",
            {"model": "test", "capabilities": [{"A2AAgent": {"bogus": 1}}]},
        )
        with pytest.raises(ConfigError):
            build_agent(load_agent_spec(path, Secrets(tmp_path)), Secrets(tmp_path))

    def test_unknown_model_fails(self, tmp_path: Path) -> None:
        path = write(tmp_path / "a.yaml", {"model": "nonsense-provider:x"})
        with pytest.raises(ConfigError):
            build_agent(load_agent_spec(path, Secrets(tmp_path)), Secrets(tmp_path))

    def test_harness_capabilities_are_declarable(self, tmp_path: Path) -> None:
        examples: dict[str, Any] = {
            "RepairToolArguments": {},
            "ToolOutputLimits": {},
            "ClampOversizedMessages": {"max_part_chars": 20000},
            "ClearToolResults": {"max_messages": 40},
            "SlidingWindowCompaction": {"max_messages": 80, "keep_messages": 40},
            "SummarizingCompaction": {"max_messages": 80},
            "WarnNearLimits": {"max_iterations": 25},
            "Planning": {},
            "SpendLimits": {},
        }
        assert set(examples) == {c.__name__ for c in HARNESS_CAPABILITIES}
        path = write(
            tmp_path / "a.yaml",
            {"model": "test", "capabilities": [{k: v} for k, v in examples.items()]},
        )
        agent = build_agent(load_agent_spec(path, Secrets(tmp_path)), Secrets(tmp_path))
        assert agent is not None

    def test_a2a_agent_is_registered(self) -> None:
        assert "A2AAgent" in {c.__name__ for c in CUSTOM_CAPABILITIES}


class TestModelResolution:
    def test_plain_model_strings_pass_through(self, tmp_path: Path) -> None:
        assert (
            resolve_model("anthropic:claude-sonnet-5", Secrets(tmp_path))
            == "anthropic:claude-sonnet-5"
        )
        assert resolve_model(None, Secrets(tmp_path)) is None

    @pytest.mark.parametrize(
        "secret_prefix", ["model", "openai_compatible"], ids=["current", "legacy"]
    )
    def test_endpoint_secrets_configure_vllm(
        self, tmp_path: Path, secret_prefix: str
    ) -> None:
        secrets = secrets_with(
            tmp_path,
            {
                f"{secret_prefix}.base_url": "http://vllm:8000/v1",
                f"{secret_prefix}.api_key": "k",
            },
        )
        model = resolve_model("vllm:Qwen/Qwen3-32B", secrets)
        assert isinstance(model, OpenAIChatModel)
        assert model.model_name == "Qwen/Qwen3-32B"
        assert str(model.client.base_url).rstrip("/") == "http://vllm:8000/v1"
        assert model.client.api_key == "k"

    def test_endpoint_secrets_never_redirect_hosted_providers(
        self, tmp_path: Path
    ) -> None:
        secrets = secrets_with(tmp_path, {"model.base_url": "http://vllm:8000/v1"})
        for model in (
            "openai:gpt-6-sol",
            "openai-chat:gpt-6-sol",
            "anthropic:claude-sonnet-5",
        ):
            assert resolve_model(model, secrets) == model

    def test_vllm_from_environment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("VLLM_BASE_URL", "http://env-vllm/v1")
        path = write(tmp_path / "a.yaml", {"model": "vllm:m"})
        agent = build_agent(load_agent_spec(path, Secrets(tmp_path)), Secrets(tmp_path))
        assert str(agent.model.client.base_url).startswith("http://env-vllm/v1")

    def test_vllm_without_endpoint_fails_clearly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("VLLM_BASE_URL", raising=False)
        path = write(tmp_path / "a.yaml", {"model": "vllm:m"})
        with pytest.raises(ConfigError, match="VLLM_BASE_URL"):
            build_agent(load_agent_spec(path, Secrets(tmp_path)), Secrets(tmp_path))


class TestLegacyAgentConfig:
    def test_openai_compat_sentinel(self, tmp_path: Path) -> None:
        raw = migrate_legacy_agent_config(
            {"model": "openai-compat", "model_id": "local-model"}
        )
        assert raw == {"model": "vllm:local-model"}

    def test_model_id_without_sentinel_is_an_error(self) -> None:
        with pytest.raises(ConfigError, match="model_id"):
            migrate_legacy_agent_config({"model": "vllm:x", "model_id": "y"})

    def test_a2a_servers_become_capabilities(self) -> None:
        raw = migrate_legacy_agent_config(
            {
                "model": "test",
                "capabilities": [{"MCP": {"url": "http://t"}}],
                "a2a_servers": [
                    {"url": "http://w/a2a", "id": "weather", "headers": {"A": "b"}},
                    {"url": "http://n/a2a"},
                ],
            }
        )
        assert raw["capabilities"] == [
            {"MCP": {"url": "http://t"}},
            {
                "A2AAgent": {
                    "url": "http://w/a2a",
                    "name": "weather",
                    "headers": {"A": "b"},
                }
            },
            {"A2AAgent": {"url": "http://n/a2a"}},
        ]

    def test_legacy_orchestrator_file_loads(self, tmp_path: Path) -> None:
        secrets = secrets_with(
            tmp_path,
            {
                "openai_compatible.base_url": "http://lm/v1",
                "openai_compatible.api_key": "k",
            },
        )
        path = write(
            tmp_path / "a.yaml",
            {
                "name": "orchestrator-agent",
                "description": "legacy",
                "model": "openai-compat",
                "model_id": "local-model",
                "instructions": "delegate",
                "model_settings": {"temperature": 0.2},
                "a2a_servers": [{"url": "http://w/a2a"}],
            },
        )
        agent = build_agent(load_agent_spec(path, secrets), secrets)
        assert isinstance(agent.model, OpenAIChatModel)
        assert agent.model.model_name == "local-model"
