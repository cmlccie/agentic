"""Tests for agentic.simple_agent.config.agent_spec."""

import os
from pathlib import Path
from unittest.mock import patch

from agentic.simple_agent.config.agent_spec import (
    _expand_headers_in_spec,
    _expand_secret_refs,
)
from agentic.simple_agent.config.server_spec import AgentSecrets

# -------------------------------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------------------------------


def _secrets_dir(tmp_path: Path, **files: str) -> Path:
    d = tmp_path / "secrets"
    d.mkdir()
    for name, value in files.items():
        (d / name).write_text(value)
    return d


def _make_secrets(tmp_path: Path, **files: str) -> AgentSecrets:
    return AgentSecrets(_secrets_dir(tmp_path, **files))


# -------------------------------------------------------------------------------------------------
# _expand_secret_refs
# -------------------------------------------------------------------------------------------------


class TestExpandSecretRefs:
    def test_expands_from_file(self, tmp_path):
        secrets = _make_secrets(tmp_path, my_token="file-value")
        result = _expand_secret_refs("Bearer ${MY_TOKEN}", secrets)
        assert result == "Bearer file-value"

    def test_falls_back_to_env_var(self, tmp_path):
        secrets = _make_secrets(tmp_path)
        with patch.dict(os.environ, {"MY_ENV_KEY": "env-value"}):
            result = _expand_secret_refs("${MY_ENV_KEY}", secrets)
        assert result == "env-value"

    def test_file_takes_precedence_over_env(self, tmp_path):
        secrets = _make_secrets(tmp_path, my_key="file-value")
        with patch.dict(os.environ, {"MY_KEY": "env-value"}):
            result = _expand_secret_refs("${MY_KEY}", secrets)
        assert result == "file-value"

    def test_unresolvable_ref_left_as_is(self, tmp_path, caplog):
        secrets = _make_secrets(tmp_path)
        result = _expand_secret_refs("${UNKNOWN_SECRET}", secrets)
        assert result == "${UNKNOWN_SECRET}"

    def test_no_refs_returns_unchanged(self, tmp_path):
        secrets = _make_secrets(tmp_path)
        result = _expand_secret_refs("plain string", secrets)
        assert result == "plain string"

    def test_multiple_refs_expanded(self, tmp_path):
        secrets = _make_secrets(tmp_path, tok_a="aaa", tok_b="bbb")
        result = _expand_secret_refs("${TOK_A}:${TOK_B}", secrets)
        assert result == "aaa:bbb"


# -------------------------------------------------------------------------------------------------
# _expand_headers_in_spec
# -------------------------------------------------------------------------------------------------


class TestExpandHeadersInSpec:
    def test_expands_mcp_headers(self, tmp_path):
        secrets = _make_secrets(tmp_path, mcp_token_a="tok-xyz")
        spec_dict = {
            "capabilities": [
                {
                    "MCP": {
                        "url": "http://tool/mcp",
                        "headers": {"Authorization": "Bearer ${MCP_TOKEN_A}"},
                    }
                }
            ]
        }
        result = _expand_headers_in_spec(spec_dict, secrets)
        assert (
            result["capabilities"][0]["MCP"]["headers"]["Authorization"]
            == "Bearer tok-xyz"
        )

    def test_no_capabilities_returns_unchanged(self, tmp_path):
        secrets = _make_secrets(tmp_path)
        spec_dict = {"model": "anthropic:claude"}
        result = _expand_headers_in_spec(spec_dict, secrets)
        assert result == {"model": "anthropic:claude"}

    def test_mcp_without_headers_returns_unchanged(self, tmp_path):
        secrets = _make_secrets(tmp_path)
        spec_dict = {"capabilities": [{"MCP": {"url": "http://tool/mcp"}}]}
        result = _expand_headers_in_spec(spec_dict, secrets)
        assert "headers" not in result["capabilities"][0]["MCP"]

    def test_non_mcp_capability_skipped(self, tmp_path):
        secrets = _make_secrets(tmp_path)
        spec_dict = {"capabilities": ["NotADict"]}
        result = _expand_headers_in_spec(spec_dict, secrets)
        assert result["capabilities"] == ["NotADict"]
