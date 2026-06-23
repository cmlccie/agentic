"""Tests for agentic.orchestrator_agent.a2a_workers."""

import asyncio
from pathlib import Path

from agentic.orchestrator_agent.a2a_workers import (
    _collect_result,
    _safe_tool_name,
    _tool_description,
    build_a2a_tools,
)
from agentic.orchestrator_agent.config.agent_spec import OrchestratorSpec
from agentic.runtime.config import AgentSecrets

# -------------------------------------------------------------------------------------------------
# _safe_tool_name
# -------------------------------------------------------------------------------------------------


class TestSafeToolName:
    def test_lowercases_and_replaces(self):
        assert _safe_tool_name("Weather Agent!") == "weather_agent"

    def test_strips_edge_separators(self):
        assert _safe_tool_name("  --Knowledge Base--  ") == "knowledge_base"

    def test_empty_fallback(self):
        assert _safe_tool_name("***") == "agent"


# -------------------------------------------------------------------------------------------------
# _tool_description
# -------------------------------------------------------------------------------------------------


class TestToolDescription:
    def test_includes_name_and_description(self):
        from a2a.types import AgentCard, AgentSkill

        card = AgentCard(
            name="Weather",
            description="Forecasts the weather.",
            version="1.0.0",
            skills=[AgentSkill(id="f", name="Forecast", description="Daily forecast")],
        )
        desc = _tool_description(card)
        assert "Weather" in desc
        assert "Forecasts the weather." in desc
        assert "Forecast: Daily forecast" in desc


# -------------------------------------------------------------------------------------------------
# _collect_result
# -------------------------------------------------------------------------------------------------


class TestCollectResult:
    def test_prefers_artifacts(self):
        assert _collect_result(["msg"], ["artifact text"]) == "artifact text"

    def test_falls_back_to_last_message(self):
        assert _collect_result(["first", "last"], []) == "last"

    def test_empty(self):
        assert _collect_result([], []) == ""


# -------------------------------------------------------------------------------------------------
# build_a2a_tools degraded mode
# -------------------------------------------------------------------------------------------------


class TestBuildA2ATools:
    def test_unreachable_servers_skipped(self, tmp_path: Path):
        secrets = AgentSecrets(tmp_path)
        spec = OrchestratorSpec.model_validate(
            {
                "name": "o",
                "model": "openai-compat",
                "a2a_servers": [{"url": "http://127.0.0.1:9/a2a"}],
            }
        )
        tools = asyncio.run(build_a2a_tools(spec, secrets))
        assert tools == []

    def test_no_servers_returns_empty(self, tmp_path: Path):
        secrets = AgentSecrets(tmp_path)
        spec = OrchestratorSpec(name="o", model="openai-compat")
        assert asyncio.run(build_a2a_tools(spec, secrets)) == []
