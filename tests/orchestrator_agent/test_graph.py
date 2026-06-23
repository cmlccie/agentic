"""Tests for agentic.orchestrator_agent.graph (model build + text extraction)."""

import asyncio
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agentic.orchestrator_agent.config.agent_spec import OrchestratorSpec
from agentic.orchestrator_agent.graph import (
    build_graph,
    build_model,
    final_ai_text,
    message_to_text,
)
from agentic.runtime.config import AgentSecrets


def _secrets(tmp_path: Path, **files: str) -> AgentSecrets:
    d = tmp_path / "secrets"
    d.mkdir()
    for name, value in files.items():
        (d / name).write_text(value)
    return AgentSecrets(d)


# -------------------------------------------------------------------------------------------------
# build_model
# -------------------------------------------------------------------------------------------------


class TestBuildModel:
    def test_openai_compat_requires_secrets(self, tmp_path):
        spec = OrchestratorSpec(name="o", model="openai-compat", model_id="m")
        with pytest.raises(RuntimeError, match="openai-compat requires"):
            build_model(spec, _secrets(tmp_path))

    def test_openai_compat_builds_chat_openai(self, tmp_path):
        from langchain_openai import ChatOpenAI

        secrets = _secrets(
            tmp_path,
            agent_model_base_url="http://127.0.0.1:1234/v1",
            agent_model_api_key="lm-studio",
        )
        spec = OrchestratorSpec(
            name="o",
            model="openai-compat",
            model_id="openai/gpt-oss-20b",
            model_settings={"temperature": 0.1},
        )
        model = build_model(spec, secrets)
        assert isinstance(model, ChatOpenAI)
        assert model.model_name == "openai/gpt-oss-20b"


# -------------------------------------------------------------------------------------------------
# message_to_text / final_ai_text
# -------------------------------------------------------------------------------------------------


class TestTextExtraction:
    def test_message_to_text_str(self):
        assert message_to_text("hello") == "hello"

    def test_message_to_text_ai_message(self):
        assert message_to_text(AIMessage(content="hi there")) == "hi there"

    def test_message_to_text_list_content(self):
        msg = AIMessage(
            content=[{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
        )
        assert message_to_text(msg) == "ab"

    def test_final_ai_text_picks_last_ai(self):
        state = {
            "messages": [
                HumanMessage(content="q"),
                AIMessage(content="first"),
                ToolMessage(content="tool out", tool_call_id="1"),
                AIMessage(content="final answer"),
            ]
        }
        assert final_ai_text(state) == "final answer"

    def test_final_ai_text_skips_empty_ai(self):
        state = {
            "messages": [
                AIMessage(content="real answer"),
                AIMessage(content=""),
            ]
        }
        assert final_ai_text(state) == "real answer"

    def test_final_ai_text_empty_state(self):
        assert final_ai_text({"messages": []}) == ""


# -------------------------------------------------------------------------------------------------
# build_graph (no downstream servers -> no network)
# -------------------------------------------------------------------------------------------------


class TestBuildGraph:
    def test_builds_with_no_tools(self, tmp_path):
        secrets = _secrets(
            tmp_path,
            agent_model_base_url="http://127.0.0.1:1234/v1",
            agent_model_api_key="lm-studio",
        )
        spec = OrchestratorSpec(
            name="o", model="openai-compat", model_id="m", a2a_servers=[]
        )
        graph = asyncio.run(build_graph(spec, secrets))
        assert graph is not None
        assert hasattr(graph, "ainvoke")
