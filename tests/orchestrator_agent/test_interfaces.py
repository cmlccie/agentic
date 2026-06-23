"""Tests for the orchestrator A2A card builder and OpenAI message conversion."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agentic.orchestrator_agent.interfaces.a2a import build_agent_card
from agentic.orchestrator_agent.interfaces.openai_compat import _messages_to_langchain
from agentic.runtime.config import ServerSpec

# -------------------------------------------------------------------------------------------------
# build_agent_card
# -------------------------------------------------------------------------------------------------


class TestBuildAgentCard:
    def _spec(self, **agent_card) -> ServerSpec:
        base = {"display_name": "Orchestrator", "description": "Routes tasks."}
        base.update(agent_card)
        return ServerSpec.model_validate({"agent_card": base})

    def test_basic_card(self):
        card = build_agent_card(self._spec(), "http://localhost:8000")
        assert card.name == "Orchestrator"
        assert card.description == "Routes tasks."
        assert card.capabilities.streaming is True
        assert card.capabilities.push_notifications is True

    def test_interface_url_includes_a2a_prefix(self):
        card = build_agent_card(self._spec(), "http://host:8000/")
        assert card.supported_interfaces[0].url == "http://host:8000/a2a"

    def test_skills_and_provider(self):
        spec = self._spec(
            skills=[
                {
                    "id": "route",
                    "name": "Route",
                    "description": "Route work to agents",
                    "tags": ["routing"],
                }
            ],
            provider={"organization": "Acme", "url": "http://acme.com"},
        )
        card = build_agent_card(spec, "http://localhost:8000")
        assert card.skills[0].id == "route"
        assert card.provider.organization == "Acme"


# -------------------------------------------------------------------------------------------------
# _messages_to_langchain
# -------------------------------------------------------------------------------------------------


class TestMessagesToLangchain:
    def test_role_mapping(self):
        msgs = [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        out = _messages_to_langchain(msgs)
        assert isinstance(out[0], SystemMessage)
        assert isinstance(out[1], HumanMessage)
        assert isinstance(out[2], AIMessage)

    def test_unknown_role_defaults_to_human(self):
        out = _messages_to_langchain([{"role": "tool", "content": "x"}])
        assert isinstance(out[0], HumanMessage)

    def test_multipart_content_joined(self):
        out = _messages_to_langchain(
            [{"role": "user", "content": [{"text": "a"}, {"text": "b"}]}]
        )
        assert out[0].content == "a b"
