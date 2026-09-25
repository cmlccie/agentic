"""Tests for agentic.simple_agent.interfaces.a2a."""

from pathlib import Path

from fastapi.testclient import TestClient
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from agentic.runtime.config import AgentSecrets, ServerSpec
from agentic.simple_agent.interfaces.a2a import build_a2a_app

# -------------------------------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------------------------------


def _server_spec() -> ServerSpec:
    return ServerSpec.model_validate(
        {
            "agent_card": {
                "display_name": "Test Agent",
                "description": "An agent used in tests.",
                "version": "1.2.3",
                "provider": {"organization": "Acme", "url": "https://acme.test"},
                "skills": [
                    {
                        "id": "echo",
                        "name": "Echo",
                        "description": "Echoes input.",
                        "tags": ["test"],
                    }
                ],
            }
        }
    )


# -------------------------------------------------------------------------------------------------
# build_a2a_app
# -------------------------------------------------------------------------------------------------


class TestBuildA2AApp:
    def test_builds_app_with_worker(self, tmp_path: Path):
        agent = Agent(TestModel(), name="test-agent")
        app = build_a2a_app(agent, _server_spec(), AgentSecrets(tmp_path))
        assert app._agent_worker is not None

    def test_agent_card_served_on_both_paths(self, tmp_path: Path):
        agent = Agent(TestModel(), name="test-agent")
        app = build_a2a_app(
            agent,
            _server_spec(),
            AgentSecrets(tmp_path),
            agent_url="http://agent.test",
        )
        with TestClient(app) as client:
            for path in ("/.well-known/agent-card.json", "/.well-known/agent.json"):
                response = client.get(path)
                assert response.status_code == 200
                card = response.json()
                assert card["name"] == "Test Agent"
                assert card["version"] == "1.2.3"
                assert [s["id"] for s in card["skills"]] == ["echo"]
