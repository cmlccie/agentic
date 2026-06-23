"""Tests for agentic.runtime.health."""

from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from agentic.runtime.health import router
from agentic.runtime.reload import AgentState, ReloadCoordinator

# -------------------------------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------------------------------


def _make_app(state: AgentState, in_flight: int = 0) -> FastAPI:
    app = FastAPI()
    coord = ReloadCoordinator()
    coord.state = state
    coord.in_flight_count = in_flight

    app_state = MagicMock()
    app_state.coordinator = coord
    app.state.app_state = app_state

    app.include_router(router)
    return app


# -------------------------------------------------------------------------------------------------
# Liveness
# -------------------------------------------------------------------------------------------------


class TestLiveness:
    def test_always_200(self):
        client = TestClient(_make_app(AgentState.RUNNING))
        assert client.get("/health/live").status_code == 200

    def test_200_when_draining(self):
        client = TestClient(_make_app(AgentState.DRAINING))
        assert client.get("/health/live").status_code == 200

    def test_200_when_reloading(self):
        client = TestClient(_make_app(AgentState.RELOADING))
        assert client.get("/health/live").status_code == 200

    def test_body_has_alive_status(self):
        client = TestClient(_make_app(AgentState.RUNNING))
        assert client.get("/health/live").json() == {"status": "alive"}


# -------------------------------------------------------------------------------------------------
# Readiness
# -------------------------------------------------------------------------------------------------


class TestReadiness:
    def test_200_when_running(self):
        client = TestClient(_make_app(AgentState.RUNNING))
        assert client.get("/health/ready").status_code == 200

    def test_body_when_running(self):
        client = TestClient(_make_app(AgentState.RUNNING, in_flight=2))
        data = client.get("/health/ready").json()
        assert data["status"] == "ready"
        assert data["in_flight"] == 2

    def test_503_when_draining(self):
        client = TestClient(
            _make_app(AgentState.DRAINING), raise_server_exceptions=False
        )
        assert client.get("/health/ready").status_code == 503

    def test_503_when_reloading(self):
        client = TestClient(
            _make_app(AgentState.RELOADING), raise_server_exceptions=False
        )
        assert client.get("/health/ready").status_code == 503

    def test_body_when_draining(self):
        client = TestClient(
            _make_app(AgentState.DRAINING, in_flight=3), raise_server_exceptions=False
        )
        data = client.get("/health/ready").json()
        assert data["status"] == "not_ready"
        assert data["state"] == "DRAINING"
        assert data["in_flight"] == 3

    def test_body_when_reloading(self):
        client = TestClient(
            _make_app(AgentState.RELOADING), raise_server_exceptions=False
        )
        data = client.get("/health/ready").json()
        assert data["state"] == "RELOADING"
