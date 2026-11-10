"""Tests for SPIFFE workload identity (`agentic.agent.workload_auth`).

The Workload API is replaced by a callable returning a fixed JWT-SVID, and the
token endpoint and gateway by `httpx2.MockTransport` handlers, so the whole
exchange runs without a SPIRE agent or network.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

import httpx2
import pytest
from openai import AsyncOpenAI
from pydantic_ai.models.openai import OpenAIChatModel

import agentic.agent.spec as spec_module
from agentic.agent import workload_auth
from agentic.agent.config import ConfigError, Secrets
from agentic.agent.spec import MCP, resolve_model
from agentic.agent.workload_auth import (
    CLIENT_ASSERTION_TYPE,
    WorkloadAuth,
    WorkloadAuthSettings,
    auth_for,
    load_settings,
)

TOKEN_URL = (
    "http://keycloak.auth.svc.cluster.local/realms/lab/protocol/openid-connect/token"
)
AUDIENCE = "https://keycloak.example/realms/lab"
GATEWAY = "http://litellm.ai-gateway.svc.cluster.local/v1"
SETTINGS = WorkloadAuthSettings(
    token_url=TOKEN_URL, audience=AUDIENCE, base_url=GATEWAY
)


@pytest.fixture(autouse=True)
def clean_state(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Start every test with workload auth off and no shared auth instance."""
    for name in (
        workload_auth.TOKEN_URL_ENV,
        workload_auth.AUDIENCE_ENV,
        workload_auth.BASE_URL_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    workload_auth._shared_auth.cache_clear()
    yield
    workload_auth._shared_auth.cache_clear()


def enable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(workload_auth.TOKEN_URL_ENV, TOKEN_URL)
    monkeypatch.setenv(workload_auth.AUDIENCE_ENV, AUDIENCE)
    monkeypatch.setenv(workload_auth.BASE_URL_ENV, GATEWAY)


class TokenEndpoint:
    """A fake token endpoint that issues ``token-1``, ``token-2``, ..."""

    def __init__(self, expires_in: int = 300) -> None:
        self.expires_in = expires_in
        self.forms: list[dict[str, list[str]]] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        assert str(request.url) == TOKEN_URL
        self.forms.append(parse_qs(request.content.decode()))
        return httpx2.Response(
            200,
            json={
                "access_token": f"token-{len(self.forms)}",
                "expires_in": self.expires_in,
            },
        )


def make_auth(endpoint: TokenEndpoint, audiences: list[str]) -> WorkloadAuth:
    def fetch_svid(audience: str) -> str:
        audiences.append(audience)
        return "the-jwt-svid"

    return WorkloadAuth(
        SETTINGS, fetch_svid=fetch_svid, transport=httpx2.MockTransport(endpoint)
    )


async def gateway_calls(
    auth: WorkloadAuth, statuses: list[int], count: int
) -> list[str | None]:
    """Make `count` gateway requests; return every Authorization header it saw."""
    seen: list[str | None] = []
    remaining = list(statuses)

    def gateway(request: httpx2.Request) -> httpx2.Response:
        seen.append(request.headers.get("Authorization"))
        return httpx2.Response(remaining.pop(0) if remaining else 200)

    async with httpx2.AsyncClient(
        auth=auth, transport=httpx2.MockTransport(gateway)
    ) as client:
        for _ in range(count):
            await client.get(f"{GATEWAY}/models")
    return seen


class TestSettings:
    def test_off_by_default(self) -> None:
        assert load_settings() is None
        assert auth_for(GATEWAY) is None

    def test_enabled_from_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        enable(monkeypatch)
        assert load_settings() == SETTINGS

    def test_incomplete_environment_is_an_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(workload_auth.TOKEN_URL_ENV, TOKEN_URL)
        with pytest.raises(ConfigError, match="AGENT_WORKLOAD_AUTH_AUDIENCE"):
            load_settings()

    def test_token_only_goes_to_the_gateway_origin(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        enable(monkeypatch)
        on_gateway = auth_for(
            "http://litellm.ai-gateway.svc.cluster.local/customer_database/mcp"
        )
        assert isinstance(on_gateway, WorkloadAuth)
        assert on_gateway is auth_for(GATEWAY)  # one shared token cache
        assert auth_for("http://other.example/mcp") is None
        assert auth_for("https://litellm.ai-gateway.svc.cluster.local/v1") is None


class TestTokenExchange:
    def test_exchanges_the_svid_and_caches_the_token(self) -> None:
        endpoint, audiences = TokenEndpoint(), []
        seen = asyncio.run(gateway_calls(make_auth(endpoint, audiences), [], 3))

        assert seen == ["Bearer token-1"] * 3
        assert audiences == [AUDIENCE]
        assert endpoint.forms == [
            {
                "grant_type": ["client_credentials"],
                "client_assertion_type": [CLIENT_ASSERTION_TYPE],
                "client_assertion": ["the-jwt-svid"],
            }
        ]

    def test_refreshes_a_token_close_to_expiry(self) -> None:
        # A lifetime inside the refresh margin is stale as soon as it is issued.
        endpoint = TokenEndpoint(expires_in=10)
        seen = asyncio.run(gateway_calls(make_auth(endpoint, []), [], 2))
        assert seen == ["Bearer token-1", "Bearer token-2"]

    def test_retries_once_with_a_new_token_on_401(self) -> None:
        endpoint = TokenEndpoint()
        seen = asyncio.run(gateway_calls(make_auth(endpoint, []), [401], 1))
        assert seen == ["Bearer token-1", "Bearer token-2"]

    def test_failed_exchange_raises_with_the_response_body(self) -> None:
        def denied(request: httpx2.Request) -> httpx2.Response:
            return httpx2.Response(401, json={"error": "invalid_client"})

        auth = WorkloadAuth(
            SETTINGS,
            fetch_svid=lambda audience: "svid",
            transport=httpx2.MockTransport(denied),
        )
        with pytest.raises(httpx2.HTTPStatusError, match="invalid_client"):
            asyncio.run(gateway_calls(auth, [], 1))


class TestModelClient:
    def test_static_key_is_used_when_workload_auth_is_off(self, tmp_path: Path) -> None:
        (tmp_path / "model.base_url").write_text(GATEWAY)
        (tmp_path / "model.api_key").write_text("sk-static")
        model = resolve_model("vllm:agent-model", Secrets(tmp_path))
        assert isinstance(model, OpenAIChatModel)
        assert model.client.api_key == "sk-static"
        assert model.client._client.auth is None

    def test_model_client_uses_workload_auth(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        enable(monkeypatch)
        (tmp_path / "model.base_url").write_text(GATEWAY)
        model = resolve_model("vllm:agent-model", Secrets(tmp_path))
        assert isinstance(model, OpenAIChatModel)
        assert model.client._client.auth is auth_for(GATEWAY)

    def test_access_token_overrides_the_sdk_api_key_header(self) -> None:
        seen: list[str | None] = []

        def gateway(request: httpx2.Request) -> httpx2.Response:
            seen.append(request.headers.get("Authorization"))
            return httpx2.Response(200, json={"object": "list", "data": []})

        async def list_models() -> None:
            http_client = httpx2.AsyncClient(
                auth=make_auth(TokenEndpoint(), []),
                transport=httpx2.MockTransport(gateway),
            )
            client = AsyncOpenAI(
                base_url=GATEWAY,
                api_key=workload_auth.PLACEHOLDER_API_KEY,
                http_client=http_client,
            )
            await client.models.list()
            await client.close()

        asyncio.run(list_models())
        assert seen == ["Bearer token-1"]


class TestMCP:
    @pytest.fixture
    def toolset_kwargs(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
        """Record the arguments the gateway `MCPToolset` is built with."""
        recorded: dict[str, Any] = {}

        def record(url: str, **kwargs: Any) -> str:
            recorded.update(kwargs, url=url)
            return "toolset"

        monkeypatch.setattr(spec_module, "MCPToolset", record)
        return recorded

    def test_gateway_mcp_server_uses_workload_auth(
        self, monkeypatch: pytest.MonkeyPatch, toolset_kwargs: dict[str, Any]
    ) -> None:
        enable(monkeypatch)
        url = "http://litellm.ai-gateway.svc.cluster.local/customer_database/mcp"
        assert MCP.from_spec(url, id="db")._build_local(url) == "toolset"
        assert toolset_kwargs["auth"] is auth_for(GATEWAY)
        assert toolset_kwargs["headers"] is None

    def test_other_mcp_servers_are_untouched(
        self, monkeypatch: pytest.MonkeyPatch, toolset_kwargs: dict[str, Any]
    ) -> None:
        enable(monkeypatch)
        url = "http://other.example/mcp"
        MCP.from_spec(url, headers={"Authorization": "Bearer x"})._build_local(url)
        assert toolset_kwargs == {}

    def test_mcp_is_untouched_when_workload_auth_is_off(
        self, toolset_kwargs: dict[str, Any]
    ) -> None:
        url = "http://litellm.ai-gateway.svc.cluster.local/customer_database/mcp"
        MCP.from_spec(url, headers={"Authorization": "Bearer k"})._build_local(url)
        assert toolset_kwargs == {}
