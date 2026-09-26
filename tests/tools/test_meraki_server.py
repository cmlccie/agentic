"""Tests for the MCP Meraki Server (HTTP mocked, no network)."""

from functools import partial

import httpx
import pytest
from fastmcp.exceptions import ToolError

from .helpers import call_tool, list_tool_names, load_server

CLIENT_PAYLOAD = {
    "id": "k74272e",
    "mac": "22:33:44:55:66:77",
    "description": "laptop",
    "status": "Online",
    "firstSeen": "2026-09-01T00:00:00Z",
    "lastSeen": "2026-09-26T00:00:00Z",
    "usage": {"sent": 10, "recv": 20},
}


@pytest.fixture
def meraki(monkeypatch):
    monkeypatch.setenv("MERAKI_API_KEY", "test-key")
    monkeypatch.setenv("MERAKI_NETWORK_ID", "N_123")
    module = load_server("meraki_server")
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/connectionStats"):
            return httpx.Response(
                200, json={"mac": "aa", "connectionStats": {"assoc": 3, "success": 2}}
            )
        return httpx.Response(200, json=[CLIENT_PAYLOAD])

    monkeypatch.setattr(
        module,
        "_http_client",
        partial(module._http_client, transport=httpx.MockTransport(handler)),
    )
    module.requests = requests
    return module


def test_import_does_not_require_credentials(monkeypatch):
    monkeypatch.delenv("MERAKI_API_KEY", raising=False)
    monkeypatch.delenv("MERAKI_NETWORK_ID", raising=False)
    module = load_server("meraki_server")
    with pytest.raises(ToolError, match="MERAKI_NETWORK_ID"):
        call_tool(module.mcp, "list_clients")


def test_tools_are_registered(meraki):
    assert list_tool_names(meraki.mcp) == {
        "list_clients",
        "get_client_details",
        "get_client_connection_stats",
        "get_client_connectivity_events",
    }


def test_list_clients(meraki):
    result = call_tool(meraki.mcp, "list_clients", {"per_page": 5})
    assert result["result"][0]["id"] == "k74272e"
    assert result["result"][0]["usage_recv"] == 20

    request = meraki.requests[0]
    assert request.url.path == "/api/v1/networks/N_123/clients"
    assert request.url.params["perPage"] == "5"
    assert request.headers["X-Cisco-Meraki-API-Key"] == "test-key"


def test_client_id_is_path_escaped(meraki):
    result = call_tool(
        meraki.mcp, "get_client_connection_stats", {"client_id": "../../orgs"}
    )
    assert result["assoc"] == 3
    assert meraki.requests[0].url.raw_path.startswith(
        b"/api/v1/networks/N_123/wireless/clients/..%2F..%2Forgs/"
    )
