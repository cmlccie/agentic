"""Tests for the MCP Provisioning Server."""

import pytest

from .helpers import call_tool, list_tool_names, load_server


@pytest.fixture(scope="module")
def provisioning():
    return load_server("provisioning_server")


def test_tools_are_registered(provisioning):
    assert list_tool_names(provisioning.mcp) == {
        "provision_server",
        "check_vlan",
        "provision_vlan",
    }


def test_provision_server(provisioning):
    arguments = {
        "server_name": "web-1",
        "cpu_cores": 4,
        "memory_gb": 16,
        "storage_gb": 100,
        "vlan_id": 42,
    }
    result = call_tool(provisioning.mcp, "provision_server", arguments)
    assert result == {**arguments, "status": "provisioning"}


def test_check_vlan(provisioning):
    assert call_tool(provisioning.mcp, "check_vlan", {"vlan_id": 42}) == {
        "result": False
    }


def test_provision_vlan(provisioning):
    arguments = {"vlan_id": 42, "name": "web", "ipv4_cidr": "10.0.42.0/24"}
    assert call_tool(provisioning.mcp, "provision_vlan", arguments) == arguments
