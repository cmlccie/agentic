#!/usr/bin/env python3
"""MCP Meraki Server — wraps Cisco Meraki Dashboard API v1 endpoints."""

import logging
import os
from typing import Any, Literal

import requests
import typer
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

import agentic.logging

agentic.logging.fancy()
logger = logging.getLogger("meraki_server")


# -------------------------------------------------------------------------------------------------
# Environment Variables
# -------------------------------------------------------------------------------------------------

MERAKI_API_KEY = os.environ.get("MERAKI_API_KEY")
MERAKI_NETWORK_ID = os.environ.get("MERAKI_NETWORK_ID")

if not MERAKI_API_KEY:
    raise EnvironmentError("Environment variable MERAKI_API_KEY is not set.")
if not MERAKI_NETWORK_ID:
    raise EnvironmentError("Environment variable MERAKI_NETWORK_ID is not set.")

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))


# -------------------------------------------------------------------------------------------------
# MCP Meraki Server
# -------------------------------------------------------------------------------------------------

mcp = FastMCP("MCP Meraki", host=HOST, port=PORT)


# -------------------------------------------------------------------------------------------------
# API Helper
# -------------------------------------------------------------------------------------------------

MERAKI_BASE_URL = "https://api.meraki.com/api/v1"


def _meraki_get(path: str, params: dict[str, Any] | None = None) -> Any:
    """Make an authenticated GET request to the Meraki Dashboard API."""
    url = f"{MERAKI_BASE_URL}{path}"
    headers = {
        "X-Cisco-Meraki-API-Key": MERAKI_API_KEY,
        "Accept": "application/json",
    }
    response = requests.get(url, headers=headers, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


# -------------------------------------------------------------------------------------------------
# Tools
# -------------------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# List Clients
# -----------------------------------------------------------------------------


class Client(BaseModel):
    id: str = Field(description="Client identifier used by other API endpoints")
    mac: str = Field(description="MAC address of the client")
    description: str | None = Field(description="Client hostname or description")
    ip: str | None = Field(description="IPv4 address")
    ip6: str | None = Field(description="IPv6 address")
    vlan: str | None = Field(description="VLAN ID the client is on")
    ssid: str | None = Field(description="Wireless SSID the client is connected to")
    switchport: str | None = Field(description="Switch port the client is connected to")
    status: str = Field(description="Client status (Online or Offline)")
    first_seen: str = Field(description="Timestamp when the client was first seen")
    last_seen: str = Field(description="Timestamp when the client was last seen")
    manufacturer: str | None = Field(description="Device manufacturer")
    os: str | None = Field(description="Operating system")
    usage_sent: float = Field(description="Data sent in bytes")
    usage_recv: float = Field(description="Data received in bytes")


@mcp.tool()
@agentic.logging.log_call(logger)
def list_clients(
    timespan: int = 86400,
    per_page: int = 100,
) -> list[Client]:
    """List clients on the network.
    Args:
        timespan: Timespan in seconds to search for clients (default: 86400 = 24h).
        per_page: Number of clients to return per page (default: 50).

    Returns:
        List of clients found on the network.
    """
    data = _meraki_get(
        f"/networks/{MERAKI_NETWORK_ID}/clients",
        params={"timespan": timespan, "perPage": per_page},
    )
    return [
        Client(
            id=c.get("id", ""),
            mac=c.get("mac", ""),
            description=c.get("description"),
            ip=c.get("ip"),
            ip6=c.get("ip6"),
            vlan=c.get("vlan"),
            ssid=c.get("ssid"),
            switchport=c.get("switchport"),
            status=c.get("status", ""),
            first_seen=c.get("firstSeen", ""),
            last_seen=c.get("lastSeen", ""),
            manufacturer=c.get("manufacturer"),
            os=c.get("os"),
            usage_sent=c.get("usage", {}).get("sent", 0),
            usage_recv=c.get("usage", {}).get("recv", 0),
        )
        for c in data
    ]


# -----------------------------------------------------------------------------
# Get Client Details
# -----------------------------------------------------------------------------


class ClientDetail(BaseModel):
    id: str = Field(description="Client identifier")
    mac: str = Field(description="MAC address of the client")
    description: str | None = Field(description="Client hostname or description")
    ip: str | None = Field(description="IPv4 address")
    ip6: str | None = Field(description="IPv6 address")
    vlan: int | None = Field(description="VLAN ID the client is on")
    ssid: str | None = Field(description="Wireless SSID the client is connected to")
    switchport: str | None = Field(description="Switch port the client is connected to")
    status: str = Field(description="Client status (Online or Offline)")
    first_seen: int = Field(description="Timestamp when the client was first seen")
    last_seen: int = Field(description="Timestamp when the client was last seen")
    manufacturer: str | None = Field(description="Device manufacturer")
    os: str | None = Field(description="Operating system")
    usage_sent: float = Field(description="Data sent in bytes")
    usage_recv: float = Field(description="Data received in bytes")
    wireless_capabilities: str | None = Field(
        description="Wireless capabilities of the client"
    )
    notes: str | None = Field(description="User-assigned notes for the client")


@mcp.tool()
@agentic.logging.log_call(logger)
def get_client_details(client_id: str) -> ClientDetail:
    """Get detailed information for a specific client.

    Args:
        client_id: The client ID returned by search_clients.

    Returns:
        Detailed client information including wireless capabilities and notes.
    """
    c = _meraki_get(f"/networks/{MERAKI_NETWORK_ID}/clients/{client_id}")
    return ClientDetail(
        id=c.get("id", ""),
        mac=c.get("mac", ""),
        description=c.get("description"),
        ip=c.get("ip"),
        ip6=c.get("ip6"),
        vlan=c.get("vlan"),
        ssid=c.get("ssid"),
        switchport=c.get("switchport"),
        status=c.get("status", ""),
        first_seen=c.get("firstSeen", ""),
        last_seen=c.get("lastSeen", ""),
        manufacturer=c.get("manufacturer"),
        os=c.get("os"),
        usage_sent=c.get("usage", {}).get("sent", 0),
        usage_recv=c.get("usage", {}).get("recv", 0),
        wireless_capabilities=c.get("wirelessCapabilities"),
        notes=c.get("notes"),
    )


# -----------------------------------------------------------------------------
# Get Client Connection Stats
# -----------------------------------------------------------------------------


class ClientConnectionStats(BaseModel):
    mac: str = Field(description="MAC address of the client")
    assoc: int = Field(description="Number of association attempts")
    auth: int = Field(description="Number of authentication attempts")
    dhcp: int = Field(description="Number of DHCP attempts")
    dns: int = Field(description="Number of DNS resolution attempts")
    success: int = Field(description="Number of successful connections")


@mcp.tool()
@agentic.logging.log_call(logger)
def get_client_connection_stats(
    client_id: str,
    timespan: int = 86400,
) -> ClientConnectionStats:
    """Get connection statistics for a specific wireless client.

    Shows where in the connection flow things are failing:
    association -> authentication -> DHCP -> DNS -> success.

    Args:
        client_id: The client ID returned by search_clients.
        timespan: Timespan in seconds (default: 86400 = 24h, max: 604800 = 7 days).

    Returns:
        Connection statistics showing counts for each connection step.
    """
    data = _meraki_get(
        f"/networks/{MERAKI_NETWORK_ID}/wireless/clients/{client_id}/connectionStats",
        params={"timespan": timespan},
    )

    stats = data.get("connectionStats", {})

    return ClientConnectionStats(
        mac=data.get("mac", ""),
        assoc=stats.get("assoc", 0),
        auth=stats.get("auth", 0),
        dhcp=stats.get("dhcp", 0),
        dns=stats.get("dns", 0),
        success=stats.get("success", 0),
    )


# -----------------------------------------------------------------------------
# Get Client Connectivity Events
# -----------------------------------------------------------------------------


class ConnectivityEvent(BaseModel):
    occurred_at: str = Field(description="Timestamp when the event occurred")
    band: str | None = Field(description="Wireless band (2.4 GHz or 5 GHz)")
    ssid_number: int | None = Field(description="SSID number")
    type: str = Field(description="Event type")
    subtype: str | None = Field(description="Event subtype with more detail")
    severity: str = Field(description="Event severity (good, info, warn, bad)")
    duration_ms: float | None = Field(description="Event duration in milliseconds")
    channel: int | None = Field(description="Wireless channel")
    rssi: int | None = Field(description="Signal strength in dBm")
    device_serial: str | None = Field(description="Serial number of the Meraki device")


@mcp.tool()
@agentic.logging.log_call(logger)
def get_client_connectivity_events(
    client_id: str,
    per_page: int = 25,
    severity: str | None = None,
) -> list[ConnectivityEvent]:
    """Get connectivity events for a specific wireless client.

    Shows the timeline of connectivity events including associations,
    disassociations, roaming, and connection failures.

    Args:
        client_id: The client ID returned by search_clients.
        per_page: Number of events to return (default: 25).
        severity: Filter by severity level (good, info, warn, bad) or None for all.

    Returns:
        List of connectivity events ordered by time.
    """
    params: dict[str, Any] = {"perPage": per_page}
    if severity:
        params["includedSeverities[]"] = severity

    data = _meraki_get(
        f"/networks/{MERAKI_NETWORK_ID}/wireless/clients/{client_id}/connectivityEvents",
        params=params,
    )

    return [
        ConnectivityEvent(
            occurred_at=e.get("occurredAt", ""),
            band=e.get("band"),
            ssid_number=e.get("ssidNumber"),
            type=e.get("type", ""),
            subtype=e.get("subtype", ""),
            severity=e.get("severity", ""),
            duration_ms=e.get("durationMs"),
            channel=e.get("channel"),
            rssi=e.get("rssi"),
            device_serial=e.get("deviceSerial"),
        )
        for e in data
    ]


# -------------------------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------------------------


def main(
    transport: Literal["stdio", "streamable-http"] = typer.Argument(default="stdio"),
):
    """Model Context Protocol (MCP) Meraki Server."""
    logger.info(f"Starting {transport} MCP Meraki Server")
    mcp.run(transport=transport)


if __name__ == "__main__":
    typer.run(main)
