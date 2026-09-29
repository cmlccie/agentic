#!/usr/bin/env python3
"""MCP Meraki Server — wraps Cisco Meraki Dashboard API v1 endpoints."""

import logging
import os
from typing import Annotated, Any, Literal
from urllib.parse import quote

import httpx
import typer
from fastmcp import FastMCP
from pydantic import BaseModel, Field

import agentic.logging

logger = logging.getLogger("meraki_server")


# -------------------------------------------------------------------------------------------------
# Environment Variables
# -------------------------------------------------------------------------------------------------

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))

HTTP_TIMEOUT = httpx.Timeout(float(os.environ.get("HTTP_TIMEOUT_S", "30")))


def _require_env(name: str) -> str:
    """Return a required environment variable, failing with a clear message."""
    value = os.environ.get(name)
    if not value:
        raise EnvironmentError(f"Environment variable {name} is not set.")
    return value


# -------------------------------------------------------------------------------------------------
# MCP Meraki Server
# -------------------------------------------------------------------------------------------------

mcp = FastMCP("MCP Meraki")


# -------------------------------------------------------------------------------------------------
# API Helper
# -------------------------------------------------------------------------------------------------

MERAKI_BASE_URL = "https://api.meraki.com/api/v1"


def _http_client(
    transport: httpx.AsyncBaseTransport | None = None,
) -> httpx.AsyncClient:
    """Create an authenticated Meraki Dashboard API client.

    Args:
        transport: Optional transport override (used by tests to mock HTTP).
    """
    return httpx.AsyncClient(
        base_url=MERAKI_BASE_URL,
        headers={
            "X-Cisco-Meraki-API-Key": _require_env("MERAKI_API_KEY"),
            "Accept": "application/json",
        },
        timeout=HTTP_TIMEOUT,
        transport=transport,
    )


async def _network_get(path: str, params: dict[str, Any] | None = None) -> Any:
    """GET a path under the configured network and return the decoded JSON body."""
    network_id = quote(_require_env("MERAKI_NETWORK_ID"), safe="")
    async with _http_client() as client:
        response = await client.get(f"/networks/{network_id}{path}", params=params)
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
async def list_clients(
    timespan: int = 86400,
    per_page: int = 100,
) -> list[Client]:
    """List clients on the network.

    Args:
        timespan: Timespan in seconds to search for clients (default: 86400 = 24h).
        per_page: Maximum number of clients to return (default: 100).

    Returns:
        List of clients found on the network.
    """
    data = await _network_get(
        "/clients", params={"timespan": timespan, "perPage": per_page}
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
    first_seen: int | None = Field(
        description="Unix timestamp when the client was first seen"
    )
    last_seen: int | None = Field(
        description="Unix timestamp when the client was last seen"
    )
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
async def get_client_details(client_id: str) -> ClientDetail:
    """Get detailed information for a specific client.

    Args:
        client_id: The client ID returned by list_clients.

    Returns:
        Detailed client information including wireless capabilities and notes.
    """
    c = await _network_get(f"/clients/{quote(client_id, safe='')}")
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
        first_seen=c.get("firstSeen"),
        last_seen=c.get("lastSeen"),
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
async def get_client_connection_stats(
    client_id: str,
    timespan: int = 86400,
) -> ClientConnectionStats:
    """Get connection statistics for a specific wireless client.

    Shows where in the connection flow things are failing:
    association -> authentication -> DHCP -> DNS -> success.

    Args:
        client_id: The client ID returned by list_clients.
        timespan: Timespan in seconds (default: 86400 = 24h, max: 604800 = 7 days).

    Returns:
        Connection statistics showing counts for each connection step.
    """
    data = await _network_get(
        f"/wireless/clients/{quote(client_id, safe='')}/connectionStats",
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
async def get_client_connectivity_events(
    client_id: str,
    per_page: int = 25,
    severity: str | None = None,
) -> list[ConnectivityEvent]:
    """Get connectivity events for a specific wireless client.

    Shows the timeline of connectivity events including associations,
    disassociations, roaming, and connection failures.

    Args:
        client_id: The client ID returned by list_clients.
        per_page: Number of events to return (default: 25).
        severity: Filter by severity level (good, info, warn, bad) or None for all.

    Returns:
        List of connectivity events ordered by time.
    """
    params: dict[str, Any] = {"perPage": per_page}
    if severity:
        params["includedSeverities[]"] = severity

    data = await _network_get(
        f"/wireless/clients/{quote(client_id, safe='')}/connectivityEvents",
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
    transport: Annotated[Literal["stdio", "http"], typer.Argument()] = "stdio",
) -> None:
    """Model Context Protocol (MCP) Meraki Server."""
    agentic.logging.fancy()

    # Fail fast at startup rather than on the first tool call.
    for name in ("MERAKI_API_KEY", "MERAKI_NETWORK_ID"):
        _require_env(name)

    logger.info("Starting %s MCP Meraki Server", transport)

    match transport:
        case "stdio":
            mcp.run(transport=transport)
        case "http":
            mcp.run(transport=transport, host=HOST, port=PORT)
        case _:
            raise typer.BadParameter(
                "Transport must be one of: stdio, http.",
                param_hint="transport",
            )


if __name__ == "__main__":
    typer.run(main)
