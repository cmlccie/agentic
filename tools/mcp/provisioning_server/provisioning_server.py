#!/usr/bin/env python3
"""MCP Provisioning Server."""

import logging
import os
from typing import Annotated, Literal

import typer
from fastmcp import FastMCP
from pydantic import BaseModel, Field

import agentic.logging

logger = logging.getLogger("provisioning_server")


HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))


# -------------------------------------------------------------------------------------------------
# MCP Provisioning Server
# -------------------------------------------------------------------------------------------------

mcp = FastMCP("MCP Provisioning")


# --------------------------------------------------------------------------------------
#  Tools
# --------------------------------------------------------------------------------------


# Define your tools here. For demonstration purposes, we'll define a simple tool.
# In a real-world scenario, these would be more complex and interact with actual systems.

# -----------------------------------------------------------------------------
# Provision Server Tool
# -----------------------------------------------------------------------------


class ProvisionedServer(BaseModel):
    server_name: str = Field(..., description="The name of the provisioned server")
    cpu_cores: int = Field(..., description="Number of CPU cores")
    memory_gb: int = Field(..., description="Memory in GB")
    storage_gb: int = Field(..., description="Storage in GB")
    vlan_id: int = Field(
        ..., description="The ID of the VLAN the server is attached to"
    )
    status: str = Field(default="provisioning", description="Status of the server")


@mcp.tool()
def provision_server(
    server_name: str,
    cpu_cores: int,
    memory_gb: int,
    storage_gb: int,
    vlan_id: int,
) -> ProvisionedServer:
    """Provisions a new server with the specified resources.
    Args:
        server_name: The name of the server to provision.
        cpu_cores: Number of CPU cores.
        memory_gb: Amount of memory in GB.
        storage_gb: Amount of storage in GB.
        vlan_id: The ID of the VLAN to attach the server to.
    Returns:
        ProvisionedServer: Details of the provisioned server.
    """
    logger.info(
        "Provisioning server '%s' with %d CPU cores, %dGB memory, and %dGB storage "
        "on VLAN %d.",
        server_name,
        cpu_cores,
        memory_gb,
        storage_gb,
        vlan_id,
    )
    # Here you would add the logic to provision the server.
    # For demonstration, we'll just return a success message.

    return ProvisionedServer(
        server_name=server_name,
        cpu_cores=cpu_cores,
        memory_gb=memory_gb,
        storage_gb=storage_gb,
        vlan_id=vlan_id,
        status="provisioning",
    )


# -----------------------------------------------------------------------------
# Check VLAN Tool
# -----------------------------------------------------------------------------


@mcp.tool()
def check_vlan(vlan_id: int) -> bool:
    """Checks if a VLAN with the specified ID exists.
    Args:
        vlan_id: The ID of the VLAN to check.
    Returns:
        bool: True if the VLAN exists, False otherwise.
    """
    logger.info("Checking existence of VLAN with ID %d.", vlan_id)
    # Here you would add the logic to check for VLAN existence.
    # For demonstration, we'll assume the VLAN does not exist.

    return False


# -----------------------------------------------------------------------------
# Provision VLAN Tool
# -----------------------------------------------------------------------------


class VLAN(BaseModel):
    vlan_id: int = Field(..., description="The ID of the VLAN")
    name: str = Field(..., description="The name of the VLAN")
    ipv4_cidr: str = Field(..., description="The CIDR block for the VLAN")


@mcp.tool()
def provision_vlan(vlan_id: int, name: str, ipv4_cidr: str) -> VLAN:
    """Provisions a new VLAN.
    Args:
        vlan_id: The ID of the VLAN to provision.
        name: The name of the VLAN.
        ipv4_cidr: The CIDR block for the VLAN.
    Returns:
        VLAN: Details of the provisioned VLAN.
    """
    logger.info(
        "Provisioning VLAN %d '%s' with CIDR block %s.", vlan_id, name, ipv4_cidr
    )
    # Here you would add the logic to provision the VLAN.
    # For demonstration, we'll just return a success message.

    return VLAN(vlan_id=vlan_id, name=name, ipv4_cidr=ipv4_cidr)


# -------------------------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------------------------


def main(
    transport: Annotated[Literal["stdio", "http"], typer.Argument()] = "stdio",
) -> None:
    """Model Context Protocol (MCP) Provisioning Server."""
    agentic.logging.fancy()
    logger.info("Starting %s MCP Provisioning Server", transport)

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
