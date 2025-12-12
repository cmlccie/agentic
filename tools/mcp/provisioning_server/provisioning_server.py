#!/usr/bin/env python3
"""MCP Weather Server."""

import logging
import os
from typing import Literal

import typer
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

import agentic.logging

agentic.logging.fancy()
logger = logging.getLogger("provisioning_server")


HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))


# -------------------------------------------------------------------------------------------------
# MCP Provisioning Server
# -------------------------------------------------------------------------------------------------

mcp = FastMCP("MCP Provisioning", host=HOST, port=PORT)


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
        f"Provisioning server '{server_name}' with {cpu_cores} CPU cores, "
        f"{memory_gb}GB memory, and {storage_gb}GB storage on VLAN '{vlan_id}'."
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
    logger.info(f"Checking existence of VLAN with ID {vlan_id}.")
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
    logger.info(f"Provisioning VLAN {vlan_id} '{name}' with CIDR block {ipv4_cidr}.")
    # Here you would add the logic to provision the VLAN.
    # For demonstration, we'll just return a success message.

    return VLAN(vlan_id=vlan_id, name=name, ipv4_cidr=ipv4_cidr)


# -------------------------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------------------------


def main(
    transport: Literal["stdio", "streamable-http"] = typer.Argument(default="stdio"),
):
    """Model Context Protocol (MCP) Provisioning Server."""
    logger.info(f"Starting {transport} MCP Provisioning Server")
    mcp.run(transport=transport)


if __name__ == "__main__":
    typer.run(main)
