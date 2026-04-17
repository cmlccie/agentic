# MCP Provisioning Server

A Model Context Protocol (MCP) server that provisions IT resources.

## Tools

- `provision_server`: Provision a new server with CPU, memory, storage, and VLAN settings.
- `check_vlan`: Check whether a VLAN ID currently exists.
- `provision_vlan`: Provision a new VLAN with a name and CIDR block.

## Container Image

This project is automatically built and published as a multi-architecture container image supporting both `amd64` and `arm64` platforms.

### Available Images

- `ghcr.io/cmlccie/agentic/tools-mcp-provisioning-server:latest`: Latest version from main branch
- `ghcr.io/cmlccie/agentic/tools-mcp-provisioning-server:main`: Main branch builds
- `ghcr.io/cmlccie/agentic/tools-mcp-provisioning-server:v*`: Tagged releases

### Use

```bash
# Run in stdio mode (default)
docker run --rm ghcr.io/cmlccie/agentic/tools-mcp-provisioning-server:latest

# Run in HTTP mode on port 8000
docker run --rm -p 8000:8000 ghcr.io/cmlccie/agentic/tools-mcp-provisioning-server:latest http
```

The HTTP MCP endpoint is exposed at `/mcp`.

## Configuration

### Required Environment Variables

- None

### Optional Environment Variables

- `HOST`: `0.0.0.0` by default. HTTP server bind host.
- `PORT`: `8000` by default. HTTP server bind port.

## Build Locally

```bash
# Build for your current platform
docker build -t tools-mcp-provisioning-server .

# Build for multiple platforms
docker buildx build --platform linux/amd64,linux/arm64 -t agentic/tools-mcp-provisioning-server:local .
```
