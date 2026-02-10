# MCP Meraki Server

A Model Context Protocol (MCP) server that wraps Cisco Meraki Dashboard API v1 endpoints for wireless client connectivity troubleshooting.

## Tools

| Tool | Description |
|------|-------------|
| `search_clients` | Search for clients on the network |
| `get_client_details` | Get detailed info for a specific client |
| `get_client_health_scores` | Get wireless health scores (0-100) |
| `get_client_connection_stats` | Get connection flow statistics (assoc/auth/DHCP/DNS/success) |
| `get_client_connectivity_events` | Get connectivity event timeline with severity filtering |

## Container Image

This project is automatically built and published as a multi-architecture container image supporting both `amd64` and `arm64` platforms.

### Available Images

- `ghcr.io/cmlccie/agentic/tools-mcp-meraki-server:latest` - Latest version from main branch
- `ghcr.io/cmlccie/agentic/tools-mcp-meraki-server:main` - Main branch builds
- `ghcr.io/cmlccie/agentic/tools-mcp-meraki-server:v*` - Tagged releases

### Use

```bash
# Run in stdio mode (default)
docker run --rm \
  -e MERAKI_API_KEY="your-api-key" \
  -e MERAKI_NETWORK_ID="your-network-id" \
  ghcr.io/cmlccie/agentic/tools-mcp-meraki-server:latest

# Run in HTTP mode on port 8000
docker run --rm -p 8000:8000 \
  -e MERAKI_API_KEY="your-api-key" \
  -e MERAKI_NETWORK_ID="your-network-id" \
  ghcr.io/cmlccie/agentic/tools-mcp-meraki-server:latest streamable-http
```

## Configuration

### Required Environment Variables

| Variable | Description |
|----------|-------------|
| `MERAKI_API_KEY` | Meraki Dashboard API key |
| `MERAKI_NETWORK_ID` | Target network ID |

### Optional Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Server bind host |
| `PORT` | `8000` | Server bind port |

## Build Locally

```bash
# Build for your current platform
docker build -t tools-mcp-meraki-server .

# Build for multiple platforms
docker buildx build --platform linux/amd64,linux/arm64 -t agentic/tools-mcp-meraki-server:local .
```
