# MCP Weather Server

A Model Context Protocol (MCP) server that provides weather information.

## Container Image

This project is automatically built and published as a multi-architecture container image supporting both `amd64` and `arm64` platforms.

### Available Images

- `ghcr.io/cmlccie/agentic/tools-mcp-weather-server:latest` - Latest version from main branch
- `ghcr.io/cmlccie/agentic/tools-mcp-weather-server:main` - Main branch builds
- `ghcr.io/cmlccie/agentic/tools-mcp-weather-server:v*` - Tagged releases

### Usage

#### Running with podman

```bash
# Run in stdio mode (default)
podman run --rm ghcr.io/cmlccie/agentic/tools-mcp-weather-server:latest

# Run in HTTP mode on port 8000
podman run --rm -p 8000:8000 ghcr.io/cmlccie/agentic/tools-mcp-weather-server:latest streamable-http
```

## Building Locally

```bash
# Build for your current platform
podman build -t tools-mcp-weather-server .

# Build for multiple platforms
podman buildx build --platform linux/amd64,linux/arm64 -t tools-mcp-weather-server .
```

## Development

This MCP server is built with:

- Python 3.13
- Alpine Linux base image
- Multi-architecture support (amd64, arm64)

See `requirements.txt` for Python dependencies and `weather_server.py` for the server implementation.
