# MCP Weather Server

A Model Context Protocol (MCP) server that provides weather information.

## Container Image

This project is automatically built and published as a multi-architecture container image supporting both `amd64` and `arm64` platforms.

### Available Images

- `ghcr.io/cmlccie/agentic/tools-mcp-weather-server:latest` - Latest version from main branch
- `ghcr.io/cmlccie/agentic/tools-mcp-weather-server:main` - Main branch builds
- `ghcr.io/cmlccie/agentic/tools-mcp-weather-server:v*` - Tagged releases

### Use

```bash
# Run in stdio mode (default)
docker run --rm ghcr.io/cmlccie/agentic/tools-mcp-weather-server:latest

# Run in HTTP mode on port 8000
docker run --rm -p 8000:8000 ghcr.io/cmlccie/agentic/tools-mcp-weather-server:latest streamable-http
```

## Build Locally

```bash
# Build for your current platform
docker build -t tools-mcp-weather-server .

# Build for multiple platforms
docker buildx build --platform linux/amd64,linux/arm64 -t agentic/tools-mcp-weather-server:local .
```
