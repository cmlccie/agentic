# Meraki Agent

A Pydantic AI agent that troubleshoots Cisco Meraki wireless client connectivity issues using MCP (Model Context Protocol) tools.

## Features

- **Client Discovery**: Search for clients by description, MAC, IP, or manufacturer
- **Health Assessment**: Check wireless health scores for quick pass/fail assessment
- **Connection Flow Analysis**: Identify where connections fail (association, authentication, DHCP, DNS)
- **Event Timeline**: Review connectivity events with severity filtering
- **OpenAI-Compatible API**: Expose the agent as an API compatible with OpenAI's chat completions endpoint
- **Streaming Support**: Both streaming and non-streaming responses

## Quick Start

Run the Meraki agent using Docker/Podman:

```bash
docker run -d \
  -p 8000:8000 \
  -e OPENAI_API_KEY="your-openai-api-key" \
  -e OPENAI_BASE_URL="your-openai-base-url" \
  -e MODEL_NAME="your-model-name" \
  -e TOOLS_MCP_MERAKI_SERVER_URL="http://meraki-server:8000" \
  ghcr.io/cmlccie/agents-meraki-agent:latest
```

## Prerequisites

- Docker or Podman
- OpenAI API access
- MCP Meraki server running (see [meraki_server](../../tools/mcp/meraki_server/README.md))

## Configuration

### Required Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | Your OpenAI API key |
| `OPENAI_BASE_URL` | OpenAI API base URL |
| `MODEL_NAME` | Model name to use (e.g., `gpt-4`) |
| `TOOLS_MCP_MERAKI_SERVER_URL` | URL of the MCP Meraki server |

### Optional Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Host to bind the API server |
| `PORT` | `8000` | Port to expose the API server |

## Usage

### API Server Mode

Start the agent as an API server:

```bash
docker run -d \
  -p 8000:8000 \
  -e OPENAI_API_KEY="your-openai-api-key" \
  -e OPENAI_BASE_URL="your-openai-base-url" \
  -e MODEL_NAME="your-model-name" \
  -e TOOLS_MCP_MERAKI_SERVER_URL="http://meraki-server:8000" \
  --name meraki-agent \
  ghcr.io/cmlccie/agents-meraki-agent:latest
```

The API will be available at `http://localhost:8000`.

#### API Endpoints

- **GET /**: Root endpoint, returns API status
- **GET /health**: Health check endpoint
- **POST /v1/chat/completions**: OpenAI-compatible chat completions endpoint

#### Example: Using curl

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model-name",
    "messages": [
      {"role": "user", "content": "Show me all clients on the network"}
    ]
  }'
```

#### Example: Using Python OpenAI client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="your-model-name",
    messages=[
        {"role": "user", "content": "Why is client DESKTOP-ABC having connectivity issues?"}
    ]
)
print(response.choices[0].message.content)
```

### CLI Mode

Start an interactive troubleshooting session:

```bash
docker run -it --rm \
  -e OPENAI_API_KEY="your-openai-api-key" \
  -e OPENAI_BASE_URL="your-openai-base-url" \
  -e MODEL_NAME="your-model-name" \
  -e TOOLS_MCP_MERAKI_SERVER_URL="http://meraki-server:8000" \
  ghcr.io/cmlccie/agents-meraki-agent:latest \
  cli
```

Example queries:

- "Show me all clients on the network"
- "Check the health of client DESKTOP-ABC"
- "Why is the client at 00:11:22:33:44:55 having connectivity issues?"
- "Show me connection stats for client k74272e over the last 7 days"

## Development

For local development and contributing to this agent, see the main [repository README](../../README.md) for setup instructions.

## Architecture

```text
Meraki Agent Container
├── FastAPI Application (OpenAI-compatible API)
│   ├── GET / (root endpoint)
│   ├── GET /health (health check)
│   └── POST /v1/chat/completions (chat endpoint)
│       ├── Streaming responses (Server-Sent Events)
│       └── Non-streaming responses (JSON)
├── Pydantic AI Agent
│   ├── OpenAI Model Integration
│   ├── System Prompt Loading
│   └── Tool Registration
├── MCP Server Integration (Meraki Dashboard API)
│   ├── Client Discovery & Details
│   ├── Wireless Health Scores
│   ├── Connection Statistics
│   └── Connectivity Events
└── CLI Interactive Mode
```
