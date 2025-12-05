# Weather Agent

A simple weather agent built with Pydantic AI that uses MCP (Model Context Protocol) servers for weather information. The agent provides natural language weather assistance with support for location lookup and weather forecasting.

## Features

- **Natural Language Interface**: Chat with the agent using plain English to get weather information
- **OpenAI-Compatible API**: Expose the agent as an API compatible with OpenAI's chat completions endpoint
- **Streaming Support**: Both streaming and non-streaming responses are supported
- **Location Intelligence**: Automatically looks up and confirms locations before providing weather data
- **MCP Server Integration**: Connects to MCP weather servers for real-time weather data
- **Configurable**: Supports multiple MCP server endpoints and custom OpenAI configurations
- **Interactive and CLI Modes**: Use interactively or with single commands

## Quick Start

Run the weather agent using Docker/Podman:

```bash
docker run -d \
  -p 8000:8000 \
  -e OPENAI_API_KEY="your-openai-api-key" \
  -e OPENAI_BASE_URL="your-openai-base-url" \
  -e MODEL_NAME="your-model-name" \
  -e TOOLS_MCP_WEATHER_SERVER_URL="http://localhost:8001" \
  ghcr.io/cmlccie/agents-weather-agent:latest
```

## Prerequisites

- Docker or Podman
- OpenAI API access
- MCP weather server running (see [weather_server](../../tools/mcp/weather_server/README.md))

## Configuration

### Required Environment Variables

| Variable                       | Description                       |
| ------------------------------ | --------------------------------- |
| `OPENAI_API_KEY`               | Your OpenAI API key               |
| `OPENAI_BASE_URL`              | OpenAI API base URL               |
| `MODEL_NAME`                   | Model name to use (e.g., `gpt-4`) |
| `TOOLS_MCP_WEATHER_SERVER_URL` | URL of the MCP weather server     |

### Optional Environment Variables

| Variable   | Default   | Description                   |
| ---------- | --------- | ----------------------------- |
| `API_HOST` | `0.0.0.0` | Host to bind the API server   |
| `API_PORT` | `8000`    | Port to expose the API server |

## Usage

### API Server Mode

Start the agent as an API server:

```bash
docker run -d \
  -p 8000:8000 \
  -e OPENAI_API_KEY="your-openai-api-key" \
  -e OPENAI_BASE_URL="your-openai-base-url" \
  -e MODEL_NAME="your-model-name" \
  -e TOOLS_MCP_WEATHER_SERVER_URL="http://localhost:8001" \
  --name weather-agent \
  ghcr.io/cmlccie/agents-weather-agent:latest
```

The API will be available at `http://localhost:8000`.

#### API Endpoints

- **GET /**: Root endpoint, returns API status
- **GET /health**: Health check endpoint
- **POST /v1/chat/completions**: OpenAI-compatible chat completions endpoint

#### Example: Using curl

Non-streaming request:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model-name",
    "messages": [
      {"role": "user", "content": "What is the weather in San Francisco?"}
    ]
  }'
```

Streaming request:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model-name",
    "messages": [
      {"role": "user", "content": "What is the weather in San Francisco?"}
    ],
    "stream": true
  }'
```

#### Example: Using Python OpenAI client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # API key not required for local server
)

# Non-streaming
response = client.chat.completions.create(
    model="your-model-name",
    messages=[
        {"role": "user", "content": "What's the weather in New York?"}
    ]
)
print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="your-model-name",
    messages=[
        {"role": "user", "content": "What's the weather in London?"}
    ],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Developer Console Mode

Start an interactive chat session with the weather agent:

```bash
docker run -it --rm \
  -e OPENAI_API_KEY="your-openai-api-key" \
  -e OPENAI_BASE_URL="your-openai-base-url" \
  -e MODEL_NAME="your-model-name" \
  -e TOOLS_MCP_WEATHER_SERVER_URL="http://localhost:8001" \
  ghcr.io/cmlccie/agents-weather-agent:latest \
  --developer-console
```

This will start an interactive session where you can ask questions like:

- "What's the weather like in San Francisco today?"
- "Will it rain in New York tomorrow?"
- "Give me the weather forecast for London this week"

Type `exit` or `quit` to end the session.

## Container Images

Pre-built container images are available at `ghcr.io/cmlccie/agents-weather-agent`:

- `latest` - Latest build from main branch
- `<version>` - Specific release version (e.g., `1.0.0`)
- `<branch>-<sha>` - Specific commit from a branch

## Development

For local development and contributing to this agent, see the main [repository README](../../README.md) for setup instructions.

## System Prompt

The agent's behavior is defined by the system prompt in `system_prompt.md`, which includes:

- Core responsibilities for weather assistance
- Location handling guidelines
- Date interpretation rules
- Weather data presentation standards
- Tool usage instructions

## Architecture

```text
Weather Agent Container
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
├── MCP Server Integration
│   ├── Connection Management
│   └── Tool Discovery
└── Developer Console
    └── Interactive Mode
```
