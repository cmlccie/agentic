# Weather Agent

A sophisticated weather agent built with Pydantic AI that uses MCP (Model Context Protocol) servers for weather information. The agent provides natural language weather assistance with support for location lookup and weather forecasting.

## Features

- **Natural Language Interface**: Chat with the agent using plain English to get weather information
- **Location Intelligence**: Automatically looks up and confirms locations before providing weather data
- **MCP Server Integration**: Connects to MCP weather servers for real-time weather data
- **Configurable**: Supports multiple MCP server endpoints and custom OpenAI configurations
- **Interactive and CLI Modes**: Use interactively or with single commands

## Prerequisites

- Python 3.11+
- OpenAI API access
- MCP weather server (included in the project)

## Environment Variables

Set the following environment variables:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export OPENAI_API_BASE_URL="your-openai-base-url"  # Optional, for custom endpoints
```

For the `openai/gpt-oss-20b` model mentioned in the requirements, you'll need to set the base URL to the appropriate endpoint that serves this model.

## Installation

This agent is part of the larger agentic project. To set up the environment:

```bash
# From the repository root
uv sync
```

## Usage

### Interactive Mode

Start an interactive chat session with the weather agent:

```bash
uv run agents/weather_agent/weather_agent.py chat --interactive
```

This will start an interactive session where you can ask questions like:

- "What's the weather like in San Francisco today?"
- "Will it rain in New York tomorrow?"
- "Give me the weather forecast for London this week"

### Single Query Mode

Get a quick weather response for a single question:

```bash
uv run agents/weather_agent/weather_agent.py chat "What's the weather in Seattle?"
```

### Custom MCP Server

Use a custom MCP server command:

```bash
uv run agents/weather_agent/weather_agent.py chat \
  --mcp-server-command "python custom_weather_server.py stdio" \
  --interactive
```

### Test Mode

Run a quick test to verify the agent is working:

```bash
uv run agents/weather_agent/weather_agent.py test
```

### Configuration

The agent can be configured with:

- **MCP Servers**: List of MCP server endpoints to connect to
- **OpenAI Settings**: API key, base URL, and model selection
- **Model Name**: Default is `openai/gpt-oss-20b`, but can be changed to other compatible models

### MCP Server Configuration

By default, the agent connects to the included weather server:

```python
MCPServerConfig(
    name="weather_server",
    command=["python", "-m", "tools.mcp.weather_server.weather_server", "stdio"],
    env=None
)
```

You can specify multiple MCP servers using the `--mcp-server-command` option multiple times.

## Agent Behavior

The weather agent follows these principles:

1. **Location Confirmation**: Always confirms location details before providing weather information
2. **Weather-Only Focus**: Politely declines non-weather related requests
3. **Date Intelligence**: Handles relative dates ("tomorrow", "this weekend") and specific date ranges
4. **User-Friendly Output**: Presents weather data in conversational, easy-to-understand format

## System Prompt

The agent's behavior is defined by the system prompt in `SYSTEM_PROMPT.md`, which includes:

- Core responsibilities for weather assistance
- Location handling guidelines
- Date interpretation rules
- Weather data presentation standards
- Tool usage instructions

## Architecture

```text
Weather Agent
├── Configuration (MCPServerConfig, WeatherAgentConfig)
├── Dependencies (WeatherAgentDeps)
├── Pydantic AI Agent
│   ├── OpenAI Model Integration
│   ├── System Prompt Loading
│   └── Tool Registration
├── MCP Server Integration
│   ├── Connection Management
│   └── Tool Discovery
└── CLI Interface
    ├── Interactive Mode
    ├── Single Query Mode
    └── Test Mode
```

## Available Tools

The agent automatically discovers and registers tools from connected MCP servers. The default weather server provides:

- `get_weather_forecast`: Get weather forecast for coordinates
- `get_locations`: Search for location information by name

## Error Handling

The agent includes comprehensive error handling for:

- Missing environment variables
- MCP server connection failures
- Tool registration errors
- API request failures
- User input validation

## Development

### Adding New Tools

Tools are automatically discovered from MCP servers. To add new weather-related functionality:

1. Implement the tool in the MCP weather server
2. The agent will automatically register it when connecting

### Custom MCP Servers

To create a custom MCP server for the agent:

1. Implement an MCP server following the MCP specification
2. Ensure it provides weather-related tools
3. Configure the agent to use your server with `--mcp-server-command`

## Logging

The agent uses the shared logging configuration from `agentic.logging`. Logging levels can be controlled with the `--log-level` option:

```bash
uv run agents/weather_agent/weather_agent.py --log-level DEBUG chat --interactive
```

## Contributing

When contributing to the weather agent:

1. Follow the Python instructions in `.github/instructions/python.instructions.md`
2. Use functional programming principles
3. Include comprehensive error handling
4. Add logging for important operations
5. Update documentation for new features

## License

This project is part of the agentic architecture components repository.
