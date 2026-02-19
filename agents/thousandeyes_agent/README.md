# ThousandEyes Agent

A Pydantic AI agent that troubleshoots application connectivity issues using ThousandEyes MCP tools. The agent follows a structured workflow — checking for outages and alerts, reviewing existing tests, running instant tests, and analyzing network paths — to diagnose connectivity problems.

## Features

- **Structured Troubleshooting Workflow**: Follows a systematic approach from outage detection through path analysis
- **Instant Test Polling**: Runs asynchronous ThousandEyes instant tests and polls for results
- **Focused Tool Subset**: Uses 10 of 30 available ThousandEyes MCP tools, keeping LLM context concise
- **Multiple Interfaces**: CLI, Agent2Agent (A2A), and OpenAI-compatible REST API

## Quick Start

Run locally with `uv run`:

```bash
uv run agents/thousandeyes_agent/thousandeyes_agent.py cli
```

Run as a container:

```bash
docker run -d \
  -p 8000:8000 \
  -e OPENAI_API_KEY="your-openai-api-key" \
  -e OPENAI_BASE_URL="your-openai-base-url" \
  -e MODEL_NAME="your-model-name" \
  -e THOUSANDEYES_MCP_URL="http://thousandeyes-mcp:8000/mcp" \
  -e THOUSANDEYES_TOKEN="your-thousandeyes-token" \
  ghcr.io/cmlccie/agents-thousandeyes-agent:latest
```

## Configuration

### Required Environment Variables

| Variable              | Description                                    |
| --------------------- | ---------------------------------------------- |
| `OPENAI_API_KEY`      | Your OpenAI API key                            |
| `OPENAI_BASE_URL`     | OpenAI API base URL                            |
| `MODEL_NAME`          | Model name to use (e.g., `gpt-4`)             |
| `THOUSANDEYES_MCP_URL` | URL of the ThousandEyes MCP server             |
| `THOUSANDEYES_TOKEN`  | ThousandEyes OAuth bearer token                |

### Optional Environment Variables

| Variable | Default   | Description                   |
| -------- | --------- | ----------------------------- |
| `HOST`   | `0.0.0.0` | Host to bind the API server   |
| `PORT`   | `8000`    | Port to expose the API server |

## Usage

### CLI Mode

Interactive troubleshooting session:

```bash
uv run agents/thousandeyes_agent/thousandeyes_agent.py cli
```

Example queries:
- "Why can't users reach https://app.example.com?"
- "Check connectivity to api.example.com from US agents."
- "Are there any outages affecting our services?"
- "Run a network test to 10.0.0.1 from our enterprise agents."

### OpenAI-Compatible API Mode

```bash
uv run agents/thousandeyes_agent/thousandeyes_agent.py openai-api
```

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model-name",
    "messages": [
      {"role": "user", "content": "Check connectivity to api.example.com"}
    ]
  }'
```

### Agent2Agent (A2A) Mode

```bash
uv run agents/thousandeyes_agent/thousandeyes_agent.py a2a http://localhost:8000
```

## ThousandEyes MCP Tool Subset

The agent uses a filtered subset of ThousandEyes MCP tools focused on connectivity troubleshooting:

| Tool | Purpose |
|------|---------|
| `list_cloud_enterprise_agents` | Find agents to run tests from |
| `list_network_app_synthetics_tests` | Find existing scheduled tests |
| `run_http_server_instant_test` | Test HTTP/HTTPS connectivity |
| `run_agent_to_server_instant_test` | Test network-layer connectivity |
| `run_dns_server_instant_test` | Test DNS resolution |
| `get_instant_test_metrics` | Poll for instant test results |
| `get_network_app_synthetics_metrics` | Get metrics from scheduled tests |
| `get_full_path_visualization` | Analyze network paths hop-by-hop |
| `list_alerts` | Check active alerts |
| `search_outages` | Check known outages |
