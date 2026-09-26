# Agentic Architecture Components

Building blocks for agentic AI systems on your own infrastructure: config-driven agents, MCP tool servers, and the images and Terraform modules to run them on Kubernetes. Agents are [Pydantic AI](https://pydantic.dev/docs/ai/) agents defined entirely by YAML; they use tools through MCP servers, delegate to other agents over the Agent2Agent (A2A) protocol, and are served through an OpenAI-compatible API and a spec-compliant A2A server. While an agent works, its activity — thinking, tool calls and results, and delegated agents' activity — streams to the caller, followed by the final answer. Self-hosted models on vLLM, SGLang, and NVIDIA NIM are first-class.

## Components

| Component                                                                       | Description                                                             |
| ------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| [Simple agent](images/simple_agent/README.md)                                   | A config-driven agent (tools via MCP); the runtime reference            |
| [Orchestrator agent](images/orchestrator_agent/README.md)                       | The same runtime configured to delegate to A2A agents                   |
| [Python base image](images/python/README.md)                                    | Base image with the `agentic` package installed                         |
| [Weather MCP server](tools/mcp/weather_server/README.md)                        | Weather forecasts and location lookup                                   |
| [Provisioning MCP server](tools/mcp/provisioning_server/README.md)              | Provisions IT resources                                                 |
| [Customer database MCP server](tools/mcp/customer_database_server/README.md)    | Read-only SQL over a PostgreSQL customer database                       |
| [Meraki MCP server](tools/mcp/meraki_server/README.md)                          | Cisco Meraki wireless client troubleshooting                            |
| [AIOps MCP server](tools/mcp/aiops_server/README.md)                            | Validation and benchmarking of model-serving deployments (vLLM, SGLang, NIM) |
| [Orchestrator Terraform module](modules/terraform-kubernetes-orchestrator-agent/README.md) | Deploys an agent of this runtime to Kubernetes                |

## An agent in two files

```yaml
# agent.yaml — what the agent is
name: weather-agent
model: vllm:Qwen/Qwen3-32B
instructions: You answer questions about the weather. Use your tools.
capabilities:
  - MCP:
      url: http://weather-mcp:8000/mcp
```

```yaml
# server.yaml — how it is served
agent_card:
  display_name: Weather Agent
  description: Answers weather questions.
```

```bash
uv run simple-agent serve --config-dir ./weather-agent --port 8080
```

Point an orchestrator at it with one capability:

```yaml
capabilities:
  - A2AAgent:
      url: http://weather-agent:8080/a2a
```

## Development

```bash
make setup    # create .venv with all dependency groups
make check    # lint + format check
make test     # run the test suite
make help     # all targets, including image builds
```

See [AGENTS.md](AGENTS.md) for the architecture and conventions.
