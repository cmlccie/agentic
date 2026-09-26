# Orchestrator Agent

A config-driven agent that coordinates a team of specialist agents over the Agent2Agent (A2A) protocol. It is the same runtime as the [simple agent](../simple_agent/README.md) — same OpenAI-compatible and A2A interfaces, configuration files, streaming, and hot reload — with its `agent.yaml` declaring remote agents as `A2AAgent` capabilities instead of (or alongside) MCP tool servers. Each remote agent becomes one tool the model can call; while a delegated agent works, its activity streams back to the orchestrator's caller, attributed to that agent, followed by the synthesized answer.

This README covers what is specific to orchestration. For the interfaces, configuration reference, storage, deployment, and CLI, see the [simple agent README](../simple_agent/README.md).

## How delegation works

```yaml
# agent.yaml
name: orchestrator-agent
model: vllm:Qwen/Qwen3-32B
instructions: |
  Coordinate the specialist agents available to you as tools...
capabilities:
  - A2AAgent:
      url: http://weather-agent:8000/a2a
  - A2AAgent:
      url: http://network-agent:8000/a2a
      name: network
      headers:
        Authorization: Bearer ${NETWORK_AGENT_TOKEN}
```

For each `A2AAgent`:

1. The remote agent's card is fetched from `<url>/.well-known/agent-card.json` the first time the agent runs, and cached. Its name becomes the tool name (`Weather Agent` → `weather_agent`) and its description and skills become the tool description. If the card can't be fetched, the tool is simply absent and the fetch is retried at most every 30 seconds, so an unavailable agent never stops the orchestrator from starting or answering.
2. When the model calls the tool, the request is sent with streaming. The remote agent's thinking, tool calls, results, and its own delegations are relayed into the orchestrator's activity stream, attributed by agent name.
3. The remote answer (a direct Message, or a Task's artifacts) is returned to the model. Failures come back to the model as text it can reason about — unreachable agent, `failed`, `rejected`, or `canceled` task, or a task that stopped early — so one flaky agent never aborts the whole run.

Follow-up calls within the same conversation reuse the remote agent's `contextId`, so the remote agent remembers earlier turns. If a remote task asks for more input (`input-required`), the model sees the question and its next call continues that same task. Cancelling the orchestrator's request (closing the stream, or `CancelTask` on the orchestrator's task) cancels the remote task too.

| `A2AAgent` argument | Default                      | Purpose                                                        |
| ------------------- | ---------------------------- | -------------------------------------------------------------- |
| `url`               | (required)                   | Base URL of the remote agent's A2A endpoint                    |
| `name`              | from the card                | Tool name                                                      |
| `description`       | from the card and its skills | Tool description                                               |
| `headers`           | none                         | HTTP headers such as `Authorization`; `${NAME}` references work |
| `timeout`           | 300                          | Seconds to wait between streamed events from the remote agent  |

Any A2A 1.0 or 0.3 agent works as a delegate, not only agents of this runtime. Agents of this runtime add the activity extension, so their inner work shows up in the orchestrator's stream.

## What callers see

Over the OpenAI API, the reasoning channel shows the orchestrator's own thinking, its delegation calls, and each delegated agent's activity, then the answer:

```text
reasoning_content:  The user wants the weather in Paris; the weather agent can help.
reasoning_content:  → weather_agent({"request": "What is the weather in Paris today?"})
reasoning_content:  [weather_agent] 💭 I should call the forecast tool.
reasoning_content:  [weather_agent] → get_forecast({"city": "Paris"})
reasoning_content:  [weather_agent] ← get_forecast: {"temp_c": 18, "sky": "sunny"}
reasoning_content:  ← weather_agent: It's 18°C and sunny in Paris.
content:            It's 18°C and sunny in Paris today.
```

Over A2A, delegation promotes the exchange to a Task, and the same activity arrives as `working` status updates marked with the `urn:agentic:a2a:activity:v1` extension; each update's `source` names the agent it came from.

## Quick start

```bash
mkdir -p config secrets
cp images/orchestrator_agent/{agent,server}.yaml config/
# edit config/agent.yaml: set the model and add an A2AAgent per downstream agent
printf 'http://host.docker.internal:8000/v1' > secrets/model.base_url

docker run --rm -p 8080:8000 \
  -v "$PWD/config:/etc/agent/config:ro" \
  -v "$PWD/secrets:/etc/agent/secrets:ro" \
  ghcr.io/cmlccie/agentic/orchestrator-agent:latest serve --public-url http://localhost:8080

curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "orchestrator-agent", "stream": true,
       "messages": [{"role": "user", "content": "What is the weather in Paris?"}]}'
```

Downstream agents must advertise a reachable address in their cards (their `--public-url`), because A2A clients send requests to the URL in the card.

## Deploy with Terraform

[`modules/terraform-kubernetes-orchestrator-agent`](../../modules/terraform-kubernetes-orchestrator-agent/README.md) renders `agent.yaml` and `server.yaml` from files into a ConfigMap, the secrets into a Secret, and a hardened Deployment and Service.

## Build

```bash
make python-base-image                                       # agentic/python:local
make orchestrator-agent BASE_IMAGE=agentic/python:local      # agentic/orchestrator-agent:local
```

## Upgrading from the LangGraph orchestrator

Earlier versions ran the orchestrator on LangChain/LangGraph. Existing configurations keep working: `a2a_servers` entries become `A2AAgent` capabilities (`id` becomes `name`), `model: openai-compat` with `model_id` becomes `model: vllm:<model_id>`, and `broker.backend: postgres` becomes the `sql` store using the existing `task_broker.database_url` secret. Each translation logs a warning showing the new form.
