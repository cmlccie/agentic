# Orchestrator Agent

A config-driven **LangGraph** supervisor that orchestrates tasks across a team of
downstream **Agent2Agent (A2A)** agents. Like the [Simple Agent](../simple_agent/README.md),
it is fully defined by configuration (`agent.yaml` + `server.yaml`) with no bespoke
Python code, runs Kubernetes-native with hot-reload, and is built on the same shared
runtime. Unlike the simple-agent (whose tools are MCP servers), the orchestrator's
"tools" are remote A2A agents: it fetches each configured agent's card, exposes it to
the supervisor model as a delegation tool (the LangChain `create_agent`
subagents-as-tools pattern), and synthesizes their responses. It simultaneously exposes
an **OpenAI-compatible REST API** under `/v1/` and a first-class **A2A server** under
`/a2a/` supporting the full task lifecycle.

## Interfaces and Endpoints

### `orchestrator-agent serve` endpoints

| Method & Path                         | Description                                          |
| ------------------------------------- | --------------------------------------------------- |
| `GET /health/live`                    | Liveness probe — always 200 while the process runs  |
| `GET /health/ready`                   | Readiness probe — 503 during drain/reload, else 200 |
| `GET /v1/models`                      | OpenAI-compatible model list                        |
| `POST /v1/chat/completions`           | OpenAI-compatible chat completions (stream or not)  |
| `GET /a2a/.well-known/agent-card.json`| A2A agent card (with `/agent.json` alias)           |
| `POST /a2a`                           | A2A JSON-RPC endpoint (`message/send`, `message/stream`, `tasks/get`, `tasks/cancel`, push-notification config) |

The A2A interface supports the full task lifecycle: short message responses,
long-running tasks with streamed status updates (SSE via `message/stream`), artifacts,
asynchronous polling via `tasks/get`, and push notifications.

## Local Testing with Docker

### 1. Create a secrets directory

```bash
mkdir -p ./secrets

# For an OpenAI-compatible provider (LM Studio, Ollama, vLLM, etc.)
printf 'http://host.docker.internal:1234/v1' > ./secrets/agent_model_base_url
printf 'lm-studio' > ./secrets/agent_model_api_key   # any non-empty placeholder

# For Anthropic (set model: anthropic:claude-sonnet-4-6 in agent.yaml)
printf 'sk-ant-...' > ./secrets/anthropic_api_key

# For OpenAI (set model: openai:gpt-4o in agent.yaml)
printf 'sk-...' > ./secrets/openai_api_key
```

### 2. Start the container

```bash
docker run --rm -p 8000:8000 \
  -v "$(pwd)/secrets:/etc/agent/secrets:ro" \
  ghcr.io/cmlccie/agentic/orchestrator-agent:latest
```

Mount your own `agent.yaml`/`server.yaml` over `/etc/agent/config` to point the
orchestrator at real downstream agents.

### 3. Check readiness

```bash
curl -s localhost:8000/health/ready
# {"status":"ready","in_flight":0}
```

### 4. Test the OpenAI API

```bash
curl -s localhost:8000/v1/models

curl -s localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"orchestrator-agent","messages":[{"role":"user","content":"Plan my day."}]}'
```

### 5. Test the A2A interface

Fetch the agent card and send a task with any A2A client (e.g. the `a2a-sdk`):

```bash
curl -s localhost:8000/a2a/.well-known/agent.json
```

```python
import asyncio, httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.helpers import get_artifact_text, new_text_message
from a2a.types import Role, SendMessageRequest

async def main():
    async with httpx.AsyncClient(timeout=120) as http:
        card = await A2ACardResolver(http, base_url="http://localhost:8000/a2a").get_agent_card()
        client = ClientFactory(ClientConfig(httpx_client=http, streaming=True)).create(card)
        req = SendMessageRequest(message=new_text_message("Plan my day.", role=Role.ROLE_USER))
        async for resp in client.send_message(req):
            if resp.HasField("artifact_update"):
                print(get_artifact_text(resp.artifact_update.artifact))
        await client.close()

asyncio.run(main())
```

## Container Images

Build locally from the repository root:

```bash
make python-base-image     # base image carries all Python dependencies
make orchestrator-agent    # builds agentic/orchestrator-agent:local
```

Published images: `ghcr.io/cmlccie/agentic/orchestrator-agent`.

## Configuration Reference

### `agent.yaml` — Agent identity and downstream agents

```yaml
name: orchestrator-agent
description: "An orchestrator that delegates tasks across specialist A2A agents"

# ── Model ──────────────────────────────────────────────────────────────────────
# Option A: first-class provider (set the matching API key secret file)
# model: anthropic:claude-sonnet-4-6
# model: openai:gpt-4o
#
# Option B: OpenAI-compatible custom endpoint (LM Studio, Ollama, vLLM, etc.)
# Requires secret files: agent_model_base_url, agent_model_api_key
model: openai-compat
model_id: local-model

instructions: |
  You are an orchestrator that coordinates a team of specialist agents...

model_settings:
  temperature: 0.2
  max_tokens: 4096

# ── Downstream A2A agents ───────────────────────────────────────────────────────
# Each agent's card is fetched at startup/reload to derive its delegation tool's
# name, description, and skills. Unreachable agents are skipped (degraded mode).
# Header values may reference secret files via ${SECRET_KEY}.
a2a_servers:
  - url: http://weather-agent/a2a
    headers:
      Authorization: "Bearer ${WEATHER_AGENT_TOKEN}"
  - url: http://network-agent/a2a
```

### `server.yaml` — Serving infrastructure

```yaml
agent_card:
  display_name: "Orchestrator Agent"
  description: "..."
  version: "1.0.0"
  provider:
    organization: ""
    url: ""
  skills: []

broker:
  backend: memory   # "memory" | "postgres"

interfaces:
  a2a: true
  openai_compat: true

reload:
  drain_timeout: 30  # seconds to drain in-flight requests before forcing reload
```

> **`broker.backend`:**
> - `memory` (default) — in-process A2A task store; ephemeral, single-replica.
> - `postgres` — persistent SQL task store (a2a-sdk `DatabaseTaskStore`) suitable
>   for multi-replica deployments. Requires the `agent_database_url` secret
>   (`postgresql+asyncpg://user:pass@host:5432/dbname`); the `tasks` table is
>   created automatically on first use.
>
> `redis` is not supported for the orchestrator's A2A task store (the a2a-sdk
> store is SQL-based); it warns and falls back to in-memory — use `postgres`.

### Secrets reference

Secrets are read from files under `/etc/agent/secrets/<key>` (Kubernetes Secret volume)
on every access — rotated values are picked up without a restart.

| Secret file              | Used for                                                        |
| ------------------------ | --------------------------------------------------------------- |
| `agent_model_base_url`   | OpenAI-compatible endpoint base URL (`model: openai-compat`)    |
| `agent_model_api_key`    | OpenAI-compatible endpoint API key (`model: openai-compat`)     |
| `anthropic_api_key`      | Anthropic API key (`model: anthropic:...`)                      |
| `openai_api_key`         | OpenAI API key (`model: openai:...`)                            |
| `<custom>`               | Any token referenced as `${CUSTOM}` in an `a2a_servers` header  |

## Kubernetes Deployment

Mount `agent.yaml`/`server.yaml` from a ConfigMap at `/etc/agent/config` and secrets
from a Secret at `/etc/agent/secrets`. **Mount whole directories — never use `subPath`**,
which bypasses Kubernetes AtomicWriter and breaks hot-reload.

```yaml
volumeMounts:
  - name: config
    mountPath: /etc/agent/config
    readOnly: true
  - name: secrets
    mountPath: /etc/agent/secrets
    readOnly: true
```

Wire probes to `/health/live` (liveness) and `/health/ready` (readiness); set the
readiness `failureThreshold` high enough to cover `reload.drain_timeout` so a reload
drains traffic instead of restarting the pod. Pass the externally reachable URL via
`--agent-url` so the published A2A agent card advertises the correct endpoint.

## Hot-Reload

Editing the mounted `agent.yaml`/`server.yaml` (or rotating a secret) triggers a
zero-downtime reload: the readiness probe goes 503 (DRAINING → RELOADING), in-flight
requests drain, the supervisor graph is rebuilt — re-fetching every downstream agent
card so added/removed `a2a_servers` take effect — then readiness returns to 200.

```bash
# Trigger a reload manually (e.g. after rotating a secret) without changing files:
kubectl exec -n agents deploy/my-orchestrator -- kill -HUP 1
```

> Interface topology (which interfaces are mounted, the published agent card) is fixed
> at startup; toggling interfaces or changing the card requires a pod restart. The model,
> instructions, and downstream agent set are hot-reloaded.

## CLI Reference

```text
orchestrator-agent serve [OPTIONS]

  --host         Bind host (default: 0.0.0.0)
  --port         Bind port (default: 8000)
  --config-dir   Config directory (default: /etc/agent/config)
  --secrets-dir  Secrets directory (default: /etc/agent/secrets)
  --agent-url    Public URL advertised in the A2A agent card (default: http://localhost:8000)
  --log-level    Log level (default: info)
```
