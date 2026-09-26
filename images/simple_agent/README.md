# Simple Agent

A config-driven [Pydantic AI](https://pydantic.dev/docs/ai/) agent served over an OpenAI-compatible API and the Agent2Agent (A2A) protocol. The agent is defined entirely by two files — `agent.yaml` (a Pydantic AI agent spec: model, instructions, and capabilities such as MCP tool servers) and `server.yaml` (the A2A agent card and serving options) — so a new agent is a new pair of files, not new code. While the agent works, its activity (thinking, tool calls and results, delegated agents' activity) streams to the caller in the OpenAI reasoning channel and as A2A task status updates, followed by the final answer.

This README is the reference for the agent runtime (`src/agentic/agent`), which the [orchestrator agent](../orchestrator_agent/README.md) shares.

- [Interfaces](#interfaces)
- [Quick start](#quick-start)
- [OpenAI-compatible API](#openai-compatible-api)
- [A2A](#a2a)
- [Configuration](#configuration)
- [Hot reload](#hot-reload)
- [Observability](#observability)
- [Kubernetes deployment](#kubernetes-deployment)
- [Build](#build)
- [CLI reference](#cli-reference)

## Interfaces

| Path                                                       | Purpose                                                            |
| ---------------------------------------------------------- | ------------------------------------------------------------------ |
| `GET /v1/models`                                           | The agent, listed as one model                                     |
| `POST /v1/chat/completions`                                | OpenAI Chat Completions (streaming and non-streaming)              |
| `POST /a2a`                                                | A2A JSON-RPC (A2A 1.0 methods plus the A2A 0.3 method names)       |
| `GET /.well-known/agent-card.json`                         | A2A agent card (also under `/a2a/` and at the legacy `agent.json`) |
| `GET /health/live`                                         | Liveness probe                                                     |
| `GET /health/ready`                                        | Readiness probe (reports the configuration generation)             |

Either interface can be switched off in `server.yaml`; both are on by default.

## Quick start

Run the published image against a vLLM (or SGLang, or NIM) server:

```bash
mkdir -p secrets
printf 'http://host.docker.internal:8000/v1' > secrets/model.base_url
printf 'not-needed' > secrets/model.api_key

docker run --rm -p 8080:8000 \
  -v "$PWD/secrets:/etc/agent/secrets:ro" \
  ghcr.io/cmlccie/agentic/simple-agent:latest serve --public-url http://localhost:8080
```

The default `agent.yaml` uses `model: vllm:local-model`; mount your own config directory at `/etc/agent/config` to change the model name, instructions, or tools. Then:

```bash
curl -s localhost:8080/health/ready
# {"status":"ready","agent":"simple-agent","generation":1}

curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "simple-agent", "messages": [{"role": "user", "content": "Hello!"}]}'
```

For local development without a container, point `--config-dir` at any directory with the two files:

```bash
uv run simple-agent serve --config-dir images/simple_agent --secrets-dir ./secrets --port 8080
uv run simple-agent chat --config-dir images/simple_agent    # terminal chat
uv run simple-agent web --config-dir images/simple_agent     # browser chat UI on :8080
```

## OpenAI-compatible API

The API follows the OpenAI Chat Completions contract, so Open WebUI, LibreChat, the `openai` SDK, and LangChain work unchanged. Conversations are stateless: send the full history each time.

**Non-streaming** responses put the answer in `message.content` and the agent's activity in `message.reasoning_content`, with token `usage`.

**Streaming** responses (`"stream": true`) are Server-Sent Events in this order:

1. A role chunk
2. `delta.reasoning_content` chunks while the agent works
3. The answer in `delta.content`
4. A chunk with `finish_reason: "stop"`
5. A usage chunk, when requested with `stream_options.include_usage`
6. `data: [DONE]`

While a slow tool runs, `: keep-alive` comment lines keep proxies from closing the connection. If the client disconnects, the agent run (and any tool call or delegated task in progress) is cancelled.

```text
data: {"choices":[{"delta":{"reasoning_content":"The user wants the weather in Paris."}}], ...}
data: {"choices":[{"delta":{"reasoning_content":"\n→ get_forecast({\"city\": \"Paris\"})\n"}}], ...}
data: {"choices":[{"delta":{"reasoning_content":"\n← get_forecast: {\"temp_c\": 18, \"sky\": \"sunny\"}\n"}}], ...}
data: {"choices":[{"delta":{"content":"It's 18°C and sunny in Paris."}}], ...}
data: {"choices":[{"delta":{},"finish_reason":"stop"}], ...}
data: [DONE]
```

`reasoning_content` is the field Open WebUI (collapsible "Thinking" panel), LibreChat, and vLLM-style clients read. Reasoning that clients send back in assistant messages (including Open WebUI's `<think>` blocks) is stripped so activity isn't fed back to the model.

**Errors** use the OpenAI error shape (`{"error": {"message", "type", "code"}}`). A failure after streaming has started is sent as an `error` event, which the `openai` SDK raises as `APIError`, so a failed run never looks like an empty answer. Agent failures carry `x-should-retry: false` so SDK retries don't re-run tools with side effects. Error messages name the exception type; details are only logged.

Other request details:

- `temperature`, `top_p`, `max_tokens`/`max_completion_tokens`, `seed`, `stop`, and the penalties are applied on top of the agent's `model_settings`.
- Images (`image_url` parts with URLs or data URIs) are passed to the model.
- Client-side `tools` and other unrecognized fields are ignored, and `n` must be 1.
- A conversation id in the `X-Conversation-Id` header (Open WebUI's `X-OpenWebUI-Chat-Id` also works) or a `conversation_id` body field ties the turns of a chat together. An orchestrator uses it to continue each delegated agent's A2A context across turns.
- Malformed messages get a `400` naming the problem, including assistant `tool_calls` without matching `tool` results.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="unused")
stream = client.chat.completions.create(
    model="simple-agent",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta
    if reasoning := getattr(delta, "reasoning_content", None):
        print(reasoning, end="", flush=True)
    if delta.content:
        print(delta.content, end="", flush=True)
```

## A2A

The A2A interface is built on the official [a2a-sdk](https://github.com/a2aproject/a2a-python) and implements A2A 1.0 over JSON-RPC, including streaming (`SendStreamingMessage`), `GetTask`, `ListTasks`, `CancelTask`, `SubscribeToTask`, and (optionally) push notifications. The A2A 0.3 method names (`message/send`, `message/stream`, `tasks/get`, ...) are accepted on the same endpoint for older clients, and the card advertises both protocol versions.

### Messages and tasks

A2A lets an agent answer with a direct **Message** (a quick reply, nothing to track) or a **Task** (tracked work with status updates, artifacts, cancellation, and polling). With the default `a2a.response_mode: auto`, each request gets the shape its work needs:

- A quick answer that uses no tools is returned as a Message.
- As soon as the agent calls a tool, or the run takes longer than `promote_after_seconds`, the exchange becomes a Task.
- A Task goes `submitted` → `working` (status updates carrying the agent's activity) → the answer as a `response` artifact → `completed`.
- A failed run becomes a `failed` task, and `CancelTask` stops the run and any tool call or delegated task in progress.

Requests that set `returnImmediately`, or continue an existing task, always get a Task. `response_mode: message` and `response_mode: task` force one shape.

Follow-up messages with the same `contextId` continue the conversation: the agent keeps the message history for each context, whether the earlier turns were Messages or Tasks.

### Activity extension

Working status updates are marked with the agent card extension `urn:agentic:a2a:activity:v1`. Their text part is a readable line for any A2A client, and their metadata carries the structured activity:

```json
{"urn:agentic:a2a:activity:v1": {"kind": "tool_call", "text": "get_forecast({\"city\": \"Paris\"})", "source": [], "tool": "get_forecast", "call_id": "call_1"}}
```

`kind` is one of `thinking`, `note`, `tool_call`, `tool_result`, `status`, or `error`; `source` attributes activity relayed from delegated agents. The orchestrator's `A2AAgent` capability uses this extension to relay a worker's activity to its own callers.

### Storage

- **`memory`** (default) keeps tasks and conversation histories in the process, with no external dependencies. Both are bounded (`max_tasks`, `max_contexts`; oldest finished entries are evicted first) and are lost on restart.
- **`sql`** keeps them in any SQLAlchemy async database. Put the DSN in the secret file named by `a2a.store.database_url_secret` (default `a2a.database_url`), for example `postgresql+asyncpg://user:pass@postgres:5432/agents` or `sqlite+aiosqlite:////data/agent.db` for single-node persistence. Tables are created on first use.

A task's live event stream (for `SendStreamingMessage`, `SubscribeToTask`, and `CancelTask`) lives in the replica that runs it, even with the `sql` store. Run the A2A interface as one replica, or give the Service session affinity (`sessionAffinity: ClientIP`) when scaling out. The OpenAI API is stateless and scales freely.

### Example

```bash
curl -s localhost:8080/a2a -H 'Content-Type: application/json' -H 'A2A-Version: 1.0' -d '{
  "jsonrpc": "2.0", "id": 1, "method": "SendMessage",
  "params": {"message": {"messageId": "m1", "role": "ROLE_USER", "parts": [{"text": "Hello!"}]}}
}'
```

From Python, use the a2a-sdk client (`A2ACardResolver`, `ClientFactory`), or point an orchestrator's `A2AAgent` capability at `http://<host>:8000/a2a`.

## Configuration

The container reads `/etc/agent/config/{agent,server}.yaml` and secret files from `/etc/agent/secrets` (override with `--config-dir` / `--secrets-dir` or `AGENT_CONFIG_DIR` / `AGENT_SECRETS_DIR`). Unknown keys in either file are errors, so typos fail loudly. Keys from earlier versions (`model: openai-compat` with `model_id`, orchestrator `a2a_servers`, `broker`, `interfaces.openai_compat`, `interfaces.ui`, `reload`) are still accepted and translated, with a warning that shows the new form.

### agent.yaml

A standard [Pydantic AI agent spec](https://pydantic.dev/docs/ai/agent-spec/):

```yaml
name: weather-agent            # also the model id on /v1/models
description: Answers weather questions
model: vllm:Qwen/Qwen3-32B
instructions: |
  You answer questions about the weather. Use your tools for current data.
model_settings:
  temperature: 0.2
  max_tokens: 4096
capabilities:
  - MCP:
      url: http://weather-mcp:8000/mcp
      headers:
        Authorization: Bearer ${WEATHER_MCP_TOKEN}
  - Thinking:
      effort: medium
  - RepairToolArguments
```

**Models.** Use any Pydantic AI model string. For self-hosted OpenAI-compatible servers — vLLM, SGLang, and NVIDIA NIM — use `vllm:<served-model-name>`: it speaks Chat Completions, parses `reasoning_content` into thinking, and picks model-family profiles (Qwen, DeepSeek, Llama, Mistral, gpt-oss, ...). Its endpoint comes from the `model.base_url` / `model.api_key` secret files, or the `VLLM_BASE_URL` / `VLLM_API_KEY` environment variables. Hosted providers (`anthropic:`, `openai:`, `openai-chat:`, `google-gla:`) read their standard API-key environment variables.

**Capabilities.** Besides the Pydantic AI built-ins (`MCP`, `Thinking`, `WebSearch`, `WebFetch`, `ToolSearch`, `PrefixTools`, `Instrumentation`, ...), `agent.yaml` can declare:

| Capability                | Purpose                                                                                    |
| ------------------------- | ------------------------------------------------------------------------------------------ |
| `A2AAgent`                | Delegate to a remote A2A agent (see the [orchestrator](../orchestrator_agent/README.md))    |
| `RepairToolArguments`     | Repair malformed JSON tool arguments (common with self-hosted models) before validation     |
| `ToolOutputLimits`        | Truncate or summarize oversized tool results at the source                                  |
| `ClampOversizedMessages`  | Clamp any single oversized message part                                                     |
| `ClearToolResults`        | Replace old tool results with placeholders as history grows                                 |
| `SlidingWindowCompaction` | Keep the most recent messages within a message or token budget                              |
| `SummarizingCompaction`   | Summarize older history with a model                                                        |
| `WarnNearLimits`          | Warn the model as it approaches iteration or token limits                                   |
| `Planning`                | Give the model a task plan it maintains while working                                       |
| `SpendLimits`             | Enforce token or cost budgets                                                               |

All but `A2AAgent` come from [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/); see its docs for their arguments.

MCP servers are connected for each run rather than held open, so a restarted MCP server never leaves the agent with a dead session.

**Secret references.** `${NAME}` anywhere in the capability arguments is replaced with the secret file `NAME` (or its lowercase form), falling back to the environment variable `NAME`. An unresolved reference is a configuration error.

### server.yaml

```yaml
agent_card:                      # required: the A2A agent card
  display_name: Weather Agent
  description: Answers weather questions.
  version: "1.0.0"
  icon_url: ""
  documentation_url: ""
  provider: {organization: Example, url: https://example.com}
  skills:
    - id: forecast
      name: Forecasts
      description: Current conditions and forecasts
      tags: [weather]
      examples: ["Will it rain in Oslo tomorrow?"]

interfaces:
  openai: true
  a2a: true

a2a:
  response_mode: auto            # auto | message | task
  promote_after_seconds: 2.0
  store:
    backend: memory              # memory | sql
    database_url_secret: a2a.database_url
    max_tasks: 10000             # memory only
    max_contexts: 1000           # memory only
    max_history_messages: 200    # history kept per A2A context
  push_notifications:
    enabled: false               # the server calls client-supplied webhooks when enabled
    allowed_hosts: []            # e.g. [hooks.example.com, .example.com]; empty refuses private IPs

streaming:
  activity: summary              # off | summary | trace
  thinking: true
  max_args_chars: 200            # summary mode truncation
  max_result_chars: 300
  redact_keys: [password, secret, token, api_key, apikey, authorization, credential]
  heartbeat_seconds: 15

auth:
  bearer_token_secret: null      # e.g. api_token
```

**Streaming verbosity.** `summary` shows tool names with short argument and result previews, `trace` shows them in full, and `off` sends only the answer. Tool arguments and results are always redacted: values under keys containing any `redact_keys` entry become `***`. Thinking can be hidden separately with `thinking: false`.

**Push notifications.** While disabled, every push URL is refused, including configs sent inline with `SendMessage`. When enabled, only `http`/`https` URLs are accepted; with `allowed_hosts` the host must match an entry (a leading dot matches subdomains), and without it loopback, private, and link-local IP addresses and `localhost` are refused. Hostnames are not resolved, so set `allowed_hosts` to restrict destinations fully.

**Authentication.** When `auth.bearer_token_secret` names a secret file, `/v1/*` and `/a2a` require `Authorization: Bearer <token>` and the agent card declares the bearer scheme. Health probes and the agent card stay public. If the secret file is missing, requests are refused (fail closed).

### Secrets reference

| File                          | Used for                                                                         |
| ----------------------------- | -------------------------------------------------------------------------------- |
| `model.base_url`              | Endpoint of a self-hosted OpenAI-compatible model server (`vllm:` models)        |
| `model.api_key`               | API key for that endpoint                                                        |
| `a2a.database_url`            | DSN for `a2a.store.backend: sql`                                                 |
| `<name from auth>`            | Bearer token clients must send, when `auth.bearer_token_secret` is set           |
| `<name>` referenced as `${NAME}` | Anything in `agent.yaml` capability arguments (MCP and A2A headers, URLs, ...) |

The earlier names `openai_compatible.base_url`, `openai_compatible.api_key`, and `task_broker.database_url` are still read.

## Hot reload

The agent watches its config and secrets directories and reloads when anything changes (and on `SIGHUP`). A reload builds a new agent from the files and swaps it in:

- Requests already in progress finish with the agent they started with, and every new request uses the new one.
- There is nothing to drain and no downtime; readiness stays up.
- If the new configuration is invalid, the error is logged and the previous configuration keeps serving. Fix the files and the next change reloads.
- Everything reloads live except `a2a.store`, which needs a restart.

```bash
kubectl exec -n agents deploy/weather-agent -- kill -HUP 1   # e.g. after rotating a secret
```

At startup an invalid configuration is fatal: the process exits with a clear error rather than serving with a broken config.

## Observability

- **Logs** go to stderr, one line per record when not attached to a terminal (so Loki, Elasticsearch, and Cloud Logging can parse them), and with Rich formatting in a terminal.
- **Traces** are exported over OTLP/HTTP when `OTEL_EXPORTER_OTLP_ENDPOINT` (or `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`) is set. They cover every agent run, model request, and tool call (OpenTelemetry GenAI conventions), plus inbound HTTP requests and outbound HTTP calls. Trace context propagates to delegated A2A agents, so an orchestrator and its workers appear in one trace. Use the standard `OTEL_*` variables (`OTEL_SERVICE_NAME`, `OTEL_EXPORTER_OTLP_HEADERS`, ...) to configure the exporter.

## Kubernetes deployment

```text
ConfigMap  weather-agent-config   agent.yaml, server.yaml  → /etc/agent/config  (no subPath)
Secret     weather-agent-secrets  model.base_url, ...      → /etc/agent/secrets (no subPath)
Deployment weather-agent          ghcr.io/cmlccie/agentic/simple-agent
Service    weather-agent          :8000  /v1  /a2a  /health
```

Mount the ConfigMap and Secret as whole directories. **Do not use `subPath`**: Kubernetes updates mounted ConfigMaps and Secrets by atomically swapping a symlink, and `subPath` mounts never see the update, so hot reload wouldn't happen.

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: weather-agent-config
  namespace: agents
data:
  agent.yaml: |
    name: weather-agent
    model: vllm:Qwen/Qwen3-32B
    instructions: You answer questions about the weather.
    capabilities:
      - MCP:
          url: http://weather-mcp:8000/mcp
  server.yaml: |
    agent_card:
      display_name: Weather Agent
      description: Answers weather questions.
---
apiVersion: v1
kind: Secret
metadata:
  name: weather-agent-secrets
  namespace: agents
stringData:
  model.base_url: http://vllm.inference.svc.cluster.local:8000/v1
  model.api_key: not-needed
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: weather-agent
  namespace: agents
spec:
  replicas: 1
  selector:
    matchLabels: {app: weather-agent}
  template:
    metadata:
      labels: {app: weather-agent}
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 10000
        seccompProfile: {type: RuntimeDefault}
      containers:
        - name: agent
          image: ghcr.io/cmlccie/agentic/simple-agent:latest
          args: [serve, --public-url=http://weather-agent.agents.svc.cluster.local:8000]
          ports: [{name: http, containerPort: 8000}]
          securityContext:
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: true
            capabilities: {drop: [ALL]}
          volumeMounts:
            - {name: config, mountPath: /etc/agent/config, readOnly: true}
            - {name: secrets, mountPath: /etc/agent/secrets, readOnly: true}
            - {name: tmp, mountPath: /tmp}
          livenessProbe:
            httpGet: {path: /health/live, port: http}
          readinessProbe:
            httpGet: {path: /health/ready, port: http}
          resources:
            requests: {cpu: 100m, memory: 256Mi}
            limits: {memory: 512Mi}
      volumes:
        - {name: config, configMap: {name: weather-agent-config}}
        - {name: secrets, secret: {secretName: weather-agent-secrets}}
        - {name: tmp, emptyDir: {}}
---
apiVersion: v1
kind: Service
metadata:
  name: weather-agent
  namespace: agents
spec:
  selector: {app: weather-agent}
  ports: [{name: http, port: 8000, targetPort: http}]
```

`--public-url` must be the address other agents and clients use to reach this one: the A2A card advertises it, and A2A clients send their requests there. The [orchestrator Terraform module](../../modules/terraform-kubernetes-orchestrator-agent/README.md) generates these resources for any agent of this runtime.

## Build

The image builds on the project's Python base image, which contains the `agentic` package:

```bash
make python-base-image                                  # agentic/python:local
make simple-agent BASE_IMAGE=agentic/python:local       # agentic/simple-agent:local
```

## CLI reference

The image's entrypoint is `simple-agent` (the same program is also installed as `orchestrator-agent`).

| Command                 | Purpose                                                                  |
| ----------------------- | ------------------------------------------------------------------------ |
| `simple-agent serve`    | Serve the OpenAI-compatible API, A2A, and health probes (the default)     |
| `simple-agent chat`     | Chat with the agent in the terminal                                      |
| `simple-agent web`      | Serve Pydantic AI's browser chat UI (for local use)                      |

`serve` options: `--host`, `--port` (8000), `--config-dir`, `--secrets-dir`, `--public-url` (alias `--agent-url`, env `AGENT_PUBLIC_URL`), `--watch/--no-watch`, and `--log-level`. Run any command with `--help` for details.
