# Simple Agent

A config-driven Pydantic AI agent that exposes three interfaces simultaneously from a single container:

- **OpenAI-compatible REST API** — drop-in replacement for `/v1/chat/completions`
- **Agent2Agent (A2A) protocol** — interoperable with other A2A agents
- **Web UI** — streaming chat interface for browser-based testing

The agent's identity — model, instructions, MCP tool servers — is defined entirely in `agent.yaml`. Infrastructure settings live in `server.yaml`. Both files are hot-reloaded at runtime: update a ConfigMap or rotate a secret, and the agent reloads without downtime.

---

## Interfaces and Endpoints

| Path | Description |
|---|---|
| `GET /health/live` | Liveness probe — always 200 while the process is alive |
| `GET /health/ready` | Readiness probe — 503 during hot-reload drain/reload cycle |
| `GET /v1/models` | List available models |
| `POST /v1/chat/completions` | OpenAI-compatible chat completions (streaming and non-streaming) |
| `GET /a2a/.well-known/agent-card.json` | A2A agent card |
| `POST /a2a/` | A2A JSON-RPC task endpoint |
| `GET /ui/` | Streaming chat web UI |
| `POST /ui/chat` | SSE chat endpoint (used by the web UI) |

All non-health endpoints return `503 Retry-After: 5` during a hot-reload drain/reload cycle.

---

## Local Testing with Docker

### 1. Create a secrets directory

The agent reads credentials from files — one file per secret — rather than environment variables. This mirrors how Kubernetes Secret volumes work.

```bash
mkdir -p /tmp/my-agent-secrets

# For an OpenAI-compatible provider (LM Studio, Ollama, vLLM, etc.)
echo "http://localhost:1234/v1" > /tmp/my-agent-secrets/agent_model_base_url
echo "your-api-key"             > /tmp/my-agent-secrets/agent_model_api_key

# For Anthropic
echo "sk-ant-..."  > /tmp/my-agent-secrets/anthropic_api_key

# For OpenAI
echo "sk-..."  > /tmp/my-agent-secrets/openai_api_key
```

### 2. Start the container

```bash
docker run -d \
  --name my-agent \
  -p 8000:8000 \
  -v /tmp/my-agent-secrets:/etc/agent/secrets:ro \
  ghcr.io/cmlccie/agentic/agents-simple:latest \
  serve --agent-url http://localhost:8000
```

> **macOS / Docker Desktop**: If your model server is running on the host machine, replace
> `http://localhost:1234` with `http://host.docker.internal:1234` in the secrets file.

### 3. Check readiness

```bash
curl http://localhost:8000/health/ready
# {"status":"ready","in_flight":0}
```

### 4. Test the API

```bash
# List available models
curl http://localhost:8000/v1/models

# Non-streaming completion
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "simple-agent",
    "messages": [{"role": "user", "content": "What is the capital of France?"}]
  }'

# Streaming completion
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "simple-agent",
    "messages": [{"role": "user", "content": "Count from 1 to 5."}],
    "stream": true
  }'
```

### 5. Open the web UI

Navigate to **[http://localhost:8000/ui/](http://localhost:8000/ui/)** in your browser. The chat UI streams responses as they arrive.

### 6. Run the terminal chat interface

The `simple-agent chat` command loads the same config and starts an interactive terminal session — useful for quick testing without a running server.

```bash
docker run -it --rm \
  -v /tmp/my-agent-secrets:/etc/agent/secrets:ro \
  ghcr.io/cmlccie/agentic/agents-simple:latest \
  chat
```

To use a custom config directory instead of the image defaults:

```bash
docker run -it --rm \
  -v /path/to/my/config:/etc/agent/config:ro \
  -v /tmp/my-agent-secrets:/etc/agent/secrets:ro \
  ghcr.io/cmlccie/agentic/agents-simple:latest \
  chat
```

### Use with the Python OpenAI client

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-required")

# Non-streaming
response = client.chat.completions.create(
    model="simple-agent",
    messages=[{"role": "user", "content": "Explain the water cycle in one paragraph."}],
)
print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="simple-agent",
    messages=[{"role": "user", "content": "Write a haiku about autumn."}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

---

## Container Images

Pre-built images are published to the GitHub Container Registry:

```
ghcr.io/cmlccie/agentic/agents-simple:latest   # latest build from main
ghcr.io/cmlccie/agentic/agents-simple:<version> # specific release (e.g. 1.0.0)
ghcr.io/cmlccie/agentic/agents-simple:<branch>-<sha> # specific commit
```

---

## Configuration Reference

The container ships with default config at `/etc/agent/config/`. Override individual files or the entire directory with volume mounts or Kubernetes ConfigMap volumes.

### `agent.yaml` — Agent identity

```yaml
name: my-agent                       # used as the model ID in /v1/models
description: "Does X for callers"

# ── Model ──────────────────────────────────────────────────────────────────────
# Option A: first-class pydantic-ai provider (no secret files needed beyond API key)
model: anthropic:claude-sonnet-4-6
# model: openai:gpt-4o
# model: google-gla:gemini-2.0-flash

# Option B: OpenAI-compatible custom endpoint (LM Studio, Ollama, vLLM, etc.)
# Requires secret files: agent_model_base_url, agent_model_api_key
model: openai-compat
model_id: my-local-model             # reported in /v1/models; passed to the API

instructions: |
  You are a helpful assistant that specialises in X.
  Always respond in plain language.

model_settings:
  temperature: 0.3
  max_tokens: 4096

# ── MCP Tool Servers (optional) ────────────────────────────────────────────────
capabilities:
  - MCP:
      url: http://tool-server-a/mcp
      id: tools-a
      # Optionally restrict which tools the agent can call:
      # allowed_tools: [search, summarize]
      # Inject secrets into request headers:
      # headers:
      #   Authorization: "Bearer ${MCP_TOKEN_A}"  # expanded from secrets dir
```

### `server.yaml` — Serving infrastructure

```yaml
agent_card:
  display_name: "My Agent"           # shown in A2A agent card and API title
  description: "Does X for callers"
  version: "1.0.0"
  icon_url: ""                       # optional URL to a PNG icon

broker:
  backend: memory                    # "memory" (default) | "redis"
  # Redis requires secret file: agent_redis_url
  # Default Redis URL if secret not present: redis://localhost:6379/0

interfaces:
  a2a: true                          # A2A protocol at /a2a/
  openai_compat: true                # OpenAI-compatible API at /v1/
  ui: true                           # Chat web UI at /ui/

reload:
  drain_timeout: 30                  # seconds to wait for in-flight requests before forcing reload
```

### Secrets reference

All secrets are read from individual files under `/etc/agent/secrets/`. File names match the secret keys (lowercased). On Kubernetes, these files are projected from a Secret volume.

| File | Required for |
|---|---|
| `anthropic_api_key` | `model: anthropic:*` providers |
| `openai_api_key` | `model: openai:*` providers |
| `agent_model_base_url` | `model: openai-compat` |
| `agent_model_api_key` | `model: openai-compat` |
| `agent_redis_url` | `broker.backend: redis` |
| `<any_key>` | MCP header injection via `${ANY_KEY}` in `agent.yaml` |

> **Local dev fallback**: If a secret file is missing, the agent falls back to the environment
> variable of the same name (uppercased). For example, `agent_model_api_key` falls back to
> `AGENT_MODEL_API_KEY`. This makes local runs without a secrets directory possible.

---

## Kubernetes Deployment

### Architecture overview

```
              Kubernetes Cluster
┌─────────────────────────────────────────────────────┐
│                                                     │
│  ConfigMap: my-agent-config                         │
│  ├── agent.yaml   ─── mounted at /etc/agent/config  │
│  └── server.yaml  ─┘                               │
│                                                     │
│  Secret: my-agent-secrets                           │
│  └── (key files) ──── mounted at /etc/agent/secrets │
│                                                     │
│  Deployment: my-agent                               │
│  └── Pod                                            │
│      └── Container: simple-agent                    │
│          ├── /health/live  ← liveness probe         │
│          ├── /health/ready ← readiness probe        │
│          ├── /v1/          ← OpenAI-compat API      │
│          ├── /a2a/         ← A2A protocol           │
│          └── /ui/          ← Web UI                 │
└─────────────────────────────────────────────────────┘
```

### ConfigMap

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: my-agent-config
  namespace: agents
data:
  agent.yaml: |
    name: my-agent
    description: "A helpful assistant"
    model: anthropic:claude-sonnet-4-6
    instructions: |
      You are a helpful assistant that specialises in answering questions about
      internal company policy. Always cite the source document when you answer.
    model_settings:
      temperature: 0.2
      max_tokens: 4096
    capabilities:
      - MCP:
          url: http://policy-docs-mcp/mcp
          id: policy-docs
          headers:
            Authorization: "Bearer ${POLICY_MCP_TOKEN}"

  server.yaml: |
    agent_card:
      display_name: "Policy Assistant"
      description: "Answers questions about company policy"
      version: "1.0.0"
    broker:
      backend: memory
    interfaces:
      a2a: true
      openai_compat: true
      ui: true
    reload:
      drain_timeout: 30
```

> **Do not use `subPath`** in your volume mounts. Kubernetes uses an atomic symlink swap
> to update ConfigMap volumes. `subPath`-mounted files are bind-mounted directly and never
> receive live updates. Mount the entire directory without `subPath`.

### Secret

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: my-agent-secrets
  namespace: agents
stringData:
  # Provider API key — only the one matching your model: provider is needed
  anthropic_api_key: "sk-ant-..."
  # openai_api_key: "sk-..."
  # agent_model_base_url: "http://..."    # only for model: openai-compat
  # agent_model_api_key: "..."            # only for model: openai-compat

  # MCP server tokens — one file per token, named to match ${PLACEHOLDER} in agent.yaml
  policy_mcp_token: "tok-..."

  # Redis URL — only needed when broker.backend: redis
  # agent_redis_url: "redis://redis:6379/0"
```

### Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-agent
  namespace: agents
spec:
  replicas: 2
  selector:
    matchLabels:
      app: my-agent
  template:
    metadata:
      labels:
        app: my-agent
    spec:
      containers:
        - name: agent
          image: ghcr.io/cmlccie/agentic/agents-simple:latest
          args:
            - serve
            - --agent-url=http://my-agent.agents.svc.cluster.local:8000
          ports:
            - name: http
              containerPort: 8000
          volumeMounts:
            # Mount the entire directory — NO subPath
            - name: config
              mountPath: /etc/agent/config
              readOnly: true
            - name: secrets
              mountPath: /etc/agent/secrets
              readOnly: true
          livenessProbe:
            httpGet:
              path: /health/live
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /health/ready
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 5
            failureThreshold: 24   # 120 s window for drain + reload
          resources:
            requests:
              cpu: "250m"
              memory: "256Mi"
            limits:
              cpu: "1"
              memory: "512Mi"
      volumes:
        - name: config
          configMap:
            name: my-agent-config    # NO subPath — mount the full directory
        - name: secrets
          secret:
            secretName: my-agent-secrets   # NO subPath — mount the full directory
```

### Service

```yaml
apiVersion: v1
kind: Service
metadata:
  name: my-agent
  namespace: agents
spec:
  selector:
    app: my-agent
  ports:
    - name: http
      port: 8000
      targetPort: 8000
```

---

## Hot-Reload

The agent watches `/etc/agent/config` and `/etc/agent/secrets` for changes. When a change is detected — or when the process receives `SIGHUP` — the agent reloads without downtime:

```
RUNNING  ──(file change or SIGHUP)──►  DRAINING  ──(in-flight==0 or timeout)──►  RELOADING
   ▲                                                                                  │
   └──────────────────────────────(reload complete)──────────────────────────────────┘
```

During DRAINING and RELOADING:
- `/health/ready` returns `503` — Kubernetes stops routing new traffic to this pod
- `/health/live` continues to return `200` — the pod is not killed
- In-flight requests are allowed to complete (up to `reload.drain_timeout` seconds)
- New requests receive `503 Retry-After: 5`

**What is hot-reloaded**: model, instructions, model settings, MCP tool servers, drain timeout.

**What requires a pod restart**: enabling or disabling interfaces (`a2a`, `openai_compat`, `ui`), changing `broker.backend`.

### Triggering a reload manually

```bash
# Via SIGHUP (triggers reload even if files haven't changed — e.g. after secret rotation)
kubectl exec -n agents deploy/my-agent -- kill -HUP 1

# Via ConfigMap update (reload is triggered automatically by file watcher)
kubectl edit configmap my-agent-config -n agents
```

---

## Redis Broker (optional)

By default the A2A broker uses in-memory storage — tasks are not shared across replicas and are lost if the pod restarts. For multi-replica deployments or durable task storage, switch to the Redis backend.

### `server.yaml` change

```yaml
broker:
  backend: redis
```

### Add the Redis URL secret

```yaml
# In your Secret:
stringData:
  agent_redis_url: "redis://my-redis:6379/0"
```

### Deploy Redis alongside the agent

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-agent-redis
  namespace: agents
spec:
  replicas: 1
  selector:
    matchLabels:
      app: my-agent-redis
  template:
    metadata:
      labels:
        app: my-agent-redis
    spec:
      containers:
        - name: redis
          image: redis:7-alpine
          ports:
            - containerPort: 6379
          resources:
            requests:
              cpu: "100m"
              memory: "64Mi"
---
apiVersion: v1
kind: Service
metadata:
  name: my-agent-redis
  namespace: agents
spec:
  selector:
    app: my-agent-redis
  ports:
    - port: 6379
      targetPort: 6379
```

Then set the secret:

```yaml
stringData:
  agent_redis_url: "redis://my-agent-redis:6379/0"
```

---

## MCP Tool Servers

Add MCP servers under `capabilities` in `agent.yaml`. The agent connects to each server at startup and reconnects on each reload cycle.

```yaml
capabilities:
  # Plain HTTP MCP server
  - MCP:
      url: http://my-tool-server/mcp
      id: my-tools

  # With restricted tool list
  - MCP:
      url: http://search-server/mcp
      id: search
      allowed_tools: [web_search, summarize]

  # With authentication header using a secret file
  - MCP:
      url: http://secured-server/mcp
      id: secured
      headers:
        Authorization: "Bearer ${MCP_TOKEN}"   # expanded from /etc/agent/secrets/mcp_token
```

Secret files referenced in `${PLACEHOLDER}` patterns are read at load time. When the agent reloads (after a ConfigMap update or SIGHUP), the headers are re-expanded from the current secret files — so rotating a token requires only updating the Secret and triggering a reload.

---

## CLI Reference

```
$ simple-agent --help

 Usage: simple-agent [OPTIONS] COMMAND [ARGS]...

 Simple Agent

╭─ Commands ─────────────────────────────────────────────────────────────────╮
│ serve   Serve all configured interfaces (default)                          │
│ chat    Interactive terminal chat                                           │
╰────────────────────────────────────────────────────────────────────────────╯
```

```
$ simple-agent serve --help

 Usage: simple-agent serve [OPTIONS]

 Start the FastAPI server with all configured interfaces.

╭─ Options ──────────────────────────────────────────────────────────────────╮
│ --host           TEXT  Bind host [default: 0.0.0.0]                        │
│ --port           INT   Bind port [default: 8000]                           │
│ --config-dir     PATH  Config directory [default: /etc/agent/config]       │
│ --secrets-dir    PATH  Secrets directory [default: /etc/agent/secrets]     │
│ --agent-url      TEXT  Public URL for A2A agent card                       │
│ --log-level      TEXT  Log level [default: info]                           │
╰────────────────────────────────────────────────────────────────────────────╯
```

```
$ simple-agent chat --help

 Usage: simple-agent chat [OPTIONS]

 Run an interactive terminal chat session with the configured agent.

╭─ Options ──────────────────────────────────────────────────────────────────╮
│ --config-dir     PATH  Config directory [default: /etc/agent/config]       │
│ --secrets-dir    PATH  Secrets directory [default: /etc/agent/secrets]     │
│ --log-level      TEXT  Log level [default: warning]                        │
╰────────────────────────────────────────────────────────────────────────────╯
```
