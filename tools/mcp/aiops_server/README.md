# MCP AIOps Server

A Model Context Protocol (MCP) server providing **deterministic, code-based validation and
benchmarking tools** for LLM model-serving deployments (SGLang, vLLM, NVIDIA NIM).

All tools are pure HTTP validators: they take an OpenAI-compatible `base_url` (a pod IP or
Kubernetes Service URL, including `/v1`) and never touch the Kubernetes API. Orchestration —
scaling deployments, isolating GPU capacity, resolving pod IPs — belongs to the caller.

## Tools

| Tool                 | Purpose                                                                    |
| -------------------- | -------------------------------------------------------------------------- |
| `get_endpoint_info`  | Readiness probe: `GET /models` + engine health endpoint                    |
| `check_inference`    | Prompt in, valid completion out (content, finish_reason, usage)            |
| `check_reasoning`    | Reasoning arrives as a structured field, not inline `<think>` leakage      |
| `check_tool_calling` | Tool calls arrive structured and JSON-Schema-valid, in `auto` and `forced` |
| `measure_tps`        | Streaming decode tokens/sec (single-stream or concurrent), TTFT            |
| `list_fixtures`      | Enumerate the versioned test-fixture library + server version              |

Every check tool returns `{passed, checks: [{name, passed, detail}], ...}` — `passed` is true
only if every itemized deterministic check passed. Transport failures set `error`.

## Architecture

```text
aiops_server.py    Entry point (typer): stdio | http transports
aiops/
├── server.py      FastMCP instance, tool registrations, /health route
├── models.py      Pydantic result models
├── http_client.py httpx client factories, SSE helpers, timeout cap (570s)
├── engines/       EngineAdapter registry — the ONLY place engines differ
└── checks/        Shared check logic: requests, timing, TPS math, pass/fail
    fixtures/      Versioned YAML fixtures (validated by FixtureSpec)
```

**Engine adapters** encode where engines genuinely differ: reasoning field candidates, leak
patterns, acceptable tool finish_reasons, and health paths. NIM 2.0 is vLLM underneath, so
`NimAdapter` subclasses `VllmAdapter` and overrides only the health path (`/v1/health/ready`).
Adapters never issue HTTP.

**Fixtures** are versioned (`-vN`): a published fixture's semantics never change — add
`-v(N+1)` instead and retain the old one until nothing references it.

## Adding a check

1. Add a fixture entry to the matching YAML in `aiops/fixtures/` (new id, `-v1`).
2. If engines diverge in response structure, add a method to `EngineAdapter` (`engines/base.py`)
   with a sensible default and override it only in the divergent adapter.
3. Implement the shared logic as a new module in `aiops/checks/` (no engine branching outside
   adapter calls).
4. Register the tool in `aiops/server.py` with a docstring (the MCP client reads it).
5. Add tests: fixture validation, adapter behavior on canned payloads, and a
   `httpx.MockTransport` end-to-end case.

## Security: probe targets

Every tool sends HTTP requests to a caller-supplied `base_url`, so the server can reach
anything its network can reach (server-side request forgery by design). Deploy it
in-cluster only, never on a public endpoint, and restrict its egress with a
NetworkPolicy.

To limit targets in the server itself, set `AIOPS_ALLOWED_TARGETS` to a comma-separated
list of entries; a tool call whose `base_url` host matches none of them fails with an
error. When the variable is unset or empty, any `http(s)` target is allowed.

| Entry form    | Example              | Matches                                    |
| ------------- | -------------------- | ------------------------------------------ |
| CIDR          | `10.0.0.0/8`         | IP-literal hosts inside the network        |
| Hostname      | `model.ns.svc`       | That exact hostname (case-insensitive)     |
| Domain suffix | `.svc.cluster.local` | Any hostname ending with the suffix        |

Hostnames are matched as written and are not resolved, so list pod CIDRs for pod-IP
targets and hostnames or suffixes for Service URLs.

## Configuration

- `HOST`: `0.0.0.0` by default. HTTP server bind host.
- `PORT`: `8000` by default. HTTP server bind port.
- `AIOPS_ALLOWED_TARGETS`: Unset by default. Optional probe-target allowlist (see above).

## Running

```bash
# stdio (default)
uv run tools/mcp/aiops_server/aiops_server.py

# Streamable HTTP on :8000 (serves /mcp and /health)
uv run tools/mcp/aiops_server/aiops_server.py http

# Container, restricted to in-cluster targets
docker run --rm -p 8000:8000 \
  -e AIOPS_ALLOWED_TARGETS="10.0.0.0/8,.svc.cluster.local" \
  ghcr.io/cmlccie/agentic/tools-mcp-aiops-server:latest http
```

## Build Locally

```bash
# Build with the repository Makefile
make tools-mcp-aiops-server

# Build for your current platform
docker build -t tools-mcp-aiops-server tools/mcp/aiops_server

# Build for multiple platforms
docker buildx build --platform linux/amd64,linux/arm64 \
  -t agentic/tools-mcp-aiops-server:local tools/mcp/aiops_server
```

## TPS methodology

- Streaming chat completion with `stream_options.include_usage`, `temperature=0`.
- `TTFT` = first content-bearing delta − request send.
- `decode_tps` = `(completion_tokens − 1) / (last delta − first delta)` — excludes prefill.
- Token counts prefer the final `usage` chunk; a chunk-count fallback is flagged via
  `token_count_source`.
- `concurrency=1`: warmup runs (discarded) then measured sequential repetitions.
  `concurrency>1`: one warmup, then one round of N parallel streams; `aggregate_tps` = total
  completion tokens / round wall-clock.

## Tests

```bash
uv run pytest tools/mcp/aiops_server/tests
```
