# AI Agent Instructions

This file provides consistent guidance to AI agents when working with code in this repository.

## Project Overview

Agentic architecture components for building Agentic AI systems (models, agents, and tools). Agents are defined by configuration, use tools through MCP tool servers and other agents through the Agent2Agent (A2A) protocol, and are exposed via an OpenAI-compatible REST API and an A2A server.

## Instruction Files

The following instruction files apply automatically to specific file types. Before making any changes to a file that matches the patterns below, read the corresponding instruction file if you have not already done so in this session. If a referenced instruction file cannot be read (e.g., it does not exist), stop and notify the user before proceeding with any changes.

| File                                                    | Applies To                | Description                      |
| ------------------------------------------------------- | ------------------------- | -------------------------------- |
| `.github/instructions/python.instructions.md`           | `**/*.py`                 | Python coding conventions        |
| `.github/instructions/containerization.instructions.md` | `**/Containerfile`        | Containerization best practices  |
| `.github/instructions/github_actions.instructions.md`   | `.github/workflows/*.yml` | GitHub Actions workflow patterns |

## Commands

All commands use `uv` (not pip or poetry) and run from the repository root.

```bash
make setup          # Reset .venv and install all dependency groups
make sync           # Sync dependencies
make upgrade        # Upgrade all dependencies to latest
make lint           # ruff check .
make format         # ruff format + ruff check --fix
make check          # lint + format --check
make test           # pytest (tests/ and the tests next to tool servers)
make clean          # Remove __pycache__, .egg-info, dist/, build/, etc.
```

Run scripts with `uv run <path-to-script>`. Run tests with `uv run pytest`. CI (`.github/workflows/ci.yml`) runs the lockfile check, lint, format check, tests, and a requirements drift check on every push and pull request.

Build container images (requires docker or podman). App images build on the published base image by default; pass `BASE_IMAGE=agentic/python:local` to build against a local base image:

```bash
make python-base-image
make simple-agent BASE_IMAGE=agentic/python:local
make orchestrator-agent BASE_IMAGE=agentic/python:local
make tools-mcp-weather-server
```

## Architecture

```text
src/agentic/agent/     Config-driven agent runtime (Pydantic AI): OpenAI API, A2A server, A2A client
src/agentic/logging.py Logging setup and the log_call decorator
images/simple_agent/   Agent image + default config (tools via MCP)
images/orchestrator_agent/  Same runtime, default config for delegating to A2A agents
images/python/         Base container image definition (installs the agentic package)
tools/mcp/             MCP tool servers that agents consume
modules/               Terraform modules for deploying agents to Kubernetes
tests/                 Tests for src/ (tests/agent) and tool servers (tests/tools)
```

### Agent Runtime (`agentic.agent`)

Agents are fully defined by configuration without bespoke Python code. The simple agent and the orchestrator agent are the same runtime; they differ only in what their `agent.yaml` declares.

1. `agent.yaml` is a Pydantic AI `AgentSpec` (model, instructions, model settings, capabilities). Besides the Pydantic AI built-in capabilities (`MCP`, `Thinking`, ...), it can declare `A2AAgent` (delegate to a remote A2A agent) and a curated set of Pydantic AI Harness capabilities (`spec.HARNESS_CAPABILITIES`). `${NAME}` references in capability arguments are expanded from secret files or environment variables.
2. `server.yaml` holds the A2A agent card and serving options (interfaces, A2A behavior and storage, activity streaming, auth). Both files reject unknown keys; legacy keys are migrated with warnings.
3. `runtime.AgentRuntime` holds the current `Snapshot` (agent + server settings) and hot-reloads on file changes or SIGHUP by building a new snapshot and swapping it in; an invalid config keeps the last good one. Interfaces read `runtime.current` once per request.
4. `streaming.activities` reduces Pydantic AI events to `Activity` items (thinking, notes, tool calls/results, relayed worker activity) and a final `Answer`; both interfaces render them.
5. Interfaces, assembled in `app.create_app`:
   - `openai_api` — `/v1/models` and `/v1/chat/completions` (activity in `reasoning_content`, then the answer)
   - `a2a_server` — spec-compliant A2A on the a2a-sdk (quick answers as Messages, promoted to Tasks with streamed activity when the agent works)
6. CLI (`cli.py`, installed as `simple-agent` and `orchestrator-agent`): `serve`, `chat`, `web`.

Model strings are standard Pydantic AI model ids. Self-hosted OpenAI-compatible servers (vLLM, SGLang, NIM) use `vllm:<model>`, with the endpoint from the `model.base_url` / `model.api_key` secrets or `VLLM_BASE_URL` / `VLLM_API_KEY`.

### MCP Tool Server Pattern

Each tool server uses `fastmcp.FastMCP` via `from fastmcp import FastMCP` and exposes `@mcp.tool()`, `@mcp.resource()`, and `@mcp.prompt()` decorated functions. Servers support `stdio` and `http` transport modes. For HTTP mode, bind settings are passed to `mcp.run(transport="http", host=HOST, port=PORT)` rather than stored on the server instance. Configure logging in the server's `main()` entry point, not at import time, so importing a server module has no side effects.

### Logging (`agentic.logging`)

- `agentic.logging.fancy(level)` — Rich handler with timestamps (stderr).
- `agentic.logging.plain(level)` — One line per record on stderr, for containers and log collectors.
- `agentic.logging.auto(level)` — `fancy` on an interactive terminal, `plain` otherwise.
- `agentic.logging.silent(level)` — Null handler, suppresses output.
- `@agentic.logging.log_call(logger, level)` — Decorator that logs calls and return values of sync and async functions, redacting secret-named arguments (`redact=`) and optionally omitting results (`log_result=False`).

## Coding Conventions

- **Python 3.13+**, type hints with Pydantic models.
- **Ruff** for linting and formatting: 88 char line length, extended rules B/I/Q.
- **uv** for all dependency management; `uv.lock` is committed.
- **Hatchling** build backend; wheel packages `src/agentic`.
- Scripts must have `#!/usr/bin/env python3` shebang and be executable.
- Each component in `tools/` and `images/` must have a `Containerfile` and a `README.md` that includes at minimum: a one-paragraph description, build instructions, and at least one usage example.
- Python dependencies are managed centrally in `pyproject.toml`; the base image's `images/python/requirements.txt` is generated from `uv.lock` with hashes. After modifying `pyproject.toml` or `uv.lock`, regenerate it with `make --always-make images/python/requirements.txt` and commit it alongside the change (CI fails on drift).
- Test agents with scripted models (`pydantic_ai.models.function.FunctionModel`) through real config files; see `tests/agent/conftest.py`.

### Style

- All items in a list must be consistently structured — either all simple phrases or all complete sentences. A list item is a complete sentence if it contains a finite verb and could stand alone as a sentence; otherwise it is a simple phrase.
- When a list contains sentences, each sentence ends with a period.
- When a list contains simple phrases, do not use periods.
- Maintain parallel structure across all items in the same list.

### Makefile

The `Makefile` in the repository root provides common developer operations. Use `make help` to see all available targets.

- Add new targets with `## Description` comments so they appear in `make help` output.
- Update the Makefile when adding new components or changing project structure.

## Containerization

See `.github/instructions/containerization.instructions.md` for the full ruleset. Key conventions:

- Name container build files `Containerfile` (not `Dockerfile`).
- Use `python:3.13-alpine` base images for smaller size.
- App images declare `ARG BASE_IMAGE=ghcr.io/cmlccie/agentic/python:latest` and build `FROM ${BASE_IMAGE}`.
- Set working directory to `/app`.
- Copy `requirements.txt` first to leverage layer caching, then install deps, then copy application code.
- Create a non-root user `appuser` (UID/GID 10000), set ownership, and switch to it with the numeric `USER 10000:10000` before the `ENTRYPOINT`.
- Make scripts executable with `RUN chmod +x` before switching users.
- Use `ENTRYPOINT` for the main command and `CMD` for default arguments.
- Expose port 8000 for HTTP services.
- Generate `requirements.txt` with `uv export` (with hashes) and install with `--require-hashes`.

## GitHub Actions Workflows

See `.github/instructions/github_actions.instructions.md` for the full ruleset. Key conventions:

- Place workflows in `.github/workflows/` with `build-<image-name>.yml` naming.
- For Python container builds, use the reusable workflow `.github/workflows/reusable-build-python-container-image.yml`.
- Trigger on push/PR to `main` with path filters, release events, and optionally `workflow_run` after base image builds.
- Guard `workflow_run` jobs with `github.event.workflow_run.conclusion == 'success'`.
- Build but don't push images on pull requests.
- Use GitHub Container Registry (`ghcr.io`) for all images.
- Use proper permissions (`contents: read`, `packages: write`, `id-token: write`, `attestations: write`).
- See `.github/workflows/build-tools-mcp-weather-server.yml` as a reference example.
