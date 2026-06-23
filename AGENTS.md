# AI Agent Instructions

This file provides consistent guidance to AI agents when working with code in this repository.

## Project Overview

Agentic architecture components for building Agentic AI systems (models, agents, and tools). Agents interact with external services through MCP tool servers and are exposed via CLI, Agent2Agent (A2A), or OpenAI-compatible REST API interfaces.

## Instruction Files

The following instruction files apply automatically to specific file types. Read them before modifying matching files.

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
make clean          # Remove __pycache__, .egg-info, dist/, build/, etc.
```

Run scripts with `uv run <path-to-script>`. Run tests with `uv run pytest`.

Build container images (requires docker or podman):

```bash
make python-base-image
make simple-agent
make tools-mcp-weather-server
```

## Architecture

```text
src/agentic/          Shared installable package (logging, simple-agent runtime)
images/simple_agent/  Simple agent container image (model + tools + system prompt via config)
tools/mcp/            MCP tool servers that agents consume (weather_server, provisioning_server)
images/python/        Base container image definition
```

### Simple Agent Pattern

Simple agents are fully defined by configuration (`agent.yaml` and `server.yaml`) without bespoke Python code. The `simple-agent` image loads an `agent.yaml` specifying the model, MCP tool server URLs, and a system prompt, then exposes the agent through multiple interfaces.

The `src/agentic/simple_agent` package implements the runtime:

1. Load `agent.yaml` and `server.yaml` from the working directory (or paths from environment variables).
2. Connect to MCP tool servers via `MCPServerStreamableHTTP`.
3. Create a `pydantic_ai.Agent` with the configured model, system prompt, and toolsets.
4. Expose the agent through three interfaces via Typer CLI subcommands:
   - `cli` — interactive console via `agent.to_cli_sync()`
   - `a2a` — Agent2Agent protocol via `agent.to_a2a()` on FastAPI/uvicorn
   - `openai_api` — OpenAI-compatible REST API via `fastapi-openai-compat`

### MCP Tool Server Pattern

Each tool server uses `fastmcp.FastMCP` via `from fastmcp import FastMCP` and exposes `@mcp.tool()`, `@mcp.resource()`, and `@mcp.prompt()` decorated functions. Servers support `stdio` and `http` transport modes. For HTTP mode, bind settings are passed to `mcp.run(transport="http", host=HOST, port=PORT)` rather than stored on the server instance.

### Logging (`agentic.logging`)

- `agentic.logging.fancy(level)` — Rich handler with timestamps.
- `agentic.logging.silent(level)` — Null handler, suppresses output.
- `@agentic.logging.log_call(logger, level)` — Decorator that logs function calls and return values.

## Coding Conventions

- **Python 3.13+**, type hints with Pydantic models.
- **Ruff** for linting and formatting: 88 char line length, extended rules B/I/Q.
- **uv** for all dependency management; `uv.lock` is committed.
- **Hatchling** build backend; wheel packages `src/agentic`.
- Scripts must have `#!/usr/bin/env python3` shebang and be executable.
- Each component in `tools/` and `images/` must be containerized (own `Containerfile`) and have a `README.md` with usage instructions.
- Python dependencies are managed centrally in `pyproject.toml`; each containerized component's `requirements.txt` is generated via `uv export`.

### Style

- All items in a list must be consistently structured — either all simple phrases or all complete sentences.
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
- Set working directory to `/app`.
- Copy `requirements.txt` first to leverage layer caching, then install deps, then copy application code.
- Create a non-root user `appuser` (UID 10000), set ownership, and switch to it before the `ENTRYPOINT`.
- Make scripts executable with `RUN chmod +x` before switching users.
- Use `ENTRYPOINT` for the main command and `CMD` for default arguments.
- Expose port 8000 for HTTP services.
- Generate `requirements.txt` with `uv export`.

## GitHub Actions Workflows

See `.github/instructions/github_actions.instructions.md` for the full ruleset. Key conventions:

- Place workflows in `.github/workflows/` with `build-<image-name>.yml` naming.
- For Python container builds, use the reusable workflow `.github/workflows/reusable-build-python-container-image.yml`.
- Trigger on push/PR to `main` with path filters, release events, and optionally `workflow_run` after base image builds.
- Use GitHub Container Registry (`ghcr.io`) for all images.
- Use proper permissions (`contents: read`, `packages: write`, `id-token: write`, `attestations: write`).
- See `.github/workflows/build-tools-mcp-weather-server.yml` as a reference example.
