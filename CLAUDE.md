# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Agentic architecture components for building AI agents using Pydantic AI and the Model Context Protocol (MCP). Agents interact with external services through MCP tool servers and are exposed via CLI, Agent2Agent (A2A), or OpenAI-compatible REST API interfaces.

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

Run scripts with `uv run <path-to-script>`. No tests are currently configured.

Build container images (requires docker or podman):
```bash
make python-base-image
make agents-weather-agent
make tools-mcp-weather-server
```

## Architecture

```
src/agentic/          Shared installable package (logging, OpenAI-compatible API wrapper)
agents/               AI agent implementations (weather_agent, provisioning_agent)
tools/mcp/            MCP tool servers that agents consume (weather_server, provisioning_server)
images/python/        Base container image definition
```

### Agent Pattern

Each agent follows a consistent structure:
1. Load config from environment variables (`OPENAI_BASE_URL`, `OPENAI_API_KEY`, `MODEL_NAME`, tool server URL).
2. Load a system prompt from a colocated `system_prompt.md` file.
3. Connect to MCP tool servers via `MCPServerStreamableHTTP`.
4. Create a `pydantic_ai.Agent` with the model, prompt, and toolsets.
5. Add dynamic system prompts via `@agent.system_prompt` decorators (e.g., injecting today's date).
6. Expose the agent through three interfaces using Typer CLI subcommands:
   - `cli` — interactive console via `agent.to_cli_sync()`
   - `a2a` — Agent2Agent protocol via `agent.to_a2a()` on FastAPI/uvicorn
   - `openai_api` — OpenAI-compatible REST API via `agentic.openai.compatible_api.OpenAICompatibleAPI`

### MCP Tool Server Pattern

Each tool server uses `mcp.server.fastmcp.FastMCP` and exposes `@mcp.tool()`, `@mcp.resource()`, and `@mcp.prompt()` decorated functions. Servers support `stdio` and `streamable-http` transport modes.

### OpenAI-Compatible API (`agentic.openai.compatible_api`)

Wraps any `Agent[Any, str]` in a FastAPI app with `/v1/chat/completions` (streaming and non-streaming), `/health`, and `/`. Converts between OpenAI message format and Pydantic AI `ModelMessage` types.

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
- Container files are named `Containerfile` (not Dockerfile), use Alpine base images, run as non-root `appuser` (UID 10000), work in `/app`, expose port 8000.
- List items use parallel structure; sentences end with periods, phrases do not.
- Makefile targets include `## Description` comments for the dynamic help system.
