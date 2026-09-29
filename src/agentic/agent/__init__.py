"""Config-driven Pydantic AI agents with OpenAI-compatible and A2A interfaces.

An agent is defined entirely by configuration:

- `agent.yaml` — a Pydantic AI agent spec: model, instructions, and
  capabilities (MCP tool servers, remote A2A agents, Harness capabilities, ...).
- `server.yaml` — the A2A agent card and serving options.

The same runtime serves "simple" agents (tools via MCP) and orchestrators
(other agents via the ``A2AAgent`` capability); the only difference is what
`agent.yaml` declares. Run it with ``simple-agent serve`` /
``orchestrator-agent serve`` (both are this package's CLI).

Modules:
    config       `server.yaml` models, secrets, YAML loading
    spec         `agent.yaml` loading, model resolution, capability registry
    runtime      the current agent snapshot and hot reload
    streaming    the activity model shared by all interfaces
    openai_api   OpenAI-compatible Chat Completions API
    a2a_server   spec-compliant A2A server (a2a-sdk)
    a2a_client   the A2AAgent capability (delegation to remote agents)
    stores       A2A task and conversation-history storage
    app          the FastAPI application
    cli          command-line interface
"""
