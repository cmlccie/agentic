# Agent Runtime — Design Record

The config-driven agent runtime in `src/agentic/agent` serves both the simple agent and the orchestrator. This record captures the decisions behind it and the alternatives that were rejected. The user-facing reference is [images/simple_agent/README.md](../images/simple_agent/README.md).

## Goals

1. **The OpenAI-compatible API is the primary client interface.** It must follow the Chat Completions contract closely, never fail silently, and stream agent activity in the reasoning channel before the answer.
2. **The A2A interface is spec compliant**, uses A2A's different workflows (direct Messages for quick exchanges, Tasks with status updates for longer work), and runs with in-memory storage and no external dependencies by default; SQL persistence is optional.
3. **One framework, one configuration schema, one code path** for simple agents and orchestrators.

## Decisions

### One runtime; an orchestrator is configuration

The orchestrator previously ran on LangChain/LangGraph (`create_agent` + checkpointer) with its own config schema, streaming, and message conversion. It is now a Pydantic AI agent whose `agent.yaml` declares `A2AAgent` capabilities. Delegation is a toolset: one tool per remote agent, named and described from the agent card. This removed four dependencies, a second streaming implementation, and an unbounded checkpointer, and gave the orchestrator the same reasoning support as simple agents (LangChain's `ChatOpenAI` drops `reasoning_content`).

### agent.yaml is a Pydantic AI AgentSpec

Custom capabilities are registered through `Agent.from_spec(custom_capability_types=...)`: `A2AAgent` plus an allowlist of Pydantic AI Harness capabilities that are safe for server agents (no filesystem/shell access, no callables, no extra packages). `RepairToolArguments` is wrapped in a dataclass because the Harness class isn't one. Unknown top-level keys are rejected (AgentSpec itself ignores them silently).

Self-hosted models use the built-in `vllm:` provider (Chat Completions, `reasoning_content` parsing, model-family profiles), with the endpoint from secret files or `VLLM_BASE_URL`. This replaced the `openai-compat` sentinel, which existed only because AgentSpec couldn't express a custom endpoint; SGLang and NIM speak the same API.

### Reload is a swap, not a drain

A reload builds a new `Snapshot` (agent + server settings) and assigns it; interfaces read `runtime.current` once per request. MCP toolsets are entered per run (not held open by the agent), so there are no sessions to drain or close, and a restarted MCP server can't leave a dead persistent session behind. An invalid configuration logs an error and keeps the last good snapshot. The previous design (RUNNING → DRAINING → RELOADING with a request-slot middleware) could strand a pod unready after a failed reload, waited the full drain timeout on idle pods, and didn't count streaming responses or A2A work.

### One activity model for every interface

`streaming.activities` maps Pydantic AI events to `Activity` items plus one final `Answer`. Text that precedes a tool call becomes a `note` in the reasoning channel; the answer always comes from the run result, so it is never duplicated or confused with preamble text. The trade-off is that the answer arrives as one chunk at the end rather than token by token, while activity streams live. Verbosity (`off`/`summary`/`trace`) and redaction are applied at render time, so interface logic (A2A promotion on the first tool call) doesn't depend on what is displayed.

Remote agents' activity crosses A2A as working status updates marked with the `urn:agentic:a2a:activity:v1` extension, and re-enters the orchestrator's event stream as `ActivityEvent` (a Pydantic AI `CapabilityEvent`) emitted from the delegation tool, with the agent name prepended to `source`.

### An in-house OpenAI API instead of fastapi-openai-compat

The library never sent `data: [DONE]`, silently truncated streams on exceptions (clients saw a normal end), didn't implement `include_usage`, leaked exception text in 500s, and turned empty reasoning into content. The API is small enough to own: request validation, message conversion, SSE framing, keep-alive comments, error events, and `x-should-retry: false` on agent failures (the `openai` SDK otherwise retries 5xx, re-running tools).

### A2A on a2a-sdk instead of fasta2a

fasta2a made every exchange a long-running task, held a reference to the agent from startup (stale after reloads), had no working Redis backend despite the configuration option, and spoke A2A 0.3 (the orchestrator's 1.x client couldn't call it). The a2a-sdk request handler lives for the whole process (it owns the live-task registry); the executor reads the current snapshot per request and the agent card is served from it.

In `auto` mode the executor starts the run immediately but holds back the reply shape: it replies with a Message if the run finishes quickly without tools, and promotes to a Task on the first tool call or after `promote_after_seconds`. Requests that set `returnImmediately`, or continue an existing task, always get a Task. Conversation history is stored per `contextId` by the runtime, because Message-only exchanges are never persisted by the SDK.

Two SDK issues are worked around, each guarded by a test: the handler never releases the internal active-task entry for Message-only replies (a leak per quick reply), and the in-memory task store is unbounded.

### Security defaults

Push notifications are off (they make the server call client-supplied URLs); when on, destinations are limited to http(s) and an optional host allowlist. Optional bearer-token auth covers `/v1` and `/a2a` and fails closed if its secret is missing; the agent card declares the scheme. Client-facing error messages name the exception type only. Tool arguments and results in activity are redacted by key.

## Known limitations

- A task's live event stream lives in the replica running it (an a2a-sdk property), so multi-replica A2A needs session affinity even with SQL storage.
- The answer is not token-streamed (see above).
- Changing `a2a.store` requires a restart.
- MCP servers using the stdio transport start a process per run.
