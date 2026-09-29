# terraform-kubernetes-orchestrator-agent

Terraform module for deploying [orchestrator-agent](../../images/orchestrator_agent) instances to Kubernetes.

The orchestrator is a config-driven Pydantic AI agent that delegates tasks across a team of downstream Agent2Agent (A2A) agents, exposing OpenAI-compatible (`/v1/`) and A2A (`/a2a`) interfaces. This module accepts file _paths_ as inputs and injects their contents into a ConfigMap at `terraform apply` time — keeping configuration as clean, version-controlled files alongside your infrastructure code. Downstream agents are declared in `agent.yaml` as `A2AAgent` capabilities. Because the simple agent and the orchestrator share one runtime and configuration schema, the module can deploy either image (set `deployment.image`).

## Pre-requisites

- A Kubernetes cluster reachable by the Kubernetes Terraform provider.
- The `kubernetes` provider configured in the calling module or root.
- The orchestrator-agent container image published to a registry accessible by your cluster. The default image is `ghcr.io/cmlccie/agentic/orchestrator-agent`.
- (Optional) A reachable PostgreSQL database if you set `a2a.store.backend: sql` for persistent A2A task and conversation storage.

## Usage

### Minimal

```hcl
provider "kubernetes" {
  config_path = "~/.kube/config"
}

module "orchestrator_agent" {
  source = "git::https://github.com/cmlccie/agentic.git//modules/terraform-kubernetes-orchestrator-agent"

  name      = "my-orchestrator"
  namespace = "agents"

  config_files = {
    agent  = "${path.module}/config/agent.yaml"
    server = "${path.module}/config/server.yaml"
  }
}
```

### With downstream A2A tokens, PostgreSQL persistence, and ingress

```hcl
module "orchestrator_agent" {
  source = "git::https://github.com/cmlccie/agentic.git//modules/terraform-kubernetes-orchestrator-agent"

  name      = "my-orchestrator"
  namespace = "agents"

  config_files = {
    agent        = "${path.module}/config/agent.yaml"
    server       = "${path.module}/config/server.yaml"
    instructions = "${path.module}/config/instructions.md"
  }

  openai_compatible = {
    base_url = var.openai_compatible_base_url
    api_key  = var.openai_compatible_api_key
  }

  task_broker = {
    database_url = var.agent_database_url # postgresql+asyncpg://user:pass@host:5432/db
  }

  # Referenced as ${WEATHER_AGENT_TOKEN} in an A2AAgent capability's headers in agent.yaml.
  agent_secrets = {
    weather_agent_token = var.weather_agent_token
  }

  deployment = {
    image_tag = "1.0.0"
    agent_url = "https://my-orchestrator.example.com"
    resources = {
      requests = { cpu = "250m", memory = "256Mi" }
      limits   = { memory = "512Mi" }
    }
  }

  ingress = {
    enabled         = true
    class_name      = "nginx"
    host            = "my-orchestrator.example.com"
    tls_secret_name = "my-orchestrator-tls"
  }

  labels = {
    "app.kubernetes.io/part-of" = "my-platform"
    "environment"               = "production"
  }
}
```

### Downstream A2A agents

Declare the agents the orchestrator delegates to in your `agent.yaml` (the file referenced by `config_files.agent`):

```yaml
capabilities:
  - A2AAgent:
      url: http://weather-agent.agents.svc.cluster.local:8000/a2a
      headers:
        Authorization: Bearer ${WEATHER_AGENT_TOKEN}
  - A2AAgent:
      url: http://network-agent.agents.svc.cluster.local:8000/a2a
```

Header values may reference secret files via `${NAME}`. Provide those tokens via `agent_secrets` (the key is lowercased to match the mounted secret filename). Each agent's card is fetched on first use to build its delegation tool; an unreachable agent has no tool until it becomes reachable. See the [orchestrator README](../../images/orchestrator_agent/README.md). Configurations written for the earlier LangGraph orchestrator (`a2a_servers`, `model: openai-compat`, `broker`) still load, with deprecation warnings.

The `openai_compatible` variable writes the `openai_compatible.base_url` and `openai_compatible.api_key` secrets, which the runtime uses as the endpoint of a self-hosted model (`model: vllm:<model>`).

### PostgreSQL persistence

The `task_broker.database_url` variable writes the secret `task_broker.database_url` (a SQLAlchemy async DSN, e.g. `postgresql+asyncpg://user:pass@host:5432/dbname`). Point the A2A store at it in `server.yaml`:

```yaml
a2a:
  store:
    backend: sql
    database_url_secret: task_broker.database_url
```

With the default `memory` backend, A2A tasks and conversation histories are kept in-process and are lost on restart. Either way, a task's live event stream lives in the replica running it, so keep one replica or enable session affinity when scaling out the A2A interface.

### Instructions file injection

When `config_files.instructions` is provided, the module reads the file's contents and injects them into the `instructions` field of the `agent.yaml` ConfigMap data at apply time. The `agent.yaml` you supply can omit the `instructions` field entirely — the module merges it in; if the file already contains an `instructions` field, the supplied instructions file takes precedence.

## What this module creates

| Resource                   | Description                                                                                                                  |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `kubernetes_config_map_v1` | Holds `agent.yaml` (incl. `A2AAgent` capabilities) and `server.yaml`; optionally merges instructions from a separate file.  |
| `kubernetes_secret_v1`     | Holds orchestrator secrets as individual files under `/etc/agent/secrets/`. Only created when at least one secret provided.  |
| `kubernetes_service_v1`    | ClusterIP Service exposing the orchestrator on the configured port.                                                         |
| `kubernetes_deployment_v1` | Runs the orchestrator container with config and secrets mounted as directories (no `sub_path`, preserving hot-reload).      |
| `kubernetes_ingress_v1`    | Optional. Created only when `ingress.enabled = true`.                                                                        |

<!-- BEGIN_TF_DOCS -->
<!-- END_TF_DOCS -->
