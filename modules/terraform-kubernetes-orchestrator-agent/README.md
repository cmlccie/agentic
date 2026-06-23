# terraform-kubernetes-orchestrator-agent

Terraform module for deploying [orchestrator-agent](../../images/orchestrator_agent) instances to Kubernetes.

The orchestrator is a LangGraph supervisor that delegates tasks across a team of downstream Agent2Agent (A2A) agents, exposing OpenAI-compatible (`/v1/`) and A2A (`/a2a/`) interfaces. Like the [simple-agent module](../terraform-kubernetes-simple-agent), this module accepts file _paths_ as inputs and injects their contents into a ConfigMap at `terraform apply` time — keeping configuration as clean, version-controlled files alongside your infrastructure code. The list of downstream A2A servers lives in `agent.yaml` (`a2a_servers`).

## Pre-requisites

- A Kubernetes cluster reachable by the Kubernetes Terraform provider.
- The `kubernetes` provider configured in the calling module or root.
- The orchestrator-agent container image published to a registry accessible by your cluster. The default image is `ghcr.io/cmlccie/agentic/orchestrator-agent`.
- (Optional) A reachable PostgreSQL database if you set `broker.backend: postgres` for persistent, multi-replica A2A task storage.

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

  secrets = {
    anthropic_api_key  = var.anthropic_api_key
    agent_database_url = var.agent_database_url # postgresql+asyncpg://user:pass@host:5432/db

    # Referenced as ${WEATHER_AGENT_TOKEN} in agent.yaml a2a_servers[].headers.
    additional = {
      weather_agent_token = var.weather_agent_token
    }
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

Configure the agents the orchestrator delegates to in your `agent.yaml` (the file referenced by `config_files.agent`):

```yaml
a2a_servers:
  - url: http://weather-agent.agents.svc.cluster.local/a2a
    headers:
      Authorization: "Bearer ${WEATHER_AGENT_TOKEN}"
  - url: http://network-agent.agents.svc.cluster.local/a2a
```

Header values may reference secret files via `${SECRET_KEY}`. Provide those tokens via `secrets.additional` (the key is lowercased to match the mounted secret filename). Each agent's card is fetched at startup/reload to build its delegation tool; unreachable agents are skipped (degraded mode).

### PostgreSQL task persistence

Set `broker.backend: postgres` in `server.yaml` and provide `secrets.agent_database_url` (a SQLAlchemy async DSN, e.g. `postgresql+asyncpg://user:pass@host:5432/dbname`) for a persistent, multi-replica-safe A2A task store. With the default `memory` backend the store is in-process and ephemeral.

### Instructions file injection

When `config_files.instructions` is provided, the module reads the file's contents and injects them into the `instructions` field of the `agent.yaml` ConfigMap data at apply time. The `agent.yaml` you supply can omit the `instructions` field entirely — the module merges it in; if the file already contains an `instructions` field, the supplied instructions file takes precedence.

## What this module creates

| Resource                   | Description                                                                                                                  |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `kubernetes_config_map_v1` | Holds `agent.yaml` (incl. `a2a_servers`) and `server.yaml`; optionally merges instructions from a separate file.            |
| `kubernetes_secret_v1`     | Holds orchestrator secrets as individual files under `/etc/agent/secrets/`. Only created when at least one secret provided.  |
| `kubernetes_service_v1`    | ClusterIP Service exposing the orchestrator on the configured port.                                                         |
| `kubernetes_deployment_v1` | Runs the orchestrator container with config and secrets mounted as directories (no `sub_path`, preserving hot-reload).      |
| `kubernetes_ingress_v1`    | Optional. Created only when `ingress.enabled = true`.                                                                        |

<!-- BEGIN_TF_DOCS -->
<!-- END_TF_DOCS -->
