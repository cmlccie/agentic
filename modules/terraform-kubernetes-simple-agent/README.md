# terraform-kubernetes-simple-agent

Terraform module for deploying [simple-agent](../../images/simple_agent) instances to Kubernetes.

Rather than maintaining `agent.yaml`, `server.yaml`, and an optional system prompt inline in Kubernetes manifests, this module accepts file _paths_ as inputs and injects their contents into a ConfigMap at `terraform apply` time — keeping configuration as clean, version-controlled files alongside your infrastructure code.

## Pre-requisites

- A Kubernetes cluster reachable by the Kubernetes Terraform provider.
- The `kubernetes` provider configured in the calling module or root.
- The simple-agent container image published to a registry accessible by your cluster. The default image is `ghcr.io/cmlccie/agentic/simple-agent`.

## Usage

### Minimal

```hcl
provider "kubernetes" {
  config_path = "~/.kube/config"
}

module "simple_agent" {
  source = "git::https://github.com/cmlccie/agentic.git//modules/terraform-kubernetes-simple-agent"

  name      = "my-agent"
  namespace = "agents"

  config_files = {
    agent  = "${path.module}/config/agent.yaml"
    server = "${path.module}/config/server.yaml"
  }
}
```

### With secrets, ingress, and a separate instructions file

```hcl
module "simple_agent" {
  source = "git::https://github.com/cmlccie/agentic.git//modules/terraform-kubernetes-simple-agent"

  name      = "my-agent"
  namespace = "agents"

  config_files = {
    agent        = "${path.module}/config/agent.yaml"
    server       = "${path.module}/config/server.yaml"
    instructions = "${path.module}/config/instructions.md"
  }

  secrets = {
    anthropic_api_key = var.anthropic_api_key
  }

  deployment = {
    image_tag = "1.0.0"
    agent_url = "https://my-agent.example.com"
    resources = {
      requests = { cpu = "250m", memory = "256Mi" }
      limits   = { memory = "512Mi" }
    }
  }

  ingress = {
    enabled         = true
    class_name      = "nginx"
    host            = "my-agent.example.com"
    tls_secret_name = "my-agent-tls"
  }

  labels = {
    "app.kubernetes.io/part-of" = "my-platform"
    "environment"               = "production"
  }
}
```

### Instructions file injection

When `config_files.instructions` is provided, the module reads the file's contents and injects them into the `instructions` field of the `agent.yaml` ConfigMap data at apply time. This lets you maintain your system prompt as a standalone, version-controlled file rather than an inline YAML string.

The `agent.yaml` you supply can omit the `instructions` field entirely — the module will merge it in. If the file already contains an `instructions` field, the supplied instructions file takes precedence.

## What this module creates

| Resource                   | Description                                                                                                             |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `kubernetes_config_map_v1` | Holds `agent.yaml` and `server.yaml`; optionally merges instructions from a separate file.                              |
| `kubernetes_secret_v1`     | Holds agent secrets as individual files under `/etc/agent/secrets/`. Only created when at least one secret is provided. |
| `kubernetes_service_v1`    | ClusterIP Service exposing the agent on the configured port.                                                            |
| `kubernetes_deployment_v1` | Runs the simple-agent container with config and secrets mounted as directories (no `sub_path`, preserving hot-reload).  |
| `kubernetes_ingress_v1`    | Optional. Created only when `ingress.enabled = true`.                                                                   |

<!-- BEGIN_TF_DOCS -->
## Requirements

| Name | Version |
| ---- | ------- |
| <a name="requirement_terraform"></a> [terraform](#requirement\_terraform) | >= 1.3.0 |
| <a name="requirement_kubernetes"></a> [kubernetes](#requirement\_kubernetes) | >= 2.20.0 |
| <a name="requirement_litellm"></a> [litellm](#requirement\_litellm) | ~> 1.4 |

## Providers

| Name | Version |
| ---- | ------- |
| <a name="provider_kubernetes"></a> [kubernetes](#provider\_kubernetes) | >= 2.20.0 |
| <a name="provider_litellm"></a> [litellm](#provider\_litellm) | ~> 1.4 |

## Modules

No modules.

## Resources

| Name | Type |
| ---- | ---- |
| [kubernetes_config_map_v1.agent](https://registry.terraform.io/providers/hashicorp/kubernetes/latest/docs/resources/config_map_v1) | resource |
| [kubernetes_deployment_v1.agent](https://registry.terraform.io/providers/hashicorp/kubernetes/latest/docs/resources/deployment_v1) | resource |
| [kubernetes_ingress_v1.agent](https://registry.terraform.io/providers/hashicorp/kubernetes/latest/docs/resources/ingress_v1) | resource |
| [kubernetes_manifest.network_policy](https://registry.terraform.io/providers/hashicorp/kubernetes/latest/docs/resources/manifest) | resource |
| [kubernetes_secret_v1.agent](https://registry.terraform.io/providers/hashicorp/kubernetes/latest/docs/resources/secret_v1) | resource |
| [kubernetes_service_v1.agent](https://registry.terraform.io/providers/hashicorp/kubernetes/latest/docs/resources/service_v1) | resource |
| [litellm_agent.agent](https://registry.terraform.io/providers/ncecere/litellm/latest/docs/resources/agent) | resource |
| [litellm_key.agent](https://registry.terraform.io/providers/ncecere/litellm/latest/docs/resources/key) | resource |

## Inputs

| Name | Description | Type | Default | Required |
| ---- | ----------- | ---- | ------- | :------: |
| <a name="input_config_files"></a> [config\_files](#input\_config\_files) | Paths to the agent configuration files. Contents are read at plan/apply time and injected into the ConfigMap. | <pre>object({<br/>    agent        = string<br/>    server       = string<br/>    instructions = optional(string)<br/>  })</pre> | n/a | yes |
| <a name="input_deployment"></a> [deployment](#input\_deployment) | Deployment configuration overrides. | <pre>object({<br/>    image     = optional(string, "ghcr.io/cmlccie/agentic/simple-agent")<br/>    image_tag = optional(string, "latest")<br/>    replicas  = optional(number, 1)<br/>    port      = optional(number, 8000)<br/>    agent_url = optional(string)<br/>    log_level = optional(string, "info")<br/>    resources = optional(object({<br/>      requests = optional(object({<br/>        cpu    = optional(string, "100m")<br/>        memory = optional(string, "128Mi")<br/>      }), {})<br/>      limits = optional(object({<br/>        cpu    = optional(string)<br/>        memory = optional(string, "512Mi")<br/>      }), {})<br/>    }), {})<br/>  })</pre> | `{}` | no |
| <a name="input_ingress"></a> [ingress](#input\_ingress) | Ingress configuration. Ingress is not deployed unless enabled = true. | <pre>object({<br/>    enabled         = optional(bool, false)<br/>    class_name      = optional(string)<br/>    annotations     = optional(map(string), {})<br/>    host            = optional(string)<br/>    path            = optional(string, "/")<br/>    path_type       = optional(string, "Prefix")<br/>    tls_secret_name = optional(string)<br/>  })</pre> | `{}` | no |
| <a name="input_labels"></a> [labels](#input\_labels) | Additional labels to apply to all Kubernetes resources. Merged with module defaults; caller values override defaults. | `map(string)` | `{}` | no |
| <a name="input_litellm_integration"></a> [litellm\_integration](#input\_litellm\_integration) | Optional LiteLLM registration. When enabled, the module creates a litellm\_key (outbound auth for this agent's own calls to LiteLLM) and a litellm\_agent (A2A registration), and injects the generated key into the agent's Kubernetes Secret automatically. Agent card fields (name, description, provider organization/url) are read from config\_files.agent, falling back to generic defaults when not present -- no separate input needed. | <pre>object({<br/>    enabled = optional(bool, false)<br/>    models  = optional(list(string), ["agent-model"])<br/>  })</pre> | `{}` | no |
| <a name="input_name"></a> [name](#input\_name) | Agent deployment name slug. Used as the base name for all Kubernetes resources. Must be a valid Kubernetes domain name (lowercase alphanumeric and hyphens, must start and end with alphanumeric). | `string` | n/a | yes |
| <a name="input_namespace"></a> [namespace](#input\_namespace) | Kubernetes namespace to deploy into. | `string` | `"default"` | no |
| <a name="input_network_policy"></a> [network\_policy](#input\_network\_policy) | Optional Kubernetes manifest content (e.g. CiliumNetworkPolicy YAML) to deploy alongside the agent. Pass raw manifest text, not a file path -- callers decide how to obtain it (file(), templatefile(), generated inline, etc). Null (default) means no policy is deployed. | `string` | `null` | no |
| <a name="input_secrets"></a> [secrets](#input\_secrets) | Agent secrets mounted as files at /etc/agent/secrets/. Only non-null values are written to the Kubernetes Secret. | <pre>object({<br/>    anthropic_api_key    = optional(string)<br/>    openai_api_key       = optional(string)<br/>    agent_model_base_url = optional(string)<br/>    agent_model_api_key  = optional(string)<br/>    agent_redis_url      = optional(string)<br/>    additional           = optional(map(string), {})<br/>  })</pre> | `{}` | no |
| <a name="input_service"></a> [service](#input\_service) | Service configuration. | <pre>object({<br/>    type = optional(string, "ClusterIP")<br/>    port = optional(number, 80)<br/>  })</pre> | `{}` | no |

## Outputs

| Name | Description |
| ---- | ----------- |
| <a name="output_ingress_hostname"></a> [ingress\_hostname](#output\_ingress\_hostname) | Ingress hostname, or null if ingress is disabled or no host was specified. |
| <a name="output_name"></a> [name](#output\_name) | Base name used for all Kubernetes resources. |
| <a name="output_namespace"></a> [namespace](#output\_namespace) | Kubernetes namespace the agent is deployed into. |
| <a name="output_service_name"></a> [service\_name](#output\_service\_name) | Name of the ClusterIP Service fronting the agent. |
<!-- END_TF_DOCS -->
