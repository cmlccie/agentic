variable "name" {
  description = "Orchestrator deployment name slug. Used as the base name for all Kubernetes resources. Must be a valid Kubernetes domain name (lowercase alphanumeric and hyphens, must start and end with alphanumeric)."
  type        = string

  validation {
    condition     = can(regex("^[a-z0-9]([a-z0-9-]*[a-z0-9])?$", var.name))
    error_message = "name must be lowercase alphanumeric with optional internal hyphens."
  }
}

variable "namespace" {
  description = "Kubernetes namespace to deploy into."
  type        = string
  default     = "default"
}

variable "labels" {
  description = "Additional labels to apply to all Kubernetes resources. Merged with module defaults; caller values override defaults."
  type        = map(string)
  default     = {}
}

variable "config_files" {
  description = "Paths to the orchestrator configuration files. Contents are read at plan/apply time and injected into the ConfigMap. The list of downstream A2A servers is part of agent.yaml."
  type = object({
    agent        = string
    server       = string
    instructions = optional(string)
  })
}

variable "secrets" {
  description = "Orchestrator secrets mounted as files at /etc/agent/secrets/. Only non-null values are written to the Kubernetes Secret. Use 'additional' for downstream A2A bearer tokens referenced as $${SECRET_KEY} in agent.yaml headers."
  sensitive   = true
  type = object({
    anthropic_api_key    = optional(string)
    openai_api_key       = optional(string)
    agent_model_base_url = optional(string)
    agent_model_api_key  = optional(string)
    agent_database_url   = optional(string)
    additional           = optional(map(string), {})
  })
  default = {}
}

variable "deployment" {
  description = "Deployment configuration overrides."
  type = object({
    image     = optional(string, "ghcr.io/cmlccie/agentic/orchestrator-agent")
    image_tag = optional(string, "latest")
    replicas  = optional(number, 1)
    port      = optional(number, 8000)
    agent_url = optional(string)
    log_level = optional(string, "info")
    resources = optional(object({
      requests = optional(object({
        cpu    = optional(string, "100m")
        memory = optional(string, "128Mi")
      }), {})
      limits = optional(object({
        cpu    = optional(string)
        memory = optional(string, "512Mi")
      }), {})
    }), {})
  })
  default = {}
}

variable "service" {
  description = "Service configuration."
  type = object({
    type = optional(string, "ClusterIP")
    port = optional(number, 80)
  })
  default = {}
}

variable "ingress" {
  description = "Ingress configuration. Ingress is not deployed unless enabled = true."
  type = object({
    enabled         = optional(bool, false)
    class_name      = optional(string)
    annotations     = optional(map(string), {})
    host            = optional(string)
    path            = optional(string, "/")
    path_type       = optional(string, "Prefix")
    tls_secret_name = optional(string)
  })
  default = {}
}
