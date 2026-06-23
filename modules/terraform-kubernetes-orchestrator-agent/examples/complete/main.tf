provider "kubernetes" {
  config_path = "~/.kube/config"
}

module "orchestrator_agent" {
  source = "../.."

  name      = "my-orchestrator"
  namespace = "agents"

  config_files = {
    agent        = "${path.module}/config/agent.yaml"
    server       = "${path.module}/config/server.yaml"
    instructions = "${path.module}/config/instructions.md"
  }

  secrets = {
    anthropic_api_key = var.anthropic_api_key

    # Persist A2A tasks in PostgreSQL (set broker.backend: postgres in server.yaml).
    agent_database_url = var.agent_database_url

    # Bearer tokens for downstream A2A agents, referenced as $${WEATHER_AGENT_TOKEN}
    # in the agent.yaml a2a_servers[].headers.
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

  service = {
    port = 80
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

variable "anthropic_api_key" {
  description = "Anthropic API key."
  type        = string
  sensitive   = true
}

variable "agent_database_url" {
  description = "SQLAlchemy async DSN for the A2A task store (PostgreSQL)."
  type        = string
  sensitive   = true
  default     = null
}

variable "weather_agent_token" {
  description = "Bearer token for the downstream weather A2A agent."
  type        = string
  sensitive   = true
  default     = null
}

output "service_name" {
  value = module.orchestrator_agent.service_name
}
