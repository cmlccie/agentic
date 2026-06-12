provider "kubernetes" {
  config_path = "~/.kube/config"
}

module "simple_agent" {
  source = "../.."

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

  service = {
    port = 80
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

variable "anthropic_api_key" {
  description = "Anthropic API key."
  type        = string
  sensitive   = true
}

output "service_name" {
  value = module.simple_agent.service_name
}
