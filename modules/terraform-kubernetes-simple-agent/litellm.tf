resource "litellm_key" "agent" {
  count = var.litellm_integration.enabled ? 1 : 0

  key_alias = var.name
  models    = var.litellm_integration.models

  metadata = {
    managed_by = "terraform"
  }
}

resource "litellm_agent" "agent" {
  count = var.litellm_integration.enabled ? 1 : 0

  agent_name = var.name

  agent_card {
    name        = local.agent_config.name
    description = try(local.agent_config.description, null)
    url         = "${var.deployment.agent_url}/a2a"
  }

  object_permission {
    models = var.litellm_integration.models
  }
}
