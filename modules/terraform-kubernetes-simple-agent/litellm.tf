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

  # NOTE: the ncecere/litellm provider's litellm_agent resource does not mark several
  # agent_card/object_permission sub-fields as Computed, even though the LiteLLM API
  # defaults them server-side when omitted (version, protocol_version, capabilities,
  # provider, skills, and the object_permission list fields). Leaving them unset causes
  # a "Provider produced inconsistent result after apply" error on create. Set every
  # one of them explicitly so the plan already matches what the API will return.
  agent_card {
    name             = local.agent_config.name
    description      = try(local.agent_config.description, null)
    url              = "${var.deployment.agent_url}/a2a"
    version          = "1.0.0"
    protocol_version = "1.0"

    capabilities {}

    provider {
      organization = "LiteLLM Proxy"
      url          = "http://models.lab.lunsford.io"
    }

    skills {
      id          = var.name
      name        = local.agent_config.name
      description = try(local.agent_config.description, null)
    }
  }

  object_permission {
    models            = var.litellm_integration.models
    mcp_servers       = []
    mcp_access_groups = []
    agents            = []
  }
}
