locals {
  # Default labels merged with caller-supplied labels; caller values override defaults.
  labels = merge(
    {
      "app.kubernetes.io/name"       = var.name
      "app.kubernetes.io/component"  = "orchestrator"
      "app.kubernetes.io/managed-by" = "terraform"
    },
    var.labels,
  )

  # Stable selector used for Deployment.spec.selector.matchLabels and Service.spec.selector.
  # Must NOT include var.labels — Deployment selectors are immutable after creation,
  # so this set must be derived only from stable inputs (var.name never changes for a given resource).
  selector_labels = {
    "app.kubernetes.io/name" = var.name
  }

  # Instructions injection: if a path is supplied, decode agent.yaml, merge in the
  # instructions key, and re-encode. yamlencode produces canonical but semantically
  # equivalent YAML (the agent uses yaml.safe_load, so format differences are harmless).
  agent_yaml_content = var.config_files.instructions != null ? yamlencode(
    merge(
      yamldecode(file(var.config_files.agent)),
      { instructions = file(var.config_files.instructions) }
    )
  ) : file(var.config_files.agent)

  # Secret aggregation: filter out null named values, merge with agent_secrets map.
  #
  # Invariant: any single object-typed secrets variable must be sourced from exactly
  # one layer -- all terragrunt `inputs`, or all `secrets.auto.tfvars`, never split
  # across both (Terraform does not deep-merge object-typed variables across variable
  # precedence levels). openai_compatible and task_broker are inputs-only; agent_secrets
  # is the sole tfvars-only variable -- that's what keeps the two sources from colliding.
  named_secrets = {
    for k, v in {
      "openai_compatible.base_url" = var.openai_compatible.base_url
      "openai_compatible.api_key"  = var.openai_compatible.api_key
      "task_broker.database_url"   = var.task_broker.database_url
    } : k => v if v != null
  }
  all_secrets = merge(local.named_secrets, var.agent_secrets)
  has_secrets = length(local.all_secrets) > 0

  # Container image and CLI args.
  image = "${var.deployment.image}:${var.deployment.image_tag}"
  agent_args = concat(
    ["serve"],
    var.deployment.agent_url != null ? ["--agent-url", var.deployment.agent_url] : [],
    ["--log-level", coalesce(var.deployment.log_level, "info")],
  )

  # Resource maps. limits is a map(string) attribute (not a sub-block), so we assign
  # null to omit it entirely when no limits are configured.
  resource_requests = {
    cpu    = try(var.deployment.resources.requests.cpu, "100m")
    memory = try(var.deployment.resources.requests.memory, "128Mi")
  }
  resource_limits_raw = {
    for k, v in {
      cpu    = try(var.deployment.resources.limits.cpu, null)
      memory = try(var.deployment.resources.limits.memory, "512Mi")
    } : k => v if v != null
  }
  resource_limits = length(local.resource_limits_raw) > 0 ? local.resource_limits_raw : null

  # Ingress output value.
  ingress_hostname = var.ingress.enabled ? var.ingress.host : null
}
