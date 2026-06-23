resource "kubernetes_config_map_v1" "agent" {
  metadata {
    name      = var.name
    namespace = var.namespace
    labels    = local.labels
  }

  data = {
    "agent.yaml"  = local.agent_yaml_content
    "server.yaml" = file(var.config_files.server)
  }
}
