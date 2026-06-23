resource "kubernetes_secret_v1" "agent" {
  count = local.has_secrets ? 1 : 0

  metadata {
    name      = var.name
    namespace = var.namespace
    labels    = local.labels
  }

  data = local.all_secrets
}
