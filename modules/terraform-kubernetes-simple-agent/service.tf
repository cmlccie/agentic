resource "kubernetes_service_v1" "agent" {
  metadata {
    name      = var.name
    namespace = var.namespace
    labels    = local.labels
  }

  spec {
    selector = local.selector_labels
    type     = var.service.type

    port {
      name        = "http"
      port        = var.service.port
      target_port = "http"
      protocol    = "TCP"
    }
  }
}
