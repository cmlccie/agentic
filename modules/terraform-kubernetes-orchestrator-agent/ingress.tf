resource "kubernetes_ingress_v1" "agent" {
  count = var.ingress.enabled ? 1 : 0

  metadata {
    name      = var.name
    namespace = var.namespace
    labels    = local.labels
    # Dual-write class annotation for controllers that predate spec.ingressClassName.
    annotations = merge(
      var.ingress.class_name != null ? { "kubernetes.io/ingress.class" = var.ingress.class_name } : {},
      var.ingress.annotations,
    )
  }

  spec {
    ingress_class_name = var.ingress.class_name

    dynamic "tls" {
      for_each = var.ingress.tls_secret_name != null ? [1] : []
      content {
        hosts       = var.ingress.host != null ? [var.ingress.host] : []
        secret_name = var.ingress.tls_secret_name
      }
    }

    rule {
      host = var.ingress.host
      http {
        path {
          path      = var.ingress.path
          path_type = var.ingress.path_type
          backend {
            service {
              name = kubernetes_service_v1.agent.metadata[0].name
              port {
                number = var.service.port
              }
            }
          }
        }
      }
    }
  }
}
