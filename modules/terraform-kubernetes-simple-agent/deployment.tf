resource "kubernetes_deployment_v1" "agent" {
  metadata {
    name      = var.name
    namespace = var.namespace
    labels    = local.labels
  }

  spec {
    replicas = var.deployment.replicas

    selector {
      match_labels = local.selector_labels
    }

    template {
      metadata {
        labels = local.labels
      }

      spec {
        security_context {
          run_as_non_root = true
          run_as_user     = 10000
          run_as_group    = 10000
          fs_group        = 10000
        }

        container {
          name              = "agent"
          image             = local.image
          image_pull_policy = var.deployment.image_tag == "latest" ? "Always" : "IfNotPresent"
          args              = local.agent_args

          port {
            name           = "http"
            container_port = var.deployment.port
            protocol       = "TCP"
          }

          resources {
            requests = local.resource_requests
            limits   = local.resource_limits
          }

          # Config volume: always mounted as a directory; sub_path must not be used —
          # it bypasses AtomicWriter and prevents hot-reload from receiving file updates.
          volume_mount {
            name       = "config"
            mount_path = "/etc/agent/config"
            read_only  = true
          }

          dynamic "volume_mount" {
            for_each = local.has_secrets ? [1] : []
            content {
              name       = "secrets"
              mount_path = "/etc/agent/secrets"
              read_only  = true
            }
          }

          liveness_probe {
            http_get {
              path = "/health/live"
              port = "http"
            }
            initial_delay_seconds = 5
            period_seconds        = 10
            timeout_seconds       = 5
            failure_threshold     = 3
          }

          readiness_probe {
            http_get {
              path = "/health/ready"
              port = "http"
            }
            initial_delay_seconds = 5
            period_seconds        = 5
            timeout_seconds       = 3
            # 12 × 5s = 60s window; covers the 30s drain_timeout default with margin.
            failure_threshold = 12
          }
        }

        volume {
          name = "config"
          config_map {
            name = kubernetes_config_map_v1.agent.metadata[0].name
          }
        }

        dynamic "volume" {
          for_each = local.has_secrets ? [1] : []
          content {
            name = "secrets"
            secret {
              secret_name = kubernetes_secret_v1.agent[0].metadata[0].name
            }
          }
        }
      }
    }
  }

  wait_for_rollout = true
}
