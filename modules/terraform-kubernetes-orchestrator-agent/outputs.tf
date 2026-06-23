output "name" {
  description = "Base name used for all Kubernetes resources."
  value       = var.name
}

output "namespace" {
  description = "Kubernetes namespace the agent is deployed into."
  value       = var.namespace
}

output "service_name" {
  description = "Name of the ClusterIP Service fronting the agent."
  value       = kubernetes_service_v1.agent.metadata[0].name
}

output "ingress_hostname" {
  description = "Ingress hostname, or null if ingress is disabled or no host was specified."
  value       = local.ingress_hostname
}
