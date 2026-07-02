locals {
  network_policy_manifests = var.network_policy != null ? {
    for manifest in provider::kubernetes::manifest_decode_multi(var.network_policy) :
    "${manifest["kind"]}|${try(manifest["metadata"]["namespace"], "")}|${manifest["metadata"]["name"]}" => manifest
  } : {}
}

resource "kubernetes_manifest" "network_policy" {
  for_each = local.network_policy_manifests

  manifest = each.value
}
