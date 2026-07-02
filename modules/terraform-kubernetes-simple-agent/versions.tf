terraform {
  required_version = ">= 1.3.0"

  required_providers {
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = ">= 2.20.0"
    }
    litellm = {
      source  = "ncecere/litellm"
      version = "~> 1.4"
    }
  }
}
