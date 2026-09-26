# --------------------------------------------------------------------------------------
# Repository Targets
# --------------------------------------------------------------------------------------

.PHONY: help setup sync upgrade clean lint format check test

help: ## Show this help message
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*##"; printf "\n"} /^[a-zA-Z0-9_.\/-]+:.*##/ { printf "  %-36s %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

setup: ## Reset and setup development environment
	rm -rf .venv
	uv sync --all-groups

sync: ## Sync all dependencies
	uv sync --all-groups

upgrade: ## Upgrade dependencies to latest versions
	uv sync --upgrade --all-groups

clean: ## Remove temporary and build artifacts
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/

lint: ## Run linting checks
	uv run ruff check .

format: ## Format code automatically
	uv run ruff format .
	uv run ruff check --fix .

check: lint ## Run all checks (lint + format check)
	uv run ruff format --check .

test: ## Run tests
	uv run pytest

# -------------------------------------------------------------------------------------------------
# Build Targets
# -------------------------------------------------------------------------------------------------

# Container engine (podman or docker); checked only when a build target runs.
CONTAINER_ENGINE ?= $(shell command -v podman 2>/dev/null || command -v docker 2>/dev/null)

# Base image for application images. Override to build against a local base image:
#   make python-base-image simple-agent BASE_IMAGE=agentic/python:local
BASE_IMAGE ?= ghcr.io/cmlccie/agentic/python:latest

# Build an application image: $(call build-image,<context-dir>,<image-name>)
build-image = $(CONTAINER_ENGINE) build --build-arg BASE_IMAGE=$(BASE_IMAGE) -f $(1)/Containerfile -t agentic/$(2):local $(1)/

container-engine:
	@test -n "$(CONTAINER_ENGINE)" || { echo "No container engine found. Please install docker or podman." >&2; exit 1; }

.PHONY: container-engine python-base-image images/python/dist simple-agent orchestrator-agent customer-database-seeder tools-mcp-weather-server tools-mcp-meraki-server tools-mcp-aiops-server tools-mcp-provisioning-server tools-mcp-customer-database-server

images/python/requirements.txt: pyproject.toml uv.lock ## Export requirements.txt for the Python Base Image
	uv export --frozen --no-dev --no-emit-project --format requirements.txt -o images/python/requirements.txt

images/python/dist: ## Build the project source distribution for the Python Base Image
	rm -rf images/python/dist
	uv build --sdist --out-dir images/python/dist

python-base-image: images/python/dist images/python/requirements.txt | container-engine ## Build the Python Base Image
	$(CONTAINER_ENGINE) build -f images/python/Containerfile -t agentic/python:local images/python/

simple-agent: | container-engine ## Build the Simple Agent
	$(call build-image,images/simple_agent,simple-agent)

orchestrator-agent: | container-engine ## Build the Orchestrator Agent
	$(call build-image,images/orchestrator_agent,orchestrator-agent)

customer-database-seeder: | container-engine ## Build the Customer Database Seeder
	$(call build-image,images/customer_database_seeder,customer-database-seeder)

tools-mcp-weather-server: | container-engine ## Build the MCP Weather Server
	$(call build-image,tools/mcp/weather_server,tools-mcp-weather-server)

tools-mcp-meraki-server: | container-engine ## Build the MCP Meraki Server
	$(call build-image,tools/mcp/meraki_server,tools-mcp-meraki-server)

tools-mcp-aiops-server: | container-engine ## Build the MCP AIOps Server
	$(call build-image,tools/mcp/aiops_server,tools-mcp-aiops-server)

tools-mcp-provisioning-server: | container-engine ## Build the MCP Provisioning Server
	$(call build-image,tools/mcp/provisioning_server,tools-mcp-provisioning-server)

tools-mcp-customer-database-server: | container-engine ## Build the MCP Customer Database Server
	$(call build-image,tools/mcp/customer_database_server,tools-mcp-customer-database-server)

# -------------------------------------------------------------------------------------------------
# Terraform Module Targets
# -------------------------------------------------------------------------------------------------

.PHONY: tf-init tf-validate tf-fmt tf-fmt-check tf-docs

tf-init: ## Initialize all Terraform modules and their examples
	@for m in modules/*/; do \
		terraform -chdir=$$m init -backend=false; \
		terraform -chdir=$${m}examples/complete init -backend=false; \
	done

tf-validate: tf-init ## Validate all Terraform modules and their examples
	@for m in modules/*/; do \
		terraform -chdir=$$m validate; \
		terraform -chdir=$${m}examples/complete validate; \
	done

tf-fmt: ## Format all Terraform files
	terraform fmt -recursive modules/

tf-fmt-check: ## Check Terraform formatting (no writes)
	terraform fmt -recursive -check modules/

tf-docs: ## Regenerate terraform-docs for all modules
	@for m in modules/*/; do \
		terraform-docs markdown table --output-file README.md --output-mode inject $$m; \
	done
