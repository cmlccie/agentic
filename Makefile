# --------------------------------------------------------------------------------------
# Container Engine Detection
# --------------------------------------------------------------------------------------

# Detect container engine (podman or docker)
CONTAINER_ENGINE := $(shell command -v docker 2>/dev/null || command -v podman 2>/dev/null)

ifeq ($(CONTAINER_ENGINE),)
$(error No container engine found. Please install docker or podman)
endif

# --------------------------------------------------------------------------------------
# Repository Targets
# --------------------------------------------------------------------------------------

.PHONY: help setup upgrade clean lint format check

help: ## Show this help message
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*##"; printf "\n"} /^[a-zA-Z_-]+:.*##/ { printf "  %-12s %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

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

# -------------------------------------------------------------------------------------------------
# Build Targets
# -------------------------------------------------------------------------------------------------

.PHONY: python-base-image agents-weather-agent tools-mcp-weather-server

images/python/requirements.txt: pyproject.toml uv.lock ## Export requirements.txt for the Python Base Image
	uv export --no-dev --no-emit-project --format requirements.txt -o images/python/requirements.txt

images/python/dist: src/ pyproject.toml ## Build the project distribution packages for the Python Base Image
	rm -rf images/python/dist
	uv build --sdist --out-dir images/python/dist

python-base-image: images/python/dist images/python/requirements.txt ## Build the Python Base Image
	$(CONTAINER_ENGINE) build -f images/python/Containerfile -t agentic/python:local images/python/

agents-weather-agent: ## Build the Weather Agent
	$(CONTAINER_ENGINE) build -f agents/weather_agent/Containerfile -t agentic/agents-weather-agent:local agents/weather_agent/

tools-mcp-weather-server: ## Build the MCP Weather Server
	$(CONTAINER_ENGINE) build -f tools/mcp/weather_server/Containerfile -t agentic/tools-mcp-weather-server:local tools/mcp/weather_server/
