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

	# Exporting updated Dependencies
	uv export -f requirements.txt --group tools_mcp_weather -o tools/mcp/weather_server/requirements.txt

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

tools-mcp-weather-server: ## Build the MCP Weather Server
	podman build -f tools/mcp/weather_server/Containerfile -t tools-mcp-weather-server:latest tools/mcp/weather_server/
