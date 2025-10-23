---
applyTo: ".github/workflows/*.yml"
---

# GitHub Actions Workflow Instructions

## Python Container Build Workflows

### Naming Convention

- Use `build-<image-name>.yml` for container build workflows
- Use descriptive names that clearly identify the service being built

### Using the Reusable Workflow

For Python container builds, use the reusable workflow `.github/workflows/build-python-container-image.yml`.

### Required Inputs

- **image-name**: The name of the container image (will be prefixed with repository name)
- **context-path**: Path to the build context relative to repository root

### Optional Inputs

- **containerfile-path**: Path to the Containerfile relative to context-path (defaults to "Containerfile")
- **platforms**: Target platforms for the build (defaults to "linux/amd64,linux/arm64")

### Example Workflow

See `.github/workflows/build-tools-mcp-weather-server.yml` for a complete working example of a Python container build workflow using the reusable workflow.

### Benefits of the Reusable Workflow

- **Consistency**: All Python container builds use the same standardized process
- **Maintainability**: Updates to the build process only need to be made in one place
- **Best Practices**: Automatically includes security features like build attestations
- **Performance**: Built-in caching and multi-platform support

### Image Tagging Strategy

The reusable workflow automatically handles image tagging with:

- Semantic versioning patterns for releases (`{{version}}`, `{{major}}.{{minor}}`, `{{major}}`)
- Branch names for development builds (`{{branch}}`)
- Commit SHA for traceability (`{{branch}}-{{sha}}`)
- `latest` tag only for default branch builds
- Pull request references for PR builds

### Automatic Features

The reusable workflow includes:

- **Multi-platform builds**: Supports both AMD64 and ARM64 architectures by default
- **GitHub Actions caching**: Optimizes build performance with layer caching
- **Build attestations**: Provides supply chain security
- **Proper permissions**: Includes all necessary permissions for container registry operations
- **Registry authentication**: Automatically handles GitHub Container Registry login
