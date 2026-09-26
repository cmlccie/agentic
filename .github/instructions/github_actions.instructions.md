---
applyTo: ".github/workflows/*.yml"
---

# GitHub Actions Workflow Instructions

## CI Workflow

`.github/workflows/ci.yml` runs on every push and pull request to `main` (no path filters) with `contents: read` permissions only. It runs these checks:

- `uv lock --check` (lockfile consistent with `pyproject.toml`)
- `uv sync --frozen --all-groups`
- `uv run ruff check .` and `uv run ruff format --check .`
- `uv run pytest` (collects `tests/` and tests next to tool servers under `tools/`)
- `make --always-make images/python/requirements.txt` followed by `git diff --exit-code` (requirements drift check)

When changing how `requirements.txt` is exported, change the Makefile target; the CI drift check uses it.

## Python Container Build Workflows

### Naming Convention

- Use `build-<image-name>.yml` for container build workflows
- Use descriptive names that clearly identify the service being built

### Using the Reusable Workflow

For Python container builds, use the reusable workflow `.github/workflows/reusable-build-python-container-image.yml`.

### Required Inputs

- **image-name**: The name of the container image (will be prefixed with repository name)
- **context-path**: Path to the build context relative to repository root

### Optional Inputs

- **containerfile-path**: Path to the Containerfile relative to context-path (defaults to "Containerfile")
- **platforms**: Target platforms for the build (defaults to "linux/amd64,linux/arm64")
- **artifact-name**: Name of a Python package artifact to download into `<context-path>/dist` (defaults to none)
- **base-image**: Image passed as the `BASE_IMAGE` build-arg (defaults to the Containerfile's `ARG BASE_IMAGE` default)

### Triggers

- Push and pull request to `main` with path filters for the component, its workflow file, and the reusable workflow
- Published releases
- `workflow_run` after "Build Python Base Image" completes on `main` (for images built `FROM` the base image)

Jobs triggered by `workflow_run` run on every completion of the upstream workflow, including failures. Guard the calling job so application images are only rebuilt on top of a successful base image build:

```yaml
jobs:
  build:
    # Skip workflow_run builds when the base image build did not succeed.
    if: ${{ github.event_name != 'workflow_run' || github.event.workflow_run.conclusion == 'success' }}
    uses: ./.github/workflows/reusable-build-python-container-image.yml
```

### Base Image Dependency

Application images build `FROM ${BASE_IMAGE}`, which defaults to the published `ghcr.io/cmlccie/agentic/python:latest`. Changes to `src/agentic`, `pyproject.toml`, or `uv.lock` reach application images through the base image: the base image workflow publishes a new `latest` on `main`, and its `workflow_run` completion rebuilds the application images. Application workflows therefore do not path-filter on `src/**`; a direct trigger would build against the previous base image.

Pull request builds of application images use the published base image, so they do not exercise unmerged `src/agentic` changes. `CI` covers those changes with tests. Pass `base-image` (for example, a digest-pinned reference) when a build must be reproducible against a specific base image.

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
- Commit SHA for traceability (`{{branch}}-{{sha}}`), except on pull request builds
- `latest` tag only for default branch builds
- Pull request references for PR builds (computed but not pushed)

### Automatic Features

The reusable workflow includes:

- **Multi-platform builds**: Supports both AMD64 and ARM64 architectures by default
- **GitHub Actions caching**: Optimizes build performance with layer caching
- **Build-only pull requests**: Builds without pushing, logging in, or attesting on `pull_request` events (works for fork PRs and keeps PR tags out of GHCR)
- **Build attestations**: Provides supply chain security for pushed images
- **Proper permissions**: Includes all necessary permissions for container registry operations
- **Registry authentication**: Automatically handles GitHub Container Registry login when pushing

Workflows with their own build steps (such as `build-a2a-inspector.yml`) follow the same rules: `push: ${{ github.event_name != 'pull_request' }}`, and skip the login and attestation steps when not pushing.
