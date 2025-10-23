---
applyTo: ".github/workflows/*.yml"
---

# GitHub Actions Workflow Instructions

## Container Build Workflows

### Naming Convention

- Use `build-<service-name>.yml` for container build workflows
- Use descriptive names that clearly identify the service being built

### Workflow Triggers

Configure appropriate triggers for the workflow:

```yaml
on:
  push:
    branches:
      - main
    paths:
      - "<service-path>/**"
      - ".github/workflows/<workflow-file>.yml"
  pull_request:
    branches:
      - main
    paths:
      - "<service-path>/**"
      - ".github/workflows/<workflow-file>.yml"
  release:
    types: [published]
```

### Environment Variables

```yaml
env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}/<service-name>
```

### Required Permissions

```yaml
permissions:
  contents: read
  packages: write
  id-token: write
  attestations: write
```

### Essential Workflow Steps

1. **Repository Checkout**

   ```yaml
   - name: Checkout repository
     uses: actions/checkout@v4
   ```

2. **Docker Buildx Setup**

   ```yaml
   - name: Set up Docker Buildx
     uses: docker/setup-buildx-action@v3
   ```

3. **Registry Login**

   ```yaml
   - name: Log in to Container Registry
     uses: docker/login-action@v3
     with:
       registry: ${{ env.REGISTRY }}
       username: ${{ github.actor }}
       password: ${{ secrets.GITHUB_TOKEN }}
   ```

4. **Metadata Extraction**

   ```yaml
   - name: Extract metadata
     id: meta
     uses: docker/metadata-action@v5
     with:
       images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
       tags: |
         type=ref,event=branch
         type=ref,event=pr
         type=semver,pattern={{version}}
         type=semver,pattern={{major}}.{{minor}}
         type=semver,pattern={{major}}
         type=sha,prefix={{branch}}-
         type=raw,value=latest,enable={{is_default_branch}}
   ```

5. **Build and Push**

   ```yaml
   - name: Build and push Docker image
     id: build
     uses: docker/build-push-action@v5
     with:
       context: ./<service-path>
       platforms: linux/amd64,linux/arm64
       push: true
       tags: ${{ steps.meta.outputs.tags }}
       labels: ${{ steps.meta.outputs.labels }}
       cache-from: type=gha
       cache-to: type=gha,mode=max
   ```

6. **Build Attestation**
   ```yaml
   - name: Generate artifact attestation
     uses: actions/attest-build-provenance@v1
     with:
       subject-name: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
       subject-digest: ${{ steps.build.outputs.digest }}
       push-to-registry: true
   ```

### Image Tagging Strategy

- Use semantic versioning patterns for releases
- Include branch names for development builds
- Add commit SHA for traceability
- Use `latest` tag only for default branch builds
- Support both AMD64 and ARM64 architectures when possible

### Performance Optimization

- Enable GitHub Actions caching for Docker layers
- Use path filters to trigger builds only when relevant files change
- Use multi-platform builds for broader compatibility
- Leverage Docker BuildKit features for faster builds

### Security Considerations

- Use build attestations for supply chain security
- Limit workflow permissions to minimum required
- Use GitHub's built-in GITHUB_TOKEN for authentication
- Pin action versions to specific commits or tags
