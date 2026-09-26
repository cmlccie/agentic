---
applyTo: "**/Containerfile"
---

# Container Image Instructions

## Container Best Practices

### Base Images

- Use official Python images with Alpine Linux variants for smaller size: `python:3.13-alpine`
- Pin specific versions for reproducibility
- Use multi-stage builds when creating production images to minimize final image size
- Build application images `FROM ${BASE_IMAGE}` with `ARG BASE_IMAGE=ghcr.io/cmlccie/agentic/python:latest` declared before `FROM`, so builds can target a specific (digest-pinned or locally built) base image

### Agentic Python Base Image

`images/python/Containerfile` provides the shared runtime for application images:

- Dependencies from a hashed `requirements.txt`, installed with `pip install --require-hashes`
- The `agentic` project sdist, installed separately with `--no-deps`
- `PYTHONDONTWRITEBYTECODE=1`, `PYTHONUNBUFFERED=1`, and `PYDANTIC_AI_NO_BANNER=1`
- A non-root `appuser` with UID and GID 10000, owning `/app`
- `WORKDIR /app` (the base image does not switch users)

### Container Structure

- Set working directory to `/app`
- Copy `requirements.txt` first to leverage layer caching
- Install dependencies before copying application code
- Copy application files and set permissions while still root user
- Create non-root user and set ownership after file operations
- Make scripts executable with `RUN chmod +x <script>` before switching users
- Use `ENTRYPOINT` for the main command and `CMD` for default arguments
- When using executable Python scripts with shebang, ENTRYPOINT can reference the script directly

### Security & Operations

- Don't run as root user when possible - create a non-root user and switch to it
- Switch users with a numeric `USER 10000:10000` (Kubernetes `runAsNonRoot` can only verify numeric UIDs)
- Set file permissions before switching to non-root user to avoid permission errors
- Use `--no-cache-dir` with pip to reduce image size
- Don't install build tools (e.g. `git`) unless a dependency needs them
- Don't write to the image filesystem at runtime, so containers can run with a read-only root filesystem (mount writable volumes such as `/tmp` instead)
- Expose only necessary ports
- Include health checks when appropriate

### Dependencies

- Generate `requirements.txt` with `uv export --frozen --no-dev --no-emit-project --format requirements.txt` (see the `images/python/requirements.txt` Makefile target)
- Include hash verification for security (`uv export` emits hashes by default; don't pass `--no-hashes`)
- Pin all dependency versions for reproducibility, including packages installed directly in a Containerfile (e.g. `pip install faker==<version>`)

## Example Containerfile Pattern

Standalone image:

```dockerfile
FROM python:3.13-alpine

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --require-hashes -r requirements.txt

# Copy and set permissions while still root
COPY <script_name>.py .
RUN chmod +x <script_name>.py

# Create non-root user and set ownership
RUN addgroup -g 10000 appuser && adduser -D -u 10000 -G appuser -s /bin/sh appuser
RUN chown -R appuser:appuser /app
USER 10000:10000

# Expose port if needed (e.g., for HTTP services)
EXPOSE 8000

ENTRYPOINT ["<script_name>.py"]
CMD ["<default_arg>"]
```

Application image built on the Agentic Python base image:

```dockerfile
ARG BASE_IMAGE=ghcr.io/cmlccie/agentic/python:latest
FROM ${BASE_IMAGE}

COPY <script_name>.py README.md ./
RUN chmod +x <script_name>.py
RUN chown -R appuser:appuser /app
USER 10000:10000

EXPOSE 8000

ENTRYPOINT ["/app/<script_name>.py"]
CMD ["stdio"]
```
