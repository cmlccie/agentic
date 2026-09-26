# Python Base Image

The base image for every Python component in this repository: `python:3.13-alpine` with the project's locked dependencies (installed from `requirements.txt` with hash verification) and the `agentic` package (installed from a source distribution built from `src/`). It creates the non-root `appuser` (UID/GID 10000) and the `/app` working directory, and sets `PYTHONUNBUFFERED`, `PYTHONDONTWRITEBYTECODE`, and `PYDANTIC_AI_NO_BANNER`. App images build `FROM` it and switch to `USER 10000:10000`; the base image itself stays root so app images can install files first.

## Build

```bash
make python-base-image      # exports requirements.txt, builds the sdist, tags agentic/python:local
```

`images/python/requirements.txt` is generated from `uv.lock` (`make --always-make images/python/requirements.txt`) and committed; CI fails if it drifts from the lockfile.

## Usage

Build an app image against the local base image:

```bash
make simple-agent BASE_IMAGE=agentic/python:local
```

Or in a Containerfile:

```dockerfile
ARG BASE_IMAGE=ghcr.io/cmlccie/agentic/python:latest
FROM ${BASE_IMAGE}
COPY my_tool.py /app/
USER 10000:10000
ENTRYPOINT ["python", "/app/my_tool.py"]
```

The published image is `ghcr.io/cmlccie/agentic/python`, rebuilt when `src/`, `pyproject.toml`, or this directory (including the generated `requirements.txt`) changes on `main`.
