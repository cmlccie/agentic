---
applyTo: "**/Containerfile"
---

# Container Image Instructions

## Container Best Practices

### Base Images

- Use official Python images with Alpine Linux variants for smaller size: `python:3.13-alpine`
- Pin specific versions for reproducibility
- Use multi-stage builds when creating production images to minimize final image size

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
- Set file permissions before switching to non-root user to avoid permission errors
- Use `--no-cache-dir` with pip to reduce image size
- Expose only necessary ports
- Include health checks when appropriate

### Dependencies

- Generate `requirements.txt` using `uv export --only-group <group-name>`
- Include hash verification for security
- Pin all dependency versions for reproducibility

## Example Containerfile Pattern

```dockerfile
FROM python:3.13-alpine

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy and set permissions while still root
COPY <script_name>.py .
RUN chmod +x <script_name>.py

# Create non-root user and set ownership
RUN adduser -D -s /bin/sh appuser
RUN chown -R appuser:appuser /app
USER appuser

# Expose port if needed (e.g., for HTTP services)
EXPOSE 8000

ENTRYPOINT ["<script_name>.py"]
CMD ["<default_arg>"]
```
