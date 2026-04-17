"""Utility helpers for OpenAI-compatible integrations."""

from urllib.parse import urlparse


def normalize_openai_base_url(base_url: str) -> str:
    """Normalize OpenAI-compatible base URL and ensure it includes /v1.

    Many local OpenAI-compatible servers expose endpoints under /v1.
    If /v1 is omitted, clients may hit /chat/completions and receive malformed
    responses.
    """

    cleaned_base_url = base_url.strip().rstrip("/")
    parsed_url = urlparse(cleaned_base_url)

    if not parsed_url.scheme or not parsed_url.netloc:
        raise EnvironmentError(
            "OPENAI_BASE_URL must be a valid URL, for example: http://127.0.0.1:1234/v1"
        )

    if parsed_url.path in ("", "/"):
        return f"{cleaned_base_url}/v1"

    return cleaned_base_url
