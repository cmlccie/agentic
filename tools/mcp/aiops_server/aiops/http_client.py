"""HTTP client helpers shared by all checks."""

import json
from typing import Any, AsyncIterator, Dict, Iterator, Optional

import httpx

# Hard server-side cap on any tool's timeout: must stay under the 600s
# request-timeout on the Ingress in front of this server.
TIMEOUT_CAP_S = 570.0

CONNECT_TIMEOUT_S = 10.0


def clamp_timeout(timeout_s: float) -> float:
    return max(1.0, min(float(timeout_s), TIMEOUT_CAP_S))


def normalize_base_url(base_url: str) -> str:
    """Normalize an OpenAI-compatible base URL (expected to include /v1)."""
    return base_url.rstrip("/")


def server_root(base_url: str) -> str:
    """Server root for non-/v1 endpoints such as engine health paths."""
    normalized = normalize_base_url(base_url)
    return normalized.removesuffix("/v1")


def _headers(api_key: Optional[str]) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def build_client(
    base_url: str,
    api_key: Optional[str] = None,
    timeout_s: float = 120.0,
    transport: Optional[httpx.BaseTransport] = None,
) -> httpx.Client:
    return httpx.Client(
        base_url=normalize_base_url(base_url),
        headers=_headers(api_key),
        timeout=httpx.Timeout(clamp_timeout(timeout_s), connect=CONNECT_TIMEOUT_S),
        transport=transport,
    )


def build_async_client(
    base_url: str,
    api_key: Optional[str] = None,
    timeout_s: float = 540.0,
    transport: Optional[httpx.AsyncBaseTransport] = None,
) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=normalize_base_url(base_url),
        headers=_headers(api_key),
        timeout=httpx.Timeout(clamp_timeout(timeout_s), connect=CONNECT_TIMEOUT_S),
        transport=transport,
    )


def _parse_sse_line(line: str) -> Optional[Dict[str, Any]]:
    line = line.strip()
    if not line.startswith("data:"):
        return None
    data = line[len("data:") :].strip()
    if not data or data == "[DONE]":
        return None
    try:
        parsed = json.loads(data)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def iter_sse_json(response: httpx.Response) -> Iterator[Dict[str, Any]]:
    """Yield parsed JSON payloads from a server-sent-events response."""
    for line in response.iter_lines():
        parsed = _parse_sse_line(line)
        if parsed is not None:
            yield parsed


async def aiter_sse_json(response: httpx.Response) -> AsyncIterator[Dict[str, Any]]:
    """Async variant of `iter_sse_json`."""
    async for line in response.aiter_lines():
        parsed = _parse_sse_line(line)
        if parsed is not None:
            yield parsed
