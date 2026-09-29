"""HTTP client helpers shared by all checks."""

import ipaddress
import json
import os
from typing import Any, AsyncIterator, Dict, Iterator, Optional

import httpx

# Hard server-side cap on any tool's timeout: must stay under the 600s
# request-timeout on the Ingress in front of this server.
TIMEOUT_CAP_S = 570.0

CONNECT_TIMEOUT_S = 10.0

# Optional comma-separated allowlist of probe targets: CIDRs (e.g. 10.0.0.0/8),
# exact hostnames (e.g. model.ns.svc), or domain suffixes with a leading dot
# (e.g. .svc.cluster.local). Unset or empty means any target is allowed.
ALLOWED_TARGETS_ENV = "AIOPS_ALLOWED_TARGETS"


def clamp_timeout(timeout_s: float) -> float:
    return max(1.0, min(float(timeout_s), TIMEOUT_CAP_S))


def _host_matches(host: str, entry: str) -> bool:
    """True when ``host`` (IP literal or hostname) matches one allowlist entry."""
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None

    try:
        network = ipaddress.ip_network(entry, strict=False)
    except ValueError:
        network = None

    if network is not None:
        return address is not None and address in network
    if address is not None:
        return False
    entry = entry.lower()
    return host == entry or (entry.startswith(".") and host.endswith(entry))


def check_target_allowed(base_url: str, allowed: Optional[str] = None) -> None:
    """Reject ``base_url`` unless its host is in the configured allowlist.

    Args:
        base_url: The caller-supplied endpoint URL.
        allowed: Comma-separated allowlist; defaults to $AIOPS_ALLOWED_TARGETS.

    Raises:
        ValueError: If the URL is not http(s), or an allowlist is configured and
            the host matches none of its entries.
    """
    url = httpx.URL(base_url)
    if url.scheme not in ("http", "https") or not url.host:
        raise ValueError(f"base_url must be an http(s) URL with a host: {base_url!r}")

    raw = os.environ.get(ALLOWED_TARGETS_ENV, "") if allowed is None else allowed
    entries = [entry.strip() for entry in raw.split(",") if entry.strip()]
    host = url.host.lower()
    if entries and not any(_host_matches(host, entry) for entry in entries):
        raise ValueError(
            f"Target host {host!r} is not allowed; permitted targets are set by "
            f"{ALLOWED_TARGETS_ENV}."
        )


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
    check_target_allowed(base_url)
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
    check_target_allowed(base_url)
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
