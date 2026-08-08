"""Endpoint readiness probe."""

import time
from typing import Optional

import httpx

from aiops.engines import EngineAdapter
from aiops.http_client import server_root
from aiops.models import EndpointInfo


def get_endpoint_info(
    client: httpx.Client, engine: EngineAdapter, base_url: str
) -> EndpointInfo:
    """Probe GET /models and the engine health endpoint."""
    started = time.perf_counter()
    try:
        response = client.get("/models")
        latency_ms = (time.perf_counter() - started) * 1000
        response.raise_for_status()
        served = [
            entry["id"]
            for entry in response.json().get("data", [])
            if isinstance(entry, dict) and "id" in entry
        ]
    except httpx.HTTPError as exc:
        return EndpointInfo(
            reachable=False,
            engine=engine.name,
            served_models=[],
            engine_health_ok=None,
            latency_ms=(time.perf_counter() - started) * 1000,
            error=f"{type(exc).__name__}: {exc}",
        )

    engine_health_ok: Optional[bool] = None
    health_path = engine.health_path()
    if health_path:
        try:
            health = client.get(f"{server_root(base_url)}{health_path}")
            engine_health_ok = health.status_code == 200
        except httpx.HTTPError:
            engine_health_ok = False

    return EndpointInfo(
        reachable=True,
        engine=engine.name,
        served_models=served,
        engine_health_ok=engine_health_ok,
        latency_ms=latency_ms,
    )
