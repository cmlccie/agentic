from __future__ import annotations

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

from .reload import AgentState

router = APIRouter(tags=["Health"])


@router.get("/health/live")
async def liveness() -> dict:
    """Kubernetes liveness probe.

    Always 200 — process is alive even during drain/reload.
    Returning non-200 causes pod restart, which we never want during hot-reload.
    """
    return {"status": "alive"}


@router.get("/health/ready", response_model=None)
async def readiness(request: Request) -> JSONResponse | dict:
    """Kubernetes readiness probe.

    503 during DRAINING and RELOADING — Kubernetes stops routing new traffic.
    200 only when RUNNING.
    """
    coordinator = request.app.state.app_state.coordinator
    state = coordinator.state

    if state == AgentState.RUNNING:
        return {"status": "ready", "in_flight": coordinator.in_flight_count}

    return JSONResponse(
        {
            "status": "not_ready",
            "state": state.name,
            "in_flight": coordinator.in_flight_count,
        },
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
    )
