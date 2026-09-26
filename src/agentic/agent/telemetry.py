"""Optional OpenTelemetry tracing, enabled by the standard OTEL environment variables.

When ``OTEL_EXPORTER_OTLP_ENDPOINT`` (or ``OTEL_EXPORTER_OTLP_TRACES_ENDPOINT``)
is set, traces are exported over OTLP/HTTP:

- every agent run, model request, and tool call (Pydantic AI's GenAI semantic
  convention spans, via `Agent.instrument_all`)
- every inbound HTTP request (FastAPI) and outbound HTTP call (httpx — MCP
  servers, model endpoints, delegated A2A agents), with trace context
  propagated so an orchestrator and its workers appear in one trace

Configure the exporter with the usual variables (``OTEL_SERVICE_NAME``,
``OTEL_EXPORTER_OTLP_HEADERS``, ``OTEL_RESOURCE_ATTRIBUTES``, ...). With none
set, this module does nothing and adds no overhead.
"""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI

log = logging.getLogger(__name__)

_ENDPOINT_VARS = ("OTEL_EXPORTER_OTLP_ENDPOINT", "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")


def telemetry_enabled() -> bool:
    """Whether an OTLP endpoint is configured."""
    return any(os.environ.get(name) for name in _ENDPOINT_VARS)


def setup_telemetry(app: FastAPI, service_name: str) -> bool:
    """Configure tracing for this process and ``app`` if an endpoint is configured.

    Returns:
        True if tracing was enabled.
    """
    if not telemetry_enabled():
        return False

    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from pydantic_ai import Agent

    resource = Resource.create(
        {SERVICE_NAME: os.environ.get("OTEL_SERVICE_NAME", service_name)}
    )
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)

    Agent.instrument_all()
    FastAPIInstrumentor.instrument_app(app, excluded_urls="health/.*")
    HTTPXClientInstrumentor().instrument()
    log.info("OpenTelemetry tracing enabled (OTLP/HTTP)")
    return True
