"""Tracing setup must work with the installed FastAPI/httpx/Pydantic AI versions.

Runs in a subprocess: OpenTelemetry installs process-global state (tracer
provider, httpx instrumentation) that would leak into other tests.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

SCRIPT = textwrap.dedent(
    """
    import asyncio, sys
    from pathlib import Path

    import httpx
    import yaml
    from opentelemetry import trace
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from agentic.agent.app import create_app

    root = Path(sys.argv[1])
    (root / "config").mkdir()
    (root / "config" / "agent.yaml").write_text(yaml.safe_dump({"model": "test", "name": "t"}))
    (root / "config" / "server.yaml").write_text(
        yaml.safe_dump({"agent_card": {"display_name": "T", "description": "d"}})
    )
    app = create_app(root / "config", root / "secrets", "http://test", watch=False)
    exporter = InMemorySpanExporter()
    trace.get_tracer_provider().add_span_processor(SimpleSpanProcessor(exporter))

    async def main():
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
                r = await c.post(
                    "/v1/chat/completions",
                    json={"model": "t", "messages": [{"role": "user", "content": "hi"}]},
                )
                assert r.status_code == 200, r.text
        names = {span.name for span in exporter.get_finished_spans()}
        assert any("chat" in n or "agent" in n for n in names), names
        assert any("POST /v1/chat/completions" in n for n in names), names
        print("ok", len(names))

    asyncio.run(main())
    """
)


def test_tracing_setup_and_a_traced_request(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "OTEL_EXPORTER_OTLP_ENDPOINT": "http://127.0.0.1:9",  # never reached
        "OTEL_BSP_EXPORT_TIMEOUT": "100",
        "OTEL_EXPORTER_OTLP_TIMEOUT": "1",
        "PYDANTIC_AI_NO_BANNER": "1",
    }
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT, str(tmp_path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.startswith("ok")
