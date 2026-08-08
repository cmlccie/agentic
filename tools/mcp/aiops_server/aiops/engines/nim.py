"""NVIDIA NIM engine adapter.

LLM NIM 2.0 is vLLM underneath (engine flags arrive via NIM_PASSTHROUGH_ARGS),
so response shapes and parser behavior match vLLM. Only the health endpoint
differs: NIM serves readiness at /v1/health/ready rather than /health.
"""

from typing import Optional

from aiops.engines.vllm import VllmAdapter


class NimAdapter(VllmAdapter):
    """NIM 2.0: vLLM semantics with a NIM-specific health endpoint."""

    name = "nim"

    def health_path(self) -> Optional[str]:
        return "/v1/health/ready"
