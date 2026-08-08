"""vLLM engine adapter."""

from aiops.engines.base import EngineAdapter


class VllmAdapter(EngineAdapter):
    """vLLM: standard OpenAI-compatible surface; health at /health."""

    name = "vllm"
