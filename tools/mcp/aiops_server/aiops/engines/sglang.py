"""SGLang engine adapter."""

from aiops.engines.base import EngineAdapter


class SglangAdapter(EngineAdapter):
    """SGLang: standard OpenAI-compatible surface; health at /health."""

    name = "sglang"
