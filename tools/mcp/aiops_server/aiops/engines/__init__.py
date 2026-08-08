"""Engine adapter registry."""

from aiops.engines.base import EngineAdapter
from aiops.engines.nim import NimAdapter
from aiops.engines.sglang import SglangAdapter
from aiops.engines.vllm import VllmAdapter

_ADAPTERS: dict[str, EngineAdapter] = {
    adapter.name: adapter for adapter in (SglangAdapter(), VllmAdapter(), NimAdapter())
}


def get_adapter(engine: str) -> EngineAdapter:
    """Return the adapter for an engine name; raise ValueError if unknown."""
    try:
        return _ADAPTERS[engine]
    except KeyError:
        raise ValueError(
            f"Unknown inferencing engine {engine!r}; expected one of "
            f"{sorted(_ADAPTERS)}."
        ) from None


__all__ = ["EngineAdapter", "get_adapter"]
