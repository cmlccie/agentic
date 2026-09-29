"""The default configs shipped in the container images must load."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentic.agent.config import Secrets
from agentic.agent.runtime import load_snapshot

IMAGES = Path(__file__).parents[2] / "images"


@pytest.mark.parametrize("image", ["simple_agent", "orchestrator_agent"])
def test_default_image_config_loads(
    image: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VLLM_BASE_URL", "http://vllm:8000/v1")
    snapshot = load_snapshot(IMAGES / image, Secrets(tmp_path))
    assert snapshot.model_name == image.replace("_", "-")
    assert snapshot.server.interfaces.openai and snapshot.server.interfaces.a2a
