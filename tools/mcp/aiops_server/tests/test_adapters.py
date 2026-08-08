"""Engine adapter behavior against canned payload fragments."""

import pytest
from aiops.engines import get_adapter


class TestAdapterRegistry:
    def test_known_engines(self):
        for engine in ("sglang", "vllm", "nim"):
            assert get_adapter(engine).name == engine

    def test_unknown_engine_raises(self):
        with pytest.raises(ValueError, match="Unknown inferencing engine"):
            get_adapter("tgi")


class TestHealthPaths:
    def test_sglang_and_vllm_use_root_health(self):
        assert get_adapter("sglang").health_path() == "/health"
        assert get_adapter("vllm").health_path() == "/health"

    def test_nim_uses_v1_health_ready(self):
        assert get_adapter("nim").health_path() == "/v1/health/ready"


class TestLeakDetection:
    @pytest.mark.parametrize(
        "content",
        [
            "<think>hmm</think> The answer is 9.",
            "<|thinking|> pondering",
            "<|channel|>analysis to=assistant",
        ],
    )
    def test_reasoning_leaks_detected(self, content):
        adapter = get_adapter("vllm")
        assert any(p.search(content) for p in adapter.reasoning_leak_patterns())

    def test_clean_content_not_flagged(self):
        adapter = get_adapter("vllm")
        content = "The farmer has 9 sheep left."
        assert not any(p.search(content) for p in adapter.reasoning_leak_patterns())
        assert not any(p.search(content) for p in adapter.tool_call_leak_patterns())

    @pytest.mark.parametrize(
        "content",
        [
            '<tool_call>{"name": "get_current_weather"}</tool_call>',
            '<|python_tag|>get_current_weather(location="Paris")',
            '```json\n{"name": "get_current_weather"}\n```',
            '<function=get_current_weather>{"location": "Paris"}</function>',
        ],
    )
    def test_tool_call_leaks_detected(self, content):
        adapter = get_adapter("sglang")
        assert any(p.search(content) for p in adapter.tool_call_leak_patterns())


class TestNimInheritsVllm:
    def test_reasoning_candidates_match(self):
        assert (
            get_adapter("nim").reasoning_field_candidates()
            == get_adapter("vllm").reasoning_field_candidates()
        )
