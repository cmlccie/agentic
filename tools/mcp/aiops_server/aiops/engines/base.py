"""Engine adapter base class.

Adapters encode the only places the OpenAI-compatible engines genuinely differ:
where reasoning content lives, what markup constitutes a parser leak, which
finish_reasons a tool call may report, and where the health endpoint is.
Adapters never issue HTTP — shared code in `aiops.checks` owns requests, timing,
and pass/fail assembly.

Extending: add a method here with a sensible default implementation and override
it only in the adapters that genuinely diverge.
"""

import re
from typing import Any, Dict, List, Optional, Set

# Reasoning markup that must not appear inline in `message.content` when a
# reasoning parser is correctly configured.
_REASONING_LEAK_PATTERNS = [
    re.compile(r"<think>", re.IGNORECASE),
    re.compile(r"</think>", re.IGNORECASE),
    re.compile(r"<\|thinking\|>"),
    re.compile(r"<\|channel\|>\s*analysis"),  # GPT-OSS harmony markup
]

# Tool-call markup that must not appear as raw text in `message.content` when a
# tool-call parser is correctly configured.
_TOOL_CALL_LEAK_PATTERNS = [
    re.compile(r"<tool_call>", re.IGNORECASE),
    re.compile(r"<\|python_tag\|>"),
    re.compile(r"<function="),
    re.compile(r"```json\s*\{\s*\"name\""),  # fenced JSON instead of tool_calls
]


class EngineAdapter:
    """Default adapter; suits OpenAI-compatible engines generally."""

    name: str = "base"

    def health_path(self) -> Optional[str]:
        """Server-root-relative health endpoint, or None if the engine has none."""
        return "/health"

    def reasoning_field_candidates(self) -> List[str]:
        """Ordered message fields that may carry structured reasoning content."""
        return ["reasoning_content", "reasoning"]

    def reasoning_leak_patterns(self) -> List[re.Pattern]:
        return _REASONING_LEAK_PATTERNS

    def acceptable_tool_finish_reasons(self) -> Set[str]:
        return {"tool_calls"}

    def tool_call_leak_patterns(self) -> List[re.Pattern]:
        return _TOOL_CALL_LEAK_PATTERNS

    def normalize_usage(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract the standard OpenAI usage object from a response payload."""
        usage = payload.get("usage")
        return usage if isinstance(usage, dict) else None
