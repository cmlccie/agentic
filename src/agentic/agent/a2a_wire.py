"""Small, pure conversions between A2A protobuf types and this package's models.

Shared by the A2A server (`a2a_server`) and the A2A client capability
(`a2a_client`) so both sides agree on how activity is encoded:

A status-update message carries activity when its metadata contains the
`ACTIVITY_EXTENSION` key::

    metadata = {"urn:agentic:a2a:activity:v1": {"kind": "tool_call",
                                                "text": "get_forecast({...})",
                                                "source": ["weather"],
                                                "tool": "get_forecast",
                                                "call_id": "call_1"}}

The message's text part holds the rendered, human-readable line (e.g. ``→
get_forecast({...})``), so generic A2A clients that know nothing about the
extension still show something sensible; the metadata carries the raw text.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from a2a.types import Message, Part, TaskState
from google.protobuf.json_format import MessageToDict

from .streaming import ACTIVITY_EXTENSION, Activity

#: Task states after which a task never changes again.
TERMINAL_STATES = frozenset(
    {
        TaskState.TASK_STATE_COMPLETED,
        TaskState.TASK_STATE_FAILED,
        TaskState.TASK_STATE_CANCELED,
        TaskState.TASK_STATE_REJECTED,
    }
)

#: Task states that pause the task until the client sends more input.
INTERRUPTED_STATES = frozenset(
    {TaskState.TASK_STATE_INPUT_REQUIRED, TaskState.TASK_STATE_AUTH_REQUIRED}
)

_ACTIVITY_KINDS = frozenset(
    {"thinking", "note", "tool_call", "tool_result", "status", "error"}
)


def state_name(state: int) -> str:
    """Readable task state, e.g. ``completed`` for ``TASK_STATE_COMPLETED``."""
    return TaskState.Name(state).removeprefix("TASK_STATE_").lower()


def parts_text(parts: Iterable[Part]) -> str:
    """Join the text and data parts of a message or artifact into plain text.

    Data parts are rendered as compact JSON; file/raw parts are summarized by
    name so the caller knows something was returned.
    """
    lines: list[str] = []
    for part in parts:
        match part.WhichOneof("content"):
            case "text":
                lines.append(part.text)
            case "data":
                lines.append(json.dumps(MessageToDict(part.data), default=str))
            case "url":
                lines.append(part.url)
            case "raw":
                lines.append(f"[{part.filename or part.media_type or 'binary data'}]")
    return "\n".join(line for line in lines if line)


def activity_metadata(item: Activity) -> dict[str, Any]:
    """Encode an activity's attributes as message metadata."""
    meta: dict[str, Any] = {
        "kind": item.kind,
        "text": item.text,
        "source": list(item.source),
    }
    if item.tool:
        meta["tool"] = item.tool
    if item.call_id:
        meta["call_id"] = item.call_id
    return {ACTIVITY_EXTENSION: meta}


def activity_from_message(message: Message) -> Activity | None:
    """Decode an activity from a status-update message, or None if it isn't one."""
    if not message.HasField("metadata"):
        return None
    meta = MessageToDict(message.metadata).get(ACTIVITY_EXTENSION)
    if not isinstance(meta, dict):
        return None
    kind = meta.get("kind")
    return Activity(
        kind=kind if kind in _ACTIVITY_KINDS else "status",
        text=str(meta.get("text", parts_text(message.parts))),
        source=tuple(str(s) for s in meta.get("source") or ()),
        tool=meta.get("tool"),
        call_id=meta.get("call_id"),
    )
