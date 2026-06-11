from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, AsyncGenerator

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, StreamingResponse

if TYPE_CHECKING:
    from ..lifespan import AppState

log = logging.getLogger(__name__)

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Agent Chat</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: #0f1117; color: #e0e0e0; height: 100vh; display: flex; flex-direction: column; }
  #header { padding: 12px 20px; background: #1a1d27; border-bottom: 1px solid #2d3048; font-weight: 600; font-size: 18px; }
  #messages { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 12px; }
  .msg { max-width: 75%; padding: 10px 14px; border-radius: 12px; line-height: 1.5; white-space: pre-wrap; word-break: break-word; }
  .msg.user { align-self: flex-end; background: #2563eb; color: #fff; border-bottom-right-radius: 3px; }
  .msg.assistant { align-self: flex-start; background: #1e2235; border-bottom-left-radius: 3px; }
  .msg.system { align-self: center; font-size: 12px; color: #888; background: none; }
  #input-row { display: flex; gap: 8px; padding: 12px 20px; background: #1a1d27; border-top: 1px solid #2d3048; }
  #prompt { flex: 1; padding: 10px 14px; border-radius: 8px; border: 1px solid #3d4060; background: #0f1117; color: #e0e0e0; font-size: 15px; resize: none; }
  #send { padding: 10px 20px; border-radius: 8px; border: none; background: #2563eb; color: #fff; font-size: 15px; cursor: pointer; }
  #send:disabled { opacity: 0.5; cursor: default; }
</style>
</head>
<body>
<div id="header">Agent Chat</div>
<div id="messages"></div>
<div id="input-row">
  <textarea id="prompt" rows="1" placeholder="Type a message…"></textarea>
  <button id="send">Send</button>
</div>
<script>
const msgs = document.getElementById('messages');
const prompt = document.getElementById('prompt');
const send = document.getElementById('send');
const history = [];

function addMsg(role, text) {
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  div.textContent = text;
  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
  return div;
}

send.addEventListener('click', () => sendMessage());
prompt.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});

async function sendMessage() {
  const text = prompt.value.trim();
  if (!text) return;
  prompt.value = '';
  send.disabled = true;

  history.push({ role: 'user', content: text });
  addMsg('user', text);

  const assistantDiv = addMsg('assistant', '');
  let assistantText = '';

  try {
    const resp = await fetch('/ui/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: history }),
    });

    if (!resp.ok) { assistantDiv.textContent = '⚠ Error: ' + resp.status; return; }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split('\\n');
      buf = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const data = line.slice(6).trim();
        if (data === '[DONE]') break;
        try {
          const chunk = JSON.parse(data);
          const delta = chunk.choices?.[0]?.delta?.content || '';
          assistantText += delta;
          assistantDiv.textContent = assistantText;
          msgs.scrollTop = msgs.scrollHeight;
        } catch {}
      }
    }

    history.push({ role: 'assistant', content: assistantText });
  } catch (err) {
    assistantDiv.textContent = '⚠ ' + err.message;
  } finally {
    send.disabled = false;
    prompt.focus();
  }
}
</script>
</body>
</html>
"""


def build_ui_router(app_state: AppState) -> APIRouter:
    """Build the testing web UI router.

    Closures read agent from app_state at call time, making the handler
    reload-safe — after a reload, new requests use the new agent.
    """
    router = APIRouter(tags=["UI"])

    @router.get("/", response_class=HTMLResponse)
    async def ui_root() -> str:
        return _HTML

    @router.post("/chat")
    async def ui_chat(body: dict[str, Any], request: Request) -> StreamingResponse:  # type: ignore[type-arg]
        agent = app_state.agent
        messages: list[dict[str, Any]] = body.get("messages", [])

        if not messages:
            return StreamingResponse(
                _sse_error("No messages provided"),
                media_type="text/event-stream",
            )

        history = _messages_to_history(messages[:-1])
        user_prompt = messages[-1].get("content") or ""

        return StreamingResponse(
            _stream_sse(agent, user_prompt, history or None),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return router


async def _stream_sse(
    agent: Any, user_prompt: str, history: Any
) -> AsyncGenerator[str, None]:
    """Yield OpenAI-style SSE chunks from agent.run_stream()."""
    async with agent.run_stream(user_prompt, message_history=history) as result:
        async for chunk in result.stream_text(delta=True):
            data = json.dumps(
                {"choices": [{"delta": {"content": chunk}, "finish_reason": None}]}
            )
            yield f"data: {data}\n\n"
    yield "data: [DONE]\n\n"


async def _sse_error(msg: str) -> AsyncGenerator[str, None]:
    data = json.dumps({"error": msg})
    yield f"data: {data}\n\n"
    yield "data: [DONE]\n\n"


def _messages_to_history(messages: list[dict[str, Any]]) -> list:
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        SystemPromptPart,
        TextPart,
        UserPromptPart,
    )

    history = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content") or ""
        if isinstance(content, list):
            content = " ".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        match role:
            case "system":
                history.append(ModelRequest(parts=[SystemPromptPart(content=content)]))
            case "user":
                history.append(ModelRequest(parts=[UserPromptPart(content=content)]))
            case "assistant":
                history.append(ModelResponse(parts=[TextPart(content=content)]))
    return history
