"""Tests for agentic.openai.compatible_api."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)

from agentic.openai.compatible_api import (
    Message,
    OpenAICompatibleAPI,
)

# -------------------------------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------------------------------


def parse_sse_chunks(text: str) -> list[ChatCompletionChunk]:
    """Parse SSE response text into a list of ChatCompletionChunk objects."""
    chunks = []
    for line in text.split("\n\n"):
        line = line.strip()
        if not line or line == "data: [DONE]":
            continue
        if line.startswith("data: "):
            chunks.append(ChatCompletionChunk.model_validate_json(line[6:]))
    return chunks


# -------------------------------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------------------------------


@pytest.fixture
def mock_agent():
    """MagicMock agent with run() and run_stream() configured."""
    agent = MagicMock()

    # Non-streaming
    mock_usage = MagicMock()
    mock_usage.input_tokens = 10
    mock_usage.output_tokens = 5
    mock_usage.total_tokens = 15

    mock_result = MagicMock()
    mock_result.output = "Hello, I am an assistant."
    mock_result.usage = MagicMock(return_value=mock_usage)

    agent.run = AsyncMock(return_value=mock_result)

    # Streaming — run_stream() returns a synchronous async context manager
    mock_stream_result = MagicMock()

    async def fake_stream_text(delta: bool = False):
        for chunk in ["Hello", ", ", "world", "!"]:
            yield chunk

    mock_stream_result.stream_text = fake_stream_text

    mock_ctx = MagicMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_stream_result)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)
    agent.run_stream = MagicMock(return_value=mock_ctx)

    return agent


@pytest.fixture
def api(mock_agent):
    return OpenAICompatibleAPI(
        agent=mock_agent,
        title="Test API",
        description="Test description.",
        model_name="test-model",
    )


@pytest.fixture
def client(api):
    return TestClient(api.app)


# -------------------------------------------------------------------------------------------------
# Message.to_model_message()
# -------------------------------------------------------------------------------------------------


class TestMessageToModelMessage:
    def test_system_role(self):
        result = Message(role="system", content="You are helpful.").to_model_message()
        assert isinstance(result, ModelRequest)
        assert isinstance(result.parts[0], SystemPromptPart)
        assert result.parts[0].content == "You are helpful."

    def test_user_role(self):
        result = Message(role="user", content="Tell me a joke.").to_model_message()
        assert isinstance(result, ModelRequest)
        assert isinstance(result.parts[0], UserPromptPart)
        assert result.parts[0].content == "Tell me a joke."

    def test_assistant_role(self):
        result = Message(role="assistant", content="Here is one.").to_model_message()
        assert isinstance(result, ModelResponse)
        assert isinstance(result.parts[0], TextPart)
        assert result.parts[0].content == "Here is one."


# -------------------------------------------------------------------------------------------------
# Service endpoints
# -------------------------------------------------------------------------------------------------


class TestServiceEndpoints:
    def test_root_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_root_includes_title(self, client, api):
        resp = client.get("/")
        assert api.app.title in resp.json()["message"]

    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_status_is_healthy(self, client):
        resp = client.get("/health")
        assert resp.json() == {"status": "healthy"}


# -------------------------------------------------------------------------------------------------
# Non-streaming chat completions
# -------------------------------------------------------------------------------------------------


class TestChatCompletionsNonStreaming:
    BASE_MESSAGES = [{"role": "user", "content": "Hello"}]

    def _post(self, client, messages=None, **kwargs):
        body = {"messages": messages or self.BASE_MESSAGES, **kwargs}
        return client.post("/v1/chat/completions", json=body)

    def test_returns_200(self, client):
        assert self._post(client).status_code == 200

    def test_response_validates_as_chat_completion(self, client):
        data = self._post(client).json()
        completion = ChatCompletion.model_validate(data)
        assert completion.object == "chat.completion"

    def test_response_has_assistant_message(self, client):
        data = self._post(client).json()
        completion = ChatCompletion.model_validate(data)
        assert completion.choices[0].message.role == "assistant"
        assert completion.choices[0].message.content == "Hello, I am an assistant."

    def test_response_finish_reason_is_stop(self, client):
        data = self._post(client).json()
        completion = ChatCompletion.model_validate(data)
        assert completion.choices[0].finish_reason == "stop"

    def test_response_includes_usage(self, client):
        data = self._post(client).json()
        completion = ChatCompletion.model_validate(data)
        assert completion.usage is not None
        assert completion.usage.prompt_tokens == 10
        assert completion.usage.completion_tokens == 5
        assert completion.usage.total_tokens == 15

    def test_response_id_starts_with_chatcmpl(self, client):
        data = self._post(client).json()
        assert data["id"].startswith("chatcmpl-")

    def test_response_has_created_and_model(self, client):
        data = self._post(client).json()
        assert "created" in data
        assert "model" in data

    def test_single_message_has_no_history(self, client, mock_agent):
        self._post(client, messages=[{"role": "user", "content": "Hi"}])
        call_args = mock_agent.run.call_args
        assert call_args.args[0] == "Hi"
        assert call_args.kwargs["message_history"] is None

    def test_multi_message_last_is_prompt(self, client, mock_agent):
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4."},
            {"role": "user", "content": "And 3+3?"},
        ]
        self._post(client, messages=messages)
        call_args = mock_agent.run.call_args
        assert call_args.args[0] == "And 3+3?"

    def test_multi_message_prior_messages_become_history(self, client, mock_agent):
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4."},
            {"role": "user", "content": "And 3+3?"},
        ]
        self._post(client, messages=messages)
        call_args = mock_agent.run.call_args
        assert call_args.kwargs["message_history"] is not None
        assert len(call_args.kwargs["message_history"]) == 3

    def test_empty_messages_returns_422(self, client):
        resp = client.post("/v1/chat/completions", json={"messages": []})
        assert resp.status_code == 422


# -------------------------------------------------------------------------------------------------
# Streaming chat completions
# -------------------------------------------------------------------------------------------------


class TestChatCompletionsStreaming:
    BASE_MESSAGES = [{"role": "user", "content": "Hi"}]

    def _post(self, client, messages=None, **kwargs):
        body = {"messages": messages or self.BASE_MESSAGES, "stream": True, **kwargs}
        return client.post("/v1/chat/completions", json=body)

    def test_returns_200(self, client):
        assert self._post(client).status_code == 200

    def test_content_type_is_event_stream(self, client):
        resp = self._post(client)
        assert "text/event-stream" in resp.headers["content-type"]

    def test_response_ends_with_done_sentinel(self, client):
        resp = self._post(client)
        assert resp.text.endswith("data: [DONE]\n\n")

    def test_chunks_validate_as_chat_completion_chunk(self, client):
        resp = self._post(client)
        chunks = parse_sse_chunks(resp.text)
        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, ChatCompletionChunk)
            assert chunk.object == "chat.completion.chunk"

    def test_first_chunk_has_assistant_role(self, client):
        resp = self._post(client)
        chunks = parse_sse_chunks(resp.text)
        assert chunks[0].choices[0].delta.role == "assistant"

    def test_final_chunk_has_finish_reason_stop(self, client):
        resp = self._post(client)
        chunks = parse_sse_chunks(resp.text)
        assert chunks[-1].choices[0].finish_reason == "stop"

    def test_concatenated_content_matches_stream(self, client):
        resp = self._post(client)
        chunks = parse_sse_chunks(resp.text)
        content = "".join(c.choices[0].delta.content or "" for c in chunks)
        assert content == "Hello, world!"

    def test_single_message_has_no_history(self, client, mock_agent):
        self._post(client, messages=[{"role": "user", "content": "Hi"}])
        call_args = mock_agent.run_stream.call_args
        assert call_args.args[0] == "Hi"
        assert call_args.kwargs["message_history"] is None

    def test_multi_message_history_passed_to_stream(self, client, mock_agent):
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Stream me something."},
        ]
        self._post(client, messages=messages)
        call_args = mock_agent.run_stream.call_args
        assert call_args.args[0] == "Stream me something."
        assert call_args.kwargs["message_history"] is not None
        assert len(call_args.kwargs["message_history"]) == 1
