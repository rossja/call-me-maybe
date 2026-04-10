"""
Tests for model backends.

Uses mocks to avoid network calls or hardware dependencies.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from call_me_maybe.config.settings import Settings
from call_me_maybe.models.base import (
    ChatMessage,
    ModelResponse,
    ToolCall,
    ToolDefinition,
)
from call_me_maybe.models.factory import create_backend
from call_me_maybe.models.remote import RemoteBackend


# ---------------------------------------------------------------------------
# ToolDefinition helper
# ---------------------------------------------------------------------------


def test_tool_definition_to_openai_schema() -> None:
    td = ToolDefinition(
        name="get_weather",
        description="Get weather for a city",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    )
    schema = td.to_openai_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "get_weather"
    assert schema["function"]["description"] == "Get weather for a city"
    assert "city" in schema["function"]["parameters"]["properties"]


# ---------------------------------------------------------------------------
# ModelResponse
# ---------------------------------------------------------------------------


def test_model_response_defaults() -> None:
    r = ModelResponse()
    assert r.text == ""
    assert r.tool_calls == []
    assert r.finish_reason == "stop"


def test_model_response_with_tool_calls() -> None:
    tc = ToolCall(id="tc1", name="search", arguments={"query": "hello"})
    r = ModelResponse(text="", tool_calls=[tc], finish_reason="tool_calls")
    assert len(r.tool_calls) == 1
    assert r.tool_calls[0].name == "search"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def test_create_backend_unknown_provider() -> None:
    s = Settings(provider="ftp")
    with pytest.raises(ValueError, match="Unknown provider"):
        create_backend(s)


def test_create_backend_remote(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    s = Settings(provider="remote")
    # Should instantiate without raising (no actual HTTP call)
    backend = create_backend(s)
    assert isinstance(backend, RemoteBackend)


# ---------------------------------------------------------------------------
# RemoteBackend – STT
# ---------------------------------------------------------------------------


@pytest.fixture()
def remote_backend(monkeypatch: pytest.MonkeyPatch) -> RemoteBackend:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    s = Settings()
    return RemoteBackend(s)


def test_remote_transcribe(remote_backend: RemoteBackend) -> None:
    """transcribe() should call the OpenAI transcriptions endpoint."""
    mock_response = MagicMock()
    mock_response.text = "  Hello world  "

    with patch.object(
        remote_backend._client.audio.transcriptions,
        "create",
        return_value=mock_response,
    ) as mock_create:
        result = remote_backend.transcribe(b"\x00" * 1024, language="en")

    assert result == "Hello world"
    mock_create.assert_called_once()
    call_kwargs = mock_create.call_args.kwargs
    assert call_kwargs["language"] == "en"


def test_remote_transcribe_no_language(remote_backend: RemoteBackend) -> None:
    mock_response = MagicMock()
    mock_response.text = "test"

    with patch.object(
        remote_backend._client.audio.transcriptions,
        "create",
        return_value=mock_response,
    ) as mock_create:
        remote_backend.transcribe(b"\x00" * 100)

    call_kwargs = mock_create.call_args.kwargs
    assert "language" not in call_kwargs


# ---------------------------------------------------------------------------
# RemoteBackend – LLM (chat)
# ---------------------------------------------------------------------------


def _make_chat_completion(content: str, tool_calls=None, finish_reason: str = "stop"):
    """Build a mock OpenAI ChatCompletion object."""
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = finish_reason

    completion = MagicMock()
    completion.choices = [choice]
    return completion


def test_remote_chat_text_reply(remote_backend: RemoteBackend) -> None:
    completion = _make_chat_completion("Paris is the capital of France.")
    with patch.object(
        remote_backend._client.chat.completions,
        "create",
        return_value=completion,
    ):
        response = remote_backend.chat(
            [ChatMessage(role="user", content="What is the capital of France?")]
        )

    assert response.text == "Paris is the capital of France."
    assert response.tool_calls == []
    assert response.finish_reason == "stop"


def test_remote_chat_with_tool_call(remote_backend: RemoteBackend) -> None:
    mock_tc = MagicMock()
    mock_tc.id = "call_abc"
    mock_tc.function.name = "get_weather"
    mock_tc.function.arguments = json.dumps({"city": "London"})

    completion = _make_chat_completion("", tool_calls=[mock_tc], finish_reason="tool_calls")
    with patch.object(
        remote_backend._client.chat.completions,
        "create",
        return_value=completion,
    ):
        response = remote_backend.chat(
            [ChatMessage(role="user", content="Weather in London?")],
            tools=[
                ToolDefinition(
                    name="get_weather",
                    description="Get weather",
                    parameters={"type": "object", "properties": {"city": {"type": "string"}}},
                )
            ],
        )

    assert len(response.tool_calls) == 1
    tc = response.tool_calls[0]
    assert tc.id == "call_abc"
    assert tc.name == "get_weather"
    assert tc.arguments == {"city": "London"}


def test_remote_chat_malformed_tool_args(remote_backend: RemoteBackend) -> None:
    """Malformed JSON in tool arguments should produce an empty dict."""
    mock_tc = MagicMock()
    mock_tc.id = "call_bad"
    mock_tc.function.name = "broken_tool"
    mock_tc.function.arguments = "not valid json {"

    completion = _make_chat_completion("", tool_calls=[mock_tc], finish_reason="tool_calls")
    with patch.object(
        remote_backend._client.chat.completions,
        "create",
        return_value=completion,
    ):
        response = remote_backend.chat(
            [ChatMessage(role="user", content="test")]
        )

    assert response.tool_calls[0].arguments == {}


def test_remote_chat_sends_system_message(remote_backend: RemoteBackend) -> None:
    """System messages should be forwarded to the API."""
    completion = _make_chat_completion("OK")
    with patch.object(
        remote_backend._client.chat.completions,
        "create",
        return_value=completion,
    ) as mock_create:
        remote_backend.chat(
            [
                ChatMessage(role="system", content="You are a test bot."),
                ChatMessage(role="user", content="hello"),
            ]
        )

    messages_sent = mock_create.call_args.kwargs["messages"]
    assert messages_sent[0]["role"] == "system"
    assert messages_sent[1]["role"] == "user"


# ---------------------------------------------------------------------------
# RemoteBackend – TTS
# ---------------------------------------------------------------------------


def test_remote_synthesize(remote_backend: RemoteBackend) -> None:
    mock_speech = MagicMock()
    mock_speech.read.return_value = b"fake-audio-bytes"

    with patch.object(
        remote_backend._client.audio.speech,
        "create",
        return_value=mock_speech,
    ) as mock_create:
        result = remote_backend.synthesize("Hello world")

    assert result == b"fake-audio-bytes"
    call_kwargs = mock_create.call_args.kwargs
    assert call_kwargs["input"] == "Hello world"
