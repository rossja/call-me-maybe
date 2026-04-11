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


# ---------------------------------------------------------------------------
# LocalMLXBackend – TTS helpers
# ---------------------------------------------------------------------------
# These tests patch away all Apple Silicon / MLX dependencies so they run
# on any platform.


def _make_local_backend(extra_settings: dict | None = None) -> "LocalMLXBackend":  # noqa: F821
    """
    Build a LocalMLXBackend with all heavy initialisation mocked out.
    """
    from call_me_maybe.models.local import LocalMLXBackend

    settings_kwargs: dict = {
        "provider": "local",
        **(extra_settings or {}),
    }
    s = Settings(**settings_kwargs)

    with (
        patch("call_me_maybe.models.local._check_apple_silicon"),
        patch("call_me_maybe.models.local._check_memory"),
        patch.object(LocalMLXBackend, "_load_llm"),
    ):
        backend = LocalMLXBackend.__new__(LocalMLXBackend)
        backend._settings = s
        backend._model = None
        backend._tokenizer = None
        backend._whisper_model = None
        backend._tts_model = None

    return backend


# ---- macOS say ---------------------------------------------------------------


def test_macos_say_passes_voice_flag(tmp_path: "pytest.TempPathFactory") -> None:
    """_macos_say_tts should pass -v <voice> to say when tts.voice is set."""
    from call_me_maybe.models.local import LocalMLXBackend

    backend = _make_local_backend({"tts": {"model": "tts-1", "voice": "Alex"}})

    # Build minimal RIFF/WAV bytes so open(wav_path, "rb") returns valid data
    import io
    import wave as wave_mod

    def fake_say(cmd, **kwargs):
        # Verify -v flag is present
        assert "-v" in cmd
        idx = cmd.index("-v")
        assert cmd[idx + 1] == "Alex"

    def fake_afconvert(cmd, **kwargs):
        wav_out = cmd[-1]
        buf = io.BytesIO()
        with wave_mod.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(b"\x00" * 100)
        with open(wav_out, "wb") as fh:
            fh.write(buf.getvalue())

    call_count = [0]

    def fake_run(cmd, **kwargs):
        if cmd[0] == "say":
            fake_say(cmd, **kwargs)
        elif cmd[0] == "afconvert":
            fake_afconvert(cmd, **kwargs)
        call_count[0] += 1
        result = MagicMock()
        result.returncode = 0
        return result

    with patch("call_me_maybe.models.local.subprocess.run", side_effect=fake_run):
        result = backend._macos_say_tts("hello")

    assert call_count[0] == 2
    assert isinstance(result, bytes)
    assert len(result) > 0


def test_macos_say_omits_voice_flag_when_empty() -> None:
    """_macos_say_tts should not pass -v when tts.voice is an empty string."""
    from call_me_maybe.models.local import LocalMLXBackend
    import io
    import wave as wave_mod

    backend = _make_local_backend({"tts": {"model": "tts-1", "voice": ""}})

    def fake_say(cmd, **kwargs):
        assert "-v" not in cmd

    def fake_afconvert(cmd, **kwargs):
        wav_out = cmd[-1]
        buf = io.BytesIO()
        with wave_mod.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(b"\x00" * 100)
        with open(wav_out, "wb") as fh:
            fh.write(buf.getvalue())

    def fake_run(cmd, **kwargs):
        if cmd[0] == "say":
            fake_say(cmd, **kwargs)
        elif cmd[0] == "afconvert":
            fake_afconvert(cmd, **kwargs)
        result = MagicMock()
        result.returncode = 0
        return result

    with patch("call_me_maybe.models.local.subprocess.run", side_effect=fake_run):
        backend._macos_say_tts("hello")


# ---- Voxtral -----------------------------------------------------------------


def _build_voxtral_mock() -> MagicMock:
    """Build a mock mlx_audio model whose generate() yields one audio chunk."""
    import numpy as np

    audio_data = (np.zeros(24000, dtype=np.float32) + 0.1)
    chunk = MagicMock()
    chunk.audio = audio_data
    chunk.sample_rate = 24000

    mock_model = MagicMock()
    mock_model.generate.return_value = iter([chunk])
    return mock_model


def test_voxtral_tts_wav_output() -> None:
    """_voxtral_tts produces valid WAV bytes from mock audio chunks."""
    import wave as wave_mod
    import io
    import numpy as np
    import sys

    backend = _make_local_backend(
        {"tts": {"model": "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit", "voice": "neutral_male"}}
    )

    audio_data = np.zeros(4800, dtype=np.float32) + 0.5
    chunk = MagicMock()
    chunk.audio = audio_data
    chunk.sample_rate = 24000

    mock_model = MagicMock()
    mock_model.generate.return_value = iter([chunk])

    mock_mlx_audio_utils = MagicMock()
    mock_mlx_audio_utils.load = MagicMock(return_value=mock_model)

    mock_mlx_audio_tts = MagicMock()
    mock_mlx_audio_tts.utils = mock_mlx_audio_utils

    mock_mlx_audio = MagicMock()
    mock_mlx_audio.tts = mock_mlx_audio_tts

    saved = {k: sys.modules.get(k) for k in ["mlx_audio", "mlx_audio.tts", "mlx_audio.tts.utils"]}
    sys.modules["mlx_audio"] = mock_mlx_audio
    sys.modules["mlx_audio.tts"] = mock_mlx_audio_tts
    sys.modules["mlx_audio.tts.utils"] = mock_mlx_audio_utils

    try:
        result = backend._voxtral_tts("hello world")
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    mock_mlx_audio_utils.load.assert_called_once_with(
        "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit"
    )
    mock_model.generate.assert_called_once_with(text="hello world", voice="neutral_male")

    buf = io.BytesIO(result)
    with wave_mod.open(buf, "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 24000
        assert wf.getnframes() == 4800


def test_voxtral_tts_model_cached() -> None:
    """_voxtral_tts should only load the model once across repeated calls."""
    import numpy as np
    import sys

    backend = _make_local_backend(
        {"tts": {"model": "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit", "voice": "neutral_male"}}
    )

    def make_chunk():
        chunk = MagicMock()
        chunk.audio = np.zeros(100, dtype=np.float32)
        chunk.sample_rate = 24000
        return chunk

    mock_model = MagicMock()
    mock_model.generate.side_effect = [iter([make_chunk()]), iter([make_chunk()])]

    mock_utils = MagicMock()
    mock_utils.load = MagicMock(return_value=mock_model)

    saved = {k: sys.modules.get(k) for k in ["mlx_audio", "mlx_audio.tts", "mlx_audio.tts.utils"]}
    sys.modules["mlx_audio.tts.utils"] = mock_utils

    try:
        backend._voxtral_tts("first")
        backend._voxtral_tts("second")
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    mock_utils.load.assert_called_once()


def test_voxtral_tts_import_error_raises() -> None:
    """_voxtral_tts should raise ImportError with helpful message when mlx-audio is missing."""
    import sys

    backend = _make_local_backend(
        {"tts": {"model": "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit", "voice": "neutral_male"}}
    )

    saved = sys.modules.get("mlx_audio.tts.utils")
    sys.modules["mlx_audio.tts.utils"] = None  # type: ignore[assignment]

    try:
        with pytest.raises(ImportError, match="mlx-audio is required"):
            backend._voxtral_tts("hello")
    finally:
        if saved is None:
            sys.modules.pop("mlx_audio.tts.utils", None)
        else:
            sys.modules["mlx_audio.tts.utils"] = saved


# ---- synthesize dispatch -----------------------------------------------------


def test_synthesize_dispatches_to_fish_when_key_set() -> None:
    """synthesize() should call _fish_audio_tts when FISH_AUDIO_API_KEY is set."""
    backend = _make_local_backend()
    backend._settings = Settings(
        **{"provider": "local", "FISH_AUDIO_API_KEY": "test-fish-key"}
    )

    with patch.object(backend, "_fish_audio_tts", return_value=b"fish-audio") as mock_fish:
        result = backend.synthesize("hello")

    mock_fish.assert_called_once_with("hello")
    assert result == b"fish-audio"


def test_synthesize_dispatches_to_voxtral_when_model_matches() -> None:
    """synthesize() should call _voxtral_tts when tts.model contains 'voxtral'."""
    backend = _make_local_backend(
        {"tts": {"model": "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit", "voice": "neutral_male"}}
    )

    with patch.object(backend, "_voxtral_tts", return_value=b"voxtral-audio") as mock_voxtral:
        result = backend.synthesize("hello")

    mock_voxtral.assert_called_once_with("hello")
    assert result == b"voxtral-audio"


def test_synthesize_dispatches_to_say_as_fallback() -> None:
    """synthesize() should fall back to _macos_say_tts when no Fish key and no Voxtral model."""
    backend = _make_local_backend({"tts": {"model": "tts-1", "voice": "Alex"}})

    with patch.object(backend, "_macos_say_tts", return_value=b"say-audio") as mock_say:
        result = backend.synthesize("hello")

    mock_say.assert_called_once_with("hello")
    assert result == b"say-audio"


def test_synthesize_fish_takes_priority_over_voxtral() -> None:
    """Fish Audio API key should take priority over a Voxtral model name."""
    backend = _make_local_backend(
        {"tts": {"model": "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit", "voice": "neutral_male"}}
    )
    backend._settings = Settings(
        **{
            "provider": "local",
            "FISH_AUDIO_API_KEY": "test-fish-key",
            "tts": {"model": "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit", "voice": "neutral_male"},
        }
    )

    with (
        patch.object(backend, "_fish_audio_tts", return_value=b"fish") as mock_fish,
        patch.object(backend, "_voxtral_tts", return_value=b"voxtral") as mock_voxtral,
    ):
        result = backend.synthesize("hello")

    mock_fish.assert_called_once_with("hello")
    mock_voxtral.assert_not_called()
    assert result == b"fish"
