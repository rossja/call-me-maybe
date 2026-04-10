"""
Remote (OpenAI-compatible API) backend.

Works with any endpoint that is compatible with the OpenAI API, including:
  - OpenRouter   (https://openrouter.ai/api/v1)
  - OpenAI       (https://api.openai.com/v1)
  - LM Studio    (http://localhost:1234/v1)
  - Ollama       (http://localhost:11434/v1)
  - ... and any other OpenAI-compatible server.

The active endpoint and model names are driven by ``config.yaml``.
The API key is read from the environment / .env file.
"""

from __future__ import annotations

import io
import json
import logging
from typing import TYPE_CHECKING

from openai import OpenAI

from call_me_maybe.models.base import (
    ChatMessage,
    ModelBackend,
    ModelResponse,
    ToolCall,
    ToolDefinition,
)

if TYPE_CHECKING:
    from call_me_maybe.config.settings import Settings

logger = logging.getLogger(__name__)


class RemoteBackend(ModelBackend):
    """
    Backend that delegates STT / LLM / TTS to a remote OpenAI-compatible API.
    """

    def __init__(self, settings: "Settings") -> None:
        self._settings = settings
        remote = settings.remote
        api_key = settings.effective_api_key or "sk-no-key-set"

        self._client = OpenAI(
            api_key=api_key,
            base_url=remote.base_url,
            timeout=remote.timeout,
            default_headers=remote.extra_headers,
        )
        logger.info(
            "RemoteBackend ready  base_url=%s  llm=%s  stt=%s  tts=%s",
            remote.base_url,
            settings.llm.model,
            settings.stt.model,
            settings.tts.model,
        )

    # ------------------------------------------------------------------
    # STT
    # ------------------------------------------------------------------

    def transcribe(self, audio_bytes: bytes, *, language: str | None = None) -> str:
        """Transcribe audio using the Whisper-compatible transcription endpoint."""
        stt = self._settings.stt
        lang = language or stt.language

        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "audio.wav"

        logger.debug("STT request: model=%s language=%s", stt.model, lang)
        kwargs: dict = {"model": stt.model, "file": audio_file}
        if lang:
            kwargs["language"] = lang

        response = self._client.audio.transcriptions.create(**kwargs)
        text = response.text.strip()
        logger.debug("STT result: %r", text)
        return text

    # ------------------------------------------------------------------
    # LLM
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolDefinition] | None = None,
    ) -> ModelResponse:
        """Send messages to the LLM and return the assistant response."""
        llm = self._settings.llm

        openai_messages = [
            self._to_openai_message(m) for m in messages
        ]

        kwargs: dict = {
            "model": llm.model,
            "messages": openai_messages,
            "temperature": llm.temperature,
            "max_tokens": llm.max_tokens,
        }
        if tools:
            kwargs["tools"] = [t.to_openai_schema() for t in tools]
            kwargs["tool_choice"] = "auto"

        logger.debug("LLM request: model=%s messages=%d", llm.model, len(messages))
        completion = self._client.chat.completions.create(**kwargs)

        choice = completion.choices[0]
        message = choice.message

        tool_calls: list[ToolCall] = []
        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append(
                    ToolCall(id=tc.id, name=tc.function.name, arguments=args)
                )

        text = message.content or ""
        logger.debug(
            "LLM response: finish_reason=%s text_len=%d tool_calls=%d",
            choice.finish_reason,
            len(text),
            len(tool_calls),
        )
        return ModelResponse(
            text=text,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
        )

    # ------------------------------------------------------------------
    # TTS
    # ------------------------------------------------------------------

    def synthesize(self, text: str) -> bytes:
        """Convert text to speech audio bytes."""
        tts = self._settings.tts
        logger.debug("TTS request: model=%s voice=%s", tts.model, tts.voice)

        response = self._client.audio.speech.create(
            model=tts.model,
            voice=tts.voice,
            input=text,
            speed=tts.speed,
            response_format=tts.audio_format,  # type: ignore[arg-type]
        )
        audio_bytes = response.read()
        logger.debug("TTS response: %d bytes", len(audio_bytes))
        return audio_bytes

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_openai_message(msg: ChatMessage) -> dict:
        result: dict = {"role": msg.role, "content": msg.content}
        if msg.tool_call_id:
            result["tool_call_id"] = msg.tool_call_id
        if msg.name:
            result["name"] = msg.name
        return result
