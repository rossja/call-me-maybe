"""Composite backend that delegates STT, LLM, and TTS to separate backends."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from call_me_maybe.models.base import (
    ChatMessage,
    ModelBackend,
    ModelResponse,
    ToolCall,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class CompositeBackend(ModelBackend):
    """
    Routes STT, LLM, and TTS to separate backend instances.

    Allows mixing local and remote backends (e.g., local LLM with remote STT).
    """

    def __init__(
        self,
        stt: ModelBackend,
        llm: ModelBackend,
        tts: ModelBackend,
    ) -> None:
        self._stt = stt
        self._llm = llm
        self._tts = tts
        logger.info(
            "CompositeBackend created: stt=%s llm=%s tts=%s",
            type(stt).__name__,
            type(llm).__name__,
            type(tts).__name__,
        )

    def transcribe(self, audio_bytes: bytes, *, language: str | None = None) -> str:
        """Delegate to the STT backend."""
        return self._stt.transcribe(audio_bytes, language=language)

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[Any] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Delegate to the LLM backend."""
        return await self._llm.chat(messages, tools=tools, **kwargs)

    async def synthesize(self, text: str) -> bytes:
        """Delegate to the TTS backend."""
        return await self._tts.synthesize(text)
