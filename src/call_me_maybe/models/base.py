"""
Abstract base classes for model backends.

All concrete backends (local MLX, remote OpenAI-compatible) must implement
the :class:`ModelBackend` protocol.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChatMessage:
    """A single message in a conversation."""

    role: str  # "system" | "user" | "assistant" | "tool"
    content: str
    # Populated when role == "tool"
    tool_call_id: str | None = None
    name: str | None = None


@dataclass
class ToolDefinition:
    """JSON-Schema description of a tool the model may call."""

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolCall:
    """A tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ModelResponse:
    """
    Normalised response from any backend.

    Either ``text`` is populated (assistant text reply) or
    ``tool_calls`` is non-empty (tool invocation requests), or both.
    """

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"


class BackendError(Exception):
    """Raised by a backend when it cannot complete a request."""


class ModelBackend(ABC):
    """Abstract interface that every backend must implement."""

    # ------------------------------------------------------------------
    # STT
    # ------------------------------------------------------------------

    @abstractmethod
    def transcribe(self, audio_bytes: bytes, *, language: str | None = None) -> str:
        """
        Convert raw audio bytes to text.

        Parameters
        ----------
        audio_bytes:
            Raw PCM or encoded audio (wav/mp3/ogg …).
        language:
            BCP-47 language code hint, or *None* for auto-detection.

        Returns
        -------
        str
            Transcribed text.
        """

    # ------------------------------------------------------------------
    # LLM
    # ------------------------------------------------------------------

    @abstractmethod
    def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolDefinition] | None = None,
    ) -> ModelResponse:
        """
        Send a conversation turn to the language model.

        Parameters
        ----------
        messages:
            Full conversation history including the latest user message.
        tools:
            Optional list of tools the model may call.

        Returns
        -------
        ModelResponse
        """

    # ------------------------------------------------------------------
    # TTS
    # ------------------------------------------------------------------

    @abstractmethod
    def synthesize(self, text: str) -> bytes:
        """
        Convert text to speech audio.

        Returns raw audio bytes (format defined by the TTS configuration).
        """
