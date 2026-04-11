"""
Local Apple Silicon / MLX backend.

Requires:
  - macOS on Apple Silicon (M-series chip)
  - At least 24 GB unified memory (configurable via config.yaml)
  - ``uv sync``

The backend:
  - Loads the LLM with mlx-lm (supports MLX-compatible text models from
    mlx-community on Hugging Face).
  - Uses mlx-whisper for speech-to-text if available, otherwise falls back to
    the ``openai-whisper`` Python package.
  - Generates TTS audio via the Fish Audio API (``FISH_AUDIO_API_KEY`` env var)
    or, as a fallback, uses macOS's built-in ``say`` command.
"""

from __future__ import annotations

import io
import json
import logging
import os
import platform
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import TYPE_CHECKING, Any

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


def _check_apple_silicon() -> None:
    """Raise RuntimeError if not running on Apple Silicon."""
    if platform.system() != "Darwin":
        raise RuntimeError(
            "The local MLX backend requires macOS on Apple Silicon."
        )
    if platform.machine() not in ("arm64", "arm"):
        raise RuntimeError(
            "The local MLX backend requires an Apple Silicon (M-series) chip. "
            f"Detected architecture: {platform.machine()}"
        )


def _check_memory(min_gb: int) -> None:
    """Emit a warning if system RAM is below the minimum requirement."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            check=True,
        )
        total_gb = int(result.stdout.strip()) / (1024**3)
        if total_gb < min_gb:
            logger.warning(
                "System RAM %.1f GiB is below the recommended minimum of %d GiB.",
                total_gb,
                min_gb,
            )
        else:
            logger.debug("System RAM %.1f GiB meets the %d GiB requirement.", total_gb, min_gb)
    except Exception:
        logger.debug("Could not determine system RAM.", exc_info=True)


class LocalMLXBackend(ModelBackend):
    """
    Backend that uses MLX for local inference on Apple Silicon.
    """

    def __init__(self, settings: "Settings") -> None:
        _check_apple_silicon()
        _check_memory(settings.local.min_memory_gb)

        self._settings = settings
        self._model: Any = None
        self._tokenizer: Any = None
        self._whisper_model: Any = None
        self._load_llm()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _load_llm(self) -> None:
        """Load the MLX language model."""
        try:
            from mlx_lm import load  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "mlx-lm is required for the local backend. "
                "Install it with: uv sync"
            ) from exc

        model_name = self._settings.llm.model
        cache_dir = self._settings.local.model_cache_dir
        logger.info("Loading local MLX model: %s", model_name)

        kwargs: dict[str, Any] = {}
        if cache_dir:
            os.environ.setdefault("HF_HOME", cache_dir)

        self._model, self._tokenizer = load(model_name, **kwargs)
        logger.info("MLX model loaded: %s", model_name)

    def _load_whisper(self) -> Any:
        """Lazily load the Whisper transcription model."""
        if self._whisper_model is not None:
            return self._whisper_model

        try:
            import mlx_whisper  # type: ignore[import]
            self._whisper_model = mlx_whisper
            logger.info("Using mlx-whisper for STT")
        except ImportError:
            try:
                import whisper  # type: ignore[import]
                self._whisper_model = whisper.load_model("base")
                logger.info("Using openai-whisper (CPU) for STT")
            except ImportError as exc:
                raise ImportError(
                    "No Whisper implementation found. Install one of:\n"
                    "  pip install mlx-whisper\n"
                    "  pip install openai-whisper"
                ) from exc
        return self._whisper_model

    # ------------------------------------------------------------------
    # STT
    # ------------------------------------------------------------------

    def transcribe(self, audio_bytes: bytes, *, language: str | None = None) -> str:
        """Transcribe audio using mlx-whisper or openai-whisper."""
        whisper_mod = self._load_whisper()
        lang = language or self._settings.stt.language

        # Write to a temporary WAV file for Whisper
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(audio_bytes)

        try:
            import mlx_whisper  # type: ignore[import]
            if isinstance(whisper_mod, type(mlx_whisper)):
                result = mlx_whisper.transcribe(
                    tmp_path,
                    path_or_hf_repo=f"mlx-community/{self._settings.stt.model}-mlx",
                    language=lang,
                )
            else:
                result = whisper_mod.transcribe(tmp_path, language=lang)
        except ImportError:
            result = whisper_mod.transcribe(tmp_path, language=lang)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        text = result.get("text", "").strip() if isinstance(result, dict) else str(result).strip()
        logger.debug("STT (local): %r", text)
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
        """Run chat with the MLX model using its chat template."""
        from mlx_lm import generate  # type: ignore[import]

        llm = self._settings.llm

        # Build prompt using the tokenizer's chat template
        raw_messages = [{"role": m.role, "content": m.content} for m in messages]
        if tools:
            # Append tool descriptions to the system prompt
            tool_descriptions = "\n".join(
                f"- {t.name}: {t.description}" for t in tools
            )
            raw_messages.insert(
                0,
                {
                    "role": "system",
                    "content": (
                        llm.system_prompt
                        + f"\n\nAvailable tools:\n{tool_descriptions}\n\n"
                        "To call a tool, respond with JSON in the format:\n"
                        '{"tool": "<name>", "arguments": {<args>}}'
                    ),
                },
            )

        prompt = self._tokenizer.apply_chat_template(
            raw_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        logger.debug("LLM (local) generating response, model=%s", llm.model)
        response_text = generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=llm.max_tokens,
            temp=llm.temperature,
            verbose=False,
        )

        # Check if the model returned a tool call in JSON
        tool_calls: list[ToolCall] = []
        if tools:
            try:
                data = json.loads(response_text.strip())
                if isinstance(data, dict) and "tool" in data:
                    tool_calls.append(
                        ToolCall(
                            id="local-0",
                            name=data["tool"],
                            arguments=data.get("arguments", {}),
                        )
                    )
                    response_text = ""
            except (json.JSONDecodeError, ValueError):
                pass

        return ModelResponse(text=response_text, tool_calls=tool_calls)

    # ------------------------------------------------------------------
    # TTS
    # ------------------------------------------------------------------

    def synthesize(self, text: str) -> bytes:
        """
        Convert text to speech.

        Uses Fish Audio API if ``FISH_AUDIO_API_KEY`` is set, otherwise falls
        back to macOS's built-in ``say`` command (AIFF → WAV).
        """
        if self._settings.fish_audio_api_key:
            return self._fish_audio_tts(text)
        return self._macos_say_tts(text)

    def _fish_audio_tts(self, text: str) -> bytes:
        """Use the Fish Audio API for TTS."""
        import httpx

        tts = self._settings.tts
        api_key = self._settings.fish_audio_api_key
        voice_id = self._settings.fish_audio_voice_id

        payload: dict[str, Any] = {
            "text": text,
            "format": tts.audio_format,
            "latency": "normal",
        }
        if voice_id:
            payload["reference_id"] = voice_id

        logger.debug("Fish Audio TTS: voice=%s", voice_id)
        with httpx.Client(timeout=60) as client:
            response = client.post(
                "https://api.fish.audio/v1/tts",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            return response.content

    def _macos_say_tts(self, text: str) -> bytes:
        """Fall back to macOS ``say`` command, returning WAV bytes."""
        tts = self._settings.tts

        with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as tmp:
            tmp_path = tmp.name

        wav_path: str | None = None
        try:
            subprocess.run(
                ["say", "-o", tmp_path, text],
                check=True,
                capture_output=True,
            )
            # Convert AIFF to PCM WAV
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_tmp:
                wav_path = wav_tmp.name
            subprocess.run(
                [
                    "afconvert",
                    "-f", "WAVE",
                    "-d", "LEI16@24000",
                    tmp_path,
                    wav_path,
                ],
                check=True,
                capture_output=True,
            )
            with open(wav_path, "rb") as fh:
                return fh.read()
        finally:
            Path(tmp_path).unlink(missing_ok=True)
            if wav_path is not None:
                Path(wav_path).unlink(missing_ok=True)
