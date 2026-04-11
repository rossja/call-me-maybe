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
  - Generates TTS audio via (in priority order):
      1. Fish Audio API (``FISH_AUDIO_API_KEY`` env var)
      2. Voxtral via mlx-audio (when ``tts.model`` contains "voxtral" and
         ``mlx-audio`` is installed); supports 20 preset voices across 9 languages
      3. macOS built-in ``say`` command (``tts.voice`` is passed as ``-v``)
"""

from __future__ import annotations

import io
import json
import logging
import os
import platform
import re
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


_THINKING_RE = re.compile(r"<\|channel>.*?<channel\|>", re.DOTALL)


def _strip_thinking(text: str) -> str:
    """Remove Gemma 4 channel/thought blocks from generated text."""
    return _THINKING_RE.sub("", text).strip()


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
        self._tts_model: Any = None
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
        self._ensure_chat_template(model_name)
        logger.info("MLX model loaded: %s", model_name)

    def _ensure_chat_template(self, model_name: str) -> None:
        """
        Guarantee the loaded tokenizer has a chat_template set.

        Resolution order (first success wins):
          1. Explicit override in settings.local.chat_template
          2. Tokenizer already has one (nothing to do)
          3. chat_template.jinja in the local HF cache snapshot directory
          4. chat_template.jinja fetched from HF Hub (exact model name)
          5. chat_template.jinja fetched from HF Hub (inferred source model,
             after stripping mlx-community prefix and quantization suffixes)
          6. tokenizer_config.json["chat_template"] from HF Hub
             (exact name, then inferred source model)
        Raises RuntimeError if no template can be found.
        """
        # 1. Config override
        if self._settings.local.chat_template:
            logger.info(
                "Using chat_template override from config for model: %s", model_name
            )
            self._apply_chat_template_string(self._settings.local.chat_template)
            return

        # 2. Tokenizer already has one
        if getattr(self._tokenizer, "has_chat_template", False):
            return

        logger.warning(
            "Model '%s' tokenizer is missing a chat_template — attempting auto-resolution.",
            model_name,
        )

        candidates = self._template_candidate_names(model_name)

        # 3. Local HF cache: chat_template.jinja
        template = self._load_local_jinja(model_name)
        if template:
            logger.warning(
                "Loaded chat_template.jinja from local HF cache for '%s'.", model_name
            )
            self._apply_chat_template_string(template)
            return

        # 4 & 5. HF Hub: chat_template.jinja (all candidates)
        for candidate in candidates:
            template = self._fetch_hf_jinja(candidate)
            if template:
                logger.warning(
                    "Fetched chat_template.jinja from HF Hub for '%s' (via '%s').",
                    model_name,
                    candidate,
                )
                self._apply_chat_template_string(template)
                return

        # 6. HF Hub: tokenizer_config.json["chat_template"] (all candidates)
        for candidate in candidates:
            template = self._fetch_hf_tokenizer_config_template(candidate)
            if template:
                logger.warning(
                    "Fetched chat_template from tokenizer_config.json on HF Hub for '%s' (via '%s').",
                    model_name,
                    candidate,
                )
                self._apply_chat_template_string(template)
                return

        raise RuntimeError(
            f"Model '{model_name}' tokenizer has no chat_template and auto-resolution "
            "failed. Set local.chat_template in config.yaml with a Jinja2 template "
            "string, or use an instruction-tuned model variant that includes one."
        )

    def _apply_chat_template_string(self, template: str) -> None:
        """Patch the chat_template onto the underlying transformers tokenizer."""
        underlying = getattr(self._tokenizer, "_tokenizer", self._tokenizer)
        underlying.chat_template = template
        self._tokenizer.has_chat_template = True

    def _template_candidate_names(self, model_name: str) -> list[str]:
        """
        Return a list of model name candidates to try when fetching a template.

        Always includes the exact name first, then attempts to derive the
        upstream source model by stripping mlx-community org prefix and
        common quantization/conversion suffixes.
        """
        candidates: list[str] = [model_name]

        # Strip mlx-community/ prefix and known suffixes to guess source model
        name = model_name
        if "/" in name:
            org, repo = name.split("/", 1)
        else:
            org, repo = "", name

        # Remove trailing quantization / conversion suffixes
        cleaned = re.sub(
            r"[-_](4bit|8bit|2bit|3bit|f16|bf16|mlx|MLX|instruct[-_]4bit|instruct[-_]8bit)$",
            "",
            repo,
            flags=re.IGNORECASE,
        )
        # Keep stripping until stable (handles e.g. foo-it-4bit -> foo-it -> foo)
        while True:
            next_cleaned = re.sub(
                r"[-_](4bit|8bit|2bit|3bit|f16|bf16|mlx|MLX|instruct[-_]4bit|instruct[-_]8bit)$",
                "",
                cleaned,
                flags=re.IGNORECASE,
            )
            if next_cleaned == cleaned:
                break
            cleaned = next_cleaned

        if cleaned != repo:
            # Try the same org with the cleaned name
            if org:
                candidates.append(f"{org}/{cleaned}")
            # Also try common upstream orgs
            for upstream_org in ("google", "meta-llama", "mistralai", "microsoft"):
                candidates.append(f"{upstream_org}/{cleaned}")

        return candidates

    def _hf_cache_snapshot_dir(self, model_name: str) -> Path | None:
        """Return the local HF hub snapshot directory for model_name, if cached."""
        cache_root = Path(
            os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
        ) / "hub"
        # HF cache uses models--org--repo naming
        dir_name = "models--" + model_name.replace("/", "--")
        snapshots_dir = cache_root / dir_name / "snapshots"
        if not snapshots_dir.is_dir():
            return None
        # Pick the newest snapshot
        snapshots = sorted(snapshots_dir.iterdir(), key=lambda p: p.stat().st_mtime)
        return snapshots[-1] if snapshots else None

    def _load_local_jinja(self, model_name: str) -> str | None:
        """Try to read chat_template.jinja from the local HF cache."""
        snapshot = self._hf_cache_snapshot_dir(model_name)
        if snapshot is None:
            return None
        jinja_path = snapshot / "chat_template.jinja"
        if jinja_path.is_file():
            return jinja_path.read_text(encoding="utf-8")
        return None

    def _fetch_hf_jinja(self, model_name: str) -> str | None:
        """Try to fetch chat_template.jinja from the HF Hub for model_name."""
        try:
            import httpx

            url = f"https://huggingface.co/{model_name}/resolve/main/chat_template.jinja"
            response = httpx.get(url, follow_redirects=True, timeout=15)
            if response.status_code == 200:
                return response.text
        except Exception:
            logger.debug(
                "Failed to fetch chat_template.jinja for '%s'.", model_name, exc_info=True
            )
        return None

    def _fetch_hf_tokenizer_config_template(self, model_name: str) -> str | None:
        """Try to extract chat_template from tokenizer_config.json on the HF Hub."""
        try:
            import httpx

            url = f"https://huggingface.co/{model_name}/resolve/main/tokenizer_config.json"
            response = httpx.get(url, follow_redirects=True, timeout=15)
            if response.status_code == 200:
                data = response.json()
                template = data.get("chat_template")
                if isinstance(template, str) and template.strip():
                    return template
        except Exception:
            logger.debug(
                "Failed to fetch tokenizer_config.json for '%s'.", model_name, exc_info=True
            )
        return None

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
            enable_thinking=False,
        )

        from mlx_lm.sample_utils import make_sampler  # type: ignore[import]

        logger.debug("LLM (local) generating response, model=%s", llm.model)
        response_text = generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=llm.max_tokens,
            sampler=make_sampler(temp=llm.temperature),
            verbose=False,
        )
        response_text = _strip_thinking(response_text)

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

        Priority:
          1. Fish Audio API if ``FISH_AUDIO_API_KEY`` is set.
          2. Voxtral via mlx-audio if ``tts.model`` contains "voxtral" and
             ``mlx-audio`` is installed.
          3. macOS built-in ``say`` command (``tts.voice`` passed as ``-v``).
        """
        if self._settings.fish_audio_api_key:
            return self._fish_audio_tts(text)
        if "voxtral" in self._settings.tts.model.lower():
            return self._voxtral_tts(text)
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

    def _voxtral_tts(self, text: str) -> bytes:
        """Use mlx-audio (Voxtral) for TTS, returning WAV bytes."""
        try:
            from mlx_audio.tts.utils import load as mlx_audio_load  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "mlx-audio is required for Voxtral TTS. "
                "Install it with: uv sync --extra local"
            ) from exc

        import numpy as np

        tts = self._settings.tts

        if self._tts_model is None:
            logger.info("Loading Voxtral TTS model: %s", tts.model)
            self._tts_model = mlx_audio_load(tts.model)
            logger.info("Voxtral TTS model loaded")

        logger.debug("Voxtral TTS: voice=%s", tts.voice)
        audio_chunks: list[Any] = []
        sample_rate: int = 24000
        for result in self._tts_model.generate(text=text, voice=tts.voice):
            audio_chunks.append(np.array(result.audio))
            sample_rate = result.sample_rate

        if not audio_chunks:
            raise RuntimeError("Voxtral TTS returned no audio chunks.")

        audio_np = np.concatenate(audio_chunks).astype(np.float32)

        # Normalise and encode as 16-bit PCM WAV
        max_val = np.abs(audio_np).max()
        if max_val > 0:
            audio_np = audio_np / max_val
        pcm = (audio_np * 32767).astype(np.int16)

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(pcm.tobytes())
        return buf.getvalue()

    def _macos_say_tts(self, text: str) -> bytes:
        """Fall back to macOS ``say`` command, returning WAV bytes."""
        tts = self._settings.tts

        with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as tmp:
            tmp_path = tmp.name

        wav_path: str | None = None
        try:
            cmd = ["say", "-o", tmp_path]
            if tts.voice:
                cmd += ["-v", tts.voice]
            cmd.append(text)
            subprocess.run(cmd, check=True, capture_output=True)
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
