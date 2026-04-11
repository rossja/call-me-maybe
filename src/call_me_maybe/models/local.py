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


_THINKING_RE = re.compile(
    r"<\|channel>.*?(?:<channel\|>|$)|<channel\|>|<\|channel>", re.DOTALL
)


def _strip_thinking(text: str) -> str:
    """Remove Gemma 4 channel/thought blocks from generated text."""
    return _THINKING_RE.sub("", text).strip()


_TOOL_CALL_PREFIX_RE = re.compile(r'<\|tool_call>call:([\w\-]+)\{')


def _extract_tool_calls(text: str) -> tuple[list[ToolCall], str]:
    """Parse tool calls from model output (native Gemma format or JSON).

    Returns ``(tool_calls, remaining_text)``.
    """
    stripped = text.strip()

    # 1. Gemma native format: <|tool_call>call:name{args}<tool_call|>
    calls: list[ToolCall] = []
    remove_spans: list[tuple[int, int]] = []
    for m in _TOOL_CALL_PREFIX_RE.finditer(stripped):
        name = m.group(1)
        brace_start = m.end() - 1
        brace_end = _find_matching_brace(stripped, brace_start)
        if brace_end is None:
            continue
        raw_args = stripped[brace_start + 1 : brace_end]
        arguments = _parse_gemma_args(raw_args)
        calls.append(ToolCall(id=f"local-{len(calls)}", name=name, arguments=arguments))
        # Mark the full span from <|tool_call> through }<tool_call|> for removal
        span_end = brace_end + 1
        suffix = "<tool_call|>"
        if stripped[span_end:span_end + len(suffix)] == suffix:
            span_end += len(suffix)
        remove_spans.append((m.start(), span_end))
    if calls:
        remaining = stripped
        for start, end in reversed(remove_spans):
            remaining = remaining[:start] + remaining[end:]
        remaining = re.sub(r'<\|tool_response>.*', '', remaining, flags=re.DOTALL)
        return (calls, remaining.strip())

    # 2. JSON format ({"name": ..., "arguments": ...} or {"tool": ..., "arguments": ...})
    tc = _try_parse_tool_json(stripped)
    if tc:
        return ([tc], "")

    # 3. JSON embedded in surrounding text
    start = stripped.find("{")
    while start != -1:
        end = _find_matching_brace(stripped, start)
        if end is not None:
            candidate = stripped[start : end + 1]
            tc = _try_parse_tool_json(candidate)
            if tc:
                before = stripped[:start].strip()
                after = stripped[end + 1 :].strip()
                remaining = f"{before} {after}".strip() if before or after else ""
                return ([tc], remaining)
        start = stripped.find("{", start + 1)

    return ([], text)


def _parse_gemma_args(raw: str) -> dict[str, Any]:
    """Recursively parse Gemma-style tool arguments into a Python dict.

    Handles quoted strings (``<|"|>val<|"|>``), nested objects (``{…}``),
    arrays (``[…]``), and bare scalars.
    """
    result: dict[str, Any] = {}
    i = 0
    n = len(raw)

    while i < n:
        # Skip whitespace / commas
        while i < n and raw[i] in " ,\n\r\t":
            i += 1
        if i >= n:
            break

        # Parse key
        key_start = i
        while i < n and raw[i] not in ":,}":
            i += 1
        key = raw[key_start:i].strip()
        if not key or i >= n or raw[i] != ":":
            break
        i += 1  # skip ':'

        # Parse value
        val, i = _parse_gemma_value(raw, i, n)
        result[key] = val

    return result


def _parse_gemma_value(raw: str, i: int, n: int) -> tuple[Any, int]:
    """Parse a single Gemma-encoded value starting at position *i*."""
    # Skip whitespace
    while i < n and raw[i] in " \n\r\t":
        i += 1
    if i >= n:
        return ("", i)

    _QUOTE = '<|"|>'
    _QLEN = len(_QUOTE)

    # Quoted string
    if raw[i:i + _QLEN] == _QUOTE:
        i += _QLEN
        end = raw.find(_QUOTE, i)
        if end == -1:
            return (raw[i:], n)
        val = raw[i:end]
        return (val, end + _QLEN)

    # Nested object
    if raw[i] == "{":
        close = _find_matching_brace(raw, i)
        if close is None:
            close = n - 1
        inner = raw[i + 1:close]
        return (_parse_gemma_args(inner), close + 1)

    # Array
    if raw[i] == "[":
        close = _find_matching_bracket(raw, i, n)
        inner = raw[i + 1:close]
        return (_parse_gemma_array(inner), close + 1)

    # Bare scalar (number / bool / unquoted string)
    val_start = i
    while i < n and raw[i] not in ",}]\n":
        i += 1
    return (_cast_scalar(raw[val_start:i].strip()), i)


def _parse_gemma_array(raw: str) -> list[Any]:
    """Parse a Gemma-encoded array body (contents between ``[`` and ``]``)."""
    items: list[Any] = []
    i, n = 0, len(raw)
    while i < n:
        while i < n and raw[i] in " ,\n\r\t":
            i += 1
        if i >= n:
            break
        val, i = _parse_gemma_value(raw, i, n)
        items.append(val)
    return items


def _find_matching_bracket(raw: str, start: int, n: int) -> int:
    depth = 0
    for i in range(start, n):
        if raw[i] == "[":
            depth += 1
        elif raw[i] == "]":
            depth -= 1
            if depth == 0:
                return i
    return n - 1


def _cast_scalar(val: str) -> Any:
    """Best-effort cast of a bare scalar value."""
    if not val:
        return val
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    low = val.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    return val


def _try_parse_tool_json(s: str) -> ToolCall | None:
    try:
        data = json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return None
    if isinstance(data, dict):
        if "tool" in data:
            return ToolCall(
                id="local-0",
                name=data["tool"],
                arguments=data.get("arguments", {}),
            )
        if "name" in data and "arguments" in data:
            return ToolCall(
                id="local-0",
                name=data["name"],
                arguments=data.get("arguments", {}),
            )
    if isinstance(data, list):
        calls = []
        for i, item in enumerate(data):
            if isinstance(item, dict) and "name" in item:
                calls.append(ToolCall(
                    id=f"local-{i}",
                    name=item["name"],
                    arguments=item.get("arguments", {}),
                ))
        if calls:
            return calls[0]  # return first; multi-call handled by caller
    return None


def _find_matching_brace(text: str, start: int) -> int | None:
    """Return the index of the closing ``}`` that matches the ``{`` at *start*."""
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
    return None


def _convert_history_for_template(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    """Convert ChatMessage history to a format compatible with Gemma's chat template.

    Gemma expects tool calls and their responses combined in a single assistant
    message with ``tool_calls`` and ``tool_responses`` keys — not as separate
    ``role="tool"`` messages.
    """
    result: list[dict[str, Any]] = []
    i = 0
    while i < len(messages):
        msg = messages[i]

        if msg.role == "assistant":
            tool_msgs: list[ChatMessage] = []
            j = i + 1
            while j < len(messages) and messages[j].role == "tool":
                tool_msgs.append(messages[j])
                j += 1

            if tool_msgs:
                tool_calls_list = []
                tool_responses_list = []
                for t in tool_msgs:
                    tool_calls_list.append({
                        "function": {
                            "name": t.name or "unknown",
                            "arguments": {},
                        }
                    })
                    try:
                        resp_data = json.loads(t.content)
                    except (json.JSONDecodeError, ValueError):
                        resp_data = t.content
                    tool_responses_list.append({
                        "name": t.name or "unknown",
                        "response": resp_data,
                    })
                result.append({
                    "role": "assistant",
                    "tool_calls": tool_calls_list,
                    "tool_responses": tool_responses_list,
                })
                i = j
            else:
                result.append({"role": msg.role, "content": msg.content})
                i += 1

        elif msg.role == "tool":
            # Orphaned tool message — shouldn't happen but handle gracefully
            result.append({
                "role": "user",
                "content": f"Tool result ({msg.name}): {msg.content}",
            })
            i += 1

        else:
            result.append({"role": msg.role, "content": msg.content})
            i += 1

    return result


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
        self._preload_stt()

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

    def _preload_stt(self) -> None:
        """Ensure the STT model weights are cached locally before first use.

        mlx_whisper downloads the model on the first transcribe() call, which
        creates a noticeable delay mid-conversation.  Calling snapshot_download
        here triggers the same HF Hub resolution at startup so the weights are
        already on disk when the first utterance arrives.
        """
        self._load_whisper()
        try:
            from huggingface_hub import snapshot_download  # type: ignore[import]
        except ImportError:
            return
        stt_model = self._settings.stt.model
        logger.info("Preloading STT model: %s", stt_model)
        snapshot_download(repo_id=stt_model)
        logger.info("STT model ready: %s", stt_model)

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
                    path_or_hf_repo=self._settings.stt.model,
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

        raw_messages = _convert_history_for_template(messages)

        template_kwargs: dict = {"enable_thinking": True}
        if llm.thinking_budget is not None:
            template_kwargs["thinking_budget"] = llm.thinking_budget

        if tools:
            template_kwargs["tools"] = [t.to_openai_schema() for t in tools]

        prompt = self._tokenizer.apply_chat_template(
            raw_messages,
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )

        from mlx_lm.sample_utils import make_sampler  # type: ignore[import]

        effective_max_tokens = llm.max_tokens
        if llm.thinking_budget is not None:
            effective_max_tokens += llm.thinking_budget

        logger.debug("LLM (local) generating response, model=%s", llm.model)
        response_text = generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=effective_max_tokens,
            sampler=make_sampler(temp=llm.temperature),
            verbose=False,
        )
        response_text = _strip_thinking(response_text)

        tool_calls: list[ToolCall] = []
        if tools:
            tool_calls, response_text = _extract_tool_calls(response_text)

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
