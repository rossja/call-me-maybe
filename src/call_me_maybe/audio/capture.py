"""
Microphone audio capture.

Records audio from the default (or configured) input device until a period
of silence is detected, then returns the buffered audio as WAV bytes.
"""

from __future__ import annotations

import io
import logging
import wave
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from call_me_maybe.config.settings import Settings

logger = logging.getLogger(__name__)

# sounddevice is an optional runtime dependency – guard import so that tests
# and CI environments without audio hardware can still import the module.
try:
    import sounddevice as sd  # type: ignore[import]
    _SD_AVAILABLE = True
except (ImportError, OSError):
    sd = None  # type: ignore[assignment]
    _SD_AVAILABLE = False


class AudioCapture:
    """
    Captures microphone audio and returns it as in-memory WAV bytes.

    Usage::

        capture = AudioCapture(settings)
        audio_bytes = capture.record()   # blocks until utterance ends
    """

    def __init__(self, settings: "Settings") -> None:
        self._cfg = settings.audio.input
        self._stt_speech_threshold = settings.stt.speech_threshold
        self._stt_silence_threshold = settings.stt.silence_threshold
        self._stt_silence_duration = settings.stt.silence_duration
        self._stt_max_duration = settings.stt.max_duration
        if not _SD_AVAILABLE:
            logger.warning(
                "sounddevice is not available – audio capture will not work. "
                "Install it with: pip install sounddevice"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self) -> bytes:
        """
        Record from the microphone until silence is detected.

        Returns
        -------
        bytes
            16-bit mono PCM audio encoded as a WAV byte string.

        Raises
        ------
        RuntimeError
            If sounddevice is not available.
        """
        if not _SD_AVAILABLE:
            raise RuntimeError(
                "sounddevice is not installed. "
                "Install it with: pip install sounddevice"
            )

        cfg = self._cfg
        sample_rate = cfg.sample_rate
        channels = cfg.channels
        chunk_size = int(sample_rate * cfg.chunk_duration)
        speech_thresh = self._settings_speech_threshold
        silence_thresh = self._settings_silence_threshold
        silence_chunks_needed = int(
            self._settings_silence_duration / cfg.chunk_duration
        )
        max_chunks = int(self._settings_max_duration / cfg.chunk_duration)

        logger.debug(
            "Starting audio capture: sr=%d ch=%d speech_thresh=%.4f silence_thresh=%.4f",
            sample_rate,
            channels,
            speech_thresh,
            silence_thresh,
        )

        frames: list[np.ndarray] = []
        silence_chunks = 0
        recording_started = False

        with sd.InputStream(
            samplerate=sample_rate,
            channels=channels,
            dtype="int16",
            device=cfg.device,
            blocksize=chunk_size,
        ) as stream:
            for i in range(max_chunks):
                chunk, _ = stream.read(chunk_size)
                rms = _rms(chunk)

                if i % 5 == 0:  # log every 0.5s at default chunk_duration
                    state = "recording" if recording_started else "waiting"
                    logger.debug(
                        "RMS=%.5f  speech=%.5f  silence=%.5f  state=%s  silent_chunks=%d/%d",
                        rms, speech_thresh, silence_thresh, state, silence_chunks, silence_chunks_needed,
                    )

                if rms > speech_thresh:
                    recording_started = True
                    silence_chunks = 0
                    frames.append(chunk.copy())
                elif recording_started:
                    frames.append(chunk.copy())
                    if rms <= silence_thresh:
                        silence_chunks += 1
                    else:
                        silence_chunks = 0
                    if silence_chunks >= silence_chunks_needed:
                        break

        if not frames:
            logger.debug("No audio captured (nothing above silence threshold)")
            return b""

        audio = np.concatenate(frames, axis=0)
        wav_bytes = _to_wav(audio, sample_rate, channels)
        logger.debug("Captured %d samples → %d WAV bytes", len(audio), len(wav_bytes))
        return wav_bytes

    @property
    def _settings_speech_threshold(self) -> float:
        return self._stt_speech_threshold

    @property
    def _settings_silence_threshold(self) -> float:
        return self._stt_silence_threshold

    @property
    def _settings_silence_duration(self) -> float:
        return self._stt_silence_duration

    @property
    def _settings_max_duration(self) -> float:
        return self._stt_max_duration

    # ------------------------------------------------------------------
    # Factory-style constructor that carries STT thresholds
    # ------------------------------------------------------------------

    @classmethod
    def from_settings(cls, settings: "Settings") -> "AudioCapture":
        obj = cls.__new__(cls)
        obj._cfg = settings.audio.input
        obj._stt_speech_threshold = settings.stt.speech_threshold
        obj._stt_silence_threshold = settings.stt.silence_threshold
        obj._stt_silence_duration = settings.stt.silence_duration
        obj._stt_max_duration = settings.stt.max_duration
        return obj


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rms(chunk: np.ndarray) -> float:
    """Root-mean-square amplitude of an audio chunk (normalised to 0–1)."""
    if chunk.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)) / 32768.0)


def _to_wav(audio: np.ndarray, sample_rate: int, channels: int) -> bytes:
    """Encode a numpy int16 array as WAV bytes."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())
    return buf.getvalue()
