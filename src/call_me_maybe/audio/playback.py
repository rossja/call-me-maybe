"""
Audio playback – plays raw audio bytes through the system speaker.

Supports WAV, MP3, and other formats.  On macOS, system ``afplay`` is used as
a fallback if sounddevice cannot decode the format natively.
"""

from __future__ import annotations

import io
import logging
import platform
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from call_me_maybe.config.settings import Settings

logger = logging.getLogger(__name__)

try:
    import sounddevice as sd  # type: ignore[import]
    _SD_AVAILABLE = True
except (ImportError, OSError):
    sd = None  # type: ignore[assignment]
    _SD_AVAILABLE = False


class AudioPlayback:
    """
    Plays audio bytes through the configured output device.

    Usage::

        playback = AudioPlayback.from_settings(settings)
        playback.play(audio_bytes)  # blocks until done
    """

    def __init__(self, settings: "Settings") -> None:
        self._cfg = settings.audio.output
        if not _SD_AVAILABLE:
            logger.warning(
                "sounddevice is not available – audio playback will not work."
            )

    @classmethod
    def from_settings(cls, settings: "Settings") -> "AudioPlayback":
        return cls(settings)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def play(self, audio_bytes: bytes) -> None:
        """
        Play audio bytes through the speaker.

        Parameters
        ----------
        audio_bytes:
            Raw audio data.  WAV is decoded natively; other formats are handled
            via ``afplay`` on macOS or written to a temp file and opened.
        """
        if not audio_bytes:
            return

        # Try to decode as WAV first (cheapest)
        try:
            data, sample_rate = _decode_wav(audio_bytes)
            self._play_array(data, sample_rate)
            return
        except Exception as exc:  # noqa: BLE001
            logger.debug("WAV decode failed, falling back to system player: %s", exc)

        # Fall back to system player
        self._play_via_system(audio_bytes)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _play_array(self, data: np.ndarray, sample_rate: int) -> None:
        """Play a numpy array through sounddevice."""
        if not _SD_AVAILABLE:
            raise RuntimeError("sounddevice is not available for playback.")
        cfg = self._cfg
        sd.play(data, samplerate=sample_rate, device=cfg.device)
        sd.wait()

    def _play_via_system(self, audio_bytes: bytes) -> None:
        """
        Write audio to a temp file and play it with the OS default player.

        On macOS, this uses ``afplay``.
        On Linux, this tries ``aplay`` (ALSA) or ``ffplay``.
        """
        suffix = _guess_suffix(audio_bytes)
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            if platform.system() == "Darwin":
                subprocess.run(["afplay", tmp_path], check=True, capture_output=True)
            else:
                # Try common Linux players in order
                for player in ("aplay", "ffplay", "mpv"):
                    try:
                        subprocess.run(
                            [player, "-nodisp", "-autoexit", tmp_path]
                            if player == "ffplay"
                            else [player, tmp_path],
                            check=True,
                            capture_output=True,
                        )
                        break
                    except (FileNotFoundError, subprocess.CalledProcessError):
                        continue
                else:
                    logger.warning("No suitable audio player found to play audio.")
        finally:
            Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _decode_wav(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decode WAV bytes into a float32 numpy array and sample rate."""
    buf = io.BytesIO(audio_bytes)
    with wave.open(buf, "rb") as wf:
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
    dtype = dtype_map.get(sample_width, np.int16)
    data = np.frombuffer(raw, dtype=dtype)
    if n_channels > 1:
        data = data.reshape(-1, n_channels)

    # Normalise to float32
    data = data.astype(np.float32) / float(np.iinfo(dtype).max)
    return data, sample_rate


def _guess_suffix(audio_bytes: bytes) -> str:
    """Guess the file extension from magic bytes."""
    if audio_bytes[:4] == b"RIFF":
        return ".wav"
    if audio_bytes[:3] == b"ID3" or audio_bytes[:2] in (b"\xff\xfb", b"\xff\xf3"):
        return ".mp3"
    if audio_bytes[:4] == b"fLaC":
        return ".flac"
    if audio_bytes[:4] == b"OggS":
        return ".ogg"
    return ".audio"
