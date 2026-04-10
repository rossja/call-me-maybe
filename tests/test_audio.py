"""
Tests for audio utilities.

These tests mock sounddevice so they work without audio hardware.
"""

from __future__ import annotations

import io
import struct
import wave

import numpy as np
import pytest

from call_me_maybe.audio.playback import _decode_wav, _guess_suffix, AudioPlayback
from call_me_maybe.audio.capture import _rms, _to_wav, AudioCapture
from call_me_maybe.config.settings import Settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_wav_bytes(samples: np.ndarray, sample_rate: int = 16000, channels: int = 1) -> bytes:
    """Create WAV bytes from a numpy int16 array."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Tests – _rms
# ---------------------------------------------------------------------------


def test_rms_silence() -> None:
    silent = np.zeros(1000, dtype=np.int16)
    assert _rms(silent) == pytest.approx(0.0)


def test_rms_loud() -> None:
    loud = np.full(1000, 32767, dtype=np.int16)
    assert _rms(loud) > 0.9


def test_rms_empty() -> None:
    assert _rms(np.array([], dtype=np.int16)) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Tests – _to_wav
# ---------------------------------------------------------------------------


def test_to_wav_roundtrip() -> None:
    samples = np.random.randint(-1000, 1000, 4000, dtype=np.int16)
    wav_bytes = _to_wav(samples, sample_rate=16000, channels=1)
    data, sr = _decode_wav(wav_bytes)
    assert sr == 16000
    assert isinstance(data, np.ndarray)
    assert len(data) == 4000


def test_to_wav_stereo() -> None:
    samples = np.random.randint(-1000, 1000, 8000, dtype=np.int16)
    wav_bytes = _to_wav(samples.reshape(-1, 2), sample_rate=44100, channels=2)
    # Should not raise
    data, sr = _decode_wav(wav_bytes)
    assert sr == 44100


# ---------------------------------------------------------------------------
# Tests – _guess_suffix
# ---------------------------------------------------------------------------


def test_guess_suffix_wav() -> None:
    assert _guess_suffix(b"RIFF\x00\x00\x00\x00WAVE") == ".wav"


def test_guess_suffix_mp3_id3() -> None:
    assert _guess_suffix(b"ID3\x03\x00") == ".mp3"


def test_guess_suffix_mp3_sync() -> None:
    assert _guess_suffix(b"\xff\xfb\x90\x00") == ".mp3"


def test_guess_suffix_flac() -> None:
    assert _guess_suffix(b"fLaC\x00\x00\x00\x22") == ".flac"


def test_guess_suffix_ogg() -> None:
    assert _guess_suffix(b"OggS\x00\x02") == ".ogg"


def test_guess_suffix_unknown() -> None:
    assert _guess_suffix(b"\x00\x01\x02\x03") == ".audio"


# ---------------------------------------------------------------------------
# Tests – _decode_wav
# ---------------------------------------------------------------------------


def test_decode_wav_mono() -> None:
    samples = np.array([0, 1000, -1000, 32767], dtype=np.int16)
    wav_bytes = make_wav_bytes(samples)
    data, sr = _decode_wav(wav_bytes)
    assert sr == 16000
    assert data.shape == (4,)
    assert data.dtype == np.float32
    # Values should be normalised between -1 and 1
    assert np.all(np.abs(data) <= 1.0)


def test_decode_wav_raises_on_invalid() -> None:
    with pytest.raises(Exception):
        _decode_wav(b"not a wav file")


# ---------------------------------------------------------------------------
# Tests – AudioPlayback.play with WAV data
# ---------------------------------------------------------------------------


def test_playback_play_silence(monkeypatch: pytest.MonkeyPatch) -> None:
    """play() with WAV bytes should call sounddevice.play and sd.wait."""
    import call_me_maybe.audio.playback as playback_mod

    played_calls = []

    class MockSD:
        @staticmethod
        def play(data, samplerate, device):
            played_calls.append({"data": data, "sr": samplerate})

        @staticmethod
        def wait():
            pass

    monkeypatch.setattr(playback_mod, "sd", MockSD())
    monkeypatch.setattr(playback_mod, "_SD_AVAILABLE", True)

    samples = np.zeros(100, dtype=np.int16)
    wav_bytes = make_wav_bytes(samples)

    settings = Settings()
    pb = AudioPlayback(settings)
    pb.play(wav_bytes)

    assert len(played_calls) == 1
    assert played_calls[0]["sr"] == 16000


def test_playback_play_empty_bytes() -> None:
    """play() with empty bytes should be a no-op."""
    settings = Settings()
    pb = AudioPlayback(settings)
    pb.play(b"")  # Should not raise


# ---------------------------------------------------------------------------
# Tests – AudioCapture construction
# ---------------------------------------------------------------------------


def test_audio_capture_from_settings() -> None:
    settings = Settings()
    capture = AudioCapture.from_settings(settings)
    assert capture._cfg.sample_rate == 16000


def test_audio_capture_no_sounddevice(monkeypatch: pytest.MonkeyPatch) -> None:
    """record() should raise RuntimeError when sounddevice is unavailable."""
    import call_me_maybe.audio.capture as capture_mod

    monkeypatch.setattr(capture_mod, "_SD_AVAILABLE", False)
    settings = Settings()
    capture = AudioCapture.from_settings(settings)

    with pytest.raises(RuntimeError, match="sounddevice is not installed"):
        capture.record()
