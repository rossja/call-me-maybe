"""
Tests for call_me_maybe.config.settings.

Exercises YAML loading, .env override precedence, API key resolution, and
config validation without requiring any real hardware or API access.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from call_me_maybe.config.settings import Settings, load_settings, _expand_env_vars


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def minimal_yaml(tmp_path: Path) -> Path:
    """Write a minimal valid config.yaml and return its path."""
    data = {
        "provider": "remote",
        "llm": {"model": "test-model"},
        "remote": {"base_url": "https://api.example.com/v1"},
    }
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.dump(data))
    return cfg


@pytest.fixture()
def full_yaml(tmp_path: Path) -> Path:
    """Write a more complete config.yaml and return its path."""
    data = {
        "provider": "remote",
        "stt": {"model": "whisper-1", "silence_threshold": 0.05, "silence_duration": 2.0},
        "llm": {
            "model": "liquid/lfm-2.5-audio-1.5b",
            "temperature": 0.8,
            "max_tokens": 256,
            "context_window_turns": 10,
        },
        "tts": {"model": "tts-1", "voice": "echo", "speed": 1.2},
        "remote": {
            "base_url": "https://openrouter.ai/api/v1",
            "timeout": 60,
        },
        "audio": {
            "input": {"sample_rate": 16000, "channels": 1},
            "output": {"sample_rate": 24000, "channels": 1},
        },
        "agent": {
            "greeting": "Hi there!",
            "wake_word": "hey maybe",
            "tools": {
                "mcp": {"enabled": False, "servers": []},
                "skills": {"enabled": False, "modules": []},
                "a2a": {"enabled": False, "agents": []},
            },
        },
    }
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.dump(data))
    return cfg


# ---------------------------------------------------------------------------
# Tests – basic loading
# ---------------------------------------------------------------------------


def test_load_defaults_when_no_yaml(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Settings should load with defaults when no config.yaml exists."""
    monkeypatch.chdir(tmp_path)  # No config.yaml here
    settings = load_settings(tmp_path / "nonexistent.yaml")
    assert settings.provider == "remote"  # default
    assert settings.llm.model == "liquid/lfm-2.5-audio-1.5b"  # default


def test_load_minimal_yaml(minimal_yaml: Path) -> None:
    """Values in config.yaml should override built-in defaults."""
    settings = load_settings(minimal_yaml)
    assert settings.provider == "remote"
    assert settings.llm.model == "test-model"
    assert settings.remote.base_url == "https://api.example.com/v1"


def test_load_full_yaml(full_yaml: Path) -> None:
    """All YAML sections should parse correctly."""
    settings = load_settings(full_yaml)
    assert settings.stt.silence_threshold == pytest.approx(0.05)
    assert settings.llm.temperature == pytest.approx(0.8)
    assert settings.llm.max_tokens == 256
    assert settings.llm.context_window_turns == 10
    assert settings.tts.voice == "echo"
    assert settings.remote.timeout == 60
    assert settings.agent.greeting == "Hi there!"
    assert settings.agent.wake_word == "hey maybe"
    assert settings.audio.input.sample_rate == 16000
    assert settings.audio.output.sample_rate == 24000


# ---------------------------------------------------------------------------
# Tests – API key resolution
# ---------------------------------------------------------------------------


def test_openrouter_key_sets_effective_key(minimal_yaml: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    settings = load_settings(minimal_yaml)
    assert settings.effective_api_key == "or-test-key"


def test_openai_key_fallback(minimal_yaml: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    monkeypatch.delenv("API_KEY", raising=False)
    settings = load_settings(minimal_yaml)
    assert settings.effective_api_key == "sk-test-key"


def test_generic_api_key_fallback(minimal_yaml: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("API_KEY", "generic-key")
    settings = load_settings(minimal_yaml)
    assert settings.effective_api_key == "generic-key"


def test_openrouter_takes_priority_over_openai(minimal_yaml: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-key")
    settings = load_settings(minimal_yaml)
    assert settings.effective_api_key == "or-key"


def test_no_api_key(minimal_yaml: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    settings = load_settings(minimal_yaml)
    assert settings.effective_api_key is None


# ---------------------------------------------------------------------------
# Tests – .env file overrides environment variables
# ---------------------------------------------------------------------------


def test_dotenv_overrides_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Values in .env should override existing environment variables."""
    # Set a key in the environment
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")

    # Write a .env file with a different value
    dotenv = tmp_path / ".env"
    dotenv.write_text("OPENROUTER_API_KEY=dotenv-key\n")
    monkeypatch.chdir(tmp_path)

    # Also write a minimal config.yaml
    cfg = tmp_path / "config.yaml"
    cfg.write_text("provider: remote\n")

    settings = load_settings(cfg)
    assert settings.effective_api_key == "dotenv-key"


# ---------------------------------------------------------------------------
# Tests – validate_provider
# ---------------------------------------------------------------------------


def test_validate_provider_remote_missing_key(minimal_yaml: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    settings = load_settings(minimal_yaml)
    with pytest.raises(ValueError, match="no API key"):
        settings.validate_provider()


def test_validate_provider_remote_with_key(minimal_yaml: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "ok-key")
    settings = load_settings(minimal_yaml)
    # Should not raise
    settings.validate_provider()


def test_validate_provider_local_no_mlx(minimal_yaml: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """If mlx is not installed, validate_provider should raise ImportError."""
    data = yaml.safe_load(minimal_yaml.read_text())
    data["provider"] = "local"
    minimal_yaml.write_text(yaml.dump(data))
    settings = load_settings(minimal_yaml)
    # mlx is not installed in the test environment
    with pytest.raises((ImportError, RuntimeError)):
        settings.validate_provider()


# ---------------------------------------------------------------------------
# Tests – expand_env_vars helper
# ---------------------------------------------------------------------------


def test_expand_env_vars_string(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_VAR", "hello")
    assert _expand_env_vars("${MY_VAR}") == "hello"


def test_expand_env_vars_nested(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_KEY", "secret")
    result = _expand_env_vars({"env": {"KEY": "${MY_KEY}"}})
    assert result == {"env": {"KEY": "secret"}}


def test_expand_env_vars_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ITEM", "world")
    result = _expand_env_vars(["hello", "${ITEM}"])
    assert result == ["hello", "world"]


def test_expand_env_vars_non_string_passthrough() -> None:
    assert _expand_env_vars(42) == 42
    assert _expand_env_vars(None) is None
    assert _expand_env_vars(True) is True


# ---------------------------------------------------------------------------
# Tests – Settings model structure
# ---------------------------------------------------------------------------


def test_settings_tool_defaults() -> None:
    """Default Settings should have all tools disabled."""
    s = Settings()
    assert s.agent.tools.mcp.enabled is False
    assert s.agent.tools.skills.enabled is False
    assert s.agent.tools.a2a.enabled is False


def test_settings_remote_defaults() -> None:
    """Remote backend defaults should include OpenRouter URL."""
    s = Settings()
    assert "openrouter" in s.remote.base_url


def test_settings_local_min_memory() -> None:
    """Default local backend minimum memory should be 24 GiB."""
    s = Settings()
    assert s.local.min_memory_gb == 24


def test_settings_log_level_default() -> None:
    s = Settings()
    assert s.log_level.lower() == "info"
