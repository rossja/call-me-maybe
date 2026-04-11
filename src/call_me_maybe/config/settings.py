"""
Settings loader for call-me-maybe.

Load order (later entries win):
  1. Built-in defaults
  2. config.yaml (from project root or path given by --config)
  3. Environment variables already set in the shell
  4. .env file values  (if a .env file is found they override shell env vars)

Sensitive values (API keys, etc.) are never stored in config.yaml –
they must come from environment variables or the .env file.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Nested config models
# ---------------------------------------------------------------------------


class STTConfig(BaseModel):
    model: str = "mlx-community/whisper-large-v3-turbo"
    language: str | None = None
    speech_threshold: float = 0.02
    silence_threshold: float = 0.01
    silence_duration: float = 1.5
    max_duration: float = 60.0


class LLMConfig(BaseModel):
    model: str
    system_prompt: str
    temperature: float = 0.5
    max_tokens: int = 512
    context_window_turns: int = 20
    thinking_budget: int | None = 512


class TTSConfig(BaseModel):
    model: str
    voice: str
    speed: float = 1.0
    audio_format: str = "mp3"


class LocalBackendConfig(BaseModel):
    min_memory_gb: int = 24
    model_cache_dir: str | None = None
    quantization: int = 4
    chat_template: str | None = None


class RemoteBackendConfig(BaseModel):
    base_url: str = "https://openrouter.ai/api/v1"
    timeout: int = 120
    extra_headers: dict[str, str] = Field(
        default_factory=lambda: {
            "HTTP-Referer": "https://github.com/rossja/call-me-maybe",
            "X-Title": "call-me-maybe",
        }
    )


class AudioInputConfig(BaseModel):
    device: int | str | None = None
    sample_rate: int = 16000
    channels: int = 1
    chunk_duration: float = 0.1


class AudioOutputConfig(BaseModel):
    device: int | str | None = None
    sample_rate: int = 24000
    channels: int = 1


class AudioConfig(BaseModel):
    input: AudioInputConfig = Field(default_factory=AudioInputConfig)
    output: AudioOutputConfig = Field(default_factory=AudioOutputConfig)


class MCPServerConfig(BaseModel):
    name: str
    command: list[str]
    env: dict[str, str] = Field(default_factory=dict)


class MCPToolsConfig(BaseModel):
    enabled: bool = False
    servers: list[MCPServerConfig] = Field(default_factory=list)


class SkillsToolsConfig(BaseModel):
    enabled: bool = False
    modules: list[str] = Field(default_factory=list)


class A2AAgentConfig(BaseModel):
    name: str
    url: str
    description: str = ""


class A2AToolsConfig(BaseModel):
    enabled: bool = False
    agents: list[A2AAgentConfig] = Field(default_factory=list)


class ToolsConfig(BaseModel):
    mcp: MCPToolsConfig = Field(default_factory=MCPToolsConfig)
    skills: SkillsToolsConfig = Field(default_factory=SkillsToolsConfig)
    a2a: A2AToolsConfig = Field(default_factory=A2AToolsConfig)


class AgentConfig(BaseModel):
    greeting: str = "Hello! I'm Maybe, your voice assistant. How can I help?"
    wake_word: str | None = None
    tools: ToolsConfig = Field(default_factory=ToolsConfig)


# ---------------------------------------------------------------------------
# Root settings (uses pydantic-settings so env vars are automatically picked up)
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """
    Root settings object.

    Config values come from config.yaml.
    Secrets come from environment variables (or .env file).
    """

    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        extra="ignore",
    )

    # Backend selection
    provider: str = "local"

    # Sub-configs (populated from YAML, not directly from env)
    stt: STTConfig = Field(default_factory=STTConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    local: LocalBackendConfig = Field(default_factory=LocalBackendConfig)
    remote: RemoteBackendConfig = Field(default_factory=RemoteBackendConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)

    # ---------------------------------------------------------------------------
    # Secrets – read exclusively from env vars / .env file
    # ---------------------------------------------------------------------------
    openrouter_api_key: str | None = Field(default=None, alias="OPENROUTER_API_KEY")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    api_key: str | None = Field(default=None, alias="API_KEY")
    fish_audio_api_key: str | None = Field(default=None, alias="FISH_AUDIO_API_KEY")
    fish_audio_voice_id: str | None = Field(default=None, alias="FISH_AUDIO_VOICE_ID")
    log_level: str = Field(default="info", alias="LOG_LEVEL")

    @model_validator(mode="after")
    def _resolve_api_key(self) -> "Settings":
        """
        Resolve the effective API key for the remote backend.

        Priority (highest first):
          1. OPENROUTER_API_KEY  (when using OpenRouter)
          2. OPENAI_API_KEY      (when using api.openai.com)
          3. API_KEY             (generic fallback)
        """
        if self.openrouter_api_key:
            self.api_key = self.openrouter_api_key
        elif self.openai_api_key and not self.api_key:
            self.api_key = self.openai_api_key
        return self

    @property
    def effective_api_key(self) -> str | None:
        """Return the API key to use for remote inference."""
        return self.api_key

    def validate_provider(self) -> None:
        """Raise ValueError if the chosen provider is misconfigured."""
        if self.provider == "remote" and not self.effective_api_key:
            raise ValueError(
                "Provider is set to 'remote' but no API key was found. "
                "Set OPENROUTER_API_KEY, OPENAI_API_KEY, or API_KEY in your "
                "environment or .env file."
            )
        if self.provider == "local":
            try:
                import mlx  # noqa: F401
            except ImportError as exc:
                raise ImportError(
                    "Provider is set to 'local' but mlx is not installed. "
                    "Install it with: uv sync"
                ) from exc


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def _find_default_config() -> Path:
    """
    Search upward from the current working directory for ``config.yaml``.

    Falls back to a path relative to this file if not found in the directory
    tree, so that the package still works when installed as a library.
    """
    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents]:
        candidate = parent / "config.yaml"
        if candidate.exists():
            return candidate
    # Fallback: project root relative to this file's location
    # (src/call_me_maybe/config/settings.py → ../../..)
    return Path(__file__).parent.parent.parent.parent / "config.yaml"


_DEFAULT_CONFIG_PATH = _find_default_config()


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file, returning an empty dict if the file does not exist."""
    if not path.exists():
        logger.debug("Config file not found at %s – using defaults", path)
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    logger.debug("Loaded config from %s", path)
    return data


def _expand_env_vars(obj: Any) -> Any:
    """
    Recursively expand ${ENV_VAR} placeholders inside string values.

    This allows MCP server env maps in config.yaml to reference env vars
    without embedding secrets in the file.
    """
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    if isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_vars(item) for item in obj]
    return obj


def load_settings(config_path: str | Path | None = None) -> Settings:
    """
    Build and return a :class:`Settings` instance.

    Parameters
    ----------
    config_path:
        Path to the YAML configuration file.  Defaults to ``config.yaml``
        in the project root.

    Returns
    -------
    Settings
        A fully resolved settings object.
    """
    # 1. Load .env file – values here override existing shell env vars.
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path, override=True)
        logger.debug("Loaded .env from %s", dotenv_path)
    else:
        logger.debug("No .env file found; using environment as-is")

    # 2. Load YAML config.
    yaml_path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    yaml_data = _load_yaml(yaml_path)
    yaml_data = _expand_env_vars(yaml_data)

    # 3. Build Settings: YAML values provide the base; env vars overlay secrets.
    settings = Settings.model_validate(yaml_data)

    # 4. Configure logging based on LOG_LEVEL.
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    return settings
