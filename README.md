# call-me-maybe

A voice-based AI agent that listens to you, takes actions via tools, and talks
back.  Runs locally on Apple Silicon using **MLX** or calls out to any
**OpenAI-compatible remote API** (e.g. OpenRouter, OpenAI, LM Studio).

---

## Features

| Capability | Details |
|---|---|
| **Speech-to-Text** | Whisper (remote API or local mlx-whisper) |
| **Language Model** | MLX Llama 3.2 local default · any OpenRouter/OpenAI model |
| **Text-to-Speech** | OpenAI TTS · Fish Audio API · macOS `say` fallback |
| **Tool use** | MCP servers · Python skill modules · A2A agents |
| **Local inference** | Apple Silicon (M-series) with MLX, ≥ 24 GB RAM |
| **Remote inference** | OpenRouter · OpenAI · any OpenAI-compatible API |
| **Configuration** | Single `config.yaml`; secrets in env vars or `.env` |

---

## Quick start

### 1 – Clone and install

```bash
git clone https://github.com/rossja/call-me-maybe.git
cd call-me-maybe
uv sync
source .venv/bin/activate
```

### 2 – Configure

Copy the example env file and fill in your API keys:

```bash
cp .env.example .env
# then edit .env
```

Review (and optionally edit) `config.yaml` to choose models, providers, and
audio devices.

### 3 – Run

```bash
# Start the voice agent loop
call-me-maybe run

# Send a single text message (no mic needed – useful for testing)
call-me-maybe chat "What is the capital of France?"

# Show the resolved configuration
call-me-maybe config
```

---

## Configuration

All agent behaviour is controlled by **`config.yaml`**.  The file is heavily
commented – open it for full documentation.  Here is a summary of the top-level
sections:

| Section | Purpose |
|---|---|
| `provider` | `"remote"` (API) or `"local"` (MLX on Apple Silicon) |
| `stt` | Speech-to-text model and silence-detection settings |
| `llm` | Language model, system prompt, temperature, context window |
| `tts` | Text-to-speech model and voice settings |
| `local` | MLX backend settings (min RAM, quantisation, cache dir) |
| `remote` | Remote API base URL, timeout, extra headers |
| `audio` | Microphone and speaker device settings |
| `agent` | Greeting, wake-word, and tool integrations |

### Sensitive values (API keys)

Secrets are **never** stored in `config.yaml`.  They must be provided as
environment variables or in a **`.env`** file.  If a `.env` file is present,
its values **override** any environment variables already set in the shell.

| Env var | Purpose |
|---|---|
| `OPENROUTER_API_KEY` | OpenRouter API key (highest priority for remote) |
| `OPENAI_API_KEY` | OpenAI API key (fallback) |
| `API_KEY` | Generic key for any other OpenAI-compatible endpoint |
| `FISH_AUDIO_API_KEY` | Fish Audio TTS API key |
| `FISH_AUDIO_VOICE_ID` | Fish Audio voice reference ID |
| `LOG_LEVEL` | Logging verbosity (`debug` / `info` / `warning`) |

See `.env.example` for the full list.

---

## Environments

### Remote API

Any [OpenAI-compatible API](https://platform.openai.com/docs/api-reference) is
supported.  **OpenRouter** is the default because it provides access to hundreds
of models through a single endpoint.

```yaml
# config.yaml
provider: "remote"
remote:
  base_url: "https://openrouter.ai/api/v1"   # change to any compatible URL
llm:
  model: "openai/gpt-4o-mini"
stt:
  model: "whisper-1"
tts:
  model: "tts-1"
  voice: "nova"
```

```bash
# .env
OPENROUTER_API_KEY=your-key-here
```

### Local Apple Silicon (MLX, default)

Requires a MacBook with an M-series chip and **at least 24 GB** of unified
memory. Run `uv sync` on Apple Silicon to install the MLX dependencies.

```yaml
# config.yaml
provider: "local"
llm:
  model: "mlx-community/Llama-3.2-3B-Instruct-4bit"
stt:
  model: "mlx-community/whisper-large-v3-turbo"
tts:
  model: "tts-1"   # local backend uses macOS 'say' unless FISH_AUDIO_API_KEY is set
```

```bash
# Optionally set Fish Audio key for high-quality TTS
# (falls back to macOS 'say' command otherwise)
# FISH_AUDIO_API_KEY=your-key-here
```

### Dependency override: mlx-lm and Voxtral (mlx-audio)

`mlx-audio` (used for Voxtral TTS) hard-pins `mlx-lm==0.31.1`, but Gemma 4
support requires `mlx-lm>=0.31.2`.  The `pyproject.toml` includes a uv
[dependency override](https://docs.astral.sh/uv/concepts/dependencies/#dependency-overrides)
to resolve this conflict:

```toml
[tool.uv]
override-dependencies = [
    "mlx-lm>=0.31.2",
]
```

This is safe because the `0.31.1→0.31.2` bump only added the `gemma4` model
files; the API used by `mlx-audio` is unchanged.  If `mlx-audio` releases a
version that officially supports `mlx-lm>=0.31.2`, this override can be
removed.

---

## Tool integrations

The agent can call external tools before returning a response.  All tool
integrations are configured under `agent.tools` in `config.yaml`.

### MCP (Model Context Protocol)

Connect to any [MCP](https://modelcontextprotocol.io) server via stdio:

```yaml
agent:
  tools:
    mcp:
      enabled: true
      servers:
        - name: "filesystem"
          command: ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
        - name: "brave-search"
          command: ["npx", "-y", "@modelcontextprotocol/server-brave-search"]
          env:
            BRAVE_API_KEY: "${BRAVE_API_KEY}"   # expanded from environment
```

### Skills (Python modules)

Expose any Python function as a tool by adding it to a module's `TOOLS` list:

```python
# myproject/skills/calculator.py

def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

add.SCHEMA = {
    "type": "object",
    "properties": {
        "a": {"type": "number"},
        "b": {"type": "number"},
    },
    "required": ["a", "b"],
}

TOOLS = [add]
```

```yaml
agent:
  tools:
    skills:
      enabled: true
      modules:
        - "myproject.skills.calculator"
```

### A2A (Agent-to-Agent)

Delegate tasks to remote agents using the
[Google A2A protocol](https://google.github.io/A2A/):

```yaml
agent:
  tools:
    a2a:
      enabled: true
      agents:
        - name: "weather-agent"
          url: "http://localhost:8001"
          description: "Returns current weather for a location"
```

---

## Project layout

```
call-me-maybe/
├── config.yaml                      # Main configuration
├── .env.example                     # Environment variable template
├── pyproject.toml                   # Project metadata and dependencies
├── requirements.txt                 # Pinned runtime dependencies
├── requirements-dev.txt             # Dev / test dependencies
├── src/
│   └── call_me_maybe/
│       ├── main.py                  # CLI entry point (typer)
│       ├── config/
│       │   └── settings.py          # Config loading (YAML + .env + env vars)
│       ├── models/
│       │   ├── base.py              # Abstract ModelBackend interface
│       │   ├── remote.py            # Remote OpenAI-compatible backend
│       │   ├── local.py             # Local MLX backend (Apple Silicon)
│       │   └── factory.py           # Backend factory
│       ├── audio/
│       │   ├── capture.py           # Microphone recording
│       │   └── playback.py          # Speaker playback
│       └── agent/
│           ├── agent.py             # Main conversation loop
│           └── tools/
│               ├── base.py          # ToolResult dataclass
│               └── registry.py      # MCP + skills + A2A tool registry
└── tests/
    ├── test_config.py
    ├── test_models.py
    ├── test_audio.py
    └── test_agent.py
```

---

## Running tests

```bash
uv sync --extra dev
source .venv/bin/activate
pytest
# or with coverage
pytest --cov
```

---

## Hardware requirements

| Environment | Minimum |
|---|---|
| Remote API | Any machine with internet access and Python ≥ 3.11 |
| Local MLX | macOS, Apple Silicon (M-series), ≥ 24 GB unified memory |

---

## License

MIT
