# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A local-first voice AI agent for Apple Silicon Macs. Records mic input, transcribes with Whisper, runs an LLM reasoning loop (with tool calls), then speaks the response. Supports both local MLX inference and remote OpenAI-compatible APIs.

## Commands

```bash
# Run the interactive voice agent loop
call-me-maybe run

# Single-turn text chat (no mic)
call-me-maybe chat "Your question here"

# Show resolved configuration
call-me-maybe config

# Install dev dependencies
uv sync --extra dev

# Run tests
pytest
pytest --cov
pytest tests/test_agent.py::TestVoiceAgent::test_process_text  # single test
```

## Architecture

### Voice Pipeline

```
AudioCapture.record()           # mic → WAV bytes (silence-based cutoff)
  → ModelBackend.transcribe()   # STT via Whisper
  → VoiceAgent._process_text()  # LLM agentic loop
      → if tool calls: ToolRegistry.execute() → feed results back, repeat
      → else: return assistant text
  → ModelBackend.synthesize()   # TTS
  → AudioPlayback.play()        # speaker
```

All pipeline steps are `async`; blocking I/O (recording, synthesis) runs via `run_in_executor` to avoid blocking the event loop.

### Backend Abstraction

`models/base.py` defines the `ModelBackend` abstract interface with three methods: `transcribe()`, `chat()`, `synthesize()`. `models/factory.py` instantiates the right backend based on `config.yaml`'s `provider` field:

- `"local"` → `LocalMLXBackend` (`models/local.py`): loads models via mlx-lm on Apple Silicon; handles Gemma 4's `<|tool_call>` native format; strips thinking blocks before passing text to TTS
- `"remote"` → `RemoteBackend` (`models/remote.py`): OpenAI SDK against any compatible endpoint (OpenRouter, OpenAI, etc.)

### Tool Registry

`agent/tools/registry.py` unifies three integration paths into a single `ToolRegistry`:
- **MCP**: stdio or HTTP Model Context Protocol servers configured under `agent.tools.mcp.servers`
  - HTTP servers default to `transport: streamable-http` (POST + SSE GET stream per MCP spec)
  - Use `transport: post` for servers that only support POST/JSON without SSE (e.g., AlphaVantage)
- **Skills**: Python modules that export `TOOLS = [ToolDefinition(...)]` — imported dynamically
- **A2A**: Remote agents reachable via HTTP (Google A2A protocol) under `agent.tools.a2a.agents`

All three expose tool definitions to the LLM and route execution through `ToolRegistry.execute(tool_name, args)`.

### Conversation State

`VoiceAgent` (`agent/agent.py`) maintains `_history: list[ChatMessage]`. The system prompt is always kept at index 0; history is auto-trimmed to `config.agent.context_window_turns` to stay within context limits. Tool descriptions are appended to the system prompt on first turn.

## Configuration

`config.yaml` at the project root is the single source of truth for non-secret settings. Load order: YAML → environment variables → `.env` file (later values win).

API key priority: `OPENROUTER_API_KEY` > `OPENAI_API_KEY` > `API_KEY`. Secrets (`FISH_AUDIO_API_KEY`, etc.) must never appear in `config.yaml` — use env vars or `.env`.

Top-level config sections: `provider`, `stt`, `llm`, `tts`, `local` (MLX), `remote` (API endpoint), `audio` (device IDs), `agent` (greeting, wake word, tool integrations).

See `config.yaml.example` for the full annotated schema.

## Gemma 4 Notes

Local Gemma 4 uses a `<|tool_call>call:{...}` format instead of the OpenAI tool-call schema — parsed specially in `local.py`. Thinking blocks (`<think>...</think>`) are stripped from Gemma 4 output before TTS synthesis to prevent the model's internal reasoning from being spoken aloud.
