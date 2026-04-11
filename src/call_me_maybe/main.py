"""
call-me-maybe  –  CLI entry point.

Usage examples::

    # Start the voice agent with default config.yaml
    call-me-maybe run

    # Use a custom config file
    call-me-maybe run --config /path/to/my-config.yaml

    # Send a single text message (useful for scripting / debugging)
    call-me-maybe chat "What is the capital of France?"

    # Check the current configuration
    call-me-maybe config
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.pretty import pprint
from rich.traceback import install as install_rich_traceback

from call_me_maybe.config import load_settings
from call_me_maybe.models.factory import create_backend
from call_me_maybe.agent.agent import VoiceAgent

install_rich_traceback(show_locals=False)
console = Console()

# Suppress HuggingFace Hub progress bars unless debug mode is active.
# Individual commands set this to "1" by default and clear it when --debug is passed.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

app = typer.Typer(
    name="call-me-maybe",
    help="A voice-based AI agent that listens, thinks, and speaks.",
    no_args_is_help=True,
)

# ---------------------------------------------------------------------------
# Common option
# ---------------------------------------------------------------------------

_config_option = typer.Option(
    None,
    "--config",
    "-c",
    help="Path to config.yaml. Defaults to config.yaml in the project root.",
    show_default=False,
)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def run(
    config: Optional[Path] = _config_option,
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging (includes audio RMS values)."),
) -> None:
    """Start the voice agent loop (listen → think → speak)."""
    if debug:
        os.environ["LOG_LEVEL"] = "DEBUG"
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"
    settings = _load(config)
    if debug:
        import logging as _logging
        _logging.getLogger().setLevel(_logging.DEBUG)
    try:
        settings.validate_provider()
    except (ValueError, ImportError) as exc:
        console.print(f"[bold red]Configuration error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    backend = create_backend(settings)
    agent = VoiceAgent(settings, backend)
    asyncio.run(agent.run())


@app.command()
def chat(
    message: str = typer.Argument(..., help="Text message to send to the agent."),
    config: Optional[Path] = _config_option,
    speak: bool = typer.Option(False, "--speak", "-s", help="Speak the reply via TTS."),
    debug: bool = typer.Option(False, "--debug", help="Enable verbose logging."),
) -> None:
    """Send a single text message to the agent and print the response."""
    if debug:
        os.environ["LOG_LEVEL"] = "DEBUG"
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"
    settings = _load(config)
    if debug:
        import logging as _logging
        _logging.getLogger().setLevel(_logging.DEBUG)
    try:
        settings.validate_provider()
    except (ValueError, ImportError) as exc:
        console.print(f"[bold red]Configuration error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    backend = create_backend(settings)
    agent = VoiceAgent(settings, backend)

    async def _run() -> str:
        return await agent.chat_text(message)

    reply = asyncio.run(_run())
    console.print(f"[bold magenta]Maybe:[/bold magenta] {reply}")
    if speak:
        asyncio.run(agent._speak(reply))


@app.command(name="config")
def show_config(
    config: Optional[Path] = _config_option,
) -> None:
    """Display the resolved configuration (secrets are redacted)."""
    settings = _load(config)
    data = settings.model_dump(exclude={"openrouter_api_key", "openai_api_key", "api_key", "fish_audio_api_key"})
    # Replace None with "<not set>" and non-None secrets with "<redacted>"
    pprint(data)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load(config_path: Optional[Path]):
    try:
        return load_settings(config_path)
    except Exception as exc:
        console.print(f"[bold red]Failed to load settings:[/bold red] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    app()
