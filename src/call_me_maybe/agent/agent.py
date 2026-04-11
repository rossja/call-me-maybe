"""
VoiceAgent – the main agent loop.

Orchestrates the full voice pipeline:

1. Capture audio from the microphone.
2. Transcribe audio → text (STT).
3. Add to conversation history and call the LLM (with tool use).
4. Execute any tool calls; feed results back to the LLM.
5. Synthesise the assistant's reply as audio (TTS).
6. Play the audio through the speaker.
7. Repeat.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import TYPE_CHECKING

from rich.console import Console
from rich.markup import escape

from call_me_maybe.audio.capture import AudioCapture
from call_me_maybe.audio.playback import AudioPlayback
from call_me_maybe.models.base import ChatMessage, ModelResponse
from call_me_maybe.agent.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from call_me_maybe.config.settings import Settings
    from call_me_maybe.models.base import ModelBackend

logger = logging.getLogger(__name__)
console = Console()


class VoiceAgent:
    """
    The top-level voice agent that drives the conversation loop.

    Parameters
    ----------
    settings:
        Fully resolved settings.
    backend:
        The model backend to use for STT / LLM / TTS.
    """

    def __init__(self, settings: "Settings", backend: "ModelBackend") -> None:
        self._settings = settings
        self._backend = backend
        self._capture = AudioCapture.from_settings(settings)
        self._playback = AudioPlayback.from_settings(settings)
        self._tools = ToolRegistry(settings)
        self._history: list[ChatMessage] = []

    # ------------------------------------------------------------------
    # Public entry-points
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """
        Start the voice agent loop.

        Initialises tools, speaks the greeting, then enters the main
        listen → think → speak loop until the user says "goodbye" or
        sends a keyboard interrupt (Ctrl-C).
        """
        await self._tools.initialise()

        try:
            await self._speak(self._settings.agent.greeting)

            console.print(
                "\n[bold green]Voice agent started.[/bold green] "
                "Press [bold]Ctrl-C[/bold] to quit.\n"
            )

            while True:
                await self._turn()

        except KeyboardInterrupt:
            console.print("\n[yellow]Goodbye![/yellow]")
        finally:
            await self._tools.close()

    async def chat_text(self, user_text: str) -> str:
        """
        Process a single text turn without audio I/O.

        Useful for testing and scripted interactions.

        Returns the assistant's text response.
        """
        await self._tools.initialise()
        try:
            return await self._process_text(user_text)
        finally:
            await self._tools.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _turn(self) -> None:
        """Execute one complete listen → respond cycle."""
        console.print("[dim]Listening…[/dim]")
        audio_bytes = await asyncio.get_event_loop().run_in_executor(
            None, self._capture.record
        )

        if not audio_bytes:
            return

        console.print("[dim]Transcribing…[/dim]")
        user_text = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._backend.transcribe(
                audio_bytes, language=self._settings.stt.language
            ),
        )

        if not user_text.strip():
            return

        console.print(f"[bold cyan]You:[/bold cyan] {escape(user_text)}")
        reply = await self._process_text(user_text)
        console.print(f"[bold magenta]Maybe:[/bold magenta] {escape(reply)}\n")
        await self._speak(reply)
        # Brief pause after TTS playback so speaker echo decays before
        # the next record() opens the microphone.
        delay = self._settings.agent.post_tts_delay
        if delay > 0:
            await asyncio.sleep(delay)

    async def _process_text(self, user_text: str) -> str:
        """
        Run the LLM reasoning loop for a given user utterance.

        Handles multi-step tool calling: the LLM may request one or more tools
        before producing its final text reply.
        """
        # Prepend system prompt if history is empty
        if not self._history:
            tool_defs = self._tools.definitions
            if tool_defs:
                tool_names = ", ".join(t.name for t in tool_defs)
                tool_note = (
                    f"\n\nTools.\nYou have the following tools available: {tool_names}. "
                    "Only use tools that are listed here."
                )
            else:
                tool_note = (
                    "\n\nTools.\nYou do not currently have any tools available. "
                    "Answer from your own knowledge. "
                    "If the user asks for something you cannot do or do not know, say so briefly and conversationally. "
                    "Do not speculate about tools you might have."
                )
            self._history.append(
                ChatMessage(role="system", content=self._settings.llm.system_prompt + tool_note)
            )

        self._history.append(ChatMessage(role="user", content=user_text))

        tools = self._tools.definitions

        # Agentic loop – continue until the model produces a text reply
        for iteration in range(10):  # prevent infinite loops
            response: ModelResponse = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._backend.chat(self._history, tools=tools or None),
            )

            if response.tool_calls:
                # Record assistant's tool-call turn
                self._history.append(
                    ChatMessage(role="assistant", content=response.text or "")
                )
                # Execute each requested tool
                for tool_call in response.tool_calls:
                    console.print(
                        f"  [dim]→ tool call: {escape(tool_call.name)}({json_preview(tool_call.arguments)})[/dim]"
                    )
                    result = await self._tools.execute(tool_call)
                    self._history.append(
                        ChatMessage(
                            role="tool",
                            content=result.content,
                            tool_call_id=result.tool_call_id,
                            name=result.name,
                        )
                    )
                # Continue loop so the LLM can see tool results
                continue

            # Final text reply
            assistant_text = _strip_thinking(response.text.strip())
            if not assistant_text:
                logger.warning("Empty response after stripping thinking block; using fallback")
                assistant_text = "Sorry, I lost my train of thought. Can you say that again?"
            self._history.append(
                ChatMessage(role="assistant", content=assistant_text)
            )

            # Trim history to the configured context window
            self._trim_history()
            return assistant_text

        # If we exhaust iterations, return a fallback
        logger.warning("Agent loop exhausted max iterations without a text reply")
        return "I'm sorry, I wasn't able to complete that request."

    async def _speak(self, text: str) -> None:
        """Synthesise and play a text response."""
        if not text.strip():
            return
        audio_bytes = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._backend.synthesize(text),
        )
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._playback.play(audio_bytes),
        )

    def _trim_history(self) -> None:
        """Keep at most N conversation turns (system message always kept)."""
        max_turns = self._settings.llm.context_window_turns
        if max_turns <= 0:
            return

        system_messages = [m for m in self._history if m.role == "system"]
        non_system = [m for m in self._history if m.role != "system"]

        # Each turn = 1 user + 1 assistant message (at minimum)
        max_messages = max_turns * 2
        if len(non_system) > max_messages:
            non_system = non_system[-max_messages:]

        self._history = system_messages + non_system


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def json_preview(data: dict, max_len: int = 80) -> str:
    s = json.dumps(data)
    if len(s) > max_len:
        return s[:max_len] + "…"
    return s


_THINKING_RE = re.compile(
    r"<\|channel>.*?(?:<channel\|>|$)|<think>.*?(?:</think>|$)",
    re.DOTALL,
)


def _strip_thinking(text: str) -> str:
    """Remove reasoning/thought blocks from model output (model-agnostic fallback)."""
    return _THINKING_RE.sub("", text).strip()
