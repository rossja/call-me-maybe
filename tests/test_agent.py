"""
Tests for the VoiceAgent and ToolRegistry.

Uses mocks to avoid hardware/network dependencies.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from call_me_maybe.agent.agent import VoiceAgent, json_preview
from call_me_maybe.agent.tools.registry import ToolRegistry, _callable_to_definition
from call_me_maybe.agent.tools.base import ToolResult
from call_me_maybe.config.settings import Settings
from call_me_maybe.models.base import (
    ChatMessage,
    ModelResponse,
    ToolCall,
    ToolDefinition,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def settings() -> Settings:
    return Settings()


@pytest.fixture()
def mock_backend() -> MagicMock:
    backend = MagicMock()
    backend.transcribe.return_value = "hello"
    backend.chat.return_value = ModelResponse(text="Hi there!", tool_calls=[])
    backend.synthesize.return_value = b"audio"
    return backend


@pytest.fixture()
def agent(settings: Settings, mock_backend: MagicMock) -> VoiceAgent:
    return VoiceAgent(settings, mock_backend)


# ---------------------------------------------------------------------------
# Tests – json_preview
# ---------------------------------------------------------------------------


def test_json_preview_short() -> None:
    assert json_preview({"key": "value"}) == '{"key": "value"}'


def test_json_preview_truncated() -> None:
    long_dict = {"key": "x" * 200}
    result = json_preview(long_dict, max_len=20)
    assert result.endswith("…")
    assert len(result) == 21  # 20 chars + ellipsis


# ---------------------------------------------------------------------------
# Tests – VoiceAgent._trim_history
# ---------------------------------------------------------------------------


def test_trim_history_respects_system_message(agent: VoiceAgent) -> None:
    agent._history = [
        ChatMessage(role="system", content="You are helpful."),
    ] + [
        ChatMessage(role="user" if i % 2 == 0 else "assistant", content=f"msg {i}")
        for i in range(50)
    ]
    # Set 5 turns max (= 10 messages)
    agent._settings.llm.context_window_turns = 5
    agent._trim_history()

    # System message must always be present
    assert agent._history[0].role == "system"
    # Non-system messages should be <= 10
    non_sys = [m for m in agent._history if m.role != "system"]
    assert len(non_sys) <= 10


def test_trim_history_unlimited(agent: VoiceAgent) -> None:
    agent._history = [
        ChatMessage(role="user", content=f"msg {i}") for i in range(100)
    ]
    agent._settings.llm.context_window_turns = 0
    agent._trim_history()
    assert len(agent._history) == 100


# ---------------------------------------------------------------------------
# Tests – VoiceAgent.chat_text
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_text_simple(agent: VoiceAgent, mock_backend: MagicMock) -> None:
    mock_backend.chat.return_value = ModelResponse(text="Paris.", tool_calls=[])
    reply = await agent.chat_text("What is the capital of France?")
    assert reply == "Paris."
    # History should now contain system + user + assistant
    roles = [m.role for m in agent._history]
    assert "system" in roles
    assert "user" in roles
    assert "assistant" in roles


@pytest.mark.asyncio
async def test_chat_text_with_tool_call(agent: VoiceAgent, mock_backend: MagicMock) -> None:
    """Agent should execute tool calls and feed results back to the LLM."""
    tool_call = ToolCall(id="tc1", name="get_time", arguments={})

    # First call returns a tool call, second returns final text
    mock_backend.chat.side_effect = [
        ModelResponse(text="", tool_calls=[tool_call], finish_reason="tool_calls"),
        ModelResponse(text="The time is noon.", tool_calls=[]),
    ]

    # Register a dummy tool in the registry
    async def get_time() -> str:
        return "12:00"

    agent._tools._definitions = [
        ToolDefinition(name="get_time", description="Get current time", parameters={})
    ]
    agent._tools._executors = {"get_time": get_time}

    reply = await agent.chat_text("What time is it?")
    assert reply == "The time is noon."
    # LLM should have been called twice (once for tool call, once after result)
    assert mock_backend.chat.call_count == 2


@pytest.mark.asyncio
async def test_chat_text_tool_not_registered(agent: VoiceAgent, mock_backend: MagicMock) -> None:
    """Unknown tools should return an error result (not crash)."""
    tool_call = ToolCall(id="tc_bad", name="nonexistent_tool", arguments={})

    mock_backend.chat.side_effect = [
        ModelResponse(text="", tool_calls=[tool_call], finish_reason="tool_calls"),
        ModelResponse(text="Sorry, can't do that.", tool_calls=[]),
    ]

    reply = await agent.chat_text("do the thing")
    # Tool error should be handled gracefully
    assert reply == "Sorry, can't do that."


@pytest.mark.asyncio
async def test_chat_text_system_message_prepended_once(agent: VoiceAgent, mock_backend: MagicMock) -> None:
    """System message should only be prepended once even across multiple turns."""
    mock_backend.chat.return_value = ModelResponse(text="OK", tool_calls=[])

    await agent.chat_text("first message")
    await agent.chat_text("second message")

    system_messages = [m for m in agent._history if m.role == "system"]
    assert len(system_messages) == 1


# ---------------------------------------------------------------------------
# Tests – ToolRegistry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_registry_no_tools_enabled(settings: Settings) -> None:
    settings.agent.tools.mcp.enabled = False
    settings.agent.tools.skills.enabled = False
    settings.agent.tools.a2a.enabled = False

    registry = ToolRegistry(settings)
    await registry.initialise()

    assert registry.definitions == []
    await registry.close()


@pytest.mark.asyncio
async def test_registry_skill_tool_registration(settings: Settings) -> None:
    """Skills module tools should be discovered and registered."""

    def my_tool(x: int) -> str:
        """Multiply x by 2."""
        return str(x * 2)

    my_tool.SCHEMA = {
        "type": "object",
        "properties": {"x": {"type": "integer"}},
        "required": ["x"],
    }

    settings.agent.tools.skills.enabled = True
    settings.agent.tools.skills.modules = ["fake_module"]

    registry = ToolRegistry(settings)

    with patch("importlib.import_module") as mock_import:
        mock_mod = MagicMock()
        mock_mod.TOOLS = [my_tool]
        mock_import.return_value = mock_mod
        await registry.initialise()

    definitions = registry.definitions
    assert len(definitions) == 1
    assert definitions[0].name == "my_tool"
    assert "Multiply x by 2" in definitions[0].description
    await registry.close()


@pytest.mark.asyncio
async def test_registry_execute_sync_skill(settings: Settings) -> None:
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    registry = ToolRegistry(settings)
    registry._definitions = [
        ToolDefinition(name="add", description="Add two numbers", parameters={})
    ]
    registry._executors = {"add": add}

    tc = ToolCall(id="tc1", name="add", arguments={"a": 3, "b": 4})
    result = await registry.execute(tc)

    assert result.is_error is False
    assert result.content == "7"


@pytest.mark.asyncio
async def test_registry_execute_async_skill(settings: Settings) -> None:
    async def fetch(url: str) -> str:
        """Fetch a URL."""
        return f"content of {url}"

    registry = ToolRegistry(settings)
    registry._definitions = [
        ToolDefinition(name="fetch", description="Fetch URL", parameters={})
    ]
    registry._executors = {"fetch": fetch}

    tc = ToolCall(id="tc2", name="fetch", arguments={"url": "http://example.com"})
    result = await registry.execute(tc)

    assert result.is_error is False
    assert "content of http://example.com" in result.content


@pytest.mark.asyncio
async def test_registry_execute_missing_tool(settings: Settings) -> None:
    registry = ToolRegistry(settings)
    tc = ToolCall(id="tc3", name="nonexistent", arguments={})
    result = await registry.execute(tc)
    assert result.is_error is True
    assert "not registered" in result.content


@pytest.mark.asyncio
async def test_registry_execute_tool_raises(settings: Settings) -> None:
    def bad_tool() -> str:
        raise ValueError("something went wrong")

    registry = ToolRegistry(settings)
    registry._definitions = [
        ToolDefinition(name="bad_tool", description="Fails", parameters={})
    ]
    registry._executors = {"bad_tool": bad_tool}

    tc = ToolCall(id="tc4", name="bad_tool", arguments={})
    result = await registry.execute(tc)
    assert result.is_error is True
    assert "something went wrong" in result.content


# ---------------------------------------------------------------------------
# Tests – _callable_to_definition
# ---------------------------------------------------------------------------


def test_callable_to_definition_with_docstring() -> None:
    def greet(name: str) -> str:
        """Say hello to name."""
        return f"Hello {name}"

    defn = _callable_to_definition(greet)
    assert defn.name == "greet"
    assert defn.description == "Say hello to name."


def test_callable_to_definition_no_docstring() -> None:
    def no_doc():
        pass

    defn = _callable_to_definition(no_doc)
    assert defn.description == ""


def test_callable_to_definition_with_schema() -> None:
    def tool_with_schema(x: int) -> int:
        """Double x."""
        return x * 2

    tool_with_schema.SCHEMA = {"type": "object", "properties": {"x": {"type": "integer"}}}
    defn = _callable_to_definition(tool_with_schema)
    assert defn.parameters == tool_with_schema.SCHEMA
