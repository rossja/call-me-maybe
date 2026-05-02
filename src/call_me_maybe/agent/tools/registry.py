"""
Tool registry that aggregates tools from MCP servers, skill modules, and
A2A agents, and exposes them as a unified list of :class:`ToolDefinition`
objects ready for the language model.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import logging
import os
from typing import TYPE_CHECKING, Any, Callable

import httpx

from call_me_maybe.models.base import ToolCall, ToolDefinition
from call_me_maybe.agent.tools.base import ToolResult

if TYPE_CHECKING:
    from call_me_maybe.config.settings import Settings

logger = logging.getLogger(__name__)


@contextlib.asynccontextmanager
async def _post_only_client(url: str):
    """Minimal MCP transport for POST-only servers (no SSE GET stream)."""
    import anyio  # type: ignore[import]
    from mcp.shared.message import SessionMessage  # type: ignore[import]
    from mcp.types import JSONRPCMessage  # type: ignore[import]

    read_stream_writer, read_stream = anyio.create_memory_object_stream(16)
    write_stream, write_stream_reader = anyio.create_memory_object_stream(16)

    async def _pump() -> None:
        async with httpx.AsyncClient(timeout=60) as http:
            async with write_stream_reader:
                async for session_message in write_stream_reader:
                    try:
                        payload = session_message.message.model_dump(
                            by_alias=True, mode="json", exclude_none=True
                        )
                        resp = await http.post(
                            url,
                            json=payload,
                            headers={
                                "Content-Type": "application/json",
                                "Accept": "application/json",
                            },
                        )
                        resp.raise_for_status()
                        if resp.content.strip():
                            data = resp.json()
                            items = data if isinstance(data, list) else [data]
                            for item in items:
                                msg = JSONRPCMessage.model_validate(item)
                                await read_stream_writer.send(SessionMessage(msg))
                    except Exception as exc:
                        logger.warning("MCP POST-only request failed: %s", exc)
                        await read_stream_writer.send(exc)
        await read_stream_writer.aclose()

    async with anyio.create_task_group() as tg:
        tg.start_soon(_pump)
        try:
            yield read_stream, write_stream
        finally:
            tg.cancel_scope.cancel()


class ToolRegistry:
    """
    Discovers and manages all available tools.

    Integrates:

    * **MCP** – Model Context Protocol servers (stdio subprocesses or remote HTTP).
    * **Skills** – Python modules that export a ``TOOLS`` list of callables.
    * **A2A** – Remote agents reachable over HTTP (Google A2A protocol).
    """

    def __init__(self, settings: "Settings") -> None:
        self._settings = settings
        self._definitions: list[ToolDefinition] = []
        self._executors: dict[str, Callable[..., Any]] = {}
        self._mcp_contexts: list[Any] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialise(self) -> None:
        """
        Discover and register all enabled tools.

        Must be awaited before calling :meth:`definitions` or :meth:`execute`.
        """
        tools_cfg = self._settings.agent.tools

        if tools_cfg.mcp.enabled and tools_cfg.mcp.servers:
            await self._load_mcp_tools()

        if tools_cfg.skills.enabled and tools_cfg.skills.modules:
            self._load_skill_tools()

        if tools_cfg.a2a.enabled and tools_cfg.a2a.agents:
            self._load_a2a_tools()

        logger.info(
            "ToolRegistry initialised: %d tools registered", len(self._definitions)
        )

    async def close(self) -> None:
        """Shut down any background MCP server processes."""
        for ctx in reversed(self._mcp_contexts):
            try:
                await ctx.__aexit__(None, None, None)
            except BaseException:
                logger.debug("Error closing MCP context", exc_info=True)
        self._mcp_contexts.clear()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def definitions(self) -> list[ToolDefinition]:
        """Return the list of registered tool definitions."""
        return list(self._definitions)

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """
        Execute a tool call and return the result.

        Parameters
        ----------
        tool_call:
            The :class:`~call_me_maybe.models.base.ToolCall` produced by the LLM.
        """
        executor = self._executors.get(tool_call.name)
        if executor is None:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content=f"Tool '{tool_call.name}' is not registered.",
                is_error=True,
            )

        try:
            result = executor(**tool_call.arguments)
            # Support both sync and async executors
            if hasattr(result, "__await__"):
                result = await result
            content = result if isinstance(result, str) else json.dumps(result)
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content=content,
            )
        except Exception as exc:
            logger.warning("Tool '%s' raised an exception: %s", tool_call.name, exc, exc_info=True)
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content=f"Error: {exc}",
                is_error=True,
            )

    # ------------------------------------------------------------------
    # MCP integration
    # ------------------------------------------------------------------

    async def _load_mcp_tools(self) -> None:
        """Connect to configured MCP servers and register their tools."""
        try:
            from mcp import ClientSession  # type: ignore[import]
        except ImportError:
            logger.warning(
                "mcp package not installed; MCP tools will not be available. "
                "Install it with: pip install mcp"
            )
            return

        for server_cfg in self._settings.agent.tools.mcp.servers:
            contexts_before = len(self._mcp_contexts)
            try:
                if server_cfg.url:
                    if getattr(server_cfg, "transport", "streamable-http") == "post":
                        await self._connect_mcp_post_server(server_cfg, ClientSession)
                    else:
                        await self._connect_mcp_remote_server(server_cfg, ClientSession)
                else:
                    await self._connect_mcp_stdio_server(server_cfg, ClientSession)
            except Exception as exc:
                logger.error(
                    "Failed to connect to MCP server '%s': %s",
                    server_cfg.name,
                    exc,
                )
                # Clean up any contexts that were partially entered for this server
                for ctx in reversed(self._mcp_contexts[contexts_before:]):
                    try:
                        await ctx.__aexit__(None, None, None)
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("Error cleaning up MCP context after failure: %s", exc, exc_info=True)
                del self._mcp_contexts[contexts_before:]

    async def _connect_mcp_stdio_server(self, server_cfg: Any, ClientSession: Any) -> None:
        """Connect to a stdio-based MCP server and register its tools."""
        import subprocess
        from mcp import StdioServerParameters  # type: ignore[import]
        from mcp.client.stdio import stdio_client  # type: ignore[import]

        params = StdioServerParameters(
            command=server_cfg.command[0],
            args=server_cfg.command[1:],
            env=server_cfg.env or None,
        )
        # Suppress subprocess stderr unless debug logging is active
        debug = os.environ.get("LOG_LEVEL", "warning").upper() == "DEBUG"
        errlog = None if debug else open(os.devnull, "w")
        ctx = stdio_client(params, errlog=errlog)
        read, write = await ctx.__aenter__()
        self._mcp_contexts.append(ctx)
        session = ClientSession(read, write)
        await session.__aenter__()
        self._mcp_contexts.append(session)
        await session.initialize()
        await self._register_mcp_tools(server_cfg, session)

    async def _connect_mcp_remote_server(self, server_cfg: Any, ClientSession: Any) -> None:
        """Connect to a remote HTTP MCP server and register its tools."""
        from mcp.client.streamable_http import streamablehttp_client  # type: ignore[import]

        ctx = streamablehttp_client(server_cfg.url)
        streams = await ctx.__aenter__()
        self._mcp_contexts.append(ctx)
        read, write = streams[0], streams[1]
        session = ClientSession(read, write)
        await session.__aenter__()
        self._mcp_contexts.append(session)
        await session.initialize()
        await self._register_mcp_tools(server_cfg, session)

    async def _connect_mcp_post_server(self, server_cfg: Any, ClientSession: Any) -> None:
        """Connect to a POST-only HTTP MCP server (no SSE GET stream)."""
        ctx = _post_only_client(server_cfg.url)
        read, write = await ctx.__aenter__()
        self._mcp_contexts.append(ctx)
        session = ClientSession(read, write)
        await session.__aenter__()
        self._mcp_contexts.append(session)
        await session.initialize()
        await self._register_mcp_tools(server_cfg, session)

    async def _register_mcp_tools(self, server_cfg: Any, session: Any) -> None:
        """Discover tools from an initialized MCP session and register them."""
        tools_response = await session.list_tools()
        for tool in tools_response.tools:
            definition = ToolDefinition(
                name=f"{server_cfg.name}__{tool.name}",
                description=tool.description or "",
                parameters=tool.inputSchema or {},
            )
            self._definitions.append(definition)

            # Build a closure to capture the correct session / tool name
            async def _call_mcp(session=session, mcp_tool_name=tool.name, **kwargs: Any) -> str:
                result = await session.call_tool(mcp_tool_name, arguments=kwargs)
                return str(result.content)

            self._executors[definition.name] = _call_mcp
            logger.debug("Registered MCP tool: %s", definition.name)

    # ------------------------------------------------------------------
    # Skills integration
    # ------------------------------------------------------------------

    def _load_skill_tools(self) -> None:
        """Import skill modules and register the tools they export."""
        for module_path in self._settings.agent.tools.skills.modules:
            try:
                mod = importlib.import_module(module_path)
            except ImportError as exc:
                logger.error("Could not import skill module '%s': %s", module_path, exc)
                continue

            tools: list[Any] = getattr(mod, "TOOLS", [])
            if not tools:
                logger.warning(
                    "Skill module '%s' has no 'TOOLS' list; skipping.", module_path
                )
                continue

            for tool_fn in tools:
                definition = _callable_to_definition(tool_fn)
                self._definitions.append(definition)
                self._executors[definition.name] = tool_fn
                logger.debug("Registered skill tool: %s", definition.name)

    # ------------------------------------------------------------------
    # A2A integration
    # ------------------------------------------------------------------

    def _load_a2a_tools(self) -> None:
        """Register remote A2A agents as callable tools."""
        for agent_cfg in self._settings.agent.tools.a2a.agents:
            definition = ToolDefinition(
                name=agent_cfg.name,
                description=agent_cfg.description or f"Delegate to {agent_cfg.name}",
                parameters={
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "The message to send to the agent",
                        }
                    },
                    "required": ["message"],
                },
            )
            self._definitions.append(definition)

            agent_url = agent_cfg.url

            async def _call_a2a(message: str, url: str = agent_url) -> str:
                async with httpx.AsyncClient(timeout=60) as client:
                    payload = {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "tasks/send",
                        "params": {
                            "message": {
                                "role": "user",
                                "parts": [{"type": "text", "text": message}],
                            }
                        },
                    }
                    response = await client.post(url, json=payload)
                    response.raise_for_status()
                    data = response.json()
                    # Extract text from A2A response
                    result = data.get("result", {})
                    parts = (
                        result.get("status", {})
                        .get("message", {})
                        .get("parts", [])
                    )
                    texts = [p.get("text", "") for p in parts if p.get("type") == "text"]
                    return " ".join(texts) or json.dumps(result)

            self._executors[definition.name] = _call_a2a
            logger.debug("Registered A2A agent: %s → %s", definition.name, agent_url)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _callable_to_definition(fn: Callable) -> ToolDefinition:
    """
    Build a :class:`ToolDefinition` from a Python callable.

    The function should have a docstring (used as ``description``) and
    optionally a ``SCHEMA`` attribute containing the JSON-Schema ``parameters``
    dict.  If ``SCHEMA`` is absent, a permissive schema accepting any kwargs
    is used.
    """
    name = fn.__name__
    description = (fn.__doc__ or "").strip().splitlines()[0] if fn.__doc__ else ""
    parameters: dict[str, Any] = getattr(fn, "SCHEMA", {"type": "object", "properties": {}})
    return ToolDefinition(name=name, description=description, parameters=parameters)
