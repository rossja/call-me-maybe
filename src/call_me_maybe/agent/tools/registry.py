"""
Tool registry that aggregates tools from MCP servers, skill modules, and
A2A agents, and exposes them as a unified list of :class:`ToolDefinition`
objects ready for the language model.
"""

from __future__ import annotations

import importlib
import json
import logging
from typing import TYPE_CHECKING, Any, Callable

import httpx

from call_me_maybe.models.base import ToolCall, ToolDefinition
from call_me_maybe.agent.tools.base import ToolResult

if TYPE_CHECKING:
    from call_me_maybe.config.settings import Settings

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Discovers and manages all available tools.

    Integrates:

    * **MCP** – Model Context Protocol servers (spawned as sub-processes).
    * **Skills** – Python modules that export a ``TOOLS`` list of callables.
    * **A2A** – Remote agents reachable over HTTP (Google A2A protocol).
    """

    def __init__(self, settings: "Settings") -> None:
        self._settings = settings
        self._definitions: list[ToolDefinition] = []
        self._executors: dict[str, Callable[..., Any]] = {}
        self._mcp_sessions: list[Any] = []

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
        for session in self._mcp_sessions:
            try:
                await session.__aexit__(None, None, None)
            except Exception:
                logger.debug("Error closing MCP session", exc_info=True)
        self._mcp_sessions.clear()

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
            from mcp import ClientSession, StdioServerParameters  # type: ignore[import]
            from mcp.client.stdio import stdio_client  # type: ignore[import]
        except ImportError:
            logger.warning(
                "mcp package not installed; MCP tools will not be available. "
                "Install it with: pip install mcp"
            )
            return

        for server_cfg in self._settings.agent.tools.mcp.servers:
            try:
                await self._connect_mcp_server(server_cfg, ClientSession, StdioServerParameters, stdio_client)
            except Exception as exc:
                logger.error(
                    "Failed to connect to MCP server '%s': %s",
                    server_cfg.name,
                    exc,
                    exc_info=True,
                )

    async def _connect_mcp_server(
        self,
        server_cfg: Any,
        ClientSession: Any,
        StdioServerParameters: Any,
        stdio_client: Any,
    ) -> None:
        """Connect to a single MCP server and register its tools."""
        params = StdioServerParameters(
            command=server_cfg.command[0],
            args=server_cfg.command[1:],
            env=server_cfg.env or None,
        )
        ctx = stdio_client(params)
        read, write = await ctx.__aenter__()
        session = ClientSession(read, write)
        await session.__aenter__()
        await session.initialize()
        self._mcp_sessions.append(session)

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
