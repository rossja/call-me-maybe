"""Tool integration sub-package."""

from call_me_maybe.agent.tools.registry import ToolRegistry
from call_me_maybe.agent.tools.base import ToolResult

__all__ = ["ToolRegistry", "ToolResult"]
