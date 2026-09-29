"""Shared helpers for MCP tool server tests."""

import asyncio
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

from fastmcp import Client

TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools" / "mcp"


def load_server(name: str) -> ModuleType:
    """Import ``tools/mcp/<name>/<name>.py`` as a fresh, standalone module.

    Tool server directories are not packages, so they are loaded by path.
    """
    spec = importlib.util.spec_from_file_location(name, TOOLS_DIR / name / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def call_tool(mcp: Any, name: str, arguments: dict[str, Any] | None = None) -> Any:
    """Call a tool through FastMCP's in-memory client and return structured output."""

    async def run() -> Any:
        async with Client(mcp) as client:
            result = await client.call_tool(name, arguments or {})
            return result.structured_content

    return asyncio.run(run())


def list_tool_names(mcp: Any) -> set[str]:
    """List tool names through FastMCP's in-memory client."""

    async def run() -> set[str]:
        async with Client(mcp) as client:
            return {tool.name for tool in await client.list_tools()}

    return asyncio.run(run())
