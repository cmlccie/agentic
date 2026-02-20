"""LangGraph FastMCP integration helpers."""

from typing import Any

import anyio
from fastmcp import Client
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model

# -------------------------------------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------------------------------------


async def mcp_tools(client: Client, server_prefix: str = "") -> list[StructuredTool]:
    """Convert a connected FastMCP Client's tools into LangChain StructuredTools.

    Cancellation propagates cleanly (anyio CancelledError is never swallowed).
    MCP-level tool errors are returned as strings so the LLM can recover.

    Args:
        client: An already-connected FastMCP Client (inside `async with` block).
        server_prefix: Optional prefix for tool names, e.g. `"files"` →
            `"files__search"`. Useful when combining tools from multiple clients.

    Returns:
        A list of LangChain StructuredTools wrapping each MCP tool.
    """
    tools = await client.list_tools()
    return [_to_lc_tool(client, t, server_prefix) for t in tools]


# --------------------------------------------------------------------------------------
# Private Helper Functions
# --------------------------------------------------------------------------------------


def _to_lc_tool(client: Client, mcp_tool: Any, prefix: str) -> StructuredTool:
    """Wrap a single MCP tool as a LangChain StructuredTool.

    Args:
        client: The connected FastMCP Client used to invoke the tool.
        mcp_tool: The MCP tool descriptor returned by `client.list_tools()`.
        prefix: Server prefix to prepend to the tool name (empty string for none).

    Returns:
        A LangChain StructuredTool that delegates calls to the MCP server.
    """
    name = f"{prefix}__{mcp_tool.name}" if prefix else mcp_tool.name

    async def invoke(**kwargs: Any) -> str:
        try:
            result = await client.call_tool_mcp(mcp_tool.name, arguments=kwargs)
            if result.isError:
                return f"[tool error] {_text(result.content)}"
            return _text(result.content)
        except anyio.get_cancelled_exc_class():
            raise  # never swallow cancellation
        except Exception as exc:  # noqa: BLE001
            return f"[error] {exc}"

    return StructuredTool(
        name=name,
        description=mcp_tool.description or name,
        args_schema=_schema_to_model(name, mcp_tool.inputSchema or {}),
        coroutine=invoke,
        func=None,  # async-only  # type: ignore[arg-type]
    )


def _text(content: list[Any]) -> str:
    """Concatenate text content from an MCP tool result.

    Args:
        content: List of content objects returned by the MCP server.

    Returns:
        A single newline-joined string of all text values.
    """
    return "\n".join(getattr(c, "text", None) or str(c) for c in content)


def _schema_to_model(name: str, schema: dict[str, Any]) -> type[BaseModel]:
    """Build a Pydantic model from an MCP tool's JSON Schema input definition.

    Args:
        name: Tool name, used as the base for the generated model class name.
        schema: JSON Schema `object` dict from the MCP tool descriptor.

    Returns:
        A dynamically created Pydantic ``BaseModel`` subclass.
    """
    _TYPES: dict[str, type] = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    required = set(schema.get("required", []))
    fields: dict[str, Any] = {
        k: (
            _TYPES.get(v.get("type", "string"), Any),  # type: ignore[misc]
            Field(..., description=v.get("description"))
            if k in required
            else Field(None, description=v.get("description")),
        )
        for k, v in schema.get("properties", {}).items()
    }
    return create_model(f"{name}_args", **fields)  # type: ignore[call-overload]
