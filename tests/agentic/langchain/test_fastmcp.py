"""Unit tests for src/agentic/langchain/fastmcp.py.

All FastMCP Client calls are mocked so no real MCP server is needed.
"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ValidationError

from agentic.langchain.fastmcp import _schema_to_model, _text, _to_lc_tool, mcp_tools

# --------------------------------------------------------------------------------------
# Helpers / Factories
# --------------------------------------------------------------------------------------


def _make_mcp_tool(
    name: str = "my_tool",
    description: str | None = "A test tool",
    input_schema: dict[str, Any] | None = None,
) -> MagicMock:
    """Create a mock MCP tool descriptor."""
    tool = MagicMock()
    tool.name = name
    tool.description = description
    tool.inputSchema = input_schema or {}
    return tool


def _make_call_result(text: str = "ok", is_error: bool = False) -> MagicMock:
    """Create a mock MCP call result."""
    content_item = MagicMock()
    content_item.text = text
    result = MagicMock()
    result.isError = is_error
    result.content = [content_item]
    return result


def _make_client(tools: list[MagicMock] | None = None) -> MagicMock:
    """Create a mock FastMCP Client."""
    client = MagicMock()
    client.list_tools = AsyncMock(return_value=tools or [])
    client.call_tool_mcp = AsyncMock(return_value=_make_call_result())
    return client


# --------------------------------------------------------------------------------------
# _text
# --------------------------------------------------------------------------------------


def test_text_returns_joined_text_attributes():
    """_text concatenates the .text attribute of each content item."""
    items = [MagicMock(text="hello"), MagicMock(text="world")]
    assert _text(items) == "hello\nworld"


def test_text_falls_back_to_str_when_no_text_attribute():
    """_text uses str() when a content item has no .text attribute."""

    class NoText:
        def __str__(self) -> str:
            return "fallback"

    items = [NoText()]
    assert _text(items) == "fallback"  # type: ignore[arg-type]


def test_text_empty_list():
    """_text on an empty list returns an empty string."""
    assert _text([]) == ""


def test_text_none_text_attribute_falls_back_to_str():
    """_text treats a falsy .text value as absent and uses str()."""
    item = MagicMock()
    item.text = None
    # getattr returns None (falsy) → falls back to str(item)
    result = _text([item])
    assert result == str(item)


# --------------------------------------------------------------------------------------
# _schema_to_model
# --------------------------------------------------------------------------------------


def test_schema_to_model_empty_schema_returns_model():
    """An empty schema produces a BaseModel subclass with no fields."""
    Model = _schema_to_model("empty_tool", {})
    assert issubclass(Model, BaseModel)
    assert Model.model_fields == {}


def test_schema_to_model_required_field_has_no_default():
    """Required fields raise ValidationError when omitted."""
    schema = {
        "properties": {"city": {"type": "string", "description": "City name"}},
        "required": ["city"],
    }
    Model = _schema_to_model("weather_tool", schema)
    with pytest.raises(ValidationError):
        Model()  # missing required field


def test_schema_to_model_optional_field_defaults_to_none():
    """Optional (non-required) fields default to None."""
    schema = {
        "properties": {"units": {"type": "string", "description": "Units"}},
        "required": [],
    }
    Model = _schema_to_model("weather_tool", schema)
    instance = Model()
    assert instance.units is None  # type: ignore[attr-defined]


def test_schema_to_model_type_mapping():
    """Each JSON Schema type maps to the correct Python type annotation."""
    schema = {
        "properties": {
            "s": {"type": "string"},
            "i": {"type": "integer"},
            "n": {"type": "number"},
            "b": {"type": "boolean"},
            "a": {"type": "array"},
            "o": {"type": "object"},
        },
        "required": ["s", "i", "n", "b", "a", "o"],
    }
    Model = _schema_to_model("types_tool", schema)
    annotations = Model.__annotations__
    assert annotations["s"] is str
    assert annotations["i"] is int
    assert annotations["n"] is float
    assert annotations["b"] is bool
    assert annotations["a"] is list
    assert annotations["o"] is dict


def test_schema_to_model_unknown_type_uses_any():
    """An unrecognized JSON Schema type falls back to Any."""
    schema = {
        "properties": {"x": {"type": "custom_type"}},
        "required": ["x"],
    }
    Model = _schema_to_model("custom_tool", schema)
    # Should not raise; field exists
    assert "x" in Model.__annotations__


def test_schema_to_model_class_name_includes_tool_name():
    """The generated model class name is derived from the tool name."""
    Model = _schema_to_model("search_docs", {})
    assert "search_docs" in Model.__name__


# --------------------------------------------------------------------------------------
# _to_lc_tool / invoke
# --------------------------------------------------------------------------------------


def test_to_lc_tool_returns_structured_tool():
    """_to_lc_tool wraps an MCP tool as a LangChain StructuredTool."""
    client = _make_client()
    mcp_tool = _make_mcp_tool(name="ping", description="Ping the server")
    lc_tool = _to_lc_tool(client, mcp_tool, prefix="")
    assert isinstance(lc_tool, StructuredTool)
    assert lc_tool.name == "ping"
    assert lc_tool.description == "Ping the server"


def test_to_lc_tool_applies_prefix():
    """When a prefix is given, the tool name is '<prefix>__<name>'."""
    client = _make_client()
    mcp_tool = _make_mcp_tool(name="search")
    lc_tool = _to_lc_tool(client, mcp_tool, prefix="files")
    assert lc_tool.name == "files__search"


def test_to_lc_tool_empty_description_falls_back_to_name():
    """An empty/None MCP description falls back to the tool name."""
    client = _make_client()
    mcp_tool = _make_mcp_tool(name="noop", description=None)
    lc_tool = _to_lc_tool(client, mcp_tool, prefix="")
    assert lc_tool.description == "noop"


async def test_invoke_returns_text_on_success():
    """invoke returns the text content when the MCP call succeeds."""
    client = _make_client()
    client.call_tool_mcp = AsyncMock(return_value=_make_call_result("42 degrees"))
    mcp_tool = _make_mcp_tool(name="get_temp")
    lc_tool = _to_lc_tool(client, mcp_tool, prefix="")

    result = await lc_tool.coroutine()  # type: ignore[misc]
    assert result == "42 degrees"


async def test_invoke_returns_error_string_on_tool_error():
    """invoke returns a '[tool error]' string when isError is True."""
    client = _make_client()
    client.call_tool_mcp = AsyncMock(
        return_value=_make_call_result("something went wrong", is_error=True)
    )
    mcp_tool = _make_mcp_tool(name="fail_tool")
    lc_tool = _to_lc_tool(client, mcp_tool, prefix="")

    result = await lc_tool.coroutine()  # type: ignore[misc]
    assert result.startswith("[tool error]")
    assert "something went wrong" in result


async def test_invoke_returns_error_string_on_exception():
    """invoke returns an '[error]' string when an unexpected exception is raised."""
    client = _make_client()
    client.call_tool_mcp = AsyncMock(side_effect=RuntimeError("network failure"))
    mcp_tool = _make_mcp_tool(name="flaky_tool")
    lc_tool = _to_lc_tool(client, mcp_tool, prefix="")

    result = await lc_tool.coroutine()  # type: ignore[misc]
    assert result.startswith("[error]")
    assert "network failure" in result


async def test_invoke_propagates_cancellation():
    """invoke re-raises CancelledError and does not swallow it."""
    client = _make_client()
    client.call_tool_mcp = AsyncMock(side_effect=asyncio.CancelledError())
    mcp_tool = _make_mcp_tool(name="slow_tool")

    with patch(
        "agentic.langchain.fastmcp.anyio.get_cancelled_exc_class",
        return_value=asyncio.CancelledError,
    ):
        lc_tool = _to_lc_tool(client, mcp_tool, prefix="")
        with pytest.raises(asyncio.CancelledError):
            await lc_tool.coroutine()  # type: ignore[misc]


async def test_invoke_passes_kwargs_to_call_tool_mcp():
    """invoke forwards keyword arguments to client.call_tool_mcp as 'arguments'."""
    client = _make_client()
    mcp_tool = _make_mcp_tool(
        name="forecast",
        input_schema={
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    )
    lc_tool = _to_lc_tool(client, mcp_tool, prefix="")

    await lc_tool.coroutine(city="London")  # type: ignore[misc]
    client.call_tool_mcp.assert_awaited_once_with(
        "forecast", arguments={"city": "London"}
    )


# --------------------------------------------------------------------------------------
# mcp_tools
# --------------------------------------------------------------------------------------


async def test_mcp_tools_returns_structured_tools_for_each_mcp_tool():
    """mcp_tools returns one StructuredTool per tool reported by the server."""
    client = _make_client(tools=[_make_mcp_tool("tool_a"), _make_mcp_tool("tool_b")])
    tools = await mcp_tools(client)
    assert len(tools) == 2
    assert all(isinstance(t, StructuredTool) for t in tools)
    names = {t.name for t in tools}
    assert names == {"tool_a", "tool_b"}


async def test_mcp_tools_applies_server_prefix():
    """mcp_tools prefixes tool names with '<server_prefix>__'."""
    client = _make_client(tools=[_make_mcp_tool("search")])
    tools = await mcp_tools(client, server_prefix="docs")
    assert tools[0].name == "docs__search"


async def test_mcp_tools_no_prefix_by_default():
    """mcp_tools leaves tool names unprefixed when no server_prefix is given."""
    client = _make_client(tools=[_make_mcp_tool("ping")])
    tools = await mcp_tools(client)
    assert tools[0].name == "ping"


async def test_mcp_tools_empty_server_returns_empty_list():
    """mcp_tools returns an empty list when the server exposes no tools."""
    client = _make_client(tools=[])
    tools = await mcp_tools(client)
    assert tools == []
