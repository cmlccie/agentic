#!/usr/bin/env python3
"""Weather Agent using LangGraph workflow.

This module demonstrates how LangGraph uses state and graph logic to build
sophisticated agents that are as simple as possible and only as sophisticated
as necessary. The code is designed to be clear, linear, and readable for
developers learning workflow patterns.

Key Design Principles:
- Small, focused model calls (reduce token usage)
- Structured JSON outputs from model (not free-form text)
- Deterministic code for formatting (tables, prompts)
- Model only for intent extraction and natural language summaries
"""

import json
import logging
import os
from datetime import date, timedelta
from pathlib import Path
from textwrap import dedent
from typing import Annotated, cast

from fastmcp import Client as FastMCPClient
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, SecretStr
from tabulate import tabulate
from typing_extensions import TypedDict

from agentic.langchain.fastmcp import mcp_tools

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------------------------------

here = Path(__file__).parent.resolve()

required_env_vars = [
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
    "MODEL_NAME",
    "TOOLS_MCP_WEATHER_SERVER_URL",
]

for var in required_env_vars:
    if var not in os.environ:
        raise EnvironmentError(f"Environment variable {var} is not set.")

OPENAI_BASE_URL = os.environ["OPENAI_BASE_URL"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
MODEL_NAME = os.environ["MODEL_NAME"]
MCP_WEATHER_SERVER_URL = os.environ["TOOLS_MCP_WEATHER_SERVER_URL"]

system_prompt_extract_intent = (here / "system_prompt_extract_intent.md").read_text()


# -------------------------------------------------------------------------------------------------
# Data Models
# -------------------------------------------------------------------------------------------------


class Coordinates(BaseModel):
    """Geographic coordinates."""

    latitude: float
    longitude: float


class DateRange(BaseModel):
    """Date range for weather forecast."""

    start_date: str = Field(description="Start date in YYYY-MM-DD format")
    end_date: str = Field(description="End date in YYYY-MM-DD format")


class LocationOption(BaseModel):
    """A location option from the geocoding API."""

    id: int
    name: str
    admin1: str | None = None  # state/province
    admin2: str | None = None  # county/district
    admin3: str | None = None  # city/town
    admin4: str | None = None  # neighborhood/suburb
    country: str
    latitude: float
    longitude: float

    def display_name(self) -> str:
        """Format location for display with relevant admin levels."""
        parts = [self.name]
        # Add most relevant admin levels (avoid redundancy with name)
        for admin in [self.admin1, self.admin2]:
            if admin and admin != self.name:
                parts.append(admin)
                break  # Usually one admin level is enough for clarity
        parts.append(self.country)
        return ", ".join(parts)

    def full_display_name(self) -> str:
        """Format location with all available admin levels."""
        parts = [self.name]
        for admin in [self.admin4, self.admin3, self.admin2, self.admin1]:
            if admin and admin not in parts:
                parts.append(admin)
        parts.append(self.country)
        return ", ".join(parts)


class ExtractedIntent(BaseModel):
    """Structured output from intent extraction node."""

    coordinates: Coordinates | None = Field(
        default=None, description="Coordinates if location was found in cache"
    )
    date_range: DateRange | None = Field(
        default=None, description="Date range if dates were specified by user"
    )
    location_query: str | None = Field(
        default=None, description="Original location text if not resolved from cache"
    )
    location_name: str | None = Field(
        default=None, description="Resolved location name for display"
    )


class ResolvedLocation(BaseModel):
    """Structured output from location resolution node."""

    coordinates: Coordinates | None = Field(
        default=None, description="Coordinates if confidently matched"
    )
    location_name: str | None = Field(
        default=None, description="Resolved location name for display"
    )
    ambiguous: bool = Field(
        default=False, description="True if user must choose from options"
    )
    selected_index: int | None = Field(
        default=None, description="Index of selected location (0-based)"
    )


class WeatherAgentState(TypedDict):
    """State for the weather agent graph."""

    # Conversation
    messages: Annotated[list[BaseMessage], add_messages]

    # Extracted intent
    coordinates: Coordinates | None
    date_range: DateRange | None
    location_name: str | None

    # Location resolution
    location_query: str | None
    cached_locations: dict[int, LocationOption] | None
    possible_locations: dict[int, LocationOption] | None

    # Weather data
    weather_data: dict | None

    # Control flow
    needs_user_input: bool
    user_prompt: str | None


# -------------------------------------------------------------------------------------------------
# LLM Configuration
# -------------------------------------------------------------------------------------------------

model = ChatOpenAI(
    model=MODEL_NAME,
    base_url=OPENAI_BASE_URL,
    api_key=SecretStr(OPENAI_API_KEY),
    temperature=0.2,
)


# -------------------------------------------------------------------------------------------------
# Context Utilities
# -------------------------------------------------------------------------------------------------


def generate_date_context() -> str:
    """Generate date context for prompts.

    Weather forecasts are available for today and the next 15 days.
    """
    today = date.today()
    date_info = [
        (today + timedelta(days=i)).strftime("%A, %B %d, %Y (%Y-%m-%d)")
        for i in range(16)
    ]

    return dedent(f"""
        Today is {today.strftime("%A, %B %d, %Y (%Y-%m-%d)")}.
        Weather forecasts are available for the following dates:
        {"\n".join("- " + d for d in date_info)}
        """).strip()


def default_date_range() -> DateRange:
    """Return today's date as the default date range."""
    today = date.today().isoformat()
    return DateRange(start_date=today, end_date=today)


def generate_current_context(state: WeatherAgentState) -> str:
    """Generate context about the current state."""

    # Build context about current location (from previous turn)
    current_context = ""
    if current_location_name and current_coordinates:
        if isinstance(current_coordinates, dict):
            lat = current_coordinates["latitude"]
            lon = current_coordinates["longitude"]
        else:
            lat = current_coordinates.latitude
            lon = current_coordinates.longitude
        current_context = dedent(f"""
            Current location context (from previous request):
            - {current_location_name} (lat: {lat}, lon: {lon})
            """).strip()


# -------------------------------------------------------------------------------------------------
# Deterministic Formatting Functions
# -------------------------------------------------------------------------------------------------


def format_location_choices(locations: list[LocationOption]) -> str:
    """Format numbered list of locations for user selection."""
    lines = ["I found multiple locations matching your request. Please select one:\n"]
    for i, loc in enumerate(locations, 1):
        lines.append(f"  {i}. {loc.display_name()}")
    lines.append("\nReply with the number of your choice.")
    return "\n".join(lines)


def format_weather_forecast(location_name: str, weather_data: dict) -> str:
    """Generate markdown table from weather data using tabulate."""
    forecast_header = f"## Weather Forecast for {location_name}"

    daily = weather_data.get("daily", {})
    if not daily:
        return "No weather data available."

    # Build table data
    table_header = ["Date", "High", "Low", "Precipitation", "Humidity", "Cloud Cover"]
    table_rows = [
        [formatted_date, high, low, precipitation, humidity, cloud_cover]
        for date, data in daily.items()
        for formatted_date in [date.fromisoformat(date).strftime("%a, %b %d")]
        for high in [f"{data.get('temperature_2m_max', 'N/A')} °F"]
        for low in [f"{data.get('temperature_2m_min', 'N/A')} °F"]
        for precipitation in [f"{data.get('precipitation_sum', 'N/A')} in"]
        for humidity in [f"{data.get('relative_humidity_2m_mean', 'N/A')}%"]
        for cloud_cover in [f"{data.get('cloud_cover_mean', 'N/A')}%"]
    ]

    table = tabulate(table_rows, headers=table_header, tablefmt="github")
    return f"{forecast_header}\n\n{table}"


# -------------------------------------------------------------------------------------------------
# Graph Nodes
# -------------------------------------------------------------------------------------------------


async def fetch_cached_locations(state: WeatherAgentState) -> dict:
    """Fetch cached locations from MCP resource."""
    # Preserve existing cached locations from state
    cached_locations = state.get("cached_locations", {}) or {}

    try:
        async with FastMCPClient(MCP_WEATHER_SERVER_URL) as client:
            result = await client.read_resource("locations://cache")

        if result:
            content = result[0]
            if hasattr(content, "text"):
                location_data = json.loads(content.text)  # type: ignore[attr-defined]
            elif hasattr(content, "blob"):
                # Handle binary content by decoding to text
                location_data = json.loads(content.blob.decode("utf-8"))  # type: ignore[attr-defined]
            else:
                location_data = json.loads(str(content))

            retrieved_locations = {
                location.id: location
                for location_dict in location_data
                for location in [LocationOption.model_validate(location_dict)]
            }

            cached_locations.update(retrieved_locations)
            return {"cached_locations": cached_locations}

    except Exception as e:  # noqa: BLE001
        logger.warning(f"Failed to fetch cached locations: {e}")

    return {"cached_locations": cached_locations}


async def extract_intent(state: WeatherAgentState) -> dict:
    """Attempt to extract location and date range from user request."""
    messages = state["messages"]
    cached_locations = state.get("cached_locations", {}) or {}
    current_location_name = state.get("location_name")
    current_coordinates = state.get("coordinates")

    # Build context about cached locations
    cached_context = ""
    if cached_locations:
        cached_list = [
            f"- {loc.display_name()} (lat: {loc.latitude}, lon: {loc.longitude})"
            for loc in cached_locations.values()
        ]
        cached_context = dedent(f"""
            Known locations from previous lookups:
            {"\n".join(cached_list)}
            """).strip()

    system_prompt = "\n".join(
        [
            system_prompt_extract_intent,
            generate_date_context(),
        ]
    )

    response = cast(
        ExtractedIntent,
        await model.with_structured_output(ExtractedIntent)
        .with_retry(stop_after_attempt=3)
        .ainvoke([SystemMessage(content=system_prompt)] + messages),
    )

    return response.model_dump()


async def lookup_location(state: WeatherAgentState) -> dict:
    """Call get_locations MCP tool to find location options."""
    location_query = state.get("location_query", "")

    try:
        async with FastMCPClient(MCP_WEATHER_SERVER_URL) as client:
            lc_tools = await mcp_tools(client)
            get_locations_tool = next(
                (t for t in lc_tools if t.name == "get_locations"), None
            )
            if get_locations_tool is None:
                raise ValueError("get_locations tool not found")

            result_text = await get_locations_tool.ainvoke({"name": location_query})
        locations_response = json.loads(result_text)
        locations = [
            LocationOption.model_validate(loc) for loc in locations_response["result"]
        ]
        return {"location_options": locations}
    except Exception as e:  # noqa: BLE001
        logger.error(f"Failed to lookup location: {e}")
        return {"location_options": []}


async def resolve_location(state: WeatherAgentState) -> dict:
    """Model attempts to match location from options."""
    location_options = state.get("location_options", [])
    location_query = state.get("location_query", "")
    messages = state["messages"]

    if not location_options:
        # No locations found
        return {
            "needs_user_input": True,
            "user_prompt": f"I couldn't find any locations matching '{location_query}'. Please try a different location name.",
            "messages": [
                AIMessage(
                    content=f"I couldn't find any locations matching '{location_query}'. Please try a different location name."
                )
            ],
        }

    # Get original user message for context
    user_message = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break

    # Build options list for the model
    options_list = [
        f"{i}: {loc.display_name()} (lat: {loc.latitude}, lon: {loc.longitude})"
        for i, loc in enumerate(location_options)
    ]

    prompt = dedent(f"""
        The user requested weather for a location. Match their request to one of these options.

        User request: {user_message}
        Location query: {location_query}

        Available locations (index: details):
        {"\n".join(options_list)}

        Instructions:
        1. If the user's request clearly matches ONE location, return its coordinates, location_name, and selected_index.
        2. If the request is ambiguous (e.g., multiple cities with same name, no state specified), set ambiguous=true.
        3. Only set ambiguous=true if you cannot confidently determine which location the user wants.
        """).strip()

    response = cast(
        ResolvedLocation,
        await model.with_structured_output(ResolvedLocation)
        .with_retry(stop_after_attempt=3)
        .ainvoke([SystemMessage(content=prompt)]),
    )

    if response.ambiguous or response.coordinates is None:
        # Need user input
        prompt_text = format_location_choices(location_options)
        return {
            "needs_user_input": True,
            "user_prompt": prompt_text,
            "messages": [AIMessage(content=prompt_text)],
        }

    return {
        "coordinates": response.coordinates,
        "location_name": response.location_name,
        "needs_user_input": False,
    }


async def fetch_weather(state: WeatherAgentState) -> dict:
    """Call get_weather_forecast MCP tool."""
    coords = state["coordinates"]
    dates = state.get("date_range") or default_date_range()

    if coords is None:
        logger.error("No coordinates available for weather fetch")
        return {"weather_data": None}

    # Handle coordinates as either dict or Coordinates object
    if isinstance(coords, dict):
        latitude = coords["latitude"]
        longitude = coords["longitude"]
    else:
        latitude = coords.latitude
        longitude = coords.longitude

    # Handle dates as either dict or DateRange object
    if isinstance(dates, dict):
        start_date = dates["start_date"]
        end_date = dates["end_date"]
    else:
        start_date = dates.start_date
        end_date = dates.end_date

    try:
        async with FastMCPClient(MCP_WEATHER_SERVER_URL) as client:
            lc_tools = await mcp_tools(client)
            weather_tool = next(
                (t for t in lc_tools if t.name == "get_weather_forecast"), None
            )
            if weather_tool is None:
                raise ValueError("get_weather_forecast tool not found")
            result_text = await weather_tool.ainvoke(
                {
                    "latitude": latitude,
                    "longitude": longitude,
                    "start_date": start_date,
                    "end_date": end_date,
                },
            )
        weather_data = json.loads(result_text)
        return {"weather_data": weather_data}
    except Exception as e:  # noqa: BLE001
        logger.error(f"Failed to fetch weather: {e}")

    return {"weather_data": None}


async def format_response(state: WeatherAgentState) -> dict:
    """Generate final response with table and summary."""
    weather_data = state.get("weather_data")
    location_name = state.get("location_name")

    if not weather_data:
        error_msg = "I'm sorry, I couldn't retrieve the weather forecast. Please try again later."
        return {
            "messages": [AIMessage(content=error_msg)],
            "needs_user_input": False,
        }

    # Generate the weather table (deterministic)
    table = format_weather_forecast(weather_data, location_name)

    # Generate a natural language summary (model call)
    daily = weather_data.get("daily", {})
    summary_prompt = dedent(f"""
        Based on this weather data, write a brief 1-2 sentence summary highlighting key conditions.

        Location: {location_name or "Requested location"}
        Weather data: {json.dumps(daily, indent=2)}

        Be concise and focus on notable conditions (temperature trends, precipitation, etc.).
        """).strip()

    summary_response = await model.ainvoke([SystemMessage(content=summary_prompt)])
    summary = summary_response.content

    # Combine table and summary
    response = f"{table}\n\n{summary}"

    return {
        "messages": [AIMessage(content=response)],
        "needs_user_input": False,
    }


async def prompt_user(state: WeatherAgentState) -> dict:
    """Prepare prompt for user input (location disambiguation)."""
    # The message has already been set by resolve_location
    return {"needs_user_input": True}


# -------------------------------------------------------------------------------------------------
# Conditional Edges
# -------------------------------------------------------------------------------------------------


def route_after_extract(state: WeatherAgentState) -> str:
    """Route based on extracted intent."""
    if state.get("coordinates"):
        return "fetch_weather"
    return "lookup_location"


def route_after_resolve(state: WeatherAgentState) -> str:
    """Route based on location resolution."""
    if state.get("needs_user_input"):
        return END
    if state.get("coordinates"):
        return "fetch_weather"
    return END


# -------------------------------------------------------------------------------------------------
# Graph Construction
# -------------------------------------------------------------------------------------------------


def create_weather_agent():
    """Create and compile the weather agent graph."""
    graph = StateGraph(WeatherAgentState)

    # Add nodes
    graph.add_node("fetch_cached_locations", fetch_cached_locations)
    graph.add_node("extract_intent", extract_intent)
    graph.add_node("lookup_location", lookup_location)
    graph.add_node("resolve_location", resolve_location)
    graph.add_node("fetch_weather", fetch_weather)
    graph.add_node("format_response", format_response)

    # Set entry point
    graph.set_entry_point("fetch_cached_locations")

    # Add edges
    graph.add_edge("fetch_cached_locations", "extract_intent")
    graph.add_conditional_edges(
        "extract_intent",
        route_after_extract,
        {"fetch_weather": "fetch_weather", "lookup_location": "lookup_location"},
    )
    graph.add_edge("lookup_location", "resolve_location")
    graph.add_conditional_edges(
        "resolve_location",
        route_after_resolve,
        {"fetch_weather": "fetch_weather", END: END},
    )
    graph.add_edge("fetch_weather", "format_response")
    graph.add_edge("format_response", END)

    return graph.compile()


# Create the compiled graph
weather_agent = create_weather_agent()


# -------------------------------------------------------------------------------------------------
# Agent Skills (for A2A)
# -------------------------------------------------------------------------------------------------

agent_skills = [
    {
        "id": "get-weather-forecast",
        "name": "Get Weather Forecast",
        "description": "Get the weather forecast for the provided location.",
        "tags": ["weather", "forecast", "location"],
        "examples": [
            "What's the weather in Knoxville, TN?",
            "Forecast for San Francisco, CA.",
            "Tell me the weather in Tokyo, Japan for the next week.",
        ],
        "input_modes": ["text/plain"],
        "output_modes": ["text/plain"],
    }
]
