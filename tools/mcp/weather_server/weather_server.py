#!/usr/bin/env python3
"""MCP Weather Server."""

import logging
import os
from collections import OrderedDict
from datetime import datetime
from typing import Annotated, Any, Dict, List, Literal, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
import typer
from fastmcp import FastMCP
from pydantic import BaseModel, Field

import agentic.logging

logger = logging.getLogger("weather_server")


HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))

OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"

HTTP_TIMEOUT = httpx.Timeout(float(os.environ.get("HTTP_TIMEOUT_S", "10")))

LOCATION_CACHE_SIZE = 256


# -------------------------------------------------------------------------------------------------
# MCP Weather Server
# -------------------------------------------------------------------------------------------------


mcp = FastMCP("MCP Weather Server")


# --------------------------------------------------------------------------------------
# HTTP Helper
# --------------------------------------------------------------------------------------


def _http_client(
    transport: httpx.AsyncBaseTransport | None = None,
) -> httpx.AsyncClient:
    """Create the HTTP client used for Open-Meteo requests.

    Args:
        transport: Optional transport override (used by tests to mock HTTP).
    """
    return httpx.AsyncClient(timeout=HTTP_TIMEOUT, transport=transport)


async def _get_json(url: str, params: dict[str, Any]) -> Any:
    """GET ``url`` with ``params`` and return the decoded JSON body."""
    async with _http_client() as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        return response.json()


# --------------------------------------------------------------------------------------
# Weather API Types
# --------------------------------------------------------------------------------------

WeatherVariables = Literal[
    "cloud_cover_max",
    "cloud_cover_mean",
    "cloud_cover_min",
    "precipitation_hours",
    "precipitation_probability_max",
    "precipitation_sum",
    "rain_sum",
    "relative_humidity_2m_max",
    "relative_humidity_2m_mean",
    "relative_humidity_2m_min",
    "showers_sum",
    "snowfall_sum",
    "sunrise",
    "sunset",
    "temperature_2m_max",
    "temperature_2m_min",
    "wind_gusts_10m_max",
    "wind_gusts_10m_min",
    "wind_speed_10m_max",
    "wind_speed_10m_min",
]

DEFAULT_WEATHER_VARIABLES: tuple[WeatherVariables, ...] = (
    "cloud_cover_mean",
    "precipitation_probability_max",
    "precipitation_sum",
    "relative_humidity_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
)

PrecipitationUnit = Literal["mm", "inch"]

TemperatureUnit = Literal["celsius", "fahrenheit"]

TimeFormat = Literal["iso8601", "unixtime"]

WindSpeedUnit = Literal["kmh", "mph", "ms", "kn"]


# --------------------------------------------------------------------------------------
#  Tools
# --------------------------------------------------------------------------------------


class WeatherForecast(BaseModel):
    """Weather forecast."""

    latitude: float = Field(..., description="Coordinate latitude in degrees.")
    longitude: float = Field(..., description="Coordinate longitude in degrees.")
    elevation: Optional[float] = Field(None, description="Elevation in meters.")
    timezone: Optional[str] = Field(
        None, description="Timezone (e.g. 'America/New_York')."
    )
    timezone_abbreviation: Optional[str] = Field(
        None, description="Timezone abbreviation (e.g. 'GMT-4')."
    )
    daily_units: Dict[str, str] = Field(
        ..., description="Units for daily weather variables."
    )
    daily: Dict[str, Dict[str, Any]] = Field(
        ..., description="Daily weather variables keyed by date."
    )


def _today(timezone: str) -> str:
    """Return today's ISO date in ``timezone`` (local time for "auto" or unknown)."""
    try:
        tz = None if timezone == "auto" else ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError):
        tz = None
    return datetime.now(tz).date().isoformat()


def _daily_by_date(
    daily_data: Dict[str, List[Any]], variables: List[str]
) -> Dict[str, Dict[str, Any]]:
    """Pivot Open-Meteo's column-oriented daily data into {date: {variable: value}}."""
    return {
        date: {
            variable: daily_data[variable][i]
            for variable in variables
            if i < len(daily_data.get(variable, []))
        }
        for i, date in enumerate(daily_data.get("time", []))
    }


@mcp.tool()
@agentic.logging.log_call(logger)
async def get_weather_forecast(
    latitude: float,
    longitude: float,
    timezone: str = "auto",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    weather_variables: Optional[List[WeatherVariables]] = None,
    time_format: TimeFormat = "iso8601",
    temperature_unit: TemperatureUnit = "fahrenheit",
    precipitation_unit: PrecipitationUnit = "inch",
    wind_speed_unit: WindSpeedUnit = "mph",
) -> WeatherForecast:
    """Get the daily weather forecast for the provided coordinates.

    When weather_variables is omitted, the forecast includes: cloud_cover_mean,
    precipitation_probability_max, precipitation_sum, relative_humidity_2m_mean,
    temperature_2m_max, and temperature_2m_min. Request sunrise, sunset, rain_sum,
    showers_sum, snowfall_sum, wind, and other variables explicitly. The units of
    every returned variable are listed in daily_units.

    Args:
        latitude: Coordinate latitude in degrees.
        longitude: Coordinate longitude in degrees.
        timezone: IANA timezone for dates and times (e.g. 'America/New_York');
            "auto" (default) uses the timezone of the coordinates.
        start_date: First forecast date as YYYY-MM-DD (default: today).
        end_date: Last forecast date as YYYY-MM-DD (default: today).
        weather_variables: Daily weather variables to include (default: the set
            listed above).
        time_format: "iso8601" (default) or "unixtime" for dates and times.
        temperature_unit: "fahrenheit" (default) or "celsius".
        precipitation_unit: "inch" (default) or "mm".
        wind_speed_unit: "mph" (default), "kmh", "ms", or "kn".

    Returns:
        WeatherForecast: Location metadata, daily_units, and daily values keyed by
        date.
    """
    today = _today(timezone)
    variables = sorted(set(weather_variables or DEFAULT_WEATHER_VARIABLES))

    data = await _get_json(
        OPEN_METEO_FORECAST_URL,
        {
            "latitude": latitude,
            "longitude": longitude,
            "timezone": timezone,
            "start_date": start_date or today,
            "end_date": end_date or today,
            "daily": ",".join(variables),
            "timeformat": time_format,
            "temperature_unit": temperature_unit,
            "precipitation_unit": precipitation_unit,
            "wind_speed_unit": wind_speed_unit,
        },
    )

    return WeatherForecast(
        latitude=data.get("latitude", latitude),
        longitude=data.get("longitude", longitude),
        elevation=data.get("elevation"),
        timezone=data.get("timezone"),
        timezone_abbreviation=data.get("timezone_abbreviation"),
        daily_units=data.get("daily_units", {}),
        daily=_daily_by_date(data.get("daily", {}), variables),
    )


# --------------------------------------------------------------------------------------
# Location Information Tool
# --------------------------------------------------------------------------------------


class LocationInfo(BaseModel):
    """Location information."""

    id: int = Field(..., description="Unique identifier for the location.")
    name: str = Field(..., description="Name of the location.")
    latitude: float = Field(..., description="Latitude of the location.")
    longitude: float = Field(..., description="Longitude of the location.")
    elevation: Optional[float] = Field(
        None, description="Elevation of the location in meters."
    )

    timezone: str = Field(..., description="Timezone of the location.")

    country: str = Field(..., description="Country of the location.")
    country_code: str = Field(
        ...,
        description="ISO-3166-1 alpha2 country code of the location (e.g. 'DE' for Germany).",
    )
    admin1: Optional[str] = Field(
        None, description="Administrative region level 1 (e.g. state or province)."
    )
    admin2: Optional[str] = Field(
        None, description="Administrative region level 2 (e.g. county or district)."
    )
    admin3: Optional[str] = Field(
        None, description="Administrative region level 3 (e.g. city or town)."
    )
    admin4: Optional[str] = Field(
        None,
        description="Administrative region level 4 (e.g. neighborhood or suburb).",
    )
    postcodes: Optional[List[str]] = Field(
        None, description="List of postcodes associated with the location."
    )

    population: Optional[int] = Field(None, description="Population of the location.")


# Most-recently-seen locations, bounded to LOCATION_CACHE_SIZE entries.
location_cache: OrderedDict[int, LocationInfo] = OrderedDict()


def _remember_locations(locations: List[LocationInfo]) -> None:
    """Add locations to the cache, evicting the least recently seen entries."""
    for location in locations:
        location_cache[location.id] = location
        location_cache.move_to_end(location.id)
    while len(location_cache) > LOCATION_CACHE_SIZE:
        location_cache.popitem(last=False)


@mcp.tool()
@agentic.logging.log_call(logger)
async def get_locations(
    name: str, country_code: Optional[str] = None, count: int = 10
) -> List[LocationInfo]:
    """Search for locations by name to get their coordinates and timezone.

    Args:
        name: Name of the location to search for (e.g. a city name).
        country_code: Optional ISO-3166-1 alpha2 country code to narrow down the
            search (e.g. 'US' for the United States).
        count: Maximum number of matching locations to return (default: 10).

    Returns:
        List[LocationInfo]: Locations matching the search criteria.
    """
    params: dict[str, Any] = {
        "name": name,
        "count": count,
        "language": "en",
        "format": "json",
    }
    if country_code is not None:
        params["countryCode"] = country_code

    data = await _get_json(OPEN_METEO_GEOCODING_URL, params)
    locations = [LocationInfo.model_validate(r) for r in data.get("results", [])]
    _remember_locations(locations)
    return locations


# --------------------------------------------------------------------------------------
# Resource
# --------------------------------------------------------------------------------------


@mcp.resource("locations://cache")
def locations_cache() -> List[LocationInfo]:
    """Cached location information.

    This resource provides access to the most recently looked-up locations,
    including their latitude and longitude coordinates, elevation, timezone, and
    administrative districts.

    Returns:
        List[LocationInfo]: A list of cached locations.
    """
    return list(location_cache.values())


# --------------------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------------------


@mcp.prompt()
def get_weather_prompt(location: str, timeframe: str) -> str:
    """Get the weather prompt."""
    return f"What is the weather forecast for {location} {timeframe}?"


# -------------------------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------------------------


def main(
    transport: Annotated[Literal["stdio", "http"], typer.Argument()] = "stdio",
) -> None:
    """Model Context Protocol (MCP) Weather Server."""
    agentic.logging.fancy()
    logger.info("Starting %s MCP Weather Server", transport)

    match transport:
        case "stdio":
            mcp.run(transport=transport)
        case "http":
            mcp.run(transport=transport, host=HOST, port=PORT)
        case _:
            raise typer.BadParameter(
                "Transport must be one of: stdio, http.",
                param_hint="transport",
            )


if __name__ == "__main__":
    typer.run(main)
