#!/usr/bin/env python3
"""MCP Weather Server."""

import argparse
import logging
from datetime import datetime
from functools import wraps
from typing import Any, Dict, Hashable, List, Literal, Optional
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

logger = logging.getLogger("weather_server")
logging.basicConfig(level=logging.INFO)


OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"

# -------------------------------------------------------------------------------------------------
# MCP Weather Server
# -------------------------------------------------------------------------------------------------

# Create an MCP server
mcp = FastMCP("MCP Weather Server", host="0.0.0.0", port=8000)


# --------------------------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------------------------


def log_call(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(
            f"Call: {func.__name__}({', '.join(map(repr, args))}, {', '.join(f'{k}={v!r}' for k, v in kwargs.items())})"
        )
        result = func(*args, **kwargs)
        logger.info(
            f"Result: {func.__name__}({', '.join(map(repr, args))}, {', '.join(f'{k}={v!r}' for k, v in kwargs.items())}) -> {result}"
        )
        return result

    return wrapper


# --------------------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------------------


@mcp.prompt()
def get_weather_prompt(location: str, timeframe: str) -> str:
    """Get the weather prompt."""
    return f"What is the weather like in {location} for {timeframe}?"


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
    daily: Dict[Hashable, Dict[Hashable, Any]] = Field(
        ..., description="Daily weather variables."
    )


@mcp.tool()
@log_call
def get_weather_forecast(
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
    """Get the weather forecast for the provided coordinates.

    Includes sunrise and sunset times; minimum, maximum, and mean temperatures (in Fahrenheit);
    rain, showers, and snowfall precipitation (in inches);
    precipitation probability (in percentage).

    Default daily variables include:
        - cloud_cover_mean
        - precipitation_probability_max
        - precipitation_sum
        - temperature_2m_max
        - temperature_2m_min

    Args:
        latitude: Coordinate latitude in degrees.
        longitude: Coordinate longitude in degrees.
        timezone: Timezone for the forecast (e.g. 'America/New_York', default is "auto").
        start_date: Start date in ISO8601 (YYYY-MM-DD) format for the forecast (default is today).
        end_date: End date in ISO8601 (YYYY-MM-DD) format for the forecast (default is today).
        daily: Set of daily weather variables to include (default is a predefined set).

    Returns:
        WeatherForecast: The weather forecast data including requested daily variables.
    """
    # Get today's date in the specified timezone
    if timezone == "auto":
        today = datetime.now().date().isoformat()
    else:
        try:
            tz = ZoneInfo(timezone)
            today = datetime.now(tz).date().isoformat()
        except Exception:
            # Fallback to local time if timezone is invalid
            today = datetime.now().date().isoformat()

    start_date = start_date or today
    end_date = end_date or today

    weather_variables = (
        list(set(weather_variables))
        if weather_variables
        else [
            "cloud_cover_mean",
            "precipitation_probability_max",
            "precipitation_sum",
            "relative_humidity_2m_mean",
            "temperature_2m_max",
            "temperature_2m_min",
        ]
    )

    request_parameters = {
        "latitude": latitude,
        "longitude": longitude,
        "timezone": timezone,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(weather_variables),
        "timeformat": time_format,
        "temperature_unit": temperature_unit,
        "precipitation_unit": precipitation_unit,
        "wind_speed_unit": wind_speed_unit,
    }

    response = requests.get(OPEN_METEO_FORECAST_URL, params=request_parameters)
    response.raise_for_status()
    data = response.json()

    # Extract daily forecast data
    daily_data = data.get("daily", {})
    daily_units = data.get("daily_units", {})

    # Create DataFrame from the daily data
    daily_forecast_dataframe = pd.DataFrame()

    # Add date column
    if "time" in daily_data:
        daily_forecast_dataframe["date"] = daily_data["time"]
        daily_forecast_dataframe.set_index("date", inplace=True)

    # Add weather variables to DataFrame
    for variable in weather_variables:
        if variable in daily_data:
            daily_forecast_dataframe[variable] = daily_data[variable]

    weather_forecast = WeatherForecast(
        latitude=data.get("latitude", latitude),
        longitude=data.get("longitude", longitude),
        elevation=data.get("elevation"),
        timezone=data.get("timezone"),
        timezone_abbreviation=data.get("timezone_abbreviation"),
        daily_units=daily_units,
        daily=daily_forecast_dataframe.to_dict(orient="index"),
    )
    return weather_forecast


# --------------------------------------------------------------------------------------
# Location Information Tool
# --------------------------------------------------------------------------------------


class LocationInfo(BaseModel):
    """Location information."""

    id: int = Field(..., description="Unique identifier for the location.")
    name: str = Field(..., description="Name of the location.")
    latitude: float = Field(..., description="Latitude of the location.")
    longitude: float = Field(..., description="Longitude of the location.")
    elevation: float = Field(..., description="Elevation of the location in meters.")

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


@mcp.tool()
@log_call
def get_locations(
    name: str, country_code: Optional[str] = None, count: int = 10
) -> List[LocationInfo]:
    """Get location information.

    Args:
        location_name: Name of the location (e.g., city name).
        country_code: Optional ISO-3166-1 alpha2 country code to narrow down the search (e.g., 'US' for the United States).

    Returns:
        List[LocationInfo]: A list of locations matching the search criteria.
    """

    request_parameters = {
        "name": name,
        "count": count,
        "language": "en",
        "format": "json",
    }

    if country_code is not None:
        request_parameters["countryCode"] = country_code

    response = requests.get(OPEN_METEO_GEOCODING_URL, params=request_parameters)
    response.raise_for_status()
    data = response.json()
    results = data.get("results", [])

    location_information = [
        LocationInfo.model_validate(location) for location in results
    ]

    return location_information


# -------------------------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------------------------


def main(transport: Literal["stdio", "sse", "streamable-http"]) -> None:
    """Main function to run the MCP server."""
    logger.info(f"Starting {transport} MCP Weather Server")
    mcp.run(transport=transport)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCP Weather Server")
    subparsers = parser.add_subparsers(dest="mode", help="Server mode")
    subparsers.add_parser("stdio", help="Run stdio stdio MCP server")
    subparsers.add_parser("streamable-http", help="Run streamable-http MCP server")
    args = parser.parse_args()

    if args.mode is None:
        parser.print_help()
        exit(1)

    try:
        main(args.mode)
    except KeyboardInterrupt:
        logger.info("MCP server stopped by user.")
