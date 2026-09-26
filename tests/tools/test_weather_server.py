"""Tests for the MCP Weather Server (HTTP mocked, no network)."""

import logging
from functools import partial

import httpx
import pytest

from .helpers import call_tool, list_tool_names, load_server

FORECAST_PAYLOAD = {
    "latitude": 40.0,
    "longitude": -75.0,
    "elevation": 10.0,
    "timezone": "America/New_York",
    "timezone_abbreviation": "EDT",
    "daily_units": {"temperature_2m_max": "°F", "sunrise": "iso8601"},
    "daily": {
        "time": ["2026-09-26", "2026-09-27"],
        "temperature_2m_max": [70.1, 72.3],
        "sunrise": ["2026-09-26T06:58"],
    },
}


def geocoding_payload(*ids: int) -> dict:
    return {
        "results": [
            {
                "id": i,
                "name": f"Place {i}",
                "latitude": 1.0,
                "longitude": 2.0,
                "timezone": "UTC",
                "country": "Nowhere",
                "country_code": "NW",
            }
            for i in ids
        ]
    }


@pytest.fixture
def weather(monkeypatch):
    module = load_server("weather_server")
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.host == "api.open-meteo.com":
            return httpx.Response(200, json=FORECAST_PAYLOAD)
        ids = range(int(request.url.params["count"]))
        return httpx.Response(200, json=geocoding_payload(*ids))

    monkeypatch.setattr(
        module,
        "_http_client",
        partial(module._http_client, transport=httpx.MockTransport(handler)),
    )
    module.requests = requests
    return module


def test_import_has_no_logging_side_effects():
    before = logging.getLogger().handlers[:]
    load_server("weather_server")
    assert logging.getLogger().handlers == before


def test_tools_are_registered(weather):
    assert list_tool_names(weather.mcp) == {"get_weather_forecast", "get_locations"}


def test_get_weather_forecast_pivots_daily_data(weather):
    result = call_tool(
        weather.mcp,
        "get_weather_forecast",
        {
            "latitude": 40.0,
            "longitude": -75.0,
            "timezone": "America/New_York",
            "weather_variables": ["temperature_2m_max", "sunrise"],
        },
    )
    assert result["daily"] == {
        "2026-09-26": {"temperature_2m_max": 70.1, "sunrise": "2026-09-26T06:58"},
        "2026-09-27": {"temperature_2m_max": 72.3},
    }
    params = weather.requests[0].url.params
    assert params["daily"] == "sunrise,temperature_2m_max"
    assert params["temperature_unit"] == "fahrenheit"


def test_get_weather_forecast_uses_default_variables(weather):
    call_tool(weather.mcp, "get_weather_forecast", {"latitude": 1, "longitude": 2})
    daily = weather.requests[0].url.params["daily"].split(",")
    assert daily == sorted(weather.DEFAULT_WEATHER_VARIABLES)


def test_get_locations_populates_bounded_cache(weather, monkeypatch):
    monkeypatch.setattr(weather, "LOCATION_CACHE_SIZE", 3)
    result = call_tool(weather.mcp, "get_locations", {"name": "Place", "count": 5})
    assert [loc["id"] for loc in result["result"]] == [0, 1, 2, 3, 4]
    assert list(weather.location_cache) == [2, 3, 4]
    assert weather.requests[0].url.params["name"] == "Place"


def test_http_client_has_timeout():
    client = load_server("weather_server")._http_client()
    assert client.timeout.read == 10.0
