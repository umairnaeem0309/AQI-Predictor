"""
Tests for OpenWeather API client.

Uses the `responses` library for HTTP mocking to test:
- Response parsing
- Weather and pollution data merging
- Error handling
- Initialization without credentials
"""

import json

import pytest
import responses
import responses as _responses_mod
from requests.exceptions import Timeout as RequestsTimeout

from src.data.exceptions import (
    APIAuthenticationError,
    APINetworkError,
    APIRateLimitError,
    APITimeoutError,
)
from src.data.openweather_client import OpenWeatherClient, _unix_to_utc
from src.data.schemas import CityConfig, DataSource, StandardObservation

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def city_config():
    """Karachi city configuration."""
    return CityConfig(
        id="karachi",
        name="Karachi",
        latitude=24.8607,
        longitude=67.0011,
    )


@pytest.fixture
def weather_response():
    """Sample OpenWeather weather response."""
    return {
        "coord": {"lon": 67.0011, "lat": 24.8607},
        "weather": [{"id": 801, "main": "Clouds", "description": "few clouds"}],
        "main": {"temp": 34.5, "humidity": 72, "pressure": 1005},
        "wind": {"speed": 4.6, "deg": 220},
        "name": "Karachi",
        "dt": 1751400000,
        "timezone": 18000,
        "cod": 200,
    }


@pytest.fixture
def pollution_response():
    """Sample OpenWeather pollution response."""
    return {
        "coord": {"lon": 67.0011, "lat": 24.8607},
        "list": [
            {
                "main": {"aqi": 4},
                "components": {
                    "co": 230.5,
                    "no2": 28.3,
                    "o3": 65.1,
                    "so2": 12.4,
                    "pm2_5": 55.8,
                    "pm10": 78.2,
                },
                "dt": 1751400000,
            }
        ],
    }


# =============================================================================
# Test Helper Functions
# =============================================================================


class TestUnixToUtc:
    """Tests for _unix_to_utc helper function."""

    def test_convert_known_timestamp(self):
        """Convert a known Unix timestamp to UTC."""
        dt = _unix_to_utc(1751400000)
        assert dt.year == 2025
        assert dt.tzinfo is not None

    def test_convert_zero_timestamp(self):
        """Convert epoch zero."""
        dt = _unix_to_utc(0)
        assert dt.year == 1970


# =============================================================================
# Test Client Initialization
# =============================================================================


class TestOpenWeatherClientInit:
    """Tests for client initialization."""

    def test_init_with_api_key(self):
        """Client initializes with API key."""
        client = OpenWeatherClient(api_key="test-key")
        assert client.api_key == "test-key"
        assert "openweathermap.org" in client.base_url

    def test_init_without_api_key(self):
        """Client initializes without API key (test/mock mode)."""
        client = OpenWeatherClient()
        assert client.api_key is None

    def test_init_custom_base_url(self):
        """Client accepts custom base URL."""
        client = OpenWeatherClient(base_url="https://custom.api.com/v2")
        assert client.base_url == "https://custom.api.com/v2"

    def test_init_custom_retry_settings(self):
        """Client accepts custom retry settings."""
        client = OpenWeatherClient(max_retries=5)
        assert client.max_retries == 5


# =============================================================================
# Test Response Parsing
# =============================================================================


class TestOpenWeatherParsing:
    """Tests for response parsing."""

    def test_parse_weather_response(self, weather_response, city_config):
        """Parse valid weather response into StandardObservation."""
        client = OpenWeatherClient(api_key="test")
        obs = client._parse_weather_response(weather_response, "karachi", "Karachi")
        assert obs is not None
        assert obs.location_id == "karachi"
        assert obs.city_name == "Karachi"
        assert obs.temperature == 34.5
        assert obs.humidity == 72.0
        assert obs.pressure == 1005.0
        assert obs.wind_speed == 4.6
        assert obs.weather_condition == "few clouds"
        assert obs.data_source == "openweather"

    def test_parse_weather_response_missing_wind(self, city_config):
        """Parse weather response with missing wind data."""
        raw = {
            "main": {"temp": 30.0, "humidity": 60, "pressure": 1010},
            "dt": 1751400000,
        }
        client = OpenWeatherClient(api_key="test")
        obs = client._parse_weather_response(raw, "karachi", "Karachi")
        assert obs is not None
        assert obs.temperature == 30.0
        assert obs.wind_speed is None

    def test_parse_weather_response_empty(self, city_config):
        """Parse empty weather response — returns observation with None values."""
        client = OpenWeatherClient(api_key="test")
        obs = client._parse_weather_response({}, "karachi", "Karachi")
        # Empty JSON parses to observation with all None fields
        assert obs is not None
        assert obs.temperature is None

    def test_parse_pollution_response(self, pollution_response):
        """Parse valid pollution response."""
        client = OpenWeatherClient(api_key="test")
        result = client._parse_pollution_response(pollution_response)
        assert result is not None
        assert result["pm25"] == 55.8
        assert result["pm10"] == 78.2
        assert result["co"] == 230.5
        assert result["no2"] == 28.3
        assert result["so2"] == 12.4
        assert result["o3"] == 65.1

    def test_parse_pollution_response_empty(self):
        """Parse empty pollution response."""
        client = OpenWeatherClient(api_key="test")
        result = client._parse_pollution_response({})
        assert result is None

    def test_parse_pollution_response_empty_list(self):
        """Parse pollution response with empty list."""
        client = OpenWeatherClient(api_key="test")
        result = client._parse_pollution_response({"list": []})
        assert result is None


# =============================================================================
# Test Merging
# =============================================================================


class TestOpenWeatherMerging:
    """Tests for weather and pollution data merging."""

    def test_merge_with_pollution(self, weather_response):
        """Merge weather observation with pollution data."""
        client = OpenWeatherClient(api_key="test")
        weather_obs = client._parse_weather_response(weather_response, "karachi", "Karachi")
        pollution_data = {"pm25": 55.8, "pm10": 78.2, "co": 230.5}

        merged = client._merge_observations(weather_obs, pollution_data)
        assert merged is not None
        assert merged.temperature == 34.5  # From weather
        assert merged.pm25 == 55.8  # From pollution
        assert merged.pm10 == 78.2  # From pollution

    def test_merge_without_pollution(self, weather_response):
        """Merge weather observation without pollution data."""
        client = OpenWeatherClient(api_key="test")
        weather_obs = client._parse_weather_response(weather_response, "karachi", "Karachi")

        merged = client._merge_observations(weather_obs, None)
        assert merged is not None
        assert merged.temperature == 34.5
        assert merged.pm25 is None


# =============================================================================
# Test Validation
# =============================================================================


class TestOpenWeatherValidation:
    """Tests for response validation."""

    def test_validate_valid_response(self):
        """Valid response passes validation."""
        client = OpenWeatherClient(api_key="test")
        assert client._validate_response({"main": {"temp": 30}}) is True

    def test_validate_response_with_weather(self):
        """Response with weather field passes validation."""
        client = OpenWeatherClient(api_key="test")
        assert client._validate_response({"weather": [{"id": 800}]}) is True

    def test_validate_empty_response(self):
        """Empty response fails validation."""
        client = OpenWeatherClient(api_key="test")
        assert client._validate_response({}) is False

    def test_validate_none_response(self):
        """None response fails validation."""
        client = OpenWeatherClient(api_key="test")
        assert client._validate_response(None) is False


# =============================================================================
# Test Full Fetch with HTTP Mocking
# =============================================================================


class TestOpenWeatherFetchData:
    """Tests for fetch_data with mocked HTTP responses."""

    @responses.activate
    def test_fetch_data_success(self, city_config, weather_response, pollution_response):
        """Successful data fetch returns merged observation."""
        responses.add(
            responses.GET,
            "https://api.openweathermap.org/data/2.5/weather",
            json=weather_response,
            status=200,
        )
        responses.add(
            responses.GET,
            "https://api.openweathermap.org/data/2.5/air_pollution",
            json=pollution_response,
            status=200,
        )

        client = OpenWeatherClient(api_key="test-key")
        observations = client.fetch_data(
            city_id="karachi",
            city_config=city_config,
        )

        assert len(observations) == 1
        obs = observations[0]
        assert obs.location_id == "karachi"
        assert obs.temperature == 34.5
        assert obs.pm25 == 55.8

    @responses.activate
    def test_fetch_data_weather_only_pollution_fails(self, city_config, weather_response):
        """Fetch succeeds when pollution endpoint fails (weather-only)."""
        responses.add(
            responses.GET,
            "https://api.openweathermap.org/data/2.5/weather",
            json=weather_response,
            status=200,
        )
        responses.add(
            responses.GET,
            "https://api.openweathermap.org/data/2.5/air_pollution",
            json={"message": "error"},
            status=500,
        )

        client = OpenWeatherClient(api_key="test-key")
        observations = client.fetch_data(
            city_id="karachi",
            city_config=city_config,
        )

        assert len(observations) == 1
        obs = observations[0]
        assert obs.temperature == 34.5
        assert obs.pm25 is None

    @responses.activate
    def test_fetch_data_timeout(self, city_config):
        """Timeout raises APITimeoutError."""
        responses.add(
            responses.GET,
            "https://api.openweathermap.org/data/2.5/weather",
            body=RequestsTimeout("mocked timeout"),
        )

        client = OpenWeatherClient(api_key="test-key", max_retries=1)
        with pytest.raises(APITimeoutError):
            client.fetch_data(
                city_id="karachi",
                city_config=city_config,
            )

    @responses.activate
    def test_fetch_data_auth_failure(self, city_config):
        """Authentication failure raises APIAuthenticationError (not retried)."""
        responses.add(
            responses.GET,
            "https://api.openweathermap.org/data/2.5/weather",
            json={"message": "Invalid API key"},
            status=401,
        )

        client = OpenWeatherClient(api_key="bad-key", max_retries=3)
        with pytest.raises(APIAuthenticationError):
            client.fetch_data(
                city_id="karachi",
                city_config=city_config,
            )

    @responses.activate
    def test_fetch_data_rate_limit_retries(self, city_config, weather_response):
        """Rate limit (429) triggers retry then succeeds."""
        responses.add(
            responses.GET,
            "https://api.openweathermap.org/data/2.5/weather",
            json={"message": "rate limit"},
            status=429,
        )
        responses.add(
            responses.GET,
            "https://api.openweathermap.org/data/2.5/weather",
            json=weather_response,
            status=200,
        )
        responses.add(
            responses.GET,
            "https://api.openweathermap.org/data/2.5/air_pollution",
            json={"list": []},
            status=200,
        )

        client = OpenWeatherClient(api_key="test-key", max_retries=2)
        observations = client.fetch_data(
            city_id="karachi",
            city_config=city_config,
        )
        assert len(observations) == 1
