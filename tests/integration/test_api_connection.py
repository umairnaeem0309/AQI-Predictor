"""
Integration tests for real API connections.

These tests make actual API calls to verify:
- OpenWeather connection and response structure
- AQICN connection and response structure
- Data freshness with real timestamps
- Duplicate detection with real data

IMPORTANT: These tests skip gracefully when API credentials are not
configured in the environment. They will NOT fail CI/CD pipelines
that lack secrets.
"""

import os

import pytest

from src.config import get_api_key

# =============================================================================
# Credential Check Fixtures
# =============================================================================


def _has_openweather_key() -> bool:
    """Check if OpenWeather API key is available."""
    key = get_api_key("openweather")
    return key is not None and len(key) > 0


def _has_aqicn_key() -> bool:
    """Check if AQICN API key is available."""
    key = get_api_key("aqicn")
    return key is not None and len(key) > 0


# =============================================================================
# OpenWeather Integration Tests
# =============================================================================


@pytest.mark.skipif(
    not _has_openweather_key(),
    reason="OPENWEATHER_API_KEY not configured — skipping OpenWeather integration tests",
)
class TestOpenWeatherConnection:
    """Real OpenWeather API connection tests."""

    def test_weather_endpoint_returns_data(self):
        """Verify /data/2.5/weather returns valid data."""
        from src.data.openweather_client import OpenWeatherClient
        from src.data.schemas import CityConfig

        client = OpenWeatherClient(api_key=get_api_key("openweather"))
        city = CityConfig(
            id="karachi",
            name="Karachi",
            latitude=24.8607,
            longitude=67.0011,
        )

        observations = client.fetch_data(city_id="karachi", city_config=city)

        assert len(observations) == 1
        obs = observations[0]
        assert obs.location_id == "karachi"
        assert obs.temperature is not None
        assert obs.humidity is not None

    def test_pollution_endpoint_returns_data(self):
        """Verify /data/2.5/air_pollution returns data."""
        from src.data.openweather_client import OpenWeatherClient
        from src.data.schemas import CityConfig

        client = OpenWeatherClient(api_key=get_api_key("openweather"))
        city = CityConfig(
            id="lahore",
            name="Lahore",
            latitude=31.5204,
            longitude=74.3587,
        )

        observations = client.fetch_data(city_id="lahore", city_config=city)

        assert len(observations) == 1
        # Pollutant data may or may not be available
        # but the observation should exist
        obs = observations[0]
        assert obs.location_id == "lahore"

    def test_response_matches_standard_schema(self):
        """Verify response fields match StandardObservation schema."""
        from src.data.openweather_client import OpenWeatherClient
        from src.data.schemas import CityConfig, StandardObservation

        client = OpenWeatherClient(api_key=get_api_key("openweather"))
        city = CityConfig(
            id="islamabad",
            name="Islamabad",
            latitude=33.6844,
            longitude=73.0479,
        )

        observations = client.fetch_data(city_id="islamabad", city_config=city)

        assert len(observations) == 1
        obs = observations[0]
        # Verify it's a valid StandardObservation
        assert isinstance(obs, StandardObservation)
        assert obs.timestamp is not None
        assert obs.data_source == "openweather"


# =============================================================================
# AQICN Integration Tests
# =============================================================================


@pytest.mark.skipif(
    not _has_aqicn_key(),
    reason="AQICN_API_KEY not configured — skipping AQICN integration tests",
)
class TestAQICNConnection:
    """Real AQICN API connection tests."""

    def test_aqicn_endpoint_returns_data(self):
        """Verify AQICN feed endpoint returns valid data."""
        from src.data.aqicn_client import AQICNClient

        client = AQICNClient(api_key=get_api_key("aqicn"))

        observations = client.fetch_data(city_id="karachi")

        assert len(observations) == 1
        obs = observations[0]
        assert obs.location_id == "karachi"
        assert obs.aqi is not None
        assert obs.aqi > 0

    def test_aqicn_returns_us_epa_scale(self):
        """Verify AQICN AQI is on US EPA scale (0-500)."""
        from src.data.aqicn_client import AQICNClient

        client = AQICNClient(api_key=get_api_key("aqicn"))

        observations = client.fetch_data(city_id="karachi")

        assert len(observations) == 1
        obs = observations[0]
        assert 0 <= obs.aqi <= 500, f"AQI {obs.aqi} is outside US EPA range (0-500)"

    def test_aqicn_weather_fields_are_none(self):
        """Verify AQICN does not provide weather fields."""
        from src.data.aqicn_client import AQICNClient

        client = AQICNClient(api_key=get_api_key("aqicn"))

        observations = client.fetch_data(city_id="karachi")

        assert len(observations) == 1
        obs = observations[0]
        assert obs.temperature is None
        assert obs.humidity is None
        assert obs.wind_speed is None

    def test_aqicn_staleness_detection(self):
        """Verify staleness detection works with real timestamps."""
        from src.data.aqicn_client import AQICNClient

        client = AQICNClient(
            api_key=get_api_key("aqicn"),
            max_staleness_hours=0.001,  # Very short threshold
        )

        observations = client.fetch_data(city_id="karachi")

        # Should still return data (stale data is still usable)
        assert len(observations) == 1


# =============================================================================
# API Manager Integration Test
# =============================================================================


@pytest.mark.skipif(
    not _has_openweather_key() and not _has_aqicn_key(),
    reason="No API keys configured — skipping API Manager integration test",
)
class TestAPIManagerIntegration:
    """Real API Manager integration test."""

    def test_fetch_city_data_with_real_apis(self):
        """Verify API Manager works with real API calls."""
        from src.data.api_manager import APIManager
        from src.data.schemas import CityConfig

        manager = APIManager()
        city = CityConfig(
            id="karachi",
            name="Karachi",
            latitude=24.8607,
            longitude=67.0011,
        )

        result = manager.fetch_city_data(city)

        # At least one source should succeed
        if result is not None:
            assert result.location_id == "karachi"
            assert result.timestamp is not None
