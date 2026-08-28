"""
Unit Tests for Open-Meteo Historical Data Providers.

Tests:
- OpenMeteoWeatherProvider: response parsing, variable mapping, DataFrame output
- OpenMeteoAirQualityProvider: response parsing, variable mapping, DataFrame output
- BaseHistoricalProvider: chunked fetching, rate limiting, error handling
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.providers.open_meteo_air_quality import OpenMeteoAirQualityProvider
from src.data.providers.open_meteo_weather import OpenMeteoWeatherProvider

# ============================================================================
# Sample API Responses
# ============================================================================

SAMPLE_WEATHER_RESPONSE = {
    "latitude": 24.86,
    "longitude": 67.0,
    "generationtime_ms": 2.5,
    "utc_offset_seconds": 0,
    "timezone": "GMT",
    "timezone_abbreviation": "GMT",
    "elevation": 8.0,
    "hourly_units": {
        "time": "iso8601",
        "temperature_2m": "°C",
        "relative_humidity_2m": "%",
        "surface_pressure": "hPa",
        "wind_speed_10m": "m/s",
        "wind_direction_10m": "°",
        "cloud_cover": "%",
        "precipitation": "mm",
    },
    "hourly": {
        "time": [
            "2023-01-01T00:00",
            "2023-01-01T01:00",
            "2023-01-01T02:00",
        ],
        "temperature_2m": [18.5, 17.8, 17.2],
        "relative_humidity_2m": [72, 75, 78],
        "surface_pressure": [1013.2, 1013.0, 1012.8],
        "wind_speed_10m": [3.5, 3.2, 2.8],
        "wind_direction_10m": [180, 185, 190],
        "cloud_cover": [45, 50, 55],
        "precipitation": [0.0, 0.0, 0.1],
    },
}

SAMPLE_AQ_RESPONSE = {
    "latitude": 24.86,
    "longitude": 67.0,
    "generationtime_ms": 3.1,
    "utc_offset_seconds": 0,
    "timezone": "GMT",
    "timezone_abbreviation": "GMT",
    "hourly_units": {
        "time": "iso8601",
        "pm2_5": "μg/m³",
        "pm10": "μg/m³",
        "carbon_monoxide": "μg/m³",
        "nitrogen_dioxide": "μg/m³",
        "sulphur_dioxide": "μg/m³",
        "ozone": "μg/m³",
        "us_aqi": "US AQI",
    },
    "hourly": {
        "time": [
            "2023-01-01T00:00",
            "2023-01-01T01:00",
            "2023-01-01T02:00",
        ],
        "pm2_5": [25.3, 27.1, 24.8],
        "pm10": [45.2, 48.5, 42.1],
        "carbon_monoxide": [450.0, 460.0, 440.0],
        "nitrogen_dioxide": [32.5, 35.1, 30.8],
        "sulphur_dioxide": [12.3, 13.0, 11.5],
        "ozone": [55.2, 52.1, 58.3],
        "us_aqi": [78, 82, 75],
        "us_aqi_pm2_5": [68, 72, 65],
        "us_aqi_pm10": [42, 45, 40],
    },
}

SAMPLE_EMPTY_RESPONSE = {
    "latitude": 24.86,
    "longitude": 67.0,
    "hourly": {},
}


# ============================================================================
# OpenMeteoWeatherProvider Tests
# ============================================================================


class TestOpenMeteoWeatherProvider:
    """Tests for the Open-Meteo Weather Provider."""

    def test_variable_mapping(self):
        """Variable mapping covers all required weather variables."""
        provider = OpenMeteoWeatherProvider()
        mapping = provider._get_variable_mapping()

        assert "temperature" in mapping
        assert "humidity" in mapping
        assert "pressure" in mapping
        assert "wind_speed" in mapping
        assert "wind_direction" in mapping
        assert "cloud_cover" in mapping
        assert "precipitation" in mapping

        # All map to Open-Meteo API names
        assert mapping["temperature"] == "temperature_2m"
        assert mapping["humidity"] == "relative_humidity_2m"
        assert mapping["pressure"] == "surface_pressure"

    def test_parse_response_basic(self):
        """Basic response parsing produces correct DataFrame."""
        provider = OpenMeteoWeatherProvider()
        df = provider._parse_response(SAMPLE_WEATHER_RESPONSE, "karachi", "Karachi")

        assert not df.empty
        assert len(df) == 3
        assert "timestamp" in df.columns
        assert "location_id" in df.columns
        assert "city_name" in df.columns
        assert "temperature" in df.columns
        assert "humidity" in df.columns
        assert "pressure" in df.columns
        assert "wind_speed" in df.columns

    def test_parse_response_values(self):
        """Parsed values match the sample response."""
        provider = OpenMeteoWeatherProvider()
        df = provider._parse_response(SAMPLE_WEATHER_RESPONSE, "karachi", "Karachi")

        # Check first row values
        assert df.iloc[0]["temperature"] == 18.5
        assert df.iloc[0]["humidity"] == 72
        assert df.iloc[0]["pressure"] == 1013.2
        assert df.iloc[0]["wind_speed"] == 3.5
        assert df.iloc[0]["wind_direction"] == 180
        assert df.iloc[0]["cloud_cover"] == 45
        assert df.iloc[0]["precipitation"] == 0.0

    def test_parse_response_timestamps_utc(self):
        """Parsed timestamps are UTC-aware datetime."""
        provider = OpenMeteoWeatherProvider()
        df = provider._parse_response(SAMPLE_WEATHER_RESPONSE, "karachi", "Karachi")

        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
        # All timestamps should be UTC
        for ts in df["timestamp"]:
            assert ts.tzinfo is not None

    def test_parse_response_metadata(self):
        """Response includes provider metadata columns."""
        provider = OpenMeteoWeatherProvider()
        df = provider._parse_response(SAMPLE_WEATHER_RESPONSE, "karachi", "Karachi")

        assert "data_source" in df.columns
        assert (df["data_source"] == "open_meteo_weather").all()
        assert "provider" in df.columns
        assert (df["provider"] == "open-meteo").all()

    def test_parse_response_empty(self):
        """Empty response produces empty DataFrame."""
        provider = OpenMeteoWeatherProvider()
        df = provider._parse_response(SAMPLE_EMPTY_RESPONSE, "karachi", "Karachi")
        assert df.empty

    def test_parse_response_missing_variable(self):
        """Missing variable in response is handled gracefully."""
        provider = OpenMeteoWeatherProvider()
        response = {
            "hourly": {
                "time": ["2023-01-01T00:00"],
                "temperature_2m": [20.0],
                # Missing other variables
            }
        }
        df = provider._parse_response(response, "karachi", "Karachi")
        assert not df.empty
        assert df.iloc[0]["temperature"] == 20.0
        assert df.iloc[0]["humidity"] is None  # Missing → None

    def test_base_url(self):
        """Base URL points to Open-Meteo archive endpoint."""
        provider = OpenMeteoWeatherProvider()
        assert "archive-api.open-meteo.com" in provider.base_url

    def test_max_days_per_request(self):
        """Weather provider supports large date range chunks."""
        provider = OpenMeteoWeatherProvider()
        assert provider.max_days_per_request >= 365


# ============================================================================
# OpenMeteoAirQualityProvider Tests
# ============================================================================


class TestOpenMeteoAirQualityProvider:
    """Tests for the Open-Meteo Air Quality Provider."""

    def test_variable_mapping(self):
        """Variable mapping covers all required pollution variables."""
        provider = OpenMeteoAirQualityProvider()
        mapping = provider._get_variable_mapping()

        assert "pm25" in mapping
        assert "pm10" in mapping
        assert "co" in mapping
        assert "no2" in mapping
        assert "so2" in mapping
        assert "o3" in mapping

        # All map to Open-Meteo API names
        assert mapping["pm25"] == "pm2_5"
        assert mapping["pm10"] == "pm10"
        assert mapping["co"] == "carbon_monoxide"

    def test_parse_response_basic(self):
        """Basic response parsing produces correct DataFrame."""
        provider = OpenMeteoAirQualityProvider()
        df = provider._parse_response(SAMPLE_AQ_RESPONSE, "karachi", "Karachi")

        assert not df.empty
        assert len(df) == 3
        assert "timestamp" in df.columns
        assert "pm25" in df.columns
        assert "pm10" in df.columns
        assert "co" in df.columns
        assert "no2" in df.columns
        assert "so2" in df.columns
        assert "o3" in df.columns

    def test_parse_response_values(self):
        """Parsed pollutant values match the sample response."""
        provider = OpenMeteoAirQualityProvider()
        df = provider._parse_response(SAMPLE_AQ_RESPONSE, "karachi", "Karachi")

        assert df.iloc[0]["pm25"] == 25.3
        assert df.iloc[0]["pm10"] == 45.2
        assert df.iloc[0]["co"] == 450.0
        assert df.iloc[0]["no2"] == 32.5
        assert df.iloc[0]["so2"] == 12.3
        assert df.iloc[0]["o3"] == 55.2

    def test_parse_response_us_aqi_reference(self):
        """US AQI reference values are stored for validation."""
        provider = OpenMeteoAirQualityProvider()
        df = provider._parse_response(SAMPLE_AQ_RESPONSE, "karachi", "Karachi")

        assert "us_aqi_open_meteo" in df.columns
        assert df.iloc[0]["us_aqi_open_meteo"] == 78
        assert "us_aqi_pm25_open_meteo" in df.columns
        assert df.iloc[0]["us_aqi_pm25_open_meteo"] == 68

    def test_parse_response_empty(self):
        """Empty response produces empty DataFrame."""
        provider = OpenMeteoAirQualityProvider()
        df = provider._parse_response(SAMPLE_EMPTY_RESPONSE, "karachi", "Karachi")
        assert df.empty

    def test_base_url(self):
        """Base URL points to Open-Meteo air quality endpoint."""
        provider = OpenMeteoAirQualityProvider()
        assert "air-quality-api.open-meteo.com" in provider.base_url

    def test_max_days_per_request(self):
        """Air quality provider respects 92-day API limit."""
        provider = OpenMeteoAirQualityProvider()
        assert provider.max_days_per_request == 92


# ============================================================================
# Base Provider Tests
# ============================================================================


class TestBaseHistoricalProvider:
    """Tests for base provider functionality."""

    def test_weather_provider_has_fetch_historical(self):
        """Weather provider inherits fetch_historical from base."""
        provider = OpenMeteoWeatherProvider()
        assert hasattr(provider, "fetch_historical")
        assert callable(provider.fetch_historical)

    def test_aq_provider_has_fetch_historical(self):
        """Air quality provider inherits fetch_historical from base."""
        provider = OpenMeteoAirQualityProvider()
        assert hasattr(provider, "fetch_historical")
        assert callable(provider.fetch_historical)

    def test_weather_provider_has_fetch_all_cities(self):
        """Weather provider inherits fetch_all_cities from base."""
        provider = OpenMeteoWeatherProvider()
        assert hasattr(provider, "fetch_all_cities")

    def test_usage_summary_empty(self):
        """Usage summary shows zero requests initially."""
        provider = OpenMeteoWeatherProvider()
        summary = provider.get_usage_summary()
        assert summary["total_requests"] == 0
        assert summary["total_errors"] == 0

    @patch("src.data.providers.open_meteo_weather.OpenMeteoWeatherProvider._fetch_chunk")
    def test_fetch_chunk_called(self, mock_fetch_chunk):
        """_fetch_chunk is called during fetch_historical."""
        mock_fetch_chunk.return_value = SAMPLE_WEATHER_RESPONSE

        provider = OpenMeteoWeatherProvider()

        df = provider.fetch_historical(
            latitude=24.86,
            longitude=67.0,
            location_id="karachi",
            city_name="Karachi",
            start_date="2023-01-01",
            end_date="2023-01-01",
        )

        assert not df.empty
        mock_fetch_chunk.assert_called_once()
