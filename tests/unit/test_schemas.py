"""
Tests for data schemas — Pydantic model validation.
"""

import pytest
from datetime import datetime, timezone

from src.data.schemas import (
    CityConfig,
    DataQualityReport,
    OpenWeatherWeatherResponse,
    OpenWeatherPollutionResponse,
    AQICNResponse,
    StandardObservation,
    DataSource,
    ValidationStatus,
)


class TestStandardObservation:
    """Tests for StandardObservation model."""

    def test_create_minimal_observation(self):
        """Minimal observation with required fields only."""
        obs = StandardObservation(
            timestamp=datetime(2026, 7, 31, 12, 0, 0, tzinfo=timezone.utc),
            location_id="karachi",
            city_name="Karachi",
            data_source="openweather",
        )
        assert obs.location_id == "karachi"
        assert obs.city_name == "Karachi"
        assert obs.data_source == "openweather"
        assert obs.temperature is None
        assert obs.aqi is None

    def test_create_full_observation(self):
        """Full observation with all fields populated."""
        obs = StandardObservation(
            timestamp=datetime(2026, 7, 31, 12, 0, 0, tzinfo=timezone.utc),
            location_id="karachi",
            city_name="Karachi",
            temperature=34.5,
            humidity=72.0,
            wind_speed=4.6,
            pressure=1005.0,
            weather_condition="few clouds",
            aqi=168,
            pm25=55.8,
            pm10=78.2,
            co=230.5,
            no2=28.3,
            so2=12.4,
            o3=65.1,
            data_source="aqicn",
        )
        assert obs.temperature == 34.5
        assert obs.aqi == 168
        assert obs.pm25 == 55.8

    def test_observation_serialization(self):
        """Observation can be serialized to dict."""
        obs = StandardObservation(
            timestamp=datetime(2026, 7, 31, 12, 0, 0, tzinfo=timezone.utc),
            location_id="karachi",
            city_name="Karachi",
            temperature=34.5,
            data_source="openweather",
        )
        d = obs.model_dump()
        assert isinstance(d, dict)
        assert d["location_id"] == "karachi"
        assert d["temperature"] == 34.5

    def test_observation_missing_required_fields(self):
        """Observation raises error when required fields are missing."""
        with pytest.raises(Exception):
            StandardObservation(
                timestamp=datetime.now(timezone.utc),
                # location_id missing
                city_name="Karachi",
                data_source="openweather",
            )


class TestCityConfig:
    """Tests for CityConfig model."""

    def test_create_city_config(self):
        """Valid city configuration."""
        city = CityConfig(
            id="karachi",
            name="Karachi",
            latitude=24.8607,
            longitude=67.0011,
        )
        assert city.id == "karachi"
        assert city.latitude == 24.8607

    def test_city_config_missing_fields(self):
        """City config raises error when required fields are missing."""
        with pytest.raises(Exception):
            CityConfig(id="karachi", name="Karachi")
            # latitude and longitude missing


class TestOpenWeatherWeatherResponse:
    """Tests for OpenWeather weather response model."""

    def test_parse_valid_response(self):
        """Parse a valid weather response."""
        raw = {
            "coord": {"lon": 67.0011, "lat": 24.8607},
            "weather": [{"id": 801, "main": "Clouds", "description": "few clouds"}],
            "main": {"temp": 34.5, "humidity": 72, "pressure": 1005},
            "wind": {"speed": 4.6, "deg": 220},
            "name": "Karachi",
            "dt": 1751400000,
            "timezone": 18000,
            "cod": 200,
        }
        parsed = OpenWeatherWeatherResponse(**raw)
        assert parsed.main is not None
        assert parsed.main.temp == 34.5
        assert parsed.weather[0].description == "few clouds"

    def test_parse_response_with_missing_fields(self):
        """Parse response with some missing optional fields."""
        raw = {
            "main": {"temp": 34.5},
            "dt": 1751400000,
        }
        parsed = OpenWeatherWeatherResponse(**raw)
        assert parsed.main.temp == 34.5
        assert parsed.wind is None
        assert parsed.weather is None


class TestAQICNResponse:
    """Tests for AQICN response model."""

    def test_parse_valid_response(self):
        """Parse a valid AQICN response."""
        raw = {
            "status": "ok",
            "data": {
                "aqi": 168,
                "iaqi": {
                    "pm25": {"v": 55.8},
                    "pm10": {"v": 78.2},
                },
                "time": {"iso": "2026-07-31T12:00:00+05:00", "v": 1751377200},
                "city": {"name": "Karachi"},
            },
        }
        parsed = AQICNResponse(**raw)
        assert parsed.status == "ok"
        assert parsed.data.aqi == 168

    def test_parse_error_response(self):
        """Parse an error response from AQICN."""
        raw = {
            "status": "error",
            "data": None,
            "data_message": "Unknown station",
        }
        parsed = AQICNResponse(**raw)
        assert parsed.status == "error"


class TestDataQualityReport:
    """Tests for DataQualityReport model."""

    def test_create_pass_report(self):
        """Create a passing quality report."""
        report = DataQualityReport(
            status="pass",
            total_records=100,
        )
        assert report.status == "pass"
        assert report.total_records == 100
        assert len(report.warnings) == 0

    def test_create_warning_report(self):
        """Create a warning quality report."""
        report = DataQualityReport(
            status="warning",
            total_records=100,
            duplicate_count=5,
            warnings=["Found 5 duplicate records"],
        )
        assert report.status == "warning"
        assert report.duplicate_count == 5

    def test_create_fail_report(self):
        """Create a failing quality report."""
        report = DataQualityReport(
            status="fail",
            total_records=0,
            errors=["Missing required column: timestamp"],
        )
        assert report.status == "fail"
        assert len(report.errors) == 1
