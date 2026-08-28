"""
Tests for API Manager — orchestration and fallback behavior.

Tests use mocked clients to verify:
- Merge logic follows data ownership rules
- Fallback works when primary source fails
- OpenWeather 1-5 AQI is NOT used as US EPA AQI
- Error handling for both sources failing
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.data.api_manager import APIManager
from src.data.exceptions import APIClientError
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
def openweather_obs():
    """Sample OpenWeather observation."""
    return StandardObservation(
        timestamp=datetime(2026, 8, 2, 12, 0, 0, tzinfo=timezone.utc),
        location_id="karachi",
        city_name="Karachi",
        temperature=34.5,
        humidity=72.0,
        wind_speed=4.6,
        pressure=1005.0,
        weather_condition="few clouds",
        data_source="openweather",
    )


@pytest.fixture
def aqicn_obs():
    """Sample AQICN observation."""
    return StandardObservation(
        timestamp=datetime(2026, 8, 2, 12, 0, 0, tzinfo=timezone.utc),
        location_id="karachi",
        city_name="Karachi",
        aqi=168,
        pm25=55.8,
        pm10=78.2,
        co=230.5,
        no2=28.3,
        so2=12.4,
        o3=65.1,
        data_source="aqicn",
    )


# =============================================================================
# Test Merge Logic
# =============================================================================


class TestMergeLogic:
    """Tests for source merging with ownership rules."""

    def test_both_sources_available(self, city_config, openweather_obs, aqicn_obs):
        """When both sources available, merge uses ownership rules."""
        ow_client = MagicMock()
        ow_client.api_key = "test"
        ow_client.fetch_data.return_value = [openweather_obs]

        aqicn_client = MagicMock()
        aqicn_client.api_key = "test"
        aqicn_client.fetch_data.return_value = [aqicn_obs]
        aqicn_client.merge_with_openweather.return_value = StandardObservation(
            timestamp=openweather_obs.timestamp,
            location_id="karachi",
            city_name="Karachi",
            # Weather from OpenWeather
            temperature=34.5,
            humidity=72.0,
            wind_speed=4.6,
            pressure=1005.0,
            weather_condition="few clouds",
            # AQI from AQICN (authoritative)
            aqi=168,
            pm25=55.8,
            pm10=78.2,
            co=230.5,
            no2=28.3,
            so2=12.4,
            o3=65.1,
            data_source="openweather",
        )

        manager = APIManager(
            openweather_client=ow_client,
            aqicn_client=aqicn_client,
        )
        result = manager.fetch_city_data(city_config)

        assert result is not None
        assert result.temperature == 34.5  # From OpenWeather
        assert result.aqi == 168  # From AQICN

    def test_only_openweather_available(self, city_config, openweather_obs):
        """When only OpenWeather available with no PM history, AQI is None."""
        from src.data.nowcast_history import NowCastHistoryManager

        ow_client = MagicMock()
        ow_client.api_key = "test"
        ow_client.fetch_data.return_value = [openweather_obs]

        aqicn_client = MagicMock()
        aqicn_client.api_key = None
        aqicn_client.fetch_data.side_effect = APIClientError("No API key")

        # Use empty NowCast history so no PM history → no derived AQI
        empty_history = NowCastHistoryManager.__new__(NowCastHistoryManager)
        empty_history.history = {}

        manager = APIManager(
            openweather_client=ow_client,
            aqicn_client=aqicn_client,
            nowcast_history=empty_history,
        )
        result = manager.fetch_city_data(city_config)

        assert result is not None
        assert result.temperature == 34.5
        assert result.aqi is None  # No PM history → cannot derive AQI
        assert result.pm25 is None

    def test_only_aqicn_available(self, city_config, aqicn_obs):
        """When only AQICN available, weather fields are None."""
        ow_client = MagicMock()
        ow_client.api_key = "test"
        ow_client.fetch_data.side_effect = APIClientError("Connection failed")

        aqicn_client = MagicMock()
        aqicn_client.api_key = "test"
        aqicn_client.fetch_data.return_value = [aqicn_obs]

        manager = APIManager(
            openweather_client=ow_client,
            aqicn_client=aqicn_client,
        )
        result = manager.fetch_city_data(city_config)

        assert result is not None
        assert result.aqi == 168
        assert result.temperature is None
        assert result.humidity is None

    def test_both_sources_fail(self, city_config):
        """When both sources fail, returns None."""
        ow_client = MagicMock()
        ow_client.api_key = "test"
        ow_client.fetch_data.side_effect = APIClientError("OpenWeather down")

        aqicn_client = MagicMock()
        aqicn_client.api_key = "test"
        aqicn_client.fetch_data.side_effect = APIClientError("AQICN down")

        manager = APIManager(
            openweather_client=ow_client,
            aqicn_client=aqicn_client,
        )
        result = manager.fetch_city_data(city_config)

        assert result is None


# =============================================================================
# Test AQI Scale Integrity
# =============================================================================


class TestAQIScaleIntegrity:
    """Verify OpenWeather 1-5 AQI is never used as US EPA AQI."""

    def test_openweather_only_no_aqi(self, city_config, openweather_obs):
        """OpenWeather AQI (1-5) must NOT appear as US EPA AQI."""
        from src.data.nowcast_history import NowCastHistoryManager

        ow_client = MagicMock()
        ow_client.api_key = "test"
        ow_client.fetch_data.return_value = [openweather_obs]

        aqicn_client = MagicMock()
        aqicn_client.api_key = None
        aqicn_client.fetch_data.side_effect = APIClientError("No API key")

        empty_history = NowCastHistoryManager.__new__(NowCastHistoryManager)
        empty_history.history = {}

        manager = APIManager(
            openweather_client=ow_client,
            aqicn_client=aqicn_client,
            nowcast_history=empty_history,
        )
        result = manager.fetch_city_data(city_config)

        # OpenWeather AQI (1-5) must NOT appear as US EPA AQI
        assert result.aqi is None

    def test_merge_uses_aqicn_aqi_not_openweather(self, city_config, openweather_obs, aqicn_obs):
        """Merge uses AQICN AQI (US EPA), not OpenWeather AQI."""
        ow_client = MagicMock()
        ow_client.api_key = "test"
        ow_client.fetch_data.return_value = [openweather_obs]

        aqicn_client = MagicMock()
        aqicn_client.api_key = "test"
        aqicn_client.fetch_data.return_value = [aqicn_obs]
        # Simulate merge that uses AQICN AQI
        aqicn_client.merge_with_openweather.return_value = StandardObservation(
            timestamp=openweather_obs.timestamp,
            location_id="karachi",
            city_name="Karachi",
            temperature=34.5,
            aqi=168,  # AQICN US EPA value
            data_source="openweather",
        )

        manager = APIManager(
            openweather_client=ow_client,
            aqicn_client=aqicn_client,
        )
        result = manager.fetch_city_data(city_config)

        # Must be AQICN value (168), not OpenWeather 1-5 scale
        assert result.aqi == 168
        assert result.aqi > 5  # Clearly not OpenWeather scale


# =============================================================================
# Test Error Handling
# =============================================================================


class TestErrorHandling:
    """Tests for error handling in API manager."""

    def test_openweather_unexpected_error(self, city_config, openweather_obs):
        """Unexpected OpenWeather error is handled gracefully."""
        ow_client = MagicMock()
        ow_client.api_key = "test"
        ow_client.fetch_data.side_effect = RuntimeError("Unexpected")

        aqicn_client = MagicMock()
        aqicn_client.api_key = "test"
        aqicn_client.fetch_data.return_value = [
            StandardObservation(
                timestamp=datetime.now(timezone.utc),
                location_id="karachi",
                city_name="Karachi",
                aqi=100,
                data_source="aqicn",
            )
        ]
        aqicn_client.merge_with_openweather.return_value = StandardObservation(
            timestamp=datetime.now(timezone.utc),
            location_id="karachi",
            city_name="Karachi",
            aqi=100,
            data_source="aqicn",
        )

        manager = APIManager(
            openweather_client=ow_client,
            aqicn_client=aqicn_client,
        )
        # Should not raise — handles error and uses AQICN fallback
        result = manager.fetch_city_data(city_config)
        assert result is not None

    def test_api_manager_init_without_clients(self):
        """API Manager can be initialized without pre-built clients."""
        with patch("src.data.api_manager.get_api_key", return_value=None):
            manager = APIManager()
            assert manager._openweather is not None
            assert manager._aqicn is not None
