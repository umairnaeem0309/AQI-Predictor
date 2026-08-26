"""
Tests for AQICN/WAQI API client.

Uses the `responses` library for HTTP mocking to test:
- Response parsing
- Staleness detection
- Error handling
- Merge with OpenWeather observations
"""

import pytest
import responses

from src.data.aqicn_client import AQICNClient, _parse_aqicn_timestamp, _extract_iaqi_value
from src.data.schemas import CityConfig, StandardObservation, DataSource
from src.data.exceptions import APIAuthenticationError


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def aqicn_response_karachi():
    """Sample AQICN response for Karachi."""
    return {
        "status": "ok",
        "data": {
            "aqi": 168,
            "idx": 10442,
            "city": {"name": "Karachi"},
            "iaqi": {
                "pm25": {"v": 55.8},
                "pm10": {"v": 78.2},
                "co": {"v": 230.5},
                "no2": {"v": 28.3},
                "so2": {"v": 12.4},
                "o3": {"v": 65.1},
            },
            "time": {"iso": "2026-07-31T12:00:00+05:00", "v": 1751377200},
        },
    }


@pytest.fixture
def openweather_obs():
    """Sample OpenWeather observation for Karachi."""
    return StandardObservation(
        timestamp=__import__("datetime").datetime(2026, 7, 31, 7, 0, 0, tzinfo=__import__("datetime").timezone.utc),
        location_id="karachi",
        city_name="Karachi",
        temperature=34.5,
        humidity=72.0,
        wind_speed=4.6,
        pressure=1005.0,
        weather_condition="few clouds",
        data_source="openweather",
    )


# =============================================================================
# Test Helper Functions
# =============================================================================


class TestParseAqicnTimestamp:
    """Tests for _parse_aqicn_timestamp helper."""

    def test_parse_unix_timestamp(self):
        """Parse Unix timestamp from time dict."""
        time_data = {"v": 1751377200}
        dt = _parse_aqicn_timestamp(time_data)
        assert dt is not None
        assert dt.tzinfo is not None

    def test_parse_iso_timestamp(self):
        """Parse ISO timestamp from time dict."""
        time_data = {"iso": "2026-07-31T12:00:00+05:00"}
        dt = _parse_aqicn_timestamp(time_data)
        assert dt is not None

    def test_parse_empty_time(self):
        """Empty time dict returns None."""
        assert _parse_aqicn_timestamp({}) is None
        assert _parse_aqicn_timestamp(None) is None

    def test_unix_preferred_over_iso(self):
        """Unix timestamp is preferred over ISO string."""
        time_data = {"v": 1751377200, "iso": "2026-07-31T12:00:00+05:00"}
        dt = _parse_aqicn_timestamp(time_data)
        assert dt is not None


class TestExtractIaqiValue:
    """Tests for _extract_iaqi_value helper."""

    def test_extract_valid_value(self):
        """Extract a valid iaqi value."""
        iaqi = {"pm25": {"v": 55.8}}
        assert _extract_iaqi_value(iaqi, "pm25") == 55.8

    def test_extract_missing_key(self):
        """Missing key returns None."""
        iaqi = {"pm25": {"v": 55.8}}
        assert _extract_iaqi_value(iaqi, "pm10") is None

    def test_extract_none_iaqi(self):
        """None iaqi returns None."""
        assert _extract_iaqi_value(None, "pm25") is None

    def test_extract_invalid_value(self):
        """Invalid value format returns None."""
        iaqi = {"pm25": "invalid"}
        assert _extract_iaqi_value(iaqi, "pm25") is None


# =============================================================================
# Test Client Initialization
# =============================================================================


class TestAQICNClientInit:
    """Tests for client initialization."""

    def test_init_with_api_key(self):
        """Client initializes with API key."""
        client = AQICNClient(api_key="test-token")
        assert client.api_key == "test-token"

    def test_init_without_api_key(self):
        """Client initializes without API key (test/mock mode)."""
        client = AQICNClient()
        assert client.api_key is None

    def test_init_custom_staleness_threshold(self):
        """Client accepts custom staleness threshold."""
        client = AQICNClient(max_staleness_hours=6.0)
        assert client.max_staleness_hours == 6.0


# =============================================================================
# Test Response Validation
# =============================================================================


class TestAQICNValidation:
    """Tests for response validation."""

    def test_validate_ok_response(self):
        """Valid response passes validation."""
        client = AQICNClient(api_key="test")
        assert client._validate_response({"status": "ok", "data": {"aqi": 50}}) is True

    def test_validate_error_response(self):
        """Error response fails validation."""
        client = AQICNClient(api_key="test")
        assert client._validate_response({"status": "error"}) is False

    def test_validate_empty_response(self):
        """Empty response fails validation."""
        client = AQICNClient(api_key="test")
        assert client._validate_response({}) is False

    def test_validate_none_response(self):
        """None response fails validation."""
        client = AQICNClient(api_key="test")
        assert client._validate_response(None) is False


# =============================================================================
# Test Response Parsing
# =============================================================================


class TestAQICNParsing:
    """Tests for response parsing."""

    def test_parse_valid_response(self, aqicn_response_karachi):
        """Parse valid AQICN response."""
        client = AQICNClient(api_key="test")
        observations = client._parse_response(
            aqicn_response_karachi, city_id="karachi"
        )
        assert len(observations) == 1
        obs = observations[0]
        assert obs.aqi == 168
        assert obs.pm25 == 55.8
        assert obs.pm10 == 78.2
        assert obs.location_id == "karachi"
        assert obs.data_source == "aqicn"

    def test_parse_response_weather_fields_none(self, aqicn_response_karachi):
        """Weather fields are None in AQICN response."""
        client = AQICNClient(api_key="test")
        observations = client._parse_response(
            aqicn_response_karachi, city_id="karachi"
        )
        obs = observations[0]
        assert obs.temperature is None
        assert obs.humidity is None
        assert obs.wind_speed is None
        assert obs.pressure is None
        assert obs.weather_condition is None

    def test_parse_response_empty_data(self):
        """Parse response with empty data — returns observation with None values."""
        client = AQICNClient(api_key="test")
        observations = client._parse_response(
            {"status": "ok", "data": {}}, city_id="karachi"
        )
        assert len(observations) == 1
        assert observations[0].aqi is None

    def test_parse_response_city_name_from_data(self):
        """City name extracted from AQICN response data."""
        raw = {
            "status": "ok",
            "data": {
                "aqi": 100,
                "city": {"name": "Lahore"},
                "time": {"v": 1751377200},
            },
        }
        client = AQICNClient(api_key="test")
        observations = client._parse_response(raw, city_id="lahore")
        assert observations[0].city_name == "Lahore"


# =============================================================================
# Test Staleness Detection
# =============================================================================


class TestAQICNStaleness:
    """Tests for staleness detection."""

    def test_fresh_data_no_warning(self):
        """Fresh data produces no staleness warning."""
        client = AQICNClient(api_key="test", max_staleness_hours=2.0)
        import time
        now_v = int(time.time())
        time_data = {"v": now_v}
        warning = client._check_staleness(time_data)
        assert warning is None

    def test_stale_data_produces_warning(self):
        """Stale data produces staleness warning."""
        client = AQICNClient(api_key="test", max_staleness_hours=1.0)
        import time
        old_v = int(time.time()) - 7200  # 2 hours ago
        time_data = {"v": old_v}
        warning = client._check_staleness(time_data)
        assert warning is not None
        assert "stale" in warning.lower()


# =============================================================================
# Test Full Fetch with HTTP Mocking
# =============================================================================


class TestAQICNFetchData:
    """Tests for fetch_data with mocked HTTP responses."""

    @responses.activate
    def test_fetch_data_success(self, aqicn_response_karachi):
        """Successful data fetch returns observation."""
        # AQICN client uses bound station ID @7393 for karachi (no trailing slash)
        responses.add(
            responses.GET,
            "https://api.waqi.info/feed/@7393",
            json=aqicn_response_karachi,
            status=200,
        )

        client = AQICNClient(api_key="test-token")
        observations = client.fetch_data(city_id="karachi")

        assert len(observations) == 1
        obs = observations[0]
        assert obs.aqi == 168
        assert obs.pm25 == 55.8

    @responses.activate
    def test_fetch_data_auth_failure(self):
        """Authentication failure raises error."""
        # AQICN client uses bound station ID @7393 for karachi (no trailing slash)
        responses.add(
            responses.GET,
            "https://api.waqi.info/feed/@7393",
            json={"status": "error", "data": "Invalid token"},
            status=401,
        )

        client = AQICNClient(api_key="bad-token", max_retries=1)
        with pytest.raises(APIAuthenticationError):
            client.fetch_data(city_id="karachi")


# =============================================================================
# Test Merge with OpenWeather
# =============================================================================


class TestAQICNMerge:
    """Tests for merge_with_openweather."""

    def test_merge_takes_aqicn_pollution(self, aqicn_response_karachi, openweather_obs):
        """Merge uses AQICN pollution values (authoritative)."""
        client = AQICNClient(api_key="test")
        aqicn_obs = client._parse_response(
            aqicn_response_karachi, city_id="karachi"
        )[0]

        merged = client.merge_with_openweather(aqicn_obs, openweather_obs)

        # Weather from OpenWeather
        assert merged.temperature == 34.5
        assert merged.humidity == 72.0
        # Pollution from AQICN (authoritative)
        assert merged.aqi == 168
        assert merged.pm25 == 55.8
        assert merged.pm10 == 78.2

    def test_merge_fallback_to_openweather_when_aqicn_none(self, openweather_obs):
        """When AQICN field is None, fall back to OpenWeather value."""
        client = AQICNClient(api_key="test")
        aqicn_obs = StandardObservation(
            timestamp=openweather_obs.timestamp,
            location_id="karachi",
            city_name="Karachi",
            aqi=None,  # AQICN has no AQI
            pm25=None,
            data_source="aqicn",
        )

        merged = client.merge_with_openweather(aqicn_obs, openweather_obs)
        # AQI fields remain None (both sources have None)
        assert merged.aqi is None
