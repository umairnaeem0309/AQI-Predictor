"""
AQICN/WAQI API Client — Fallback data source.

Responsibilities:
- Fetch AQI and pollutant data from AQICN API
- Detect data staleness (AQICN updates infrequently)
- Authoritative source for: AQI (US EPA scale), pm25, pm10, co, no2, so2, o3
- Limited weather data (set to None; OpenWeather is authoritative)

Data ownership:
- AQI/pollution fields: AQICN is authoritative when available
- Weather fields: Not available from AQICN (set to None)
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from src.data.base_client import BaseAPIClient
from src.data.schemas import (
    CityConfig,
    AQICNResponse,
    StandardObservation,
    DataSource,
)
from src.data.exceptions import APIValidationError, StalenessWarning

logger = logging.getLogger(__name__)

# Maximum acceptable data age in hours before warning
DEFAULT_MAX_STALENESS_HOURS = 2.0


def _parse_aqicn_timestamp(time_data: Optional[Dict[str, Any]]) -> Optional[datetime]:
    """Parse AQICN timestamp to UTC datetime.

    AQICN provides:
    - iso: ISO 8601 string with explicit timezone offset (e.g. 2026-08-26T11:00:00+05:00)
    - v: Unix timestamp (NOTE: AQICN Unix timestamps are unreliable — they often
      represent local clock time misinterpreted as UTC, causing a timezone-offset
      discrepancy. Always prefer the ISO string with explicit offset.)
    - tz: Timezone offset string (e.g. '+05:00')

    IMPORTANT: The ISO string is the authoritative timestamp source.
    The Unix timestamp (time.v) MUST NOT be used as primary because AQICN
    encodes local clock time into the Unix value without applying the offset.

    Args:
        time_data: AQICN time dictionary.

    Returns:
        UTC datetime or None.
    """
    if not time_data:
        return None

    # PREFER ISO string — it contains the explicit timezone offset
    # AQICN Unix timestamps are unreliable (local time stored as UTC)
    if "iso" in time_data and time_data["iso"]:
        try:
            dt = datetime.fromisoformat(time_data["iso"])
            # If timezone-aware, convert to UTC
            if dt.tzinfo is not None:
                return dt.astimezone(timezone.utc)
            # If naive, treat as UTC
            return dt.replace(tzinfo=timezone.utc)
        except Exception as e:
            logger.warning("Failed to parse AQICN ISO timestamp '%s': %s", time_data["iso"], e)

    # Fallback to Unix timestamp only if ISO unavailable
    # Log warning because this path uses the unreliable AQICN Unix value
    if "v" in time_data and time_data["v"]:
        logger.warning(
            "AQICN ISO timestamp unavailable, falling back to Unix timestamp. "
            "Note: AQICN Unix timestamps may be offset by the timezone value."
        )
        return datetime.fromtimestamp(time_data["v"], tz=timezone.utc)

    return None


def _extract_iaqi_value(iaqi: Optional[Dict[str, Any]], key: str) -> Optional[float]:
    """Extract a value from AQICN iaqi dictionary.

    AQICN iaqi format: {"pm25": {"v": 35.5}, "pm10": {"v": 50.0}}

    Args:
        iaqi: AQICN iaqi dictionary.
        key: Key to extract (e.g., "pm25", "pm10").

    Returns:
        Numeric value or None.
    """
    if not iaqi or key not in iaqi:
        return None

    value = iaqi[key]
    if isinstance(value, dict) and "v" in value:
        try:
            return float(value["v"])
        except (ValueError, TypeError):
            return None

    return None


# Bound station IDs for Pakistani cities — AQICN UID-based stations
# Verified: these are actual monitoring stations in Pakistan
# City-level feeds may return different/stale data; bound stations are preferred
#
# IMPORTANT: AQICN Pakistani stations may have very infrequent updates.
# The ISO timestamp is authoritative; never use the Unix timestamp as primary
# because AQICN encodes local clock time as if it were UTC.
CITY_STATION_MAP = {
    "karachi": "@11790",    # Karachi US Consulate, Pakistan (uid=11790)
    "lahore": "@11765",     # Lahore US Embassy, Pakistan (uid=11765)
    "islamabad": "@11739",  # Islamabad US Embassy, Pakistan (uid=11739)
}


class AQICNClient(BaseAPIClient):
    """Client for AQICN/WAQI API.

    Fetches AQI and pollutant data. Includes staleness detection
    because AQICN ground stations update infrequently.

    Uses bound station IDs (e.g. @7393) instead of city-level feeds
    (/feed/karachi/) because city-level feeds return stale cached data.

    Usage:
        client = AQICNClient(api_key="your-token")
        observations = client.fetch_data(city_id="karachi")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.waqi.info",
        timeout: int = 30,
        max_retries: int = 3,
        max_staleness_hours: float = DEFAULT_MAX_STALENESS_HOURS,
    ):
        """Initialize AQICN client.

        Args:
            api_key: AQICN/WAQI API token. None for test/mock mode.
            base_url: API base URL.
            timeout: Request timeout in seconds.
            max_retries: Maximum retry attempts.
            max_staleness_hours: Maximum acceptable data age before warning.
        """
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.max_staleness_hours = max_staleness_hours

    def _build_request(self, **kwargs) -> tuple:
        """Build AQICN request parameters.

        Uses bound station IDs for known cities to get fresh data.
        Falls back to city-level feed for unknown cities.

        Expected kwargs:
            city_id (str): City identifier (used in AQICN path).

        Returns:
            Tuple of (endpoint, params).
        """
        city_id: str = kwargs.get("city_id", "")
        # Use bound station ID if available for fresh data
        station_id = CITY_STATION_MAP.get(city_id.lower(), city_id)
        endpoint = f"feed/{station_id}"
        params = {"token": self.api_key}
        return endpoint, params

    def _validate_response(self, raw_json: Dict[str, Any]) -> bool:
        """Validate AQICN response structure.

        Args:
            raw_json: Raw JSON response.

        Returns:
            True if response is valid.
        """
        if not raw_json:
            return False

        # Check status field
        status = raw_json.get("status")
        if status != "ok":
            logger.warning(
                "AQICN response status is '%s': %s",
                status,
                raw_json.get("data_message", "No message"),
            )
            return False

        # Check data field exists
        data = raw_json.get("data")
        if not data:
            return False

        return True

    def _check_staleness(self, time_data: Optional[Dict[str, Any]]) -> Optional[str]:
        """Check if AQICN data is stale.

        AQICN ground stations may update only every few hours.
        Stale data should trigger a warning but is still returned.

        Args:
            time_data: AQICN time dictionary.

        Returns:
            Staleness warning message or None.
        """
        dt = _parse_aqicn_timestamp(time_data)
        if dt is None:
            return None

        now = datetime.now(timezone.utc)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        age_hours = (now - dt).total_seconds() / 3600

        if age_hours > self.max_staleness_hours:
            warning = (
                f"AQICN data is stale: {age_hours:.1f} hours old "
                f"(threshold: {self.max_staleness_hours} hours)"
            )
            logger.warning(warning)
            return warning

        return None

    def _parse_response(
        self,
        raw_json: Dict[str, Any],
        **kwargs,
    ) -> List[StandardObservation]:
        """Parse AQICN response into StandardObservation.

        AQICN provides:
        - AQI (US EPA scale, 0-500)
        - Pollutant values via iaqi dictionary
        - Station time (may be stale)
        - City name and coordinates

        Does NOT provide:
        - Temperature, humidity, wind, pressure, weather condition
        (these are set to None; OpenWeather is authoritative)

        Args:
            raw_json: Parsed JSON from AQICN API.
            **kwargs: Must include city_id.

        Returns:
            List containing a single StandardObservation.
        """
        city_id: str = kwargs.get("city_id", "")

        parsed = AQICNResponse(**raw_json)
        data = parsed.data

        if not data:
            logger.warning("AQICN response has no data for %s", city_id)
            return []

        # Parse timestamp — convert Pydantic model to dict for parser
        time_dict = data.time.model_dump() if data.time else None
        dt_utc = _parse_aqicn_timestamp(time_dict)
        if dt_utc is None:
            dt_utc = datetime.now(timezone.utc)
            logger.warning("No timestamp in AQICN response for %s, using current time", city_id)

        # Check staleness
        staleness_warning = self._check_staleness(time_dict)
        if staleness_warning:
            # Data is stale but still usable
            pass

        # Extract AQI (US EPA scale)
        aqi = data.aqi

        # Extract pollutant values from iaqi
        iaqi = data.iaqi
        pm25 = _extract_iaqi_value(iaqi, "pm25")
        pm10 = _extract_iaqi_value(iaqi, "pm10")
        co = _extract_iaqi_value(iaqi, "co")
        no2 = _extract_iaqi_value(iaqi, "no2")
        so2 = _extract_iaqi_value(iaqi, "so2")
        o3 = _extract_iaqi_value(iaqi, "o3")

        # Get city name from response
        city_name = city_id.title()
        if data.city and "name" in data.city:
            city_name = data.city["name"]

        # Determine training validity based on source freshness
        collected_at = datetime.now(timezone.utc)
        is_training_valid = True
        staleness_reason = None
        if staleness_warning:
            is_training_valid = False
            staleness_reason = staleness_warning
            logger.warning(
                "AQICN observation for %s marked as NOT training-valid: %s",
                city_id, staleness_reason,
            )

        observation = StandardObservation(
            timestamp=dt_utc,
            location_id=city_id,
            city_name=city_name,
            # Weather fields: not available from AQICN (OpenWeather is authoritative)
            temperature=None,
            humidity=None,
            wind_speed=None,
            pressure=None,
            weather_condition=None,
            # AQI/pollution fields: AQICN is authoritative
            aqi=aqi,
            pm25=pm25,
            pm10=pm10,
            co=co,
            no2=no2,
            so2=so2,
            o3=o3,
            data_source=DataSource.AQICN,
            raw_response_time=dt_utc,
            collected_at=collected_at,
            is_training_valid=is_training_valid,
            staleness_reason=staleness_reason,
        )

        return [observation]

    def merge_with_openweather(
        self,
        aqicn_obs: StandardObservation,
        openweather_obs: StandardObservation,
    ) -> StandardObservation:
        """Merge AQICN observation with OpenWeather observation.

        Data ownership rules:
        - Weather fields: from OpenWeather (authoritative)
        - AQI/pollution fields: from AQICN (authoritative when available)
        - If AQICN field is None, fall back to OpenWeather value

        Args:
            aqicn_obs: AQICN observation (authoritative for AQI/pollution).
            openweather_obs: OpenWeather observation (authoritative for weather).

        Returns:
            Merged StandardObservation.
        """
        # Start with OpenWeather data (authoritative for weather)
        merged = openweather_obs.model_dump()
        # Indicate both sources contributed
        merged["data_source"] = DataSource.OPENWEATHER_AQICN.value

        # Override AQI/pollution with AQICN values where available
        aqicn_dict = aqicn_obs.model_dump()

        for field in ["aqi", "pm25", "pm10", "co", "no2", "so2", "o3"]:
            aqicn_value = aqicn_dict.get(field)
            if aqicn_value is not None:
                merged[field] = aqicn_value
            # If AQICN value is None, keep OpenWeather value

        # Use AQICN timestamp if it's more recent
        if aqicn_obs.timestamp and openweather_obs.timestamp:
            if aqicn_obs.timestamp > openweather_obs.timestamp:
                merged["raw_response_time"] = aqicn_obs.raw_response_time

        # Propagate freshness metadata:
        # If AQICN source is stale, the merged observation is NOT training-valid
        merged["is_training_valid"] = (
            openweather_obs.is_training_valid and aqicn_obs.is_training_valid
        )
        if aqicn_obs.staleness_reason:
            merged["staleness_reason"] = aqicn_obs.staleness_reason

        # collected_at = time of the most recent API call
        if openweather_obs.collected_at and aqicn_obs.collected_at:
            merged["collected_at"] = max(openweather_obs.collected_at, aqicn_obs.collected_at)
        elif openweather_obs.collected_at:
            merged["collected_at"] = openweather_obs.collected_at
        elif aqicn_obs.collected_at:
            merged["collected_at"] = aqicn_obs.collected_at

        return StandardObservation(**merged)
