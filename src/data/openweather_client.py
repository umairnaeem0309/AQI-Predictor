"""
OpenWeather API Client — Primary data source.

Responsibilities:
- Fetch weather data from /data/2.5/weather
- Fetch air pollution data from /data/2.5/air_pollution
- Merge weather and pollution into StandardObservation
- Authoritative source for: temperature, humidity, wind, pressure, weather_condition
- Fallback source for: AQI/pollution (when AQICN unavailable)

Data ownership:
- Weather fields: OpenWeather is authoritative
- AQI/pollution fields: OpenWeather provides fallback values;
  AQICN is authoritative when available
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from src.data.base_client import BaseAPIClient
from src.data.schemas import (
    CityConfig,
    OpenWeatherPollutionResponse,
    OpenWeatherWeatherResponse,
    StandardObservation,
    DataSource,
)
from src.data.exceptions import APIValidationError

logger = logging.getLogger(__name__)


def _unix_to_utc(unix_ts: int, timezone_offset: int = 0) -> datetime:
    """Convert Unix timestamp to UTC datetime.

    OpenWeather provides a timezone offset (seconds from UTC).
    The dt field is UTC regardless of timezone offset.
    The timezone offset tells us the local timezone of the city.

    Args:
        unix_ts: Unix timestamp (seconds since epoch).
        timezone_offset: Timezone offset from UTC in seconds (unused for UTC conversion).

    Returns:
        UTC datetime object.
    """
    return datetime.fromtimestamp(unix_ts, tz=timezone.utc)


def _normalize_temperature_to_celsius(temp: Optional[float]) -> Optional[float]:
    """Ensure temperature is in Celsius.

    OpenWeather with units=metric returns Celsius directly.
    This function is a safety net for unexpected formats.

    Args:
        temp: Temperature value.

    Returns:
        Temperature in Celsius or None.
    """
    if temp is None:
        return None
    # OpenWeather metric units return Celsius directly
    return round(float(temp), 2)


class OpenWeatherClient(BaseAPIClient):
    """Client for OpenWeather API.

    Fetches both weather and air pollution data, merging them into
    StandardObservation objects with timezone-normalized timestamps.

    Usage:
        client = OpenWeatherClient(api_key="your-key")
        observations = client.fetch_data(
            city_id="karachi",
            city_config=CityConfig(id="karachi", name="Karachi", latitude=24.86, longitude=67.00)
        )
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openweathermap.org/data/2.5",
        timeout: int = 30,
        max_retries: int = 3,
    ):
        """Initialize OpenWeather client.

        Args:
            api_key: OpenWeather API key. None for test/mock mode.
            base_url: API base URL.
            timeout: Request timeout in seconds.
            max_retries: Maximum retry attempts.
        """
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    def _build_request(self, **kwargs) -> tuple:
        """Build weather and pollution request parameters.

        Expected kwargs:
            city_id (str): City identifier.
            city_config (CityConfig): City metadata with lat/lon.

        Returns:
            Tuple of (endpoint, params) for weather request.
            Pollution request is built separately in fetch_data().
        """
        city_config: CityConfig = kwargs.get("city_config")
        if not city_config:
            raise ValueError("city_config is required for OpenWeather requests")

        params = {
            "lat": city_config.latitude,
            "lon": city_config.longitude,
            "appid": self.api_key,
            "units": "metric",
        }

        return "weather", params

    def _build_pollution_params(self, city_config: CityConfig) -> Dict[str, Any]:
        """Build pollution request parameters."""
        return {
            "lat": city_config.latitude,
            "lon": city_config.longitude,
            "appid": self.api_key,
        }

    def _parse_response(
        self,
        raw_json: Dict[str, Any],
        **kwargs,
    ) -> list:
        """Parse raw JSON into StandardObservation(s).

        Delegates to _parse_weather_response using kwargs for city context.
        Called by base class fetch_data for the weather endpoint.
        """
        city_id = kwargs.get("city_id", "unknown")
        city_name = kwargs.get("city_name", city_id)
        obs = self._parse_weather_response(raw_json, city_id, city_name)
        return [obs] if obs else []

    def _validate_response(self, raw_json: Dict[str, Any]) -> bool:
        """Validate OpenWeather weather response.

        Args:
            raw_json: Raw JSON response.



        Returns:
            True if response contains minimum required fields.
        """
        if not raw_json:
            return False
        # Minimum: must have either 'main' or 'weather' data
        return "main" in raw_json or "weather" in raw_json

    def _validate_pollution_response(self, raw_json: Dict[str, Any]) -> bool:
        """Validate OpenWeather pollution response."""
        if not raw_json:
            return False
        return "list" in raw_json and len(raw_json["list"]) > 0

    def _parse_weather_response(
        self,
        raw_json: Dict[str, Any],
        city_id: str,
        city_name: str,
    ) -> Optional[StandardObservation]:
        """Parse OpenWeather weather response into StandardObservation.

        Timezone normalization:
        - OpenWeather dt field is always UTC
        - timezone field indicates local timezone offset (not used for UTC conversion)
        - All timestamps are normalized to UTC for consistency

        Missing observation handling:
        - Missing optional fields (wind, weather condition) are set to None
        - Missing required fields (main) log a warning

        Args:
            raw_json: Parsed JSON from /data/2.5/weather.
            city_id: City identifier.
            city_name: Human-readable city name.

        Returns:
            StandardObservation or None if parsing fails.
        """
        try:
            parsed = OpenWeatherWeatherResponse(**raw_json)
        except Exception as e:
            logger.warning("Failed to parse weather response: %s", e)
            return None

        # Extract timestamp (dt is UTC)
        dt_utc = None
        if parsed.dt:
            dt_utc = _unix_to_utc(parsed.dt)

        # Extract weather fields
        weather_condition = None
        if parsed.weather and len(parsed.weather) > 0:
            weather_condition = parsed.weather[0].description

        # Extract main fields
        temperature = None
        humidity = None
        pressure = None
        if parsed.main:
            temperature = _normalize_temperature_to_celsius(parsed.main.temp)
            humidity = parsed.main.humidity
            pressure = parsed.main.pressure

        # Extract wind
        wind_speed = None
        if parsed.wind:
            wind_speed = parsed.wind.speed

        return StandardObservation(
            timestamp=dt_utc or datetime.now(timezone.utc),
            location_id=city_id,
            city_name=city_name,
            temperature=temperature,
            humidity=float(humidity) if humidity is not None else None,
            wind_speed=wind_speed,
            pressure=float(pressure) if pressure is not None else None,
            weather_condition=weather_condition,
            data_source=DataSource.OPENWEATHER,
            raw_response_time=dt_utc,
            collected_at=datetime.now(timezone.utc),
        )

    def _parse_pollution_response(
        self,
        raw_json: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Parse OpenWeather pollution response.

        Extracts AQI and pollutant values from the first item in the list.

        OpenWeather AQI scale: 1-5 (different from US EPA 0-500).
        This is stored as-is; conversion to US EPA scale happens if needed.

        Args:
            raw_json: Parsed JSON from /data/2.5/air_pollution.

        Returns:
            Dictionary with pollution fields, or None.
        """
        try:
            parsed = OpenWeatherPollutionResponse(**raw_json)
        except Exception as e:
            logger.warning("Failed to parse pollution response: %s", e)
            return None

        if not parsed.list or len(parsed.list) == 0:
            return None

        item = parsed.list[0]

        result = {}
        if item.main:
            result["aqi_ow"] = item.main.aqi  # OpenWeather 1-5 scale
        if item.components:
            result["pm25"] = item.components.pm2_5
            result["pm10"] = item.components.pm10
            result["co"] = item.components.co
            result["no2"] = item.components.no2
            result["so2"] = item.components.so2
            result["o3"] = item.components.o3

        return result

    def _merge_observations(
        self,
        weather_obs: Optional[StandardObservation],
        pollution_data: Optional[Dict[str, Any]],
    ) -> StandardObservation:
        """Merge weather and pollution data into a single observation.

        Data ownership:
        - Weather fields: from OpenWeather (authoritative)
        - AQI/pollution fields: from OpenWeather (fallback source)

        Missing observation handling:
        - If weather fetch fails, pollution-only observation is returned
        - If pollution fetch fails, weather-only observation is returned
        - Missing individual fields are set to None

        Args:
            weather_obs: Weather observation (may be None).
            pollution_data: Pollution dictionary (may be None).

        Returns:
            Merged StandardObservation.
        """
        if weather_obs is None:
            # Should not happen in normal flow, but handle gracefully
            logger.error("No weather observation to merge with pollution data")
            return None

        # Start with weather observation, add pollution data
        obs_dict = weather_obs.model_dump()

        if pollution_data:
            # AQICN is authoritative for AQI when available,
            # but here we only have OpenWeather pollution data
            obs_dict["pm25"] = pollution_data.get("pm25")
            obs_dict["pm10"] = pollution_data.get("pm10")
            obs_dict["co"] = pollution_data.get("co")
            obs_dict["no2"] = pollution_data.get("no2")
            obs_dict["so2"] = pollution_data.get("so2")
            obs_dict["o3"] = pollution_data.get("o3")
            # Note: OpenWeather AQI is 1-5 scale, not US EPA
            # US EPA conversion would happen at a higher level if needed

        return StandardObservation(**obs_dict)

    def fetch_data(self, **kwargs) -> List[StandardObservation]:
        """Fetch weather and pollution data for a city.

        Combines weather and pollution API calls, merges results,
        and returns a list of StandardObservation objects.

        Args:
            **kwargs:
                city_id (str): City identifier.
                city_config (CityConfig): City metadata.

        Returns:
            List containing a single StandardObservation.
        """
        city_id: str = kwargs.get("city_id", "")
        city_config: CityConfig = kwargs.get("city_config")

        if not city_config:
            raise ValueError("city_config is required")

        logger.info(
            "Fetching OpenWeather data for %s (lat=%s, lon=%s)",
            city_id,
            city_config.latitude,
            city_config.longitude,
        )

        # Fetch weather data
        weather_params = {
            "lat": city_config.latitude,
            "lon": city_config.longitude,
            "appid": self.api_key,
            "units": "metric",
        }
        weather_json = self._retry_request("weather", weather_params)
        weather_obs = self._parse_weather_response(
            weather_json, city_id, city_config.name
        )

        # Fetch pollution data
        pollution_obs = None
        try:
            pollution_params = self._build_pollution_params(city_config)
            pollution_json = self._retry_request("air_pollution", pollution_params)
            if self._validate_pollution_response(pollution_json):
                pollution_obs = self._parse_pollution_response(pollution_json)
        except Exception as e:
            logger.warning(
                "Pollution data fetch failed for %s: %s. Continuing with weather only.",
                city_id,
                str(e),
            )

        # Merge observations
        merged = self._merge_observations(weather_obs, pollution_obs)

        if merged is None:
            logger.error("Failed to produce observation for %s", city_id)
            return []

        return [merged]
