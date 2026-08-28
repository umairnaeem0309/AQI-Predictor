"""
Open-Meteo Historical Weather Provider.

Fetches historical hourly weather data from Open-Meteo's /v1/archive endpoint.
No API key required for non-commercial use.

Available data:
- IFS 9km: 2017 to present (highest resolution)
- ERA5 0.25°: 1940 to present (longest history)
- ERA5-Land 0.1°: 1950 to present (surface conditions)

Variables available:
- temperature_2m: Air temperature at 2m (°C)
- relative_humidity_2m: Relative humidity (%)
- surface_pressure: Surface pressure (hPa)
- wind_speed_10m: Wind speed at 10m (km/h or m/s)
- wind_direction_10m: Wind direction at 10m (°)
- cloud_cover: Total cloud cover (%)
- precipitation: Precipitation (mm)

API documentation: https://open-meteo.com/en/docs/historical-weather-api
"""

import logging
from typing import Any, Dict, Optional

import pandas as pd

from src.data.providers.base_provider import BaseHistoricalProvider

logger = logging.getLogger(__name__)


class OpenMeteoWeatherProvider(BaseHistoricalProvider):
    """Provider for Open-Meteo Historical Weather API.

    Fetches hourly weather data from the /v1/archive endpoint.
    Supports ERA5 (1940+), ERA5-Land (1950+), and IFS (2017+) datasets.

    Usage:
        provider = OpenMeteoWeatherProvider()
        df = provider.fetch_historical(
            latitude=24.86, longitude=67.00,
            location_id="karachi", city_name="Karachi",
            start_date="2021-01-01", end_date="2026-08-27",
        )
    """

    base_url = "https://archive-api.open-meteo.com/v1"
    max_days_per_request = 365  # Open-Meteo supports large ranges
    request_delay = 0.3  # Generous rate limit for free tier

    # Internal column names → API variable names
    VARIABLE_MAPPING = {
        "temperature": "temperature_2m",
        "humidity": "relative_humidity_2m",
        "pressure": "surface_pressure",
        "wind_speed": "wind_speed_10m",
        "wind_direction": "wind_direction_10m",
        "cloud_cover": "cloud_cover",
        "precipitation": "precipitation",
    }

    # Requested variables
    HOURLY_VARIABLES = [
        "temperature_2m",
        "relative_humidity_2m",
        "surface_pressure",
        "wind_speed_10m",
        "wind_direction_10m",
        "cloud_cover",
        "precipitation",
    ]

    def _get_variable_mapping(self) -> Dict[str, str]:
        return self.VARIABLE_MAPPING.copy()

    def _fetch_chunk(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """Fetch weather data chunk from Open-Meteo /v1/archive.

        Args:
            latitude: Location latitude.
            longitude: Location longitude.
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            **kwargs:
                dataset (str): 'best_match' (default), 'era5', 'era5_land', 'ecmwf_ifs'.
                wind_speed_unit (str): 'kmh' (default), 'ms', 'mph'.
                temperature_unit (str): 'celsius' (default), 'fahrenheit'.
                precipitation_unit (str): 'mm' (default), 'inch'.

        Returns:
            Raw API response dictionary.
        """
        dataset = kwargs.get("dataset", "best_match")

        # Map dataset name to endpoint
        endpoint_map = {
            "best_match": "archive",
            "era5": "archive",
            "era5_land": "archive",
            "ecmwf_ifs": "archive",
        }
        endpoint = endpoint_map.get(dataset, "archive")

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(self.HOURLY_VARIABLES),
            "timezone": "GMT",
            "wind_speed_unit": kwargs.get("wind_speed_unit", "ms"),
            "temperature_unit": kwargs.get("temperature_unit", "celsius"),
            "precipitation_unit": kwargs.get("precipitation_unit", "mm"),
        }

        url = f"{self.base_url}/{endpoint}"
        return self._make_request(url, params)

    def _parse_response(
        self,
        raw_json: Dict[str, Any],
        location_id: str,
        city_name: str,
    ) -> pd.DataFrame:
        """Parse Open-Meteo archive response into DataFrame.

        The response format is:
        {
            "latitude": ...,
            "longitude": ...,
            "hourly": {
                "time": ["2021-01-01T00:00", ...],
                "temperature_2m": [5.2, ...],
                "relative_humidity_2m": [85, ...],
                ...
            }
        }

        Args:
            raw_json: Raw API response.
            location_id: City identifier.
            city_name: Human-readable city name.

        Returns:
            DataFrame with standardized weather columns.
        """
        hourly = raw_json.get("hourly", {})
        if not hourly or "time" not in hourly:
            logger.warning("No hourly data in response for %s", city_name)
            return pd.DataFrame()

        timestamps = hourly["time"]
        n = len(timestamps)

        data = {
            "timestamp": pd.to_datetime(timestamps, utc=True),
            "location_id": location_id,
            "city_name": city_name,
        }

        # Map API variables to internal names
        for internal_name, api_name in self.VARIABLE_MAPPING.items():
            values = hourly.get(api_name)
            if values is not None and len(values) == n:
                data[internal_name] = values
            else:
                data[internal_name] = [None] * n
                logger.debug(
                    "Variable %s not available in response for %s",
                    api_name,
                    city_name,
                )

        df = pd.DataFrame(data)

        # Add metadata
        df["data_source"] = "open_meteo_weather"
        df["provider"] = "open-meteo"
        df["dataset"] = raw_json.get("dataset", "unknown")

        # Convert wind speed from km/h to m/s if needed
        # Open-Meteo returns km/h by default, but we requested 'ms'
        # No conversion needed when wind_speed_unit='ms'

        logger.debug(
            "Parsed %d weather records for %s",
            len(df),
            city_name,
        )

        return df
