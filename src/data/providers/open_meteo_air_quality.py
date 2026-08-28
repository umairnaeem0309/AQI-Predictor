"""
Open-Meteo Air Quality Provider.

Fetches historical hourly air quality data from Open-Meteo's /v1/air-quality endpoint.
No API key required for non-commercial use.

Available data:
- CAMS Global: Aug 2022 to present (45km, 3-hourly → interpolated to hourly)
- CAMS European: 2013 to present (11km, hourly) — Europe only

Variables available:
- pm2_5: PM2.5 concentration (μg/m³)
- pm10: PM10 concentration (μg/m³)
- carbon_monoxide: CO concentration (μg/m³)
- nitrogen_dioxide: NO2 concentration (μg/m³)
- sulphur_dioxide: SO2 concentration (μg/m³)
- ozone: O3 concentration (μg/m³)
- us_aqi: US AQI (for validation; project uses own EPA calculation)

API documentation: https://open-meteo.com/en/docs/air-quality-api
"""

import logging
from typing import Any, Dict, Optional

import pandas as pd

from src.data.providers.base_provider import BaseHistoricalProvider

logger = logging.getLogger(__name__)


class OpenMeteoAirQualityProvider(BaseHistoricalProvider):
    """Provider for Open-Meteo Air Quality API.

    Fetches hourly air quality data from the /v1/air-quality endpoint.
    Supports CAMS Global (Aug 2022+) and CAMS European (2013+) datasets.

    Usage:
        provider = OpenMeteoAirQualityProvider()
        df = provider.fetch_historical(
            latitude=24.86, longitude=67.00,
            location_id="karachi", city_name="Karachi",
            start_date="2022-08-01", end_date="2026-08-27",
        )
    """

    base_url = "https://air-quality-api.open-meteo.com/v1"
    max_days_per_request = 92  # API constraint for air quality
    request_delay = 0.5

    # Internal column names → API variable names
    VARIABLE_MAPPING = {
        "pm25": "pm2_5",
        "pm10": "pm10",
        "co": "carbon_monoxide",
        "no2": "nitrogen_dioxide",
        "so2": "sulphur_dioxide",
        "o3": "ozone",
    }

    # Requested hourly variables
    HOURLY_VARIABLES = [
        "pm2_5",
        "pm10",
        "carbon_monoxide",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "ozone",
        "us_aqi",  # For validation only — project uses own EPA calc
        "us_aqi_pm2_5",  # For validation
        "us_aqi_pm10",  # For validation
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
        """Fetch air quality data chunk from Open-Meteo /v1/air-quality.

        Args:
            latitude: Location latitude.
            longitude: Location longitude.
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            **kwargs:
                domain (str): 'auto' (default), 'cams_europe', 'cams_global'.
                cell_selection (str): 'nearest' (default), 'land', 'sea'.

        Returns:
            Raw API response dictionary.
        """
        domain = kwargs.get("domain", "auto")
        cell_selection = kwargs.get("cell_selection", "nearest")

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(self.HOURLY_VARIABLES),
            "timezone": "GMT",
            "domains": domain,
            "cell_selection": cell_selection,
        }

        url = f"{self.base_url}/air-quality"
        return self._make_request(url, params)

    def _parse_response(
        self,
        raw_json: Dict[str, Any],
        location_id: str,
        city_name: str,
    ) -> pd.DataFrame:
        """Parse Open-Meteo air quality response into DataFrame.

        The response format is:
        {
            "latitude": ...,
            "longitude": ...,
            "hourly": {
                "time": ["2022-08-01T00:00", ...],
                "pm2_5": [15.2, ...],
                "pm10": [25.1, ...],
                ...
            }
        }

        Args:
            raw_json: Raw API response.
            location_id: City identifier.
            city_name: Human-readable city name.

        Returns:
            DataFrame with standardized pollution columns.
        """
        hourly = raw_json.get("hourly", {})
        if not hourly or "time" not in hourly:
            logger.warning("No hourly air quality data in response for %s", city_name)
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

        # Store Open-Meteo US AQI values for validation reference
        us_aqi_values = hourly.get("us_aqi")
        if us_aqi_values is not None and len(us_aqi_values) == n:
            data["us_aqi_open_meteo"] = us_aqi_values
        else:
            data["us_aqi_open_meteo"] = [None] * n

        us_aqi_pm25 = hourly.get("us_aqi_pm2_5")
        if us_aqi_pm25 is not None and len(us_aqi_pm25) == n:
            data["us_aqi_pm25_open_meteo"] = us_aqi_pm25
        else:
            data["us_aqi_pm25_open_meteo"] = [None] * n

        us_aqi_pm10 = hourly.get("us_aqi_pm10")
        if us_aqi_pm10 is not None and len(us_aqi_pm10) == n:
            data["us_aqi_pm10_open_meteo"] = us_aqi_pm10
        else:
            data["us_aqi_pm10_open_meteo"] = [None] * n

        df = pd.DataFrame(data)

        # Add metadata
        df["data_source"] = "open_meteo_air_quality"
        df["provider"] = "open-meteo"

        logger.debug(
            "Parsed %d air quality records for %s",
            len(df),
            city_name,
        )

        return df
