"""
API Manager — Orchestrates multi-source data collection.

Responsibilities:
- Coordinate OpenWeather and AQICN client calls
- Merge results following data ownership rules
- Run validation pipeline
- Generate API audit trail (raw response storage)
- Handle fallback when primary source fails

Does NOT contain: HTTP logic, parsing logic, or client implementation.
All HTTP/parsing is delegated to the respective API clients.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.config import PROJECT_ROOT, get_api_key, mask_key
from src.data.openweather_client import OpenWeatherClient
from src.data.aqicn_client import AQICNClient
from src.data.schemas import CityConfig, StandardObservation, DataSource
from src.data.validators import full_validation, drop_duplicates
from src.data.exceptions import APIClientError

logger = logging.getLogger(__name__)

# Raw data storage path
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
API_AUDIT_DIR = RAW_DATA_DIR / "api_audit"


def _ensure_audit_directory() -> None:
    """Create API audit directory if it doesn't exist."""
    API_AUDIT_DIR.mkdir(parents=True, exist_ok=True)


def _save_api_audit(
    source: str,
    city_id: str,
    response_data: dict,
    timestamp: Optional[datetime] = None,
) -> Path:
    """Save raw API response for audit trail.

    Args:
        source: API source name (openweather, aqicn).
        city_id: City identifier.
        response_data: Raw JSON response data.
        timestamp: Response timestamp. Defaults to now.

    Returns:
        Path to saved audit file.
    """
    _ensure_audit_directory()

    if timestamp is None:
        timestamp = datetime.now(timezone.utc)

    filename = f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{source}_{city_id}.json"
    filepath = API_AUDIT_DIR / filename

    with open(filepath, "w") as f:
        json.dump(response_data, f, indent=2, default=str)

    logger.debug("Saved API audit: %s", filepath)
    return filepath


class APIManager:
    """Orchestrates data collection from multiple API sources.

    Follows data ownership rules:
    - OpenWeather: authoritative for weather fields
    - AQICN: authoritative for AQI/pollution fields (US EPA scale)

    If OpenWeather weather fails:
    - AQICN may provide AQI/pollution only
    - Weather fields remain None

    If AQICN is unavailable:
    - AQI/pollution fields remain None
    - Do NOT use OpenWeather 1-5 AQI scale as substitute

    Usage:
        manager = APIManager()
        df = manager.fetch_all_cities()
    """

    def __init__(
        self,
        openweather_client: Optional[OpenWeatherClient] = None,
        aqicn_client: Optional[AQICNClient] = None,
    ):
        """Initialize API manager.

        Args:
            openweather_client: Pre-configured OpenWeather client.
                If None, creates one from environment credentials.
            aqicn_client: Pre-configured AQICN client.
                If None, creates one from environment credentials.
        """
        if openweather_client is not None:
            self._openweather = openweather_client
        else:
            ow_key = get_api_key("openweather")
            self._openweather = OpenWeatherClient(api_key=ow_key)

        if aqicn_client is not None:
            self._aqicn = aqicn_client
        else:
            aqicn_key = get_api_key("aqicn")
            self._aqicn = AQICNClient(api_key=aqicn_key)

        logger.info(
            "API Manager initialized — OpenWeather: %s, AQICN: %s",
            "available" if self._openweather.api_key else "unavailable",
            "available" if self._aqicn.api_key else "unavailable",
        )

    def fetch_city_data(self, city_config: CityConfig) -> Optional[StandardObservation]:
        """Fetch and merge data for a single city.

        Flow: client calls → merge → validation → output

        Args:
            city_config: City metadata (id, name, lat, lon).

        Returns:
            Merged StandardObservation or None if all sources fail.
        """
        logger.info("Fetching data for %s", city_config.name)

        openweather_obs = None
        aqicn_obs = None

        # --- OpenWeather call ---
        try:
            ow_observations = self._openweather.fetch_data(
                city_id=city_config.id,
                city_config=city_config,
            )
            if ow_observations:
                openweather_obs = ow_observations[0]
                logger.info(
                    "OpenWeather data received for %s — temp=%s, humidity=%s",
                    city_config.name,
                    openweather_obs.temperature,
                    openweather_obs.humidity,
                )
        except APIClientError as e:
            logger.warning(
                "OpenWeather failed for %s: %s", city_config.name, str(e)
            )
        except Exception as e:
            logger.error(
                "Unexpected error from OpenWeather for %s: %s",
                city_config.name,
                str(e),
            )

        # --- AQICN call ---
        try:
            aqicn_observations = self._aqicn.fetch_data(
                city_id=city_config.id,
            )
            if aqicn_observations:
                aqicn_obs = aqicn_observations[0]
                logger.info(
                    "AQICN data received for %s — AQI=%s",
                    city_config.name,
                    aqicn_obs.aqi,
                )
        except APIClientError as e:
            logger.warning(
                "AQICN failed for %s: %s", city_config.name, str(e)
            )
        except Exception as e:
            logger.error(
                "Unexpected error from AQICN for %s: %s",
                city_config.name,
                str(e),
            )

        # --- Merge following ownership rules ---
        merged = self._merge_sources(openweather_obs, aqicn_obs, city_config)

        if merged is None:
            logger.error(
                "All sources failed for %s — no data available", city_config.name
            )
            return None

        logger.info(
            "Merged observation for %s — source=%s, AQI=%s, temp=%s",
            city_config.name,
            merged.data_source,
            merged.aqi,
            merged.temperature,
        )

        return merged

    def _merge_sources(
        self,
        openweather_obs: Optional[StandardObservation],
        aqicn_obs: Optional[StandardObservation],
        city_config: CityConfig,
    ) -> Optional[StandardObservation]:
        """Merge observations from OpenWeather and AQICN.

        Data ownership rules:
        - Weather fields: from OpenWeather (authoritative)
        - AQI/pollution fields: from AQICN (authoritative)
        - If AQICN field is None, OpenWeather value is NOT used as substitute
          (OpenWeather AQI is 1-5 scale, different from US EPA)

        If both sources fail, returns None.

        Args:
            openweather_obs: OpenWeather observation (may be None).
            aqicn_obs: AQICN observation (may be None).
            city_config: City metadata.

        Returns:
            Merged StandardObservation or None.
        """
        if openweather_obs is None and aqicn_obs is None:
            return None

        if openweather_obs is not None and aqicn_obs is not None:
            # Both sources available — merge with ownership rules
            merged = self._aqicn.merge_with_openweather(aqicn_obs, openweather_obs)
            return merged

        if openweather_obs is not None:
            # Only OpenWeather available — weather fields only, no AQI
            # Do NOT use OpenWeather 1-5 AQI as US EPA AQI
            obs_dict = openweather_obs.model_dump()
            obs_dict["aqi"] = None  # Explicitly: OpenWeather AQI is not US EPA
            obs_dict["pm25"] = None
            obs_dict["pm10"] = None
            obs_dict["co"] = None
            obs_dict["no2"] = None
            obs_dict["so2"] = None
            obs_dict["o3"] = None
            return StandardObservation(**obs_dict)

        if aqicn_obs is not None:
            # Only AQICN available — AQI/pollution only, no weather
            obs_dict = aqicn_obs.model_dump()
            obs_dict["temperature"] = None
            obs_dict["humidity"] = None
            obs_dict["wind_speed"] = None
            obs_dict["pressure"] = None
            obs_dict["weather_condition"] = None
            return StandardObservation(**obs_dict)

        return None

    def fetch_all_cities(
        self,
        city_configs: Optional[List[CityConfig]] = None,
    ) -> pd.DataFrame:
        """Fetch data for all configured cities.

        Args:
            city_configs: List of cities to fetch. If None, loads from config.yaml.

        Returns:
            DataFrame with validated observations for all cities.
        """
        if city_configs is None:
            from src.config import load_config

            config = load_config()
            city_configs = [
                CityConfig(**city) for city in config.get("cities", [])
            ]

        observations = []
        for city in city_configs:
            obs = self.fetch_city_data(city)
            if obs is not None:
                observations.append(obs)

        if not observations:
            logger.warning("No data fetched for any city")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame([obs.model_dump() for obs in observations])

        # Drop duplicates
        df = drop_duplicates(df)

        # Validate
        report = full_validation(df)
        logger.info(
            "Data collection complete: %d cities, validation=%s",
            len(df),
            report.status.value,
        )

        return df
