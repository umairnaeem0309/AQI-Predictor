"""
Data Schemas — Pydantic models for API responses and internal data contracts.

Schema ownership:
- WeatherResponse / PollutionResponse: OpenWeather raw response structure
- AQICNResponse: AQICN raw response structure
- StandardObservation: Normalized observation returned by all clients
- CityConfig: City metadata from config.yaml
- DataQualityReport: Validation result summary
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# =============================================================================
# OpenWeather Raw Response Schemas
# =============================================================================


class OpenWeatherMain(BaseModel):
    """Main weather data from OpenWeather /data/2.5/weather."""

    temp: float = Field(..., description="Temperature in Celsius (metric)")
    feels_like: Optional[float] = Field(None, description="Feels-like temperature")
    temp_min: Optional[float] = Field(None, description="Minimum temperature")
    temp_max: Optional[float] = Field(None, description="Maximum temperature")
    pressure: Optional[int] = Field(None, description="Atmospheric pressure (hPa)")
    humidity: Optional[int] = Field(None, description="Relative humidity (%)")


class OpenWeatherWind(BaseModel):
    """Wind data from OpenWeather."""

    speed: Optional[float] = Field(None, description="Wind speed (m/s)")
    deg: Optional[int] = Field(None, description="Wind direction (degrees)")
    gust: Optional[float] = Field(None, description="Wind gust (m/s)")


class OpenWeatherWeatherCondition(BaseModel):
    """Weather condition from OpenWeather."""

    id: Optional[int] = Field(None, description="Weather condition ID")
    main: Optional[str] = Field(None, description="Group weather parameters")
    description: Optional[str] = Field(None, description="Weather description")
    icon: Optional[str] = Field(None, description="Weather icon ID")


class OpenWeatherCoord(BaseModel):
    """Coordinates from OpenWeather response."""

    lon: float = Field(..., description="Longitude")
    lat: float = Field(..., description="Latitude")


class OpenWeatherWeatherResponse(BaseModel):
    """Raw response from OpenWeather /data/2.5/weather endpoint."""

    coord: Optional[OpenWeatherCoord] = None
    weather: Optional[List[OpenWeatherWeatherCondition]] = None
    main: Optional[OpenWeatherMain] = None
    wind: Optional[OpenWeatherWind] = None
    name: Optional[str] = Field(None, description="City name")
    dt: Optional[int] = Field(None, description="Data calculation time (Unix timestamp)")
    timezone: Optional[int] = Field(None, description="Timezone shift from UTC (seconds)")
    cod: Optional[int] = Field(None, description="API response code")


class OpenWeatherPollutionComponent(BaseModel):
    """Pollution components from OpenWeather air pollution API."""

    co: Optional[float] = Field(None, description="Carbon monoxide (μg/m³)")
    no: Optional[float] = Field(None, description="Nitrogen monoxide (μg/m³)")
    no2: Optional[float] = Field(None, description="Nitrogen dioxide (μg/m³)")
    o3: Optional[float] = Field(None, description="Ozone (μg/m³)")
    so2: Optional[float] = Field(None, description="Sulfur dioxide (μg/m³)")
    pm2_5: Optional[float] = Field(None, description="Fine particulate matter (μg/m³)")
    pm10: Optional[float] = Field(None, description="Coarse particulate matter (μg/m³)")
    nh3: Optional[float] = Field(None, description="Ammonia (μg/m³)")


class OpenWeatherPollutionMain(BaseModel):
    """Pollution index from OpenWeather air pollution API."""

    aqi: int = Field(..., description="Air Quality Index (1-5 scale, OpenWeather)")
    main: Optional[str] = Field(None, description="Air quality level label")


class OpenWeatherPollutionItem(BaseModel):
    """Single pollution measurement from OpenWeather."""

    main: Optional[OpenWeatherPollutionMain] = None
    components: Optional[OpenWeatherPollutionComponent] = None
    dt: Optional[int] = Field(None, description="Measurement time (Unix timestamp)")


class OpenWeatherPollutionResponse(BaseModel):
    """Raw response from OpenWeather /data/2.5/air_pollution endpoint."""

    coord: Optional[OpenWeatherCoord] = None
    list: Optional[List[OpenWeatherPollutionItem]] = None


# =============================================================================
# AQICN Raw Response Schema
# =============================================================================


class AQICNAqi(BaseModel):
    """AQI data from AQICN response."""

    idx: Optional[int] = Field(None, description="Station ID")
    aqi: Optional[int] = Field(None, description="Air Quality Index (US EPA scale)")
    attributions: Optional[List[Dict[str, Any]]] = None


class AQICNTime(BaseModel):
    """Timestamp from AQICN response."""

    iso: Optional[str] = Field(None, description="ISO 8601 timestamp")
    v: Optional[int] = Field(None, description="Unix timestamp")


class AQICNDaqi(BaseModel):
    """Daily AQI forecast from AQICN."""

    o3: Optional[List[Optional[int]]] = None
    pm25: Optional[List[Optional[int]]] = None


class AQICNData(BaseModel):
    """Data payload from AQICN response."""

    aqi: Optional[int] = Field(None, description="Air Quality Index (US EPA scale)")
    idx: Optional[int] = Field(None, description="Station ID")
    city: Optional[Dict[str, Any]] = None
    iaqi: Optional[Dict[str, Any]] = None
    time: Optional[AQICNTime] = None
    forecast: Optional[AQICNDaqi] = None


class AQICNResponse(BaseModel):
    """Raw response from AQICN/WAQI API."""

    status: Optional[str] = Field(None, description="API status (ok/error)")
    data: Optional[AQICNData] = None
    data_message: Optional[str] = Field(None, description="Error message if status=error")


# =============================================================================
# Standard Observation Schema
# =============================================================================


class DataSource(str, Enum):
    """Data source identifier."""

    OPENWEATHER = "openweather"
    AQICN = "aqicn"
    OPENWEATHER_AQICN = "openweather+aqicn"


class StandardObservation(BaseModel):
    """Normalized observation returned by all API clients.

    This is the canonical data contract between data collection
    and feature engineering layers.
    """

    timestamp: datetime = Field(..., description="Observation time (UTC)")
    location_id: str = Field(..., description="City identifier (karachi, lahore, islamabad)")
    city_name: str = Field(..., description="Human-readable city name")

    # Weather fields (authoritative source: OpenWeather)
    temperature: Optional[float] = Field(None, description="Temperature (°C)")
    humidity: Optional[float] = Field(None, description="Relative humidity (%)")
    wind_speed: Optional[float] = Field(None, description="Wind speed (m/s)")
    pressure: Optional[float] = Field(None, description="Atmospheric pressure (hPa)")
    weather_condition: Optional[str] = Field(None, description="Weather description")

    # AQI fields (authoritative source: AQICN when available, OpenWeather as fallback)
    aqi: Optional[int] = Field(None, description="Air Quality Index (US EPA scale)")
    pm25: Optional[float] = Field(None, description="PM2.5 concentration (μg/m³)")
    pm10: Optional[float] = Field(None, description="PM10 concentration (μg/m³)")
    co: Optional[float] = Field(None, description="Carbon monoxide (μg/m³)")
    no2: Optional[float] = Field(None, description="Nitrogen dioxide (μg/m³)")
    so2: Optional[float] = Field(None, description="Sulfur dioxide (μg/m³)")
    o3: Optional[float] = Field(None, description="Ozone (μg/m³)")

    # Metadata
    data_source: DataSource = Field(..., description="Source API identifier")
    raw_response_time: Optional[datetime] = Field(
        None,
        description="Source provider observation timestamp (before timezone normalization)",
    )
    collected_at: Optional[datetime] = Field(
        None, description="Local collection timestamp (when API call was made)"
    )
    is_training_valid: bool = Field(
        True,
        description="Whether this observation meets freshness requirements for training. "
        "False if source data is stale (e.g. AQICN returning cached observations).",
    )
    staleness_reason: Optional[str] = Field(
        None,
        description="Reason why observation was marked as not training-valid",
    )
    aqi_dominant_pollutant: Optional[str] = Field(
        None,
        description="Pollutant producing the selected AQI sub-index (pm25 or pm10). "
        "Set when AQI is derived from PM NowCast methodology.",
    )
    # NowCast audit fields
    pm25_nowcast: Optional[float] = Field(
        None,
        description="PM2.5 NowCast concentration (ug/m3). "
        "Weighted 12-hour average from EPA methodology.",
    )
    pm10_nowcast: Optional[float] = Field(
        None,
        description="PM10 NowCast concentration (ug/m3). "
        "Weighted 12-hour average from EPA methodology.",
    )
    pm25_aqi_subindex: Optional[int] = Field(
        None,
        description="PM2.5 AQI sub-index calculated from NowCast concentration.",
    )
    pm10_aqi_subindex: Optional[int] = Field(
        None,
        description="PM10 AQI sub-index calculated from NowCast concentration.",
    )
    aqi_standard: Optional[str] = Field(
        None,
        description="AQI standard used (e.g. US_EPA).",
    )
    aqi_method: Optional[str] = Field(
        None,
        description="AQI calculation method (e.g. PM_NOWCAST).",
    )
    aqi_method_version: Optional[str] = Field(
        None,
        description="AQI methodology version (e.g. EPA-454/B-24-002_MAY_2024).",
    )
    aqi_derived: Optional[bool] = Field(
        None,
        description="True if AQI was derived from pollutant concentrations.",
    )
    aqi_source: Optional[str] = Field(
        None,
        description="Source of pollutant data for AQI derivation (e.g. openweather_pollutants).",
    )

    class Config:
        use_enum_values = True


# =============================================================================
# Configuration Schemas
# =============================================================================


class CityConfig(BaseModel):
    """City metadata from config.yaml."""

    id: str = Field(..., description="City identifier")
    name: str = Field(..., description="Human-readable city name")
    latitude: float = Field(..., description="Latitude")
    longitude: float = Field(..., description="Longitude")


# =============================================================================
# Data Quality Report
# =============================================================================


class ValidationStatus(str, Enum):
    """Validation result status."""

    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"


class DataQualityReport(BaseModel):
    """Result of running validation checks on a DataFrame."""

    status: ValidationStatus = Field(..., description="Overall validation status")
    total_records: int = Field(0, description="Total records checked")
    missing_values: Dict[str, int] = Field(
        default_factory=dict, description="Column -> count of missing values"
    )
    duplicate_count: int = Field(0, description="Number of duplicate records")
    staleness_hours: Optional[float] = Field(None, description="Age of newest record in hours")
    warnings: List[str] = Field(default_factory=list, description="Warning messages")
    errors: List[str] = Field(default_factory=list, description="Error messages")
