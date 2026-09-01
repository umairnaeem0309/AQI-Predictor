"""
Feature Store Schemas — Pydantic models for feature group definitions and metadata.

Includes:
- FeatureSchema: Column definitions for feature groups
- DatasetMetadata: Dataset type and approval status
- FeatureGroupMetadata: Feature group configuration
- LineageMetadata: Mandatory lineage tracking
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class DatasetType(str, Enum):
    """Dataset type classification."""

    SYNTHETIC_TEST = "synthetic_test_data"
    REAL_TRAINING = "real_training_data"
    REAL_VALIDATION = "real_validation_data"
    REAL_TEST = "real_test_data"


class FeatureColumn(BaseModel):
    """Definition of a single feature column."""

    name: str = Field(..., description="Column name")
    dtype: str = Field(..., description="Data type (string, float, int, timestamp)")
    description: Optional[str] = Field(None, description="Human-readable description")
    nullable: bool = Field(True, description="Whether column can be null")
    is_primary_key: bool = Field(False, description="Whether column is part of primary key")
    is_event_time: bool = Field(False, description="Whether column is the event time")


class FeatureSchema(BaseModel):
    """Schema definition for a feature group."""

    name: str = Field(..., description="Feature group name")
    version: int = Field(1, description="Schema version")
    columns: List[FeatureColumn] = Field(..., description="Column definitions")
    primary_key: List[str] = Field(..., description="Primary key columns")
    event_time: str = Field(..., description="Event time column")
    description: Optional[str] = Field(None, description="Feature group description")

    def get_column_names(self) -> List[str]:
        """Get all column names."""
        return [col.name for col in self.columns]

    def get_column_types(self) -> Dict[str, str]:
        """Get column name to type mapping."""
        return {col.name: col.dtype for col in self.columns}


class DatasetMetadata(BaseModel):
    """Metadata about the dataset being inserted."""

    dataset_version: str = Field(..., description="Dataset version identifier")
    dataset_type: DatasetType = Field(..., description="Type of dataset")
    approved_for_training: bool = Field(False, description="Whether approved for model training")
    approved_for_evaluation: bool = Field(
        False, description="Whether approved for model evaluation"
    )
    source: Optional[str] = Field(None, description="Data source (e.g., openweather, synthetic)")
    generation_timestamp: Optional[str] = Field(None, description="When dataset was generated")
    record_count: Optional[int] = Field(None, description="Number of records")
    feature_count: Optional[int] = Field(None, description="Number of features")


class LineageMetadata(BaseModel):
    """Mandatory lineage metadata for every feature group insertion."""

    feature_version: str = Field(..., description="Feature definition version")
    schema_version: str = Field(..., description="Schema version")
    source_dataset_version: str = Field(..., description="Source dataset version")
    creation_timestamp: str = Field(..., description="When features were created (UTC ISO)")
    dataset_type: str = Field(..., description="Dataset type classification")


class FeatureGroupMetadata(BaseModel):
    """Complete metadata for a feature group."""

    model_config = ConfigDict(use_enum_values=True, protected_namespaces=())

    name: str = Field(..., description="Feature group name")
    version: int = Field(1, description="Feature group version")
    schema_: FeatureSchema = Field(..., alias="schema", description="Feature schema")
    lineage: Optional[LineageMetadata] = Field(None, description="Lineage tracking")
    description: Optional[str] = Field(None, description="Description")
    online_enabled: bool = Field(
        False, description="Whether online store is enabled (offline only initially)"
    )


# =============================================================================
# Pre-defined Feature Group Schemas
# =============================================================================

# Standard feature columns for AQI features
AQI_FEATURE_COLUMNS = [
    FeatureColumn(
        name="location_id",
        dtype="string",
        is_primary_key=True,
        description="City identifier",
    ),
    FeatureColumn(
        name="timestamp",
        dtype="timestamp",
        is_event_time=True,
        is_primary_key=True,
        description="Observation time (UTC)",
    ),
    FeatureColumn(name="city_name", dtype="string", description="City name"),
    FeatureColumn(name="temperature", dtype="float", description="Temperature (°C)"),
    FeatureColumn(name="humidity", dtype="float", description="Relative humidity (%)"),
    FeatureColumn(name="wind_speed", dtype="float", description="Wind speed (m/s)"),
    FeatureColumn(name="pressure", dtype="float", description="Atmospheric pressure (hPa)"),
    FeatureColumn(name="weather_condition", dtype="string", description="Weather description"),
    FeatureColumn(name="aqi", dtype="float", description="Air Quality Index (US EPA)"),
    FeatureColumn(name="pm25", dtype="float", description="PM2.5 (μg/m³)"),
    FeatureColumn(name="pm10", dtype="float", description="PM10 (μg/m³)"),
    FeatureColumn(name="co", dtype="float", description="CO (μg/m³)"),
    FeatureColumn(name="no2", dtype="float", description="NO2 (μg/m³)"),
    FeatureColumn(name="so2", dtype="float", description="SO2 (μg/m³)"),
    FeatureColumn(name="o3", dtype="float", description="O3 (μg/m³)"),
    FeatureColumn(name="data_source", dtype="string", description="Source API"),
    FeatureColumn(name="weather_available", dtype="int", description="Weather data available flag"),
    FeatureColumn(name="aqi_available", dtype="int", description="AQI data available flag"),
    # Time features
    FeatureColumn(name="hour", dtype="int", description="Hour of day (0-23)"),
    FeatureColumn(name="day_of_week", dtype="int", description="Day of week (0=Mon, 6=Sun)"),
    FeatureColumn(name="month", dtype="int", description="Month (1-12)"),
    FeatureColumn(
        name="season",
        dtype="int",
        description="Season (0=Winter, 1=Spring, 2=Summer, 3=Fall)",
    ),
    FeatureColumn(name="is_weekend", dtype="int", description="Weekend flag"),
    FeatureColumn(name="hour_sin", dtype="float", description="Cyclical hour encoding (sin)"),
    FeatureColumn(name="hour_cos", dtype="float", description="Cyclical hour encoding (cos)"),
    # Lag features
    FeatureColumn(name="aqi_lag_1h", dtype="float", description="AQI 1 hour ago"),
    FeatureColumn(name="aqi_lag_6h", dtype="float", description="AQI 6 hours ago"),
    FeatureColumn(name="aqi_lag_12h", dtype="float", description="AQI 12 hours ago"),
    FeatureColumn(name="aqi_lag_24h", dtype="float", description="AQI 24 hours ago"),
    FeatureColumn(name="aqi_lag_48h", dtype="float", description="AQI 48 hours ago"),
    FeatureColumn(name="aqi_lag_72h", dtype="float", description="AQI 72 hours ago"),
    FeatureColumn(name="pm25_lag_1h", dtype="float", description="PM2.5 1 hour ago"),
    FeatureColumn(name="pm25_lag_24h", dtype="float", description="PM2.5 24 hours ago"),
    FeatureColumn(name="temperature_lag_1h", dtype="float", description="Temperature 1 hour ago"),
    FeatureColumn(
        name="temperature_lag_24h",
        dtype="float",
        description="Temperature 24 hours ago",
    ),
    FeatureColumn(name="humidity_lag_1h", dtype="float", description="Humidity 1 hour ago"),
    FeatureColumn(name="humidity_lag_24h", dtype="float", description="Humidity 24 hours ago"),
    # Rolling features
    FeatureColumn(name="aqi_rolling_mean_6h", dtype="float", description="AQI mean over 6h"),
    FeatureColumn(name="aqi_rolling_mean_12h", dtype="float", description="AQI mean over 12h"),
    FeatureColumn(name="aqi_rolling_mean_24h", dtype="float", description="AQI mean over 24h"),
    FeatureColumn(name="aqi_rolling_std_24h", dtype="float", description="AQI std over 24h"),
    FeatureColumn(name="aqi_rolling_min_24h", dtype="float", description="AQI min over 24h"),
    FeatureColumn(name="aqi_rolling_max_24h", dtype="float", description="AQI max over 24h"),
    FeatureColumn(name="pm25_rolling_mean_6h", dtype="float", description="PM2.5 mean over 6h"),
    FeatureColumn(name="pm25_rolling_mean_24h", dtype="float", description="PM2.5 mean over 24h"),
    FeatureColumn(
        name="temperature_rolling_mean_24h",
        dtype="float",
        description="Temperature mean over 24h",
    ),
    FeatureColumn(
        name="humidity_rolling_mean_24h",
        dtype="float",
        description="Humidity mean over 24h",
    ),
    # Derived features
    FeatureColumn(name="aqi_change_rate_1h", dtype="float", description="AQI change per hour"),
    FeatureColumn(name="aqi_change_rate_6h", dtype="float", description="AQI change per 6h"),
    FeatureColumn(name="aqi_change_rate_24h", dtype="float", description="AQI change per 24h"),
    FeatureColumn(name="aqi_trend_24h", dtype="float", description="AQI trend direction"),
    FeatureColumn(name="pm25_pm10_ratio", dtype="float", description="PM2.5/PM10 ratio"),
    FeatureColumn(name="no2_so2_ratio", dtype="float", description="NO2/SO2 ratio"),
    FeatureColumn(name="o3_no2_ratio", dtype="float", description="O3/NO2 ratio"),
    FeatureColumn(
        name="temp_humidity_interaction",
        dtype="float",
        description="Temperature × humidity",
    ),
    FeatureColumn(name="wind_cooling_effect", dtype="float", description="Wind chill effect"),
    FeatureColumn(
        name="aqi_deviation_from_24h_avg",
        dtype="float",
        description="AQI deviation from daily avg",
    ),
]

# Target columns
AQI_TARGET_COLUMNS = [
    FeatureColumn(name="location_id", dtype="string", is_primary_key=True),
    FeatureColumn(name="timestamp", dtype="timestamp", is_event_time=True, is_primary_key=True),
    FeatureColumn(name="target_aqi_24h", dtype="float", description="AQI 24 hours ahead"),
    FeatureColumn(name="target_aqi_48h", dtype="float", description="AQI 48 hours ahead"),
    FeatureColumn(name="target_aqi_72h", dtype="float", description="AQI 72 hours ahead"),
]

# Pre-defined schemas
AQI_FEATURES_SCHEMA = FeatureSchema(
    name="aqi_features",
    version=1,
    columns=AQI_FEATURE_COLUMNS,
    primary_key=["location_id", "timestamp"],
    event_time="timestamp",
    description="AQI prediction features with weather, pollution, and derived variables",
)

AQI_TARGETS_SCHEMA = FeatureSchema(
    name="aqi_targets",
    version=1,
    columns=AQI_TARGET_COLUMNS,
    primary_key=["location_id", "timestamp"],
    event_time="timestamp",
    description="AQI target variables for 24h, 48h, 72h forecasting",
)


def get_feature_group_name(base_name: str, dataset_type: DatasetType) -> str:
    """Generate feature group name with dataset type suffix.

    Naming convention:
    - aqi_features_test (synthetic/test data)
    - aqi_features_prod (real production data)
    - aqi_targets_test
    - aqi_targets_prod

    Args:
        base_name: Base feature group name (e.g., 'aqi_features').
        dataset_type: Type of dataset.

    Returns:
        Feature group name with suffix.
    """
    suffix = "_test" if dataset_type == DatasetType.SYNTHETIC_TEST else "_prod"
    return f"{base_name}{suffix}"
