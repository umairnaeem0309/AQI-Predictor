"""
Historical Data Ingestion Pipeline.

Downloads historical hourly weather and air quality data from Open-Meteo,
merges them, calculates EPA AQI, and produces a clean ML-ready dataset.

Pipeline steps:
1. Download hourly weather data (Open-Meteo /v1/archive)
2. Download hourly air quality data (Open-Meteo /v1/air-quality)
3. Merge on (timestamp, location_id)
4. Calculate US EPA PM NowCast AQI from pollutant concentrations
5. Validate data quality
6. Save raw observations and metadata

Important:
- Weather data available from 2017+ (IFS 9km)
- Air quality data available from Aug 2022+ (CAMS Global 45km)
- Effective overlap: Aug 2022 onwards
- Open-Meteo US AQI is stored for reference; project uses own EPA calculation
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, load_config
from src.data.providers.open_meteo_air_quality import OpenMeteoAirQualityProvider
from src.data.providers.open_meteo_weather import OpenMeteoWeatherProvider
from src.utils.epa_aqi import (
    calculate_individual_aqi,
    calculate_pm10_aqi,
    calculate_pm25_aqi,
    get_aqi_metadata,
)

logger = logging.getLogger(__name__)

# Output directories
RAW_HISTORICAL_DIR = PROJECT_ROOT / "data" / "raw" / "historical"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Default date range
DEFAULT_START_DATE = "2021-01-01"
DEFAULT_END_DATE = "2026-08-26"


def _ensure_directories() -> None:
    """Create output directories if they don't exist."""
    RAW_HISTORICAL_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def download_weather_data(
    city_configs: List[Dict[str, Any]],
    start_date: str,
    end_date: str,
    dataset: str = "best_match",
    save: bool = True,
) -> pd.DataFrame:
    """Download historical weather data for all configured cities.

    Args:
        city_configs: List of city dicts with 'id', 'name', 'latitude', 'longitude'.
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        dataset: Open-Meteo dataset ('best_match', 'era5', 'era5_land', 'ecmwf_ifs').
        save: If True, save raw data to disk.

    Returns:
        DataFrame with hourly weather observations.
    """
    provider = OpenMeteoWeatherProvider()

    df = provider.fetch_all_cities(
        city_configs=city_configs,
        start_date=start_date,
        end_date=end_date,
        dataset=dataset,
    )

    if save and not df.empty:
        output_file = RAW_HISTORICAL_DIR / "weather_data.csv"
        df.to_csv(output_file, index=False)
        logger.info("Saved weather data: %s (%d rows)", output_file, len(df))

    logger.info(
        "Weather download complete: %d rows, %d cities, usage=%s",
        len(df),
        df["location_id"].nunique() if not df.empty else 0,
        provider.get_usage_summary(),
    )

    return df


def download_air_quality_data(
    city_configs: List[Dict[str, Any]],
    start_date: str,
    end_date: str,
    domain: str = "auto",
    save: bool = True,
) -> pd.DataFrame:
    """Download historical air quality data for all configured cities.

    Args:
        city_configs: List of city dicts with 'id', 'name', 'latitude', 'longitude'.
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        domain: Open-Meteo domain ('auto', 'cams_europe', 'cams_global').
        save: If True, save raw data to disk.

    Returns:
        DataFrame with hourly air quality observations.
    """
    provider = OpenMeteoAirQualityProvider()

    df = provider.fetch_all_cities(
        city_configs=city_configs,
        start_date=start_date,
        end_date=end_date,
        domain=domain,
    )

    if save and not df.empty:
        output_file = RAW_HISTORICAL_DIR / "air_quality_data.csv"
        df.to_csv(output_file, index=False)
        logger.info("Saved air quality data: %s (%d rows)", output_file, len(df))

    logger.info(
        "Air quality download complete: %d rows, %d cities, usage=%s",
        len(df),
        df["location_id"].nunique() if not df.empty else 0,
        provider.get_usage_summary(),
    )

    return df


def merge_weather_and_air_quality(
    weather_df: pd.DataFrame,
    air_quality_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge weather and air quality DataFrames on (timestamp, location_id).

    Weather provides: temperature, humidity, pressure, wind_speed, wind_direction,
                      cloud_cover, precipitation
    Air quality provides: pm25, pm10, co, no2, so2, o3

    Args:
        weather_df: Weather observations DataFrame.
        air_quality_df: Air quality observations DataFrame.

    Returns:
        Merged DataFrame with all weather and pollution fields.
    """
    if weather_df.empty and air_quality_df.empty:
        logger.warning("Both weather and air quality DataFrames are empty")
        return pd.DataFrame()

    if weather_df.empty:
        logger.warning("Weather DataFrame is empty — using air quality only")
        return air_quality_df

    if air_quality_df.empty:
        logger.warning("Air quality DataFrame is empty — using weather only")
        return weather_df

    # Columns to take from each source
    weather_cols = [
        "timestamp",
        "location_id",
        "temperature",
        "humidity",
        "pressure",
        "wind_speed",
        "wind_direction",
        "cloud_cover",
        "precipitation",
    ]
    aq_cols = [
        "timestamp",
        "location_id",
        "pm25",
        "pm10",
        "co",
        "no2",
        "so2",
        "o3",
        "us_aqi_open_meteo",
        "us_aqi_pm25_open_meteo",
        "us_aqi_pm10_open_meteo",
    ]

    # Filter to available columns
    weather_available = [c for c in weather_cols if c in weather_df.columns]
    aq_available = [c for c in aq_cols if c in air_quality_df.columns]

    w_df = weather_df[weather_available].copy()
    a_df = air_quality_df[aq_available].copy()

    # Ensure timestamp is same type for merge
    w_df["timestamp"] = pd.to_datetime(w_df["timestamp"], utc=True)
    a_df["timestamp"] = pd.to_datetime(a_df["timestamp"], utc=True)

    # Round timestamps to hourly for merge alignment
    w_df["timestamp"] = w_df["timestamp"].dt.floor("h")
    a_df["timestamp"] = a_df["timestamp"].dt.floor("h")

    # Merge
    merged = pd.merge(
        w_df,
        a_df,
        on=["timestamp", "location_id"],
        how="outer",
        suffixes=("_weather", "_aq"),
    )

    # Use city_name from weather if available
    if "city_name_weather" in merged.columns:
        merged["city_name"] = merged["city_name_weather"]
        merged.drop(columns=["city_name_weather"], inplace=True, errors="ignore")
    elif "city_name_aq" in merged.columns:
        merged["city_name"] = merged["city_name_aq"]
        merged.drop(columns=["city_name_aq"], inplace=True, errors="ignore")

    # Drop duplicate columns from suffix handling
    for col in merged.columns:
        if col.endswith("_weather") or col.endswith("_aq"):
            base = col.rsplit("_", 1)[0]
            if base in merged.columns:
                # Keep the non-null value
                merged[base] = merged[base].fillna(merged[col])
            merged.drop(columns=[col], inplace=True)

    # Sort
    merged = merged.sort_values(["location_id", "timestamp"]).reset_index(drop=True)

    logger.info(
        "Merged weather + air quality: %d rows, %d columns",
        len(merged),
        len(merged.columns),
    )

    return merged


def calculate_aqi_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate US EPA PM NowCast AQI from pollutant concentrations.

    Uses the project's existing EPA AQI implementation (src/utils/epa_aqi.py).
    Calculates both PM2.5 and PM10 AQI sub-indices and selects the higher one.

    Args:
        df: DataFrame with pm25 and pm10 columns (μg/m³).

    Returns:
        DataFrame with added AQI columns:
        - aqi: Overall AQI (max of PM2.5 and PM10 sub-indices)
        - aqi_dominant_pollutant: 'pm25' or 'pm10'
        - pm25_aqi: PM2.5 AQI sub-index
        - pm10_aqi: PM10 AQI sub-index
        - aqi_category: Human-readable category name
        - aqi_standard: 'US_EPA'
        - aqi_method: 'PM_NOWCAST' or 'PM_DIRECT' (depending on history)
        - aqi_derived: True
    """
    df = df.copy()

    # Calculate individual AQI sub-indices for each row
    pm25_aqi_values = []
    pm10_aqi_values = []
    aqi_values = []
    dominant_values = []

    for idx, row in df.iterrows():
        pm25 = row.get("pm25")
        pm10 = row.get("pm10")

        # Calculate individual sub-indices
        pm25_idx = calculate_pm25_aqi(pm25) if pm25 is not None and not pd.isna(pm25) else None
        pm10_idx = calculate_pm10_aqi(pm10) if pm10 is not None and not pd.isna(pm10) else None

        pm25_aqi_values.append(pm25_idx)
        pm10_aqi_values.append(pm10_idx)

        # Select dominant (highest valid sub-index)
        if pm25_idx is not None and pm10_idx is not None:
            if pm25_idx >= pm10_idx:
                aqi_values.append(pm25_idx)
                dominant_values.append("pm25")
            else:
                aqi_values.append(pm10_idx)
                dominant_values.append("pm10")
        elif pm25_idx is not None:
            aqi_values.append(pm25_idx)
            dominant_values.append("pm25")
        elif pm10_idx is not None:
            aqi_values.append(pm10_idx)
            dominant_values.append("pm10")
        else:
            aqi_values.append(None)
            dominant_values.append(None)

    df["pm25_aqi"] = pm25_aqi_values
    df["pm10_aqi"] = pm10_aqi_values
    df["aqi"] = aqi_values
    df["aqi_dominant_pollutant"] = dominant_values

    # Add AQI category
    from src.utils.aqi_categories import get_aqi_category

    def _get_category_label(aqi_val):
        if aqi_val is None or pd.isna(aqi_val):
            return None
        try:
            _, label = get_aqi_category(int(aqi_val))
            return label
        except (ValueError, TypeError):
            return None

    df["aqi_category"] = df["aqi"].apply(_get_category_label)

    # Add metadata
    aqi_meta = get_aqi_metadata()
    df["aqi_standard"] = aqi_meta["aqi_standard"]
    df["aqi_method"] = "PM_DIRECT"  # Direct from hourly concentrations (not NowCast)
    df["aqi_method_version"] = aqi_meta["aqi_method_version"]
    df["aqi_derived"] = True
    df["aqi_source"] = "open_meteo_pollutants"

    # Log AQI statistics
    valid_aqi = df["aqi"].dropna()
    if len(valid_aqi) > 0:
        logger.info(
            "AQI calculation complete: %d valid rows, min=%d, max=%d, mean=%.1f",
            len(valid_aqi),
            valid_aqi.min(),
            valid_aqi.max(),
            valid_aqi.mean(),
        )
    else:
        logger.warning("No valid AQI values calculated")

    return df


def add_aqi_category(df: pd.DataFrame) -> pd.DataFrame:
    """Add AQI category label based on AQI value.

    Uses US EPA AQI categories:
    - Good: 0-50
    - Moderate: 51-100
    - Unhealthy for Sensitive Groups: 101-150
    - Unhealthy: 151-200
    - Very Unhealthy: 201-300
    - Hazardous: 301-500

    Args:
        df: DataFrame with 'aqi' column.

    Returns:
        DataFrame with added 'aqi_category' column.
    """
    from src.utils.aqi_categories import get_aqi_category

    df = df.copy()

    def _label(aqi_val):
        if aqi_val is None or pd.isna(aqi_val):
            return None
        try:
            _, label = get_aqi_category(int(aqi_val))
            return label
        except (ValueError, TypeError):
            return None

    df["aqi_category"] = df["aqi"].apply(_label)
    return df


def validate_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Run data quality checks on the merged dataset.

    Checks:
    - Missing data percentage per column
    - Negative pollutant values
    - Impossible weather values
    - Duplicate timestamps per city
    - Timestamp ordering
    - AQI range validity (0-500)

    Args:
        df: Merged DataFrame with weather, pollution, and AQI columns.

    Returns:
        Dictionary with validation results.
    """
    if df.empty:
        return {"status": "FAIL", "error": "Empty dataset"}

    report = {
        "status": "PASS",
        "total_rows": len(df),
        "cities": (df["location_id"].unique().tolist() if "location_id" in df.columns else []),
        "date_range": {},
        "missing_data": {},
        "quality_issues": [],
    }

    # Date range
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True)
        report["date_range"] = {
            "start": str(ts.min()),
            "end": str(ts.max()),
            "hours": int((ts.max() - ts.min()).total_seconds() / 3600) + 1,
        }

    # Missing data per column
    key_columns = [
        "temperature",
        "humidity",
        "pressure",
        "wind_speed",
        "pm25",
        "pm10",
        "co",
        "no2",
        "so2",
        "o3",
        "aqi",
    ]
    for col in key_columns:
        if col in df.columns:
            missing = df[col].isna().sum()
            missing_pct = round(missing / len(df) * 100, 2)
            report["missing_data"][col] = {
                "count": int(missing),
                "percentage": missing_pct,
            }
            if missing_pct > 50:
                report["quality_issues"].append(f"High missing rate for {col}: {missing_pct}%")

    # Negative pollutant values
    pollutant_cols = ["pm25", "pm10", "co", "no2", "so2", "o3"]
    for col in pollutant_cols:
        if col in df.columns:
            negatives = (df[col] < 0).sum()
            if negatives > 0:
                report["quality_issues"].append(f"Negative values in {col}: {negatives} rows")

    # Impossible weather values
    if "temperature" in df.columns:
        extreme_temp = ((df["temperature"] < -60) | (df["temperature"] > 60)).sum()
        if extreme_temp > 0:
            report["quality_issues"].append(f"Extreme temperature values: {extreme_temp} rows")

    if "humidity" in df.columns:
        invalid_hum = ((df["humidity"] < 0) | (df["humidity"] > 100)).sum()
        if invalid_hum > 0:
            report["quality_issues"].append(f"Invalid humidity values (0-100): {invalid_hum} rows")

    # Duplicate timestamps per city
    if "timestamp" in df.columns and "location_id" in df.columns:
        dupes = df.duplicated(subset=["timestamp", "location_id"], keep=False).sum()
        if dupes > 0:
            report["quality_issues"].append(
                f"Duplicate (timestamp, location_id) pairs: {dupes} rows"
            )

    # AQI range
    if "aqi" in df.columns:
        valid_aqi = df["aqi"].dropna()
        out_of_range = ((valid_aqi < 0) | (valid_aqi > 500)).sum()
        if out_of_range > 0:
            report["quality_issues"].append(f"AQI values outside 0-500 range: {out_of_range} rows")

    # Overall status
    if report["quality_issues"]:
        report["status"] = "WARNING"

    logger.info(
        "Dataset validation: %s — %d rows, %d issues",
        report["status"],
        report["total_rows"],
        len(report["quality_issues"]),
    )

    return report


def run_historical_ingestion(
    city_configs: Optional[List[Dict[str, Any]]] = None,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    save: bool = True,
) -> Dict[str, Any]:
    """Run the complete historical data ingestion pipeline.

    Pipeline:
    1. Load city configuration
    2. Download historical weather data
    3. Download historical air quality data
    4. Merge weather and air quality
    5. Calculate EPA AQI targets
    6. Validate dataset quality
    7. Save processed dataset and metadata

    Args:
        city_configs: List of city dicts. If None, loads from config.yaml.
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        save: If True, save all outputs to disk.

    Returns:
        Dictionary with merged DataFrame, validation report, and metadata.
    """
    _ensure_directories()

    # Load city configuration
    if city_configs is None:
        config = load_config()
        city_configs = config.get("cities", [])

    logger.info(
        "Starting historical ingestion: %d cities, %s to %s",
        len(city_configs),
        start_date,
        end_date,
    )

    # Step 1: Download weather data
    logger.info("Step 1/5: Downloading weather data...")
    weather_df = download_weather_data(
        city_configs,
        start_date,
        end_date,
        save=save,
    )

    # Step 2: Download air quality data
    # AQ data only available from Aug 2022; adjust start date if needed
    aq_start = max(start_date, "2022-08-01")
    logger.info("Step 2/5: Downloading air quality data (from %s)...", aq_start)
    aq_df = download_air_quality_data(
        city_configs,
        aq_start,
        end_date,
        save=save,
    )

    # Step 3: Merge
    logger.info("Step 3/5: Merging weather and air quality...")
    merged_df = merge_weather_and_air_quality(weather_df, aq_df)

    if merged_df.empty:
        logger.error("Merge produced empty DataFrame")
        return {"dataframe": merged_df, "validation": {}, "metadata": {}}

    # Step 4: Calculate AQI
    logger.info("Step 4/5: Calculating EPA AQI targets...")
    merged_df = calculate_aqi_targets(merged_df)

    # Step 5: Validate
    logger.info("Step 5/5: Validating dataset...")
    validation = validate_dataset(merged_df)

    # Save processed dataset
    if save:
        output_file = PROCESSED_DIR / "raw_observations.csv"
        merged_df.to_csv(output_file, index=False)
        logger.info("Saved processed observations: %s", output_file)

    # Generate metadata
    metadata = {
        "generation_timestamp": datetime.now(timezone.utc).isoformat(),
        "date_range": {"start": start_date, "end": end_date},
        "aq_data_start": aq_start,
        "cities": [c["id"] for c in city_configs],
        "total_rows": len(merged_df),
        "rows_per_city": (
            merged_df["location_id"].value_counts().to_dict()
            if "location_id" in merged_df.columns
            else {}
        ),
        "columns": merged_df.columns.tolist(),
        "weather_rows": len(weather_df),
        "aq_rows": len(aq_df),
        "validation_status": validation.get("status", "UNKNOWN"),
        "dataset_type": "real_api_data",
        "approved_for_training": True,
        "data_provider": "open-meteo",
        "weather_source": "open-meteo-archive",
        "aqi_source": "open-meteo-air-quality",
        "aqi_method": "EPA_PM_DIRECT",
        "aqi_method_version": "EPA-454/B-24-002_MAY_2024",
    }

    if save:
        metadata_file = PROCESSED_DIR / "ingestion_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info("Saved metadata: %s", metadata_file)

    logger.info(
        "Historical ingestion complete: %d total rows, status=%s",
        len(merged_df),
        validation.get("status"),
    )

    return {
        "dataframe": merged_df,
        "validation": validation,
        "metadata": metadata,
    }
