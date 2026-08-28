"""
Feature Engineering Pipeline — Transforms raw observations into ML-ready features.

Feature categories:
- Time-based: hour, day, month, season, cyclical encoding
- Lag: historical values at specific offsets
- Rolling: windowed statistics over historical data
- Derived: ratios, change rates, interactions

Data leakage prevention:
- All features use ONLY data available at or before observation time
- Lag features use shift() which guarantees no future data
- Rolling features use closed='left' to exclude current period
- Current AQI at prediction time IS used (documented below)

Feature metadata:
- Every generated dataset includes version, schema, timestamp info
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Feature version — increment when feature definitions change
FEATURE_VERSION = "1.0.0"
SCHEMA_VERSION = "1.0"

# US EPA AQI cap (values above 500 are beyond the scale)
US_EPA_AQI_MAX = 500


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features from timestamp column.

    Time features are available immediately (at observation time t).
    No data leakage: time features only use the timestamp itself.

    Args:
        df: DataFrame with 'timestamp' column (datetime).

    Returns:
        DataFrame with added time features.
    """
    df = df.copy()

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Basic time features
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.weekday
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Season mapping (Northern Hemisphere — Pakistan)
    # Winter: Dec-Feb, Spring: Mar-May, Summer: Jun-Aug, Fall: Sep-Nov
    season_map = {
        12: 0,
        1: 0,
        2: 0,  # Winter
        3: 1,
        4: 1,
        5: 1,  # Spring
        6: 2,
        7: 2,
        8: 2,  # Summer
        9: 3,
        10: 3,
        11: 3,  # Fall
    }
    df["season"] = df["month"].map(season_map)

    # Cyclical encoding (avoids artificial discontinuity at hour 23→0)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    logger.debug(
        "Added time features: hour, day_of_week, month, season, is_weekend, hour_sin, hour_cos"
    )
    return df


def add_lag_features(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    lag_hours: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Add lag features — historical values at specific time offsets.

    Lag features are available at prediction time t (they use data from t-N).
    Data leakage prevention: shift() guarantees no future data is used.

    Available lags per column:
    - aqi: [1, 6, 12, 24, 48, 72] hours
    - pm25: [1, 24] hours
    - temperature: [1, 24] hours
    - humidity: [1, 24] hours

    Args:
        df: DataFrame sorted by (location_id, timestamp).
        columns: Columns to create lag features for.
        lag_hours: Lag offsets in hours.

    Returns:
        DataFrame with added lag features.
    """
    df = df.copy()

    if columns is None:
        columns = ["aqi", "pm25", "temperature", "humidity"]

    if lag_hours is None:
        lag_hours = [1, 6, 12, 24, 48, 72]

    for col in columns:
        if col not in df.columns:
            continue

        for lag in lag_hours:
            feature_name = f"{col}_lag_{lag}h"

            # Shift by lag hours within each location group
            # This preserves the time-series ordering per city
            df[feature_name] = df.groupby("location_id")[col].shift(lag)

            missing_count = df[feature_name].isna().sum()
            if missing_count > 0:
                logger.debug(
                    "Lag feature %s: %d missing values (insufficient history)",
                    feature_name,
                    missing_count,
                )

    logger.debug("Added lag features for columns: %s", columns)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    windows: Optional[Dict[str, List[str]]] = None,
) -> pd.DataFrame:
    """Add rolling window statistics.

    Rolling features are available at prediction time t (they use data
    from the window ending at t-1).
    Data leakage prevention: closed='left' excludes current period.

    Available windows:
    - aqi: mean(6h, 12h, 24h), std(24h), min(24h), max(24h)
    - pm25: mean(6h, 24h)
    - temperature: mean(24h)
    - humidity: mean(24h)

    Args:
        df: DataFrame sorted by (location_id, timestamp).
        columns: Columns to create rolling features for.
        windows: Dict mapping column to list of (window, agg_function) tuples.

    Returns:
        DataFrame with added rolling features.
    """
    df = df.copy()

    if columns is None:
        columns = ["aqi", "pm25", "temperature", "humidity"]

    if windows is None:
        windows = {
            "aqi": [
                ("6h", "mean"),
                ("12h", "mean"),
                ("24h", "mean"),
                ("24h", "std"),
                ("24h", "min"),
                ("24h", "max"),
            ],
            "pm25": [("6h", "mean"), ("24h", "mean")],
            "temperature": [("24h", "mean")],
            "humidity": [("24h", "mean")],
        }

    # Ensure timestamp is datetime for time-based rolling
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    for col in columns:
        if col not in df.columns:
            continue

        if col not in windows:
            continue

        for window, agg_func in windows[col]:
            feature_name = f"{col}_rolling_{agg_func}_{window}"

            # Rolling window within each location group
            # closed='left' excludes current period (no future data leakage)
            # Time-based rolling requires a DatetimeIndex; set timestamp
            # as index within each group, roll, then restore.
            parts = []
            for loc_id, group in df.groupby("location_id"):
                if len(group) == 0:
                    continue
                g = group.set_index("timestamp")[[col]].sort_index()
                rolling = g[col].rolling(window=window, min_periods=1, closed="left")
                if agg_func == "mean":
                    result = rolling.mean()
                elif agg_func == "std":
                    result = rolling.std()
                elif agg_func == "min":
                    result = rolling.min()
                elif agg_func == "max":
                    result = rolling.max()
                else:
                    continue
                result.index = group.index
                parts.append(result)

            if parts:
                df[feature_name] = pd.concat(parts).sort_index()

    logger.debug("Added rolling features for columns: %s", columns)
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features — ratios, change rates, and interactions.

    Feature usefulness will be experimentally evaluated during model
    training. Features that do not improve model performance may be
    removed in future iterations.

    Available derived features:
    - aqi_change_rate_1h/6h/24h: AQI change speed
    - aqi_trend_24h: short vs long term direction
    - pm25_pm10_ratio: particle size distribution
    - no2_so2_ratio: industrial vs traffic signature
    - o3_no2_ratio: photochemical activity
    - temp_humidity_interaction: heat index approximation
    - wind_cooling_effect: wind-chill approximation
    - aqi_deviation_from_24h_avg: deviation from recent average

    Args:
        df: DataFrame with base and lag features.

    Returns:
        DataFrame with added derived features.
    """
    df = df.copy()

    # --- AQI change rates ---
    if "aqi" in df.columns:
        if "aqi_lag_1h" in df.columns:
            df["aqi_change_rate_1h"] = df["aqi"] - df["aqi_lag_1h"]

        if "aqi_lag_6h" in df.columns:
            df["aqi_change_rate_6h"] = (df["aqi"] - df["aqi_lag_6h"]) / 6

        if "aqi_lag_24h" in df.columns:
            df["aqi_change_rate_24h"] = (df["aqi"] - df["aqi_lag_24h"]) / 24

    # --- AQI trend (short vs long term) ---
    if "aqi_rolling_mean_6h" in df.columns and "aqi_rolling_mean_24h" in df.columns:
        df["aqi_trend_24h"] = df["aqi_rolling_mean_6h"] - df["aqi_rolling_mean_24h"]

    # --- Pollutant ratios ---
    if "pm25" in df.columns and "pm10" in df.columns:
        # Avoid division by zero
        df["pm25_pm10_ratio"] = np.where(df["pm10"] > 0, df["pm25"] / df["pm10"], np.nan)

    if "no2" in df.columns and "so2" in df.columns:
        df["no2_so2_ratio"] = np.where(df["so2"] > 0, df["no2"] / df["so2"], np.nan)

    if "o3" in df.columns and "no2" in df.columns:
        df["o3_no2_ratio"] = np.where(df["no2"] > 0, df["o3"] / df["no2"], np.nan)

    # --- Weather interactions ---
    if "temperature" in df.columns and "humidity" in df.columns:
        df["temp_humidity_interaction"] = df["temperature"] * df["humidity"] / 100

    if "temperature" in df.columns and "wind_speed" in df.columns:
        df["wind_cooling_effect"] = df["temperature"] - (df["wind_speed"] * 2)

    # --- AQI deviation from average ---
    if "aqi" in df.columns and "aqi_rolling_mean_24h" in df.columns:
        df["aqi_deviation_from_24h_avg"] = df["aqi"] - df["aqi_rolling_mean_24h"]

    logger.debug("Added derived features")
    return df


def cap_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Cap extreme outlier values in processed features.

    Note: Raw values are preserved in the original columns.
    Capping only applies to feature columns used by the model.

    Args:
        df: DataFrame with features.

    Returns:
        DataFrame with capped outliers.
    """
    df = df.copy()

    # AQI: cap at US EPA maximum (500) in feature columns
    aqi_feature_cols = [c for c in df.columns if c.startswith("aqi") and "raw" not in c]
    for col in aqi_feature_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            over_limit = (df[col] > US_EPA_AQI_MAX).sum()
            if over_limit > 0:
                logger.warning(
                    "Feature %s: %d values above US EPA limit (500) — capping",
                    col,
                    over_limit,
                )
                df[col] = df[col].clip(upper=US_EPA_AQI_MAX)

    # Humidity: must be 0-100
    humidity_cols = [c for c in df.columns if "humidity" in c]
    for col in humidity_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            invalid = ((df[col] < 0) | (df[col] > 100)).sum()
            if invalid > 0:
                logger.warning(
                    "Feature %s: %d values outside [0, 100] — setting to NaN",
                    col,
                    invalid,
                )
                df.loc[(df[col] < 0) | (df[col] > 100), col] = np.nan

    return df


def engineer_features(
    df: pd.DataFrame,
    feature_version: str = FEATURE_VERSION,
) -> pd.DataFrame:
    """Full feature engineering pipeline.

    Pipeline order:
    1. Sort by (location_id, timestamp)
    2. Add time features
    3. Add lag features
    4. Add rolling features
    5. Add derived features
    6. Cap outliers
    7. Add feature metadata

    Missing value strategy:
    - Missing values are PRESERVED throughout the pipeline
    - Imputation is decided during model training experiments
    - NaN values flow through lag, rolling, and derived features

    Args:
        df: Raw DataFrame with StandardObservation columns.
        feature_version: Version string for this feature set.

    Returns:
        DataFrame with all engineered features and metadata.
    """
    if df.empty:
        logger.warning("Empty DataFrame passed to feature engineering")
        return df

    logger.info(
        "Starting feature engineering: %d rows, %d columns",
        len(df),
        len(df.columns),
    )

    # Sort by location and timestamp (critical for lag/rolling correctness)
    df = df.sort_values(["location_id", "timestamp"]).reset_index(drop=True)

    # Feature engineering pipeline
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_derived_features(df)
    df = cap_outliers(df)

    # Add feature metadata
    df.attrs["feature_version"] = feature_version
    df.attrs["schema_version"] = SCHEMA_VERSION
    df.attrs["generation_timestamp"] = datetime.utcnow().isoformat()
    df.attrs["source_row_count"] = len(df)
    df.attrs["feature_count"] = len(df.columns)

    # Log metadata as a row in the DataFrame for tracking
    # (metadata is stored in df.attrs, not as data rows)

    logger.info(
        "Feature engineering complete: %d rows, %d features, version=%s",
        len(df),
        len(df.columns),
        feature_version,
    )

    return df


def get_feature_metadata(df: pd.DataFrame) -> Dict[str, Any]:
    """Extract feature metadata from DataFrame attributes.

    Args:
        df: DataFrame with feature metadata in .attrs.

    Returns:
        Dictionary with feature version, schema, timestamp, and counts.
    """
    return {
        "feature_version": df.attrs.get("feature_version", "unknown"),
        "schema_version": df.attrs.get("schema_version", "unknown"),
        "generation_timestamp": df.attrs.get("generation_timestamp", "unknown"),
        "source_row_count": df.attrs.get("source_row_count", len(df)),
        "feature_count": df.attrs.get("feature_count", len(df.columns)),
    }
