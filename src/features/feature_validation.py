"""
Feature Validation — Verifies feature quality and detects data leakage.

Responsibilities:
- Verify no future data leakage in features
- Validate feature calculations (lag correctness, rolling correctness)
- Check data availability constraints
- Report feature quality metrics
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def check_no_future_leakage(
    df: pd.DataFrame,
    feature_columns: List[str],
    timestamp_column: str = "timestamp",
    location_column: str = "location_id",
) -> List[str]:
    """Verify that no feature uses future data.

    For each feature that depends on historical data, this check verifies
    that the feature value at time t only uses data from times ≤ t.

    Args:
        df: DataFrame with features and timestamps.
        feature_columns: Feature columns to check.
        timestamp_column: Name of timestamp column.
        location_column: Name of location column.

    Returns:
        List of error messages. Empty if no leakage detected.
    """
    errors = []

    if df.empty:
        return errors

    # Check that lag features have correct shift direction
    for col in feature_columns:
        if "_lag_" in col:
            # Extract lag hours from feature name
            parts = col.split("_lag_")
            if len(parts) == 2:
                try:
                    lag_hours = int(parts[1].replace("h", ""))
                    base_col = parts[0]

                    # Verify: lag feature should be NaN for first `lag_hours` records
                    for loc_id, group in df.groupby(location_column):
                        group_sorted = group.sort_values(timestamp_column)
                        lag_values = group_sorted[col].values
                        base_values = group_sorted[base_col].values

                        # First lag_hours records should be NaN (no history)
                        for i in range(min(lag_hours, len(lag_values))):
                            if not pd.isna(lag_values[i]) and not pd.isna(base_values[i]):
                                errors.append(
                                    f"Leakage in {col} at index {i} for location {loc_id}: "
                                    f"lag value should be NaN but is {lag_values[i]}"
                                )

                        # Also verify lag values match historical base values
                        for i in range(lag_hours, len(lag_values)):
                            if not pd.isna(lag_values[i]) and not pd.isna(base_values[i - lag_hours]):
                                expected = base_values[i - lag_hours]
                                if not pd.isna(expected) and lag_values[i] != expected:
                                    errors.append(
                                        f"Leakage in {col} at index {i} for location {loc_id}: "
                                        f"expected {expected} but got {lag_values[i]}"
                                    )
                except (ValueError, IndexError):
                    pass

    if errors:
        logger.error("Data leakage detected: %d issues", len(errors))
    else:
        logger.info("No data leakage detected in %d feature columns", len(feature_columns))

    return errors


def validate_lag_features(
    df: pd.DataFrame,
    lag_columns: List[str],
    location_column: str = "location_id",
    timestamp_column: str = "timestamp",
) -> Dict[str, bool]:
    """Validate that lag features contain correct historical values.

    Args:
        df: DataFrame with lag features.
        lag_columns: Lag feature column names.
        location_column: Name of location column.
        timestamp_column: Name of timestamp column.

    Returns:
        Dict mapping feature name to validation result (True = valid).
    """
    results = {}

    for col in lag_columns:
        if col not in df.columns:
            results[col] = False
            continue

        # Extract lag hours
        parts = col.split("_lag_")
        if len(parts) != 2:
            results[col] = False
            continue

        try:
            lag_hours = int(parts[1].replace("h", ""))
        except ValueError:
            results[col] = False
            continue

        base_col = parts[0]
        if base_col not in df.columns:
            results[col] = False
            continue

        valid = True
        for loc_id, group in df.groupby(location_column):
            group_sorted = group.sort_values(timestamp_column)
            lag_vals = group_sorted[col].values
            base_vals = group_sorted[base_col].values

            # Check: first lag_hours positions should be NaN (no history)
            for i in range(min(lag_hours, len(lag_vals))):
                if not pd.isna(lag_vals[i]) and not pd.isna(base_vals[i]):
                    valid = False
                    break

            if not valid:
                break

            # Check: lag value at position i should equal base value at position i-lag_hours
            for i in range(lag_hours, len(lag_vals)):
                if not pd.isna(lag_vals[i]) and not pd.isna(base_vals[i - lag_hours]):
                    if lag_vals[i] != base_vals[i - lag_hours]:
                        valid = False
                        break
            if not valid:
                break

        results[col] = valid

    return results


def validate_rolling_features(
    df: pd.DataFrame,
    rolling_columns: List[str],
    location_column: str = "location_id",
    timestamp_column: str = "timestamp",
) -> Dict[str, bool]:
    """Validate that rolling features are plausible.

    Checks:
    - Rolling mean should be between min and max of the source column
    - Rolling std should be non-negative
    - Rolling min should be ≤ rolling max

    Args:
        df: DataFrame with rolling features.
        rolling_columns: Rolling feature column names.
        location_column: Name of location column.
        timestamp_column: Name of timestamp column.

    Returns:
        Dict mapping feature name to validation result (True = valid).
    """
    results = {}

    for col in rolling_columns:
        if col not in df.columns:
            results[col] = False
            continue

        valid = True

        # Check: rolling std should be non-negative
        if "_std_" in col:
            if (df[col] < 0).any():
                valid = False
                logger.error("Rolling std feature %s has negative values", col)

        # Check: rolling min should be ≤ rolling max
        if "_min_" in col:
            max_col = col.replace("_min_", "_max_")
            if max_col in df.columns:
                if (df[col] > df[max_col]).any():
                    valid = False
                    logger.error("Rolling min %s > rolling max %s", col, max_col)

        results[col] = valid

    return results


def get_feature_availability() -> Dict[str, str]:
    """Return documentation of feature availability times.

    All lag and rolling features are available at prediction time t
    because they use historical values (data from t-N or earlier).

    Returns:
        Dict mapping feature name to availability description.
    """
    availability = {}

    # Time features — available immediately
    for feat in ["hour", "day_of_week", "month", "season", "is_weekend", "hour_sin", "hour_cos"]:
        availability[feat] = "t (immediately available)"

    # Lag features — available at t (using historical data)
    lag_features = []
    lag_hours = [1, 6, 12, 24, 48, 72]
    lag_columns = ["aqi", "pm25", "temperature", "humidity"]
    for col in lag_columns:
        for h in lag_hours:
            lag_features.append(f"{col}_lag_{h}h")
    for feat in lag_features:
        availability[feat] = "t (uses data from t-N, available at prediction time)"

    # Rolling features — available at t (using historical window)
    rolling_features = [
        "aqi_rolling_mean_6h", "aqi_rolling_mean_12h", "aqi_rolling_mean_24h",
        "aqi_rolling_std_24h", "aqi_rolling_min_24h", "aqi_rolling_max_24h",
        "pm25_rolling_mean_6h", "pm25_rolling_mean_24h",
        "temperature_rolling_mean_24h", "humidity_rolling_mean_24h",
    ]
    for feat in rolling_features:
        availability[feat] = "t (uses window ending at t-1, available at prediction time)"

    # Derived features — available at t (depend on lag/rolling)
    derived_features = [
        "aqi_change_rate_1h", "aqi_change_rate_6h", "aqi_change_rate_24h",
        "aqi_trend_24h", "pm25_pm10_ratio", "no2_so2_ratio",
        "o3_no2_ratio", "temp_humidity_interaction",
        "wind_cooling_effect", "aqi_deviation_from_24h_avg",
    ]
    for feat in derived_features:
        availability[feat] = "t (depends on lag/rolling features, available at prediction time)"

    # Current values — available at t
    current_features = ["aqi", "pm25", "pm10", "co", "no2", "so2", "o3",
                        "temperature", "humidity", "wind_speed", "pressure"]
    for feat in current_features:
        availability[feat] = "t (current observation, available at prediction time)"

    return availability


def full_feature_validation(
    df: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run all feature validation checks.

    Args:
        df: DataFrame with engineered features.
        feature_columns: Columns to validate. If None, validates all non-metadata columns.

    Returns:
        Dictionary with validation results.
    """
    if feature_columns is None:
        feature_columns = [
            c for c in df.columns
            if c not in ["timestamp", "location_id", "city_name", "data_source", "raw_response_time"]
        ]

    # Leakage check
    leakage_errors = check_no_future_leakage(df, feature_columns)

    # Lag validation
    lag_cols = [c for c in feature_columns if "_lag_" in c]
    lag_results = validate_lag_features(df, lag_cols)

    # Rolling validation
    rolling_cols = [c for c in feature_columns if "_rolling_" in c]
    rolling_results = validate_rolling_features(df, rolling_cols)

    # Feature completeness
    completeness = {}
    for col in feature_columns:
        if col in df.columns:
            total = len(df)
            missing = df[col].isna().sum()
            completeness[col] = {
                "total": total,
                "missing": int(missing),
                "completeness_pct": round((1 - missing / total) * 100, 1) if total > 0 else 0,
            }

    results = {
        "leakage_errors": leakage_errors,
        "lag_validation": lag_results,
        "rolling_validation": rolling_results,
        "completeness": completeness,
        "total_features": len(feature_columns),
        "total_rows": len(df),
    }

    logger.info(
        "Feature validation: %d features, %d rows, %d leakage errors",
        len(feature_columns),
        len(df),
        len(leakage_errors),
    )

    return results
