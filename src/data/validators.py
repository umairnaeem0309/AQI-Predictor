"""
Data Validators — Quality checks for collected data.

Responsibilities:
- Schema validation: check DataFrame matches expected columns/types
- Staleness detection: reject data older than threshold
- Duplicate prevention: detect by (timestamp, location_id)
- Missing value reporting: count nulls per column
- Full validation pipeline: run all checks, return DataQualityReport
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Set

import pandas as pd

from src.data.schemas import DataQualityReport, ValidationStatus

logger = logging.getLogger(__name__)

# Required columns for a valid StandardObservation DataFrame
REQUIRED_COLUMNS = ["timestamp", "location_id"]

RECOMMENDED_COLUMNS = [
    "city_name",
    "temperature",
    "humidity",
    "wind_speed",
    "pressure",
    "aqi",
    "pm25",
    "pm10",
    "data_source",
]


def validate_schema(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
) -> List[str]:
    """Check that DataFrame contains expected columns.

    Args:
        df: DataFrame to validate.
        required_columns: List of required column names.
            Defaults to REQUIRED_COLUMNS.

    Returns:
        List of error messages. Empty if validation passes.
    """
    if required_columns is None:
        required_columns = REQUIRED_COLUMNS

    errors = []
    for col in required_columns:
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")

    return errors


def check_staleness(
    df: pd.DataFrame,
    max_age_hours: float = 2.0,
    timestamp_column: str = "timestamp",
) -> Optional[str]:
    """Check if data is newer than the staleness threshold.

    Args:
        df: DataFrame to check.
        max_age_hours: Maximum acceptable data age in hours.
        timestamp_column: Name of the timestamp column.

    Returns:
        Warning message if data is stale, None if fresh.
    """
    if df.empty or timestamp_column not in df.columns:
        return None

    # Ensure timestamp column is datetime
    timestamps = pd.to_datetime(df[timestamp_column], utc=True)
    newest = timestamps.max()
    now = datetime.now(timezone.utc)

    if newest.tzinfo is None:
        newest = newest.replace(tzinfo=timezone.utc)

    age_hours = (now - newest).total_seconds() / 3600

    if age_hours > max_age_hours:
        warning = (
            f"Data is stale: newest record is {age_hours:.1f} hours old "
            f"(threshold: {max_age_hours} hours)"
        )
        logger.warning(warning)
        return warning

    return None


def check_duplicates(
    df: pd.DataFrame,
    key_columns: Optional[List[str]] = None,
) -> int:
    """Count duplicate records by key columns.

    Args:
        df: DataFrame to check.
        key_columns: Columns to check for duplicates.
            Defaults to ["timestamp", "location_id"].

    Returns:
        Number of duplicate rows (excluding first occurrence).
    """
    if key_columns is None:
        key_columns = ["timestamp", "location_id"]

    # Only check columns that exist in the DataFrame
    existing_keys = [col for col in key_columns if col in df.columns]

    if not existing_keys:
        return 0

    duplicate_mask = df.duplicated(subset=existing_keys, keep="first")
    count = int(duplicate_mask.sum())

    if count > 0:
        logger.warning("Found %d duplicate records by keys %s", count, existing_keys)

    return count


def check_missing_values(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
) -> dict:
    """Count missing values per column.

    Args:
        df: DataFrame to check.
        columns: Columns to check. Defaults to all columns.

    Returns:
        Dictionary mapping column name to count of missing values.
    """
    if columns is None:
        columns = df.columns.tolist()

    missing = {}
    for col in columns:
        if col in df.columns:
            count = int(df[col].isna().sum())
            if count > 0:
                missing[col] = count

    return missing


def drop_duplicates(
    df: pd.DataFrame,
    key_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Remove duplicate records, keeping the first occurrence.

    Args:
        df: DataFrame to deduplicate.
        key_columns: Columns to check for duplicates.
            Defaults to ["timestamp", "location_id"].

    Returns:
        DataFrame with duplicates removed.
    """
    if key_columns is None:
        key_columns = ["timestamp", "location_id"]

    existing_keys = [col for col in key_columns if col in df.columns]

    if not existing_keys:
        return df

    original_len = len(df)
    cleaned = df.drop_duplicates(subset=existing_keys, keep="first").reset_index(drop=True)
    removed = original_len - len(cleaned)

    if removed > 0:
        logger.info("Removed %d duplicate records", removed)

    return cleaned


def full_validation(
    df: pd.DataFrame,
    max_staleness_hours: float = 2.0,
    required_columns: Optional[List[str]] = None,
    recommend_columns: Optional[List[str]] = None,
) -> DataQualityReport:
    """Run all validation checks on a DataFrame.

    Args:
        df: DataFrame to validate.
        max_staleness_hours: Maximum acceptable data age in hours.
        required_columns: Required columns. Defaults to REQUIRED_COLUMNS.
        recommend_columns: Recommended columns. Defaults to RECOMMENDED_COLUMNS.

    Returns:
        DataQualityReport with all check results.
    """
    if required_columns is None:
        required_columns = REQUIRED_COLUMNS
    if recommend_columns is None:
        recommend_columns = RECOMMENDED_COLUMNS

    warnings = []
    errors = []
    status = ValidationStatus.PASS

    # 1. Schema validation
    schema_errors = validate_schema(df, required_columns)
    if schema_errors:
        errors.extend(schema_errors)
        status = ValidationStatus.FAIL

    # 2. Staleness check
    staleness_warning = check_staleness(df, max_staleness_hours)
    staleness_hours = None
    if staleness_warning:
        warnings.append(staleness_warning)
        if status != ValidationStatus.FAIL:
            status = ValidationStatus.WARNING

        # Calculate actual staleness for report
        if not df.empty and "timestamp" in df.columns:
            timestamps = pd.to_datetime(df["timestamp"], utc=True)
            now = datetime.now(timezone.utc)
            newest = timestamps.max()
            if newest.tzinfo is None:
                newest = newest.replace(tzinfo=timezone.utc)
            staleness_hours = (now - newest).total_seconds() / 3600

    # 3. Duplicate check
    duplicate_count = check_duplicates(df)
    if duplicate_count > 0:
        warnings.append(f"Found {duplicate_count} duplicate records")
        if status != ValidationStatus.FAIL:
            status = ValidationStatus.WARNING

    # 4. Missing values check
    missing_values = check_missing_values(df, df.columns.tolist())

    # Report missing values in recommended columns
    for col in recommend_columns:
        if col in missing_values and missing_values[col] > 0:
            pct = missing_values[col] / len(df) * 100 if len(df) > 0 else 0
            warnings.append(f"Column '{col}' has {missing_values[col]} missing values ({pct:.1f}%)")

    report = DataQualityReport(
        status=status,
        total_records=len(df),
        missing_values=missing_values,
        duplicate_count=duplicate_count,
        staleness_hours=staleness_hours,
        warnings=warnings,
        errors=errors,
    )

    logger.info(
        "Validation complete: status=%s, records=%d, warnings=%d, errors=%d",
        status.value,
        report.total_records,
        len(warnings),
        len(errors),
    )

    return report
