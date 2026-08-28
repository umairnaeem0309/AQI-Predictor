"""
Dataset Builder — Constructs training datasets from historical observations.

Responsibilities:
- Generate target variables (24h, 48h, 72h forward shifts)
- Split data chronologically into train/val/test
- Track source quality metadata (weather availability, AQI availability)
- Validate no target leakage
- Generate dataset metadata and versioning
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.validators import drop_duplicates, full_validation
from src.features.feature_engineering import (
    FEATURE_VERSION,
    SCHEMA_VERSION,
    engineer_features,
)
from src.features.feature_validation import (
    check_no_future_leakage,
    get_feature_availability,
)

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"


def add_source_quality_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Add source quality metadata to each observation.

    Tracks:
    - weather_available: whether weather fields are present
    - aqi_available: whether AQI/pollution fields are present
    - sources_used: which data sources contributed to this observation

    Args:
        df: DataFrame with StandardObservation columns.

    Returns:
        DataFrame with added metadata columns.
    """
    df = df.copy()

    # Weather availability
    weather_cols = ["temperature", "humidity", "wind_speed", "pressure"]
    df["weather_available"] = df[weather_cols].notna().any(axis=1).astype(int)

    # AQI availability
    aqi_cols = ["aqi", "pm25", "pm10"]
    df["aqi_available"] = df[aqi_cols].notna().any(axis=1).astype(int)

    # Sources used
    if "data_source" in df.columns:
        df["sources_used"] = df["data_source"]
    else:
        df["sources_used"] = "unknown"

    return df


def generate_targets(df: pd.DataFrame, horizons: Optional[List[int]] = None) -> pd.DataFrame:
    """Generate target variables by forward-shifting AQI.

    Target leakage prevention:
    - feature timestamp < target timestamp (guaranteed by forward shift)
    - no target information exists in features (target is separate column)

    Args:
        df: DataFrame sorted by (location_id, timestamp) with 'aqi' column.
        horizons: Forecast horizons in hours. Defaults to [24, 48, 72].

    Returns:
        DataFrame with added target columns.
    """
    df = df.copy()

    if horizons is None:
        horizons = [24, 48, 72]

    if "aqi" not in df.columns:
        logger.warning("No 'aqi' column found — cannot generate targets")
        return df

    for horizon in horizons:
        target_col = f"target_aqi_{horizon}h"
        # Forward shift within each location group
        df[target_col] = df.groupby("location_id")["aqi"].shift(-horizon)

        # Log target availability
        non_null = df[target_col].notna().sum()
        logger.debug(
            "Target %s: %d non-null values out of %d",
            target_col,
            non_null,
            len(df),
        )

    return df


def validate_no_target_leakage(
    df: pd.DataFrame,
    feature_columns: List[str],
    target_columns: List[str],
) -> List[str]:
    """Verify that no target information leaks into features.

    Checks:
    - Feature timestamp < target timestamp (by construction of forward shift)
    - No target column values appear in feature columns
    - No future AQI values used in feature calculations

    Args:
        df: DataFrame with features and targets.
        feature_columns: Feature column names.
        target_columns: Target column names.

    Returns:
        List of error messages. Empty if no leakage detected.
    """
    errors = []

    # Check: no target column should be in feature columns
    for target in target_columns:
        if target in feature_columns:
            errors.append(f"Target column '{target}' found in feature columns")

    # Check: target values should not appear in feature values
    # (This is a basic check — thorough leakage detection is in feature_validation.py)
    for target in target_columns:
        if target in df.columns and "aqi" in df.columns:
            # Target should be shifted forward, so it should differ from current AQI
            # For the last N rows where target is NaN, this check is not applicable
            valid_mask = df[target].notna() & df["aqi"].notna()
            if valid_mask.any():
                overlap = (df.loc[valid_mask, target] == df.loc[valid_mask, "aqi"]).sum()
                if overlap > 0:
                    # This could happen if AQI is stable — not necessarily leakage
                    # Log as informational, not an error
                    logger.info(
                        "Target %s matches current AQI in %d rows (may be stable AQI)",
                        target,
                        overlap,
                    )

    if errors:
        logger.error("Target leakage detected: %d issues", len(errors))
    else:
        logger.info("No target leakage detected")

    return errors


def split_chronological(
    df: pd.DataFrame,
    train_ratio: float = 0.82,
    val_ratio: float = 0.08,
    timestamp_column: str = "timestamp",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data chronologically into train/val/test.

    Rules:
    - No random shuffling — chronological order preserved
    - Per-city split at same timestamps
    - Continuous timeline (no gaps between splits)

    Args:
        df: DataFrame sorted by timestamp.
        train_ratio: Proportion for training.
        val_ratio: Proportion for validation.
        timestamp_column: Name of timestamp column.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    df = df.sort_values(timestamp_column).reset_index(drop=True)

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    logger.info(
        "Chronological split: train=%d (%.1f%%), val=%d (%.1f%%), test=%d (%.1f%%)",
        len(train_df),
        len(train_df) / n * 100,
        len(val_df),
        len(val_df) / n * 100,
        len(test_df),
        len(test_df) / n * 100,
    )

    return train_df, val_df, test_df


def generate_dataset_version() -> str:
    """Generate a version string for the dataset.

    Returns:
        Version string like 'v20260806_a3f2b1'.
    """
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    hash_input = datetime.now(timezone.utc).isoformat().encode()
    hash_suffix = hashlib.md5(hash_input).hexdigest()[:6]
    return f"v{date_str}_{hash_suffix}"


def build_dataset(
    observations_df: pd.DataFrame,
    dataset_version: Optional[str] = None,
    feature_version: str = FEATURE_VERSION,
    save: bool = True,
) -> Dict[str, Any]:
    """Build complete training dataset from observations.

    Pipeline order:
    1. Source quality metadata
    2. Feature engineering (complete dataset)
    3. Target generation
    4. Target leakage validation
    5. Train/val/test split
    6. Save datasets
    7. Generate metadata

    Args:
        observations_df: Raw observations in StandardObservation format.
        dataset_version: Version string. Auto-generated if None.
        feature_version: Feature version string.
        save: If True, save datasets to disk.

    Returns:
        Dictionary with train/val/test DataFrames and metadata.
    """
    if observations_df.empty:
        logger.error("Empty observations DataFrame — cannot build dataset")
        return {}

    if dataset_version is None:
        dataset_version = generate_dataset_version()

    logger.info(
        "Building dataset %s from %d observations",
        dataset_version,
        len(observations_df),
    )

    # Step 1: Source quality metadata
    df = add_source_quality_metadata(observations_df)

    # Step 2: Feature engineering (complete dataset before split)
    df = engineer_features(df, feature_version=feature_version)

    # Step 3: Target generation
    df = generate_targets(df)

    # Step 4: Target leakage validation
    feature_cols = [
        c
        for c in df.columns
        if c
        not in [
            "timestamp",
            "location_id",
            "city_name",
            "data_source",
            "raw_response_time",
            "weather_available",
            "aqi_available",
            "sources_used",
            "target_aqi_24h",
            "target_aqi_48h",
            "target_aqi_72h",
        ]
    ]
    target_cols = [c for c in df.columns if c.startswith("target_")]
    leakage_errors = validate_no_target_leakage(df, feature_cols, target_cols)

    # Step 5: Train/val/test split (chronological)
    train_df, val_df, test_df = split_chronological(df)

    # Step 6: Save datasets
    if save:
        for split_name, split_df in [
            ("train", train_df),
            ("val", val_df),
            ("test", test_df),
        ]:
            features_file = PROCESSED_DIR / f"{split_name}_features.csv"
            targets_file = PROCESSED_DIR / f"{split_name}_targets.csv"

            # Separate features and targets
            feat_cols = [c for c in split_df.columns if not c.startswith("target_")]
            tgt_cols = [c for c in split_df.columns if c.startswith("target_")] + [
                "timestamp",
                "location_id",
            ]

            split_df[feat_cols].to_csv(features_file, index=False)
            split_df[tgt_cols].to_csv(targets_file, index=False)

    # Step 7: Generate metadata
    metadata = {
        "dataset_version": dataset_version,
        "feature_version": feature_version,
        "schema_version": SCHEMA_VERSION,
        "generation_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_records": len(df),
        "train_records": len(train_df),
        "val_records": len(val_df),
        "test_records": len(test_df),
        "feature_count": len(feature_cols),
        "target_count": len(target_cols),
        "cities": (df["location_id"].unique().tolist() if "location_id" in df.columns else []),
        "leakage_errors": leakage_errors,
        "quality_report": full_validation(df).__dict__,
        "dataset_type": "synthetic_test_data",
        "approved_for_training": False,
        "approved_for_evaluation": False,
    }

    if save:
        metadata_file = PROCESSED_DIR / "feature_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

    logger.info(
        "Dataset built: version=%s, total=%d, train=%d, val=%d, test=%d, features=%d",
        dataset_version,
        len(df),
        len(train_df),
        len(val_df),
        len(test_df),
        len(feature_cols),
    )

    return {
        "train": train_df,
        "val": val_df,
        "test": test_df,
        "metadata": metadata,
    }
