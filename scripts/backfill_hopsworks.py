#!/usr/bin/env python3
"""
Backfill Hopsworks Feature Store with historical data.

Loads the 4-year historical dataset into Hopsworks feature groups.
This only needs to run once to initialize the feature store.

Usage:
    python scripts/backfill_hopsworks.py
"""

import sys
import logging
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_environment
load_environment()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("backfill_hopsworks")


def main():
    # 1. Load historical data
    features_file = PROJECT_ROOT / "data" / "processed" / "train_features.csv"
    targets_file = PROJECT_ROOT / "data" / "processed" / "train_targets.csv"

    if not features_file.exists():
        logger.error(f"Features file not found: {features_file}")
        return 1

    logger.info("Loading historical features...")
    features_df = pd.read_csv(features_file)
    
    # Convert timestamp to datetime for Hopsworks
    if "timestamp" in features_df.columns:
        features_df["timestamp"] = pd.to_datetime(features_df["timestamp"], utc=True)
    
    # Fix NaN values for Hopsworks string columns
    for col in features_df.select_dtypes(include=["object"]).columns:
        features_df[col] = features_df[col].fillna("")
    
    # Drop columns that Hopsworks schema doesn't expect
    extra_cols = ["aqi_category", "aqi_standard", "aqi_method", "aqi_method_version", 
                  "aqi_derived", "aqi_source"]
    features_df = features_df.drop(columns=[c for c in extra_cols if c in features_df.columns])
    
    logger.info(f"Loaded {len(features_df)} feature rows, {len(features_df.columns)} columns")

    logger.info("Loading historical targets...")
    targets_df = pd.read_csv(targets_file)
    
    # Convert timestamp to datetime for Hopsworks
    if "timestamp" in targets_df.columns:
        targets_df["timestamp"] = pd.to_datetime(targets_df["timestamp"], utc=True)
    
    logger.info(f"Loaded {len(targets_df)} target rows")

    # 2. Connect to Hopsworks
    from src.feature_store import get_feature_store
    store = get_feature_store()
    logger.info(f"Connected to: {store.__class__.__name__}")

    if store.__class__.__name__ != "HopsworksStore":
        logger.error("Not connected to Hopsworks. Check HOPSWORKS_HOST.")
        return 1

    # 3. Create dataset metadata
    from src.feature_store.schemas import DatasetMetadata, DatasetType

    metadata = DatasetMetadata(
        dataset_version="v1.0_historical_4years",
        dataset_type=DatasetType.REAL_TRAINING,
        approved_for_training=True,
        source="open-meteo-historical",
        generation_timestamp="2026-08-27T00:00:00Z",
        record_count=len(features_df),
        feature_count=len(features_df.columns),
    )

    # 4. Insert features (chunked to avoid timeouts)
    CHUNK_SIZE = 5000
    total_chunks = (len(features_df) + CHUNK_SIZE - 1) // CHUNK_SIZE

    logger.info(f"Inserting {len(features_df)} rows in {total_chunks} chunks...")

    success = False
    for i in range(0, len(features_df), CHUNK_SIZE):
        chunk = features_df.iloc[i:i + CHUNK_SIZE]
        chunk_num = i // CHUNK_SIZE + 1
        logger.info(f"  Chunk {chunk_num}/{total_chunks}: {len(chunk)} rows")

        result = store.insert_features(
            "aqi_features_prod", chunk, metadata, version=1
        )
        if result:
            success = True
        else:
            logger.warning(f"  Chunk {chunk_num} insert returned False")

    # 5. Insert targets
    logger.info(f"Inserting {len(targets_df)} target rows...")
    store.insert_targets(
        "aqi_targets_prod", targets_df, metadata, version=1
    )

    if success:
        logger.info("✅ Hopsworks backfill complete!")
    else:
        logger.warning("⚠️ Some chunks failed. Check logs.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
