#!/usr/bin/env python3
"""
Hopsworks Feature Ingestion Pipeline.

Fetches historical weather + air quality data from Open-Meteo,
engineers features AND targets, and stores them in a SINGLE
Hopsworks Feature Group.

This replaces the old approach of storing features and targets
in separate local CSV files.

Usage:
    python scripts/ingest_to_hopsworks.py
    python scripts/ingest_to_hopsworks.py --start-date 2022-08-01 --end-date 2026-08-31
    python scripts/ingest_to_hopsworks.py --force-recreate
"""

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_environment

load_environment()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ingest_to_hopsworks")

# City configurations
CITIES = {
    "karachi": {
        "name": "Karachi",
        "latitude": 24.8607,
        "longitude": 67.0011,
    },
    "lahore": {
        "name": "Lahore",
        "latitude": 31.5204,
        "longitude": 74.3587,
    },
    "islamabad": {
        "name": "Islamabad",
        "latitude": 33.6844,
        "longitude": 73.0479,
    },
}


def fetch_historical_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch historical weather + air quality from Open-Meteo.

    Returns:
        DataFrame with raw hourly observations for all cities.
    """
    from src.data.providers.open_meteo_air_quality import OpenMeteoAirQualityProvider
    from src.data.providers.open_meteo_weather import OpenMeteoWeatherProvider

    weather_provider = OpenMeteoWeatherProvider()
    aqi_provider = OpenMeteoAirQualityProvider()

    all_weather = []
    all_aqi = []

    for city_id, city_info in CITIES.items():
        logger.info(f"Fetching data for {city_info['name']}...")

        # Weather
        weather_df = weather_provider.fetch_historical(
            latitude=city_info["latitude"],
            longitude=city_info["longitude"],
            location_id=city_id,
            city_name=city_info["name"],
            start_date=start_date,
            end_date=end_date,
        )
        all_weather.append(weather_df)

        # Air quality
        aqi_df = aqi_provider.fetch_historical(
            latitude=city_info["latitude"],
            longitude=city_info["longitude"],
            location_id=city_id,
            city_name=city_info["name"],
            start_date=start_date,
            end_date=end_date,
        )
        all_aqi.append(aqi_df)

    weather_all = pd.concat(all_weather, ignore_index=True)
    aqi_all = pd.concat(all_aqi, ignore_index=True)

    logger.info(f"Weather rows: {len(weather_all)}, AQI rows: {len(aqi_all)}")

    # Merge weather + air quality on (timestamp, location_id)
    df = pd.merge(
        weather_all,
        aqi_all,
        on=["timestamp", "location_id", "city_name"],
        how="inner",
        suffixes=("", "_aqi"),
    )

    # Drop duplicate columns from merge
    duplicate_suffix_cols = [c for c in df.columns if c.endswith("_aqi")]
    df = df.drop(columns=duplicate_suffix_cols, errors="ignore")

    logger.info(f"Merged dataset: {len(df)} rows, {len(df.columns)} columns")
    return df


def engineer_features_and_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features and create target variables.

    This is the SINGLE source of truth for feature engineering.
    Features and targets are stored together in Hopsworks.
    """
    from src.features.feature_engineering import (
        add_lag_features,
        add_rolling_features,
        add_time_features,
    )
    from src.utils.epa_aqi import calculate_pm10_aqi, calculate_pm25_aqi

    # Ensure timestamp is datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Sort for chronological processing
    df = df.sort_values(["location_id", "timestamp"]).reset_index(drop=True)

    # Calculate EPA AQI from pollutant concentrations
    logger.info("Calculating EPA AQI from pollutant concentrations...")
    df["pm25_aqi"] = df["pm25"].apply(lambda x: calculate_pm25_aqi(x) if pd.notna(x) else None)
    df["pm10_aqi"] = df["pm10"].apply(lambda x: calculate_pm10_aqi(x) if pd.notna(x) else None)
    # AQI = max(PM2.5 AQI, PM10 AQI)
    df["aqi"] = df[["pm25_aqi", "pm10_aqi"]].max(axis=1)

    # Time features
    df = add_time_features(df)

    # Lag features
    df = add_lag_features(df)

    # Rolling features
    df = add_rolling_features(df)

    # AQI-specific lags
    for lag in [1, 6, 12, 24, 48, 72]:
        df[f"aqi_lag_{lag}h"] = df.groupby("location_id")["aqi"].shift(lag)

    # PM lags
    for lag in [1, 24]:
        df[f"pm25_lag_{lag}h"] = df.groupby("location_id")["pm25"].shift(lag)

    # TARGETS: Forward-shift AQI by 24h, 48h, 72h
    logger.info("Creating target variables (forward-shifted AQI)...")
    for horizon, col_name in [
        (24, "target_aqi_24h"),
        (48, "target_aqi_48h"),
        (72, "target_aqi_72h"),
    ]:
        df[col_name] = df.groupby("location_id")["aqi"].shift(-horizon)

    # Drop rows with NaN targets (trailing rows that can't have targets)
    target_cols = ["target_aqi_24h", "target_aqi_48h", "target_aqi_72h"]
    before_drop = len(df)
    df = df.dropna(subset=target_cols)
    logger.info(
        f"Dropped {before_drop - len(df)} rows with missing targets, " f"{len(df)} remaining"
    )

    # Clean up intermediate columns
    cols_to_drop = ["pm25_aqi", "pm10_aqi", "data_source", "provider"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

    # Drop any remaining string columns (except location_id and city_name)
    string_cols = df.select_dtypes(include=["object"]).columns
    non_meta_strings = [c for c in string_cols if c not in ["location_id", "city_name"]]
    df = df.drop(columns=non_meta_strings, errors="ignore")

    # Fill NaN with 0 for numeric columns (Hopsworks requirement)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0.0)

    # Replace inf with 0
    df = df.replace([np.inf, -np.inf], 0.0)

    logger.info(f"Final dataset: {len(df)} rows, {len(df.columns)} columns")
    return df


def create_hopsworks_feature_group(
    df: pd.DataFrame,
    group_name: str = "aqi_features_prod",
    version: int = 1,
    force_recreate: bool = False,
) -> bool:
    """Store features + targets in a SINGLE Hopsworks Feature Group.

    This is the core requirement: features AND targets in one group,
    not separate files or groups.
    """
    import hopsworks

    from src.feature_store.schemas import DatasetMetadata, DatasetType

    host = os.environ.get("HOPSWORKS_HOST")
    api_key = os.environ.get("HOPSWORKS_API_KEY")
    project_name = os.environ.get("HOPSWORKS_PROJECT", "AQI_Predictor")

    if not host or not api_key:
        logger.error("HOPSWORKS_HOST and HOPSWORKS_API_KEY must be set")
        return False

    # Connect
    project = hopsworks.login(
        host=host,
        api_key_value=api_key,
        project=project_name,
    )
    fs = project.get_feature_store()

    # Delete existing if force recreate
    if force_recreate:
        try:
            fg = fs.get_feature_group(name=group_name, version=version)
            fg.delete()
            logger.info(f"Deleted existing feature group: {group_name} v{version}")
        except Exception:
            pass

    # Prepare DataFrame for Hopsworks
    df_hops = df.copy()

    # Ensure timestamp is datetime
    df_hops["timestamp"] = pd.to_datetime(df_hops["timestamp"], utc=True)

    # Hopsworks requires string columns to not have NaN
    for col in df_hops.select_dtypes(include=["object"]).columns:
        df_hops[col] = df_hops[col].fillna("")

    # Define primary key and event time
    primary_key = ["location_id", "timestamp"]
    event_time = "timestamp"

    # Create or get feature group with ALL columns (features + targets)
    fg = fs.get_or_create_feature_group(
        name=group_name,
        version=version,
        primary_key=primary_key,
        event_time=event_time,
        description=(
            "AQI prediction features + targets. "
            "Features: weather, pollution, time, lags, rolling, derived. "
            "Targets: target_aqi_24h, target_aqi_48h, target_aqi_72h."
        ),
        online_enabled=False,
        time_travel_format="HUDI",
    )

    # Insert data
    logger.info(f"Inserting {len(df_hops)} rows into {group_name} v{version}...")
    fg.insert(
        df_hops,
        write_options={"hoodie.bulkinsert.shuffle.parallelism": 1},
    )

    logger.info(f"✅ Stored {len(df_hops)} rows in Hopsworks: {group_name} v{version}")
    logger.info(f"   Columns: {len(df_hops.columns)}")
    logger.info(f"   Features: {len([c for c in df_hops.columns if not c.startswith('target_')])}")
    logger.info(f"   Targets: {len([c for c in df_hops.columns if c.startswith('target_')])}")

    return True


def create_feature_view(
    group_name: str = "aqi_features_prod",
    version: int = 1,
    view_name: str = "aqi_feature_view",
) -> bool:
    """Create a Hopsworks Feature View that designates the target column.

    This implements the Feature View requirement:
    - Selects all features from the feature group
    - Explicitly designates the target column as the label
    """
    import hopsworks

    host = os.environ.get("HOPSWORKS_HOST")
    api_key = os.environ.get("HOPSWORKS_API_KEY")
    project_name = os.environ.get("HOPSWORKS_PROJECT", "AQI_Predictor")

    project = hopsworks.login(
        host=host,
        api_key_value=api_key,
        project=project_name,
    )
    fs = project.get_feature_store()

    # Get the feature group
    fg = fs.get_feature_group(name=group_name, version=version)

    # Target columns
    target_columns = ["target_aqi_24h", "target_aqi_48h", "target_aqi_72h"]

    # Create feature view with labels
    # The label parameter tells Hopsworks which columns are targets
    try:
        fv = fs.get_feature_view(name=view_name, version=version)
        logger.info(f"Feature view '{view_name}' v{version} already exists")
    except Exception:
        fv = fs.create_feature_view(
            name=view_name,
            version=version,
            query=fg.select_all(),
            labels=target_columns,
            description="AQI prediction feature view with 24h/48h/72h targets",
        )
        logger.info(f"✅ Created feature view: {view_name} v{version}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Ingest historical data into Hopsworks Feature Store"
    )
    parser.add_argument(
        "--start-date",
        default="2022-08-01",
        help="Start date (YYYY-MM-DD). Default: 2022-08-01",
    )
    parser.add_argument(
        "--end-date",
        default=datetime.now().strftime("%Y-%m-%d"),
        help="End date (YYYY-MM-DD). Default: today",
    )
    parser.add_argument(
        "--force-recreate",
        action="store_true",
        help="Delete and recreate the feature group",
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip API fetch, use local CSV if available",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("HOPSWORKS FEATURE INGESTION PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Date range: {args.start_date} to {args.end_date}")
    logger.info(f"Force recreate: {args.force_recreate}")

    # Step 1: Fetch or load data
    if args.skip_fetch:
        # Load from local CSV as fallback
        features_file = PROJECT_ROOT / "data" / "processed" / "train_features.csv"
        targets_file = PROJECT_ROOT / "data" / "processed" / "train_targets.csv"

        if features_file.exists() and targets_file.exists():
            logger.info("Loading from local CSV files...")
            features_df = pd.read_csv(features_file)
            targets_df = pd.read_csv(targets_file)
            df = pd.merge(features_df, targets_df, on=["timestamp", "location_id"], how="inner")
        else:
            logger.error("Local CSV files not found. Cannot use --skip-fetch.")
            return 1
    else:
        # Fetch from Open-Meteo API
        logger.info("Fetching historical data from Open-Meteo...")
        df = fetch_historical_data(args.start_date, args.end_date)

        if df.empty:
            logger.error("No data fetched from Open-Meteo")
            return 1

    # Step 2: Engineer features + targets
    logger.info("Engineering features and targets...")
    df = engineer_features_and_targets(df)

    if len(df) < 100:
        logger.error(f"Too few rows after engineering: {len(df)}")
        return 1

    # Step 3: Store in Hopsworks (features + targets together)
    logger.info("Storing in Hopsworks Feature Store...")
    success = create_hopsworks_feature_group(
        df,
        group_name="aqi_features_prod",
        version=1,
        force_recreate=args.force_recreate,
    )

    if not success:
        logger.error("Failed to store in Hopsworks")
        return 1

    # Step 4: Create Feature View with target designation
    logger.info("Creating Feature View...")
    create_feature_view(
        group_name="aqi_features_prod",
        version=1,
        view_name="aqi_feature_view",
    )

    # Step 5: Save local backup
    backup_dir = PROJECT_ROOT / "data" / "processed"
    backup_dir.mkdir(parents=True, exist_ok=True)

    # Save features (without targets) for API fallback
    feature_cols = [c for c in df.columns if not c.startswith("target_")]
    df[feature_cols].to_csv(backup_dir / "train_features.csv", index=False)

    # Save targets separately for API fallback
    target_cols = ["timestamp", "location_id"] + [c for c in df.columns if c.startswith("target_")]
    df[target_cols].to_csv(backup_dir / "train_targets.csv", index=False)

    logger.info(f"Local backup saved to {backup_dir}")

    # Summary
    logger.info("=" * 60)
    logger.info("INGESTION COMPLETE")
    logger.info(f"  Total rows: {len(df)}")
    logger.info(f"  Cities: {df['location_id'].nunique()}")
    logger.info(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    logger.info(f"  Features: {len([c for c in df.columns if not c.startswith('target_')])}")
    logger.info(f"  Targets: {len([c for c in df.columns if c.startswith('target_')])}")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
