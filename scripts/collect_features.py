#!/usr/bin/env python3
"""
Hourly Feature Collection Pipeline.

Collects current weather + pollution from Open-Meteo,
engineers features, and stores in the feature store.

Designed to run every hour via GitHub Actions or cron.

Usage:
    python scripts/collect_features.py
    python scripts/collect_features.py --city karachi
    python scripts/collect_features.py --dry-run
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_environment

load_environment()

from src.data.live_fetcher import (
    CITIES,
    build_features_for_prediction,
    fetch_current_aqi,
    fetch_current_pollution,
    fetch_current_weather,
    fetch_historical_for_features,
)
from src.features.feature_engineering import (
    add_lag_features,
    add_rolling_features,
    add_time_features,
)
from src.utils.epa_aqi import calculate_pm10_aqi, calculate_pm25_aqi

# =============================================================================
# Hopsworks feature-group schema (aqi_features_prod v1) — columns the live
# collector must produce so inserts into Hopsworks actually succeed.
# =============================================================================
FG_SCHEMA_COLUMNS = [
    "timestamp", "location_id", "city_name",
    "temperature", "humidity", "pressure", "wind_speed", "wind_direction",
    "cloud_cover", "precipitation",
    "pm25", "pm10", "co", "no2", "so2", "o3",
    "us_aqi_open_meteo", "us_aqi_pm25_open_meteo", "us_aqi_pm10_open_meteo",
    "aqi", "aqi_lag_1h", "aqi_lag_6h", "aqi_lag_12h", "aqi_lag_24h",
    "aqi_lag_48h", "aqi_lag_72h", "aqi_rolling_mean_6h", "aqi_rolling_mean_12h",
    "aqi_rolling_mean_24h", "aqi_rolling_std_24h", "aqi_rolling_min_24h",
    "aqi_rolling_max_24h",
    "pm25_lag_1h", "pm25_lag_6h", "pm25_lag_12h", "pm25_lag_24h",
    "pm25_lag_48h", "pm25_lag_72h", "pm25_rolling_mean_6h", "pm25_rolling_mean_24h",
    "temperature_lag_1h", "temperature_lag_6h", "temperature_lag_12h",
    "temperature_lag_24h", "temperature_lag_48h", "temperature_lag_72h",
    "temperature_rolling_mean_24h",
    "humidity_lag_1h", "humidity_lag_6h", "humidity_lag_12h", "humidity_lag_24h",
    "humidity_lag_48h", "humidity_lag_72h", "humidity_rolling_mean_24h",
    "hour", "day_of_week", "month", "is_weekend", "season", "hour_sin", "hour_cos",
    "target_aqi_24h", "target_aqi_48h", "target_aqi_72h",
]

# Sensor columns present in the collector's raw record (kept as audit columns
# in the local backup, but NOT part of the Hopsworks feature-group schema).
RAW_AUDIT_COLUMNS = ["pm25_aqi", "pm10_aqi", "data_source", "collected_at",
                     "is_training_valid", "weather_available", "aqi_available"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("collect_features")

# Feature store paths
FEATURES_DIR = PROJECT_ROOT / "data" / "processed" / "features"
HEALTH_LOG = PROJECT_ROOT / "data" / "collection_health.json"


def collect_one_round(city_ids=None, dry_run=False):
    """
    Collect one round of features for all cities.

    Returns:
        dict with collection results
    """
    if city_ids is None:
        city_ids = list(CITIES.keys())

    start_time = time.time()
    results = {
        "scheduled_time": datetime.now(timezone.utc).isoformat(),
        "start_time": None,
        "end_time": None,
        "duration_seconds": 0,
        "cities_attempted": len(city_ids),
        "cities_succeeded": 0,
        "observations_collected": 0,
        "training_valid_observations": 0,
        "invalid_observations": 0,
        "stale_observations": 0,
        "duplicates_rejected": 0,
        "openweather_requests": 0,
        "aqicn_requests": 0,
        "retry_count": 0,
        "failed_requests": 0,
        "hopsworks_persisted": False,
        "local_persisted": False,
        "status": "success",
        "errors": [],
        "city_results": {},
    }

    results["start_time"] = datetime.now(timezone.utc).isoformat()

    all_records = []

    for city_id in city_ids:
        city_name = CITIES[city_id]["name"]
        logger.info(f"Collecting features for {city_name}...")

        try:
            # 1. Fetch current weather
            weather = fetch_current_weather(city_id)
            results["openweather_requests"] += 1
            time.sleep(0.5)  # Rate limiting

            # 2. Fetch current pollution
            pollution = fetch_current_pollution(city_id)
            results["openweather_requests"] += 1
            time.sleep(0.5)

            # 3. Fetch current AQI (for reference/validation)
            aqi_data = fetch_current_aqi(city_id)
            results["openweather_requests"] += 1
            time.sleep(0.5)

            # 4. Fetch historical data for lag/rolling features
            hist_df = fetch_historical_for_features(city_id, hours=96)
            results["openweather_requests"] += 2  # weather + pollution archive

            # 5. Build the current observation record
            # Use the floored hour (UTC) as the observation timestamp so live
            # rows align with the hourly-bucket convention used by the
            # historical ingest (e.g. 17:00), and so Hopsworks upserts on
            # (location_id, timestamp) deduplicate retries within the same hour.
            now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
            record = {
                "timestamp": now,
                "location_id": city_id,
                "city_name": city_name,
                # Weather
                "temperature": weather.get("temperature"),
                "humidity": weather.get("humidity"),
                "pressure": weather.get("pressure"),
                "wind_speed": weather.get("wind_speed"),
                "wind_direction": weather.get("wind_direction"),
                "cloud_cover": weather.get("cloud_cover"),
                "precipitation": weather.get("precipitation"),
                # Pollution
                "pm25": pollution.get("pm25"),
                "pm10": pollution.get("pm10"),
                "co": pollution.get("co"),
                "no2": pollution.get("no2"),
                "so2": pollution.get("so2"),
                "o3": pollution.get("o3"),
                # AQI reference
                "us_aqi": aqi_data.get("us_aqi"),
                "us_aqi_pm25": aqi_data.get("us_aqi_pm25"),
                "us_aqi_pm10": aqi_data.get("us_aqi_pm10"),
                # Source metadata
                "data_source": "open-meteo",
                "weather_available": 1 if weather else 0,
                "aqi_available": 1 if aqi_data else 0,
                "collected_at": now.isoformat(),
            }

            # 6. Calculate EPA AQI from pollutants
            pm25_val = record.get("pm25")
            pm10_val = record.get("pm10")

            if pm25_val is not None and pd.notna(pm25_val):
                record["pm25_aqi"] = calculate_pm25_aqi(pm25_val)
            else:
                record["pm25_aqi"] = None

            if pm10_val is not None and pd.notna(pm10_val):
                record["pm10_aqi"] = calculate_pm10_aqi(pm10_val)
            else:
                record["pm10_aqi"] = None

            # Use higher AQI as the target
            aqi_values = [
                v for v in [record.get("pm25_aqi"), record.get("pm10_aqi")] if v is not None
            ]
            record["aqi"] = max(aqi_values) if aqi_values else None

            # 7. Determine training validity
            # Valid if: has AQI, has weather, and AQI is reasonable (0-500)
            has_aqi = record["aqi"] is not None and 0 <= record["aqi"] <= 500
            has_weather = record["temperature"] is not None and record["humidity"] is not None
            record["is_training_valid"] = has_aqi and has_weather

            if record["is_training_valid"]:
                results["training_valid_observations"] += 1
            else:
                results["invalid_observations"] += 1

            results["observations_collected"] += 1
            results["cities_succeeded"] += 1

            # Store city result
            results["city_results"][city_id] = {
                "aqi": record.get("aqi"),
                "pm25": pm25_val,
                "pm10": pm10_val,
                "temperature": record.get("temperature"),
                "training_valid": record["is_training_valid"],
            }

            record["_history_df"] = hist_df
            all_records.append(record)
            logger.info(
                f"  {city_name}: AQI={record.get('aqi')}, "
                f"PM2.5={pm25_val}, PM10={pm10_val}, "
                f"Valid={record['is_training_valid']}"
            )

        except Exception as e:
            logger.error(f"Failed to collect for {city_name}: {e}")
            results["errors"].append({"city": city_id, "error": str(e)})
            results["failed_requests"] += 1

    # 8. Persist to feature store (Hopsworks PRIMARY, Local FALLBACK)
    if all_records and not dry_run:
        # Each collector record carries its city's recent history (fetched above)
        # so we can engineer the full feature group schema (lags, rolling, time).
        fg_rows = []
        for rec in all_records:
            fg_rows.append(_build_engineered_row(rec))
        df = pd.concat(fg_rows, ignore_index=True) if fg_rows else pd.DataFrame(columns=FG_SCHEMA_COLUMNS)

        # Ensure timestamp is datetime for Hopsworks
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        # Try Hopsworks first (PRIMARY)
        try:
            from src.feature_store import get_feature_store
            from src.feature_store.schemas import DatasetMetadata, DatasetType

            store = get_feature_store()
            logger.info(f"Using feature store: {store.__class__.__name__}")

            metadata = DatasetMetadata(
                dataset_version=f"hourly_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}",
                dataset_type=DatasetType.REAL_TRAINING,
                approved_for_training=True,
                source="open-meteo",
                generation_timestamp=datetime.now(timezone.utc).isoformat(),
                record_count=len(df),
                feature_count=len(df.columns),
            )

            # Insert features
            success = store.insert_features("aqi_features_prod", df, metadata, version=1)

            if success:
                results["hopsworks_persisted"] = True
                logger.info(f"✅ Persisted {len(df)} records to Hopsworks Feature Store")
            else:
                logger.warning("Hopsworks insert returned False")

        except Exception as e:
            logger.warning(f"Hopsworks persistence failed: {e}")

        # Always save locally as backup (full raw + engineered audit columns)
        FEATURES_DIR.mkdir(parents=True, exist_ok=True)
        features_file = FEATURES_DIR / "hourly_observations.parquet"

        # Drop the internal _history_df (DataFrame) before persistence —
        # it cannot be serialized to parquet and is only needed for
        # feature engineering at collect time.
        backup_records = [
            {k: v for k, v in r.items() if k != "_history_df"} for r in all_records
        ]
        raw_backup = pd.DataFrame(backup_records)
        if "timestamp" in raw_backup.columns:
            raw_backup["timestamp"] = pd.to_datetime(raw_backup["timestamp"], utc=True)

        if features_file.exists():
            existing_df = pd.read_parquet(features_file)
            raw_backup = pd.concat([existing_df, raw_backup], ignore_index=True)
            raw_backup = raw_backup.drop_duplicates(
                subset=["timestamp", "location_id"], keep="last"
            )
            raw_backup = raw_backup.sort_values(["location_id", "timestamp"]).reset_index(drop=True)

        raw_backup.to_parquet(features_file, index=False)
        results["local_persisted"] = True
        logger.info(f"✅ Persisted {len(raw_backup)} records to local Parquet (backup)")

        # Save metadata
        meta = {
            "last_collection": datetime.now(timezone.utc).isoformat(),
            "total_records": len(raw_backup),
            "cities": list(raw_backup["location_id"].unique()),
            "training_valid": int(raw_backup["is_training_valid"].sum()),
        }
        meta_file = FEATURES_DIR / "collection_metadata.json"
        with open(meta_file, "w") as f:
            json.dump(meta, f, indent=2)

    # 9. Update health log
    results["end_time"] = datetime.now(timezone.utc).isoformat()
    start_dt = datetime.fromisoformat(results["start_time"])
    end_dt = datetime.fromisoformat(results["end_time"])
    results["duration_seconds"] = round((end_dt - start_dt).total_seconds(), 1)

    if not dry_run:
        _update_health_log(results)

    return results


def _update_health_log(results):
    """Append to the collection health log."""
    HEALTH_LOG.parent.mkdir(parents=True, exist_ok=True)

    # Load existing log
    log_entries = []
    if HEALTH_LOG.exists():
        with open(HEALTH_LOG) as f:
            log_entries = json.load(f)

    # Append new entry (keep last 168 entries = 7 days of hourly)
    log_entries.append(results)
    log_entries = log_entries[-168:]

    with open(HEALTH_LOG, "w") as f:
        json.dump(log_entries, f, indent=2, default=str)

    logger.info(f"Health log updated: {len(log_entries)} entries")


def _build_engineered_row(record: dict) -> pd.DataFrame:
    """Build ONE row matching the Hopsworks aqi_features_prod v1 schema.

    Uses the record's recent pollution/weather history (attached as
    ``_history_df``) to compute the exact feature set the feature group
    expects: EPA AQI sub-indices, time features, lag features (1/6/12/24/48/72h)
    and rolling features, matching ``ingest_to_hopsworks.py``.

    Args:
        record: Collector record containing current observation + history.

    Returns:
        Single-row DataFrame aligned to FG_SCHEMA_COLUMNS.
    """
    from src.features.feature_engineering import (
        add_lag_features,
        add_rolling_features,
        add_time_features,
    )
    from src.utils.epa_aqi import calculate_pm10_aqi, calculate_pm25_aqi

    hist = record.get("_history_df")
    if hist is None or hist.empty:
        # Insufficient history — row cannot have lag/rolling features.
        # Return an empty DataFrame so the caller can emit an honest warning.
        return pd.DataFrame(columns=FG_SCHEMA_COLUMNS)

    df = hist.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["location_id", "timestamp"]).reset_index(drop=True)

    # Ensure the current observation is present as the latest row
    now = pd.Timestamp(record["timestamp"]).tz_localize("UTC") if pd.Timestamp(record["timestamp"]).tzinfo is None else pd.Timestamp(record["timestamp"])
    cur = {
        "timestamp": now,
        "location_id": record["location_id"],
        "city_name": record["city_name"],
        "temperature": record.get("temperature"),
        "humidity": record.get("humidity"),
        "pressure": record.get("pressure"),
        "wind_speed": record.get("wind_speed"),
        "wind_direction": record.get("wind_direction"),
        "cloud_cover": record.get("cloud_cover"),
        "precipitation": record.get("precipitation"),
        "pm25": record.get("pm25"),
        "pm10": record.get("pm10"),
        "co": record.get("co"),
        "no2": record.get("no2"),
        "so2": record.get("so2"),
        "o3": record.get("o3"),
        "us_aqi_open_meteo": record.get("us_aqi"),
        "us_aqi_pm25_open_meteo": record.get("us_aqi_pm25"),
        "us_aqi_pm10_open_meteo": record.get("us_aqi_pm10"),
    }
    if df["timestamp"].max() >= now:
        # History already contains the current hour; just use it as-is
        pass
    else:
        df = pd.concat([df, pd.DataFrame([cur])], ignore_index=True)

    df = df.sort_values(["location_id", "timestamp"]).reset_index(drop=True)
    # No future observations may be used for the CURRENT row: keep only t <= now
    df = df[df["timestamp"] <= now].reset_index(drop=True)

    # 1) EPA AQI sub-indices + dominant AQI (matches ingest_to_hopsworks.py)
    df["pm25_aqi"] = df["pm25"].apply(lambda x: calculate_pm25_aqi(x) if pd.notna(x) else None)
    df["pm10_aqi"] = df["pm10"].apply(lambda x: calculate_pm10_aqi(x) if pd.notna(x) else None)
    df["aqi"] = df[["pm25_aqi", "pm10_aqi"]].max(axis=1)

    # 2) Time features (hour, day_of_week, month, season, cyclical)
    df = add_time_features(df)

    # 3) Lag features — shift within each location (available at t)
    df = add_lag_features(df)

    # 4) Rolling features — time-based, closed='left' (no leakage)
    df = add_rolling_features(df)

    # 5) AQI-specific lags (matches ingest pipeline)
    for lag in [1, 6, 12, 24, 48, 72]:
        df[f"aqi_lag_{lag}h"] = df.groupby("location_id")["aqi"].shift(lag)

    # 6) PM lag features (matches ingest pipeline)
    for lag in [1, 24]:
        df[f"pm25_lag_{lag}h"] = df.groupby("location_id")["pm25"].shift(lag)

    # Take the current observation row (latest)
    row = df.iloc[[-1]].copy()

    # Stamp the row with the CURRENT observation values explicitly. The
    # Open-Meteo history already contains the current hour, so the current
    # observation may not have been appended above; these assignments make
    # the persisted row represent exactly the current collection round.
    row["timestamp"] = now
    row["location_id"] = record["location_id"]
    row["city_name"] = record["city_name"]
    raw_cols = [
        "temperature", "humidity", "pressure", "wind_speed", "wind_direction",
        "cloud_cover", "precipitation", "pm25", "pm10", "co", "no2", "so2", "o3",
    ]
    for col in raw_cols:
        row[col] = record.get(col)
    row["us_aqi_open_meteo"] = record.get("us_aqi")
    row["us_aqi_pm25_open_meteo"] = record.get("us_aqi_pm25")
    row["us_aqi_pm10_open_meteo"] = record.get("us_aqi_pm10")

    # Targets are unknown at collection time (future observations do not
    # exist yet). They are backfilled deterministically by the training
    # pipeline once future hours accumulate in the feature group.
    for col in ["target_aqi_24h", "target_aqi_48h", "target_aqi_72h"]:
        row[col] = None

    # Align to the feature-group schema: add any missing columns as empty,
    # then reorder to exactly FG_SCHEMA_COLUMNS.
    for col in FG_SCHEMA_COLUMNS:
        if col not in row.columns:
            row[col] = None
    row = row[FG_SCHEMA_COLUMNS]

    # Type-cast each numeric column to the EXACT type Hopsworks expects for
    # aqi_features_prod v1 (verified against the live feature-group schema):
    #   double  -> float64 (weather, pollutants, AQI, lags, rolling, targets)
    #   bigint  -> int64   (humidity, wind_direction, cloud_cover, season)
    #   int     -> int32   (hour, day_of_week, month, is_weekend)
    # pandas infers int64 for value ranges with no NaN, which Hopsworks
    # rejects as 'bigint vs double' / 'double vs bigint' mismatches.
    BIGINT_COLS = {"humidity", "wind_direction", "cloud_cover", "season"}
    INT_COLS = {"hour", "day_of_week", "month", "is_weekend"}
    numeric_cols = [
        c for c in FG_SCHEMA_COLUMNS if c not in ("timestamp", "location_id", "city_name")
    ]
    for col in numeric_cols:
        arr = pd.to_numeric(row[col], errors="coerce")
        if col in INT_COLS:
            row[col] = arr.astype("int32") if not arr.isna().any() else arr
        elif col in BIGINT_COLS:
            row[col] = arr.astype("int64") if not arr.isna().any() else arr
        else:
            row[col] = arr.astype("float64")

    return row


def main():
    parser = argparse.ArgumentParser(description="Collect hourly features")
    parser.add_argument("--city", type=str, help="Collect for specific city only")
    parser.add_argument("--dry-run", action="store_true", help="Don't persist data")
    args = parser.parse_args()

    city_ids = [args.city.lower()] if args.city else None

    results = collect_one_round(city_ids=city_ids, dry_run=args.dry_run)

    # Print summary
    print("\n" + "=" * 60)
    print("COLLECTION ROUND SUMMARY")
    print("=" * 60)
    print(f"Cities attempted:  {results['cities_attempted']}")
    print(f"Cities succeeded:  {results['cities_succeeded']}")
    print(f"Observations:      {results['observations_collected']}")
    print(f"Training valid:    {results['training_valid_observations']}")
    print(f"Invalid:           {results['invalid_observations']}")
    print(f"OpenWeather calls: {results['openweather_requests']}")
    print(f"Duration:          {results['duration_seconds']}s")
    print(f"Status:            {results['status']}")
    if results["errors"]:
        print(f"Errors:            {len(results['errors'])}")
    print("=" * 60)

    return 0 if results["status"] == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
