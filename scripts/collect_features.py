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
            now = datetime.now(timezone.utc)
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
            aqi_values = [v for v in [record.get("pm25_aqi"), record.get("pm10_aqi")] if v is not None]
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

    # 8. Persist to feature store
    if all_records and not dry_run:
        df = pd.DataFrame(all_records)

        # Save to local feature store (Parquet)
        FEATURES_DIR.mkdir(parents=True, exist_ok=True)
        features_file = FEATURES_DIR / "hourly_observations.parquet"

        if features_file.exists():
            existing_df = pd.read_parquet(features_file)
            # Deduplicate by timestamp + location_id
            df = pd.concat([existing_df, df], ignore_index=True)
            df = df.drop_duplicates(subset=["timestamp", "location_id"], keep="last")
            df = df.sort_values(["location_id", "timestamp"]).reset_index(drop=True)

            # Count duplicates rejected
            results["duplicates_rejected"] = len(all_records) - len(
                df[df["timestamp"].isin([r["timestamp"] for r in all_records])]
            )

        df.to_parquet(features_file, index=False)
        results["local_persisted"] = True
        logger.info(f"Persisted {len(df)} total records to local feature store")

        # Save metadata
        meta = {
            "last_collection": datetime.now(timezone.utc).isoformat(),
            "total_records": len(df),
            "cities": list(df["location_id"].unique()),
            "training_valid": int(df["is_training_valid"].sum()),
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
