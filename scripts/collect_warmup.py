#!/usr/bin/env python3
"""
Collect 7-day historical pollution warm-up for all 3 cities.
Stores as warm-up context data for NowCast initialization.
"""
import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")
from src.config import load_environment
load_environment()

import pandas as pd
import requests

ow_key = os.environ.get("OPENWEATHER_API_KEY")

CITIES = {
    "karachi": {"lat": 24.8607, "lon": 67.0011, "name": "Karachi"},
    "lahore": {"lat": 31.5204, "lon": 74.3587, "name": "Lahore"},
    "islamabad": {"lat": 33.6844, "lon": 73.0479, "name": "Islamabad"},
}

output_dir = project_root / "data" / "warmup"
output_dir.mkdir(parents=True, exist_ok=True)

now = int(time.time())
seven_days_ago = now - (7 * 24 * 3600)

print("=" * 90)
print("7-DAY POLLUTION WARM-UP COLLECTION")
print("=" * 90)
print(f"Start: {datetime.now(timezone.utc).isoformat()}")
print(f"Period: {datetime.fromtimestamp(seven_days_ago, tz=timezone.utc).isoformat()} to {datetime.fromtimestamp(now, tz=timezone.utc).isoformat()}")
print()

all_warmup = []

for city_id, city_info in CITIES.items():
    print(f"--- {city_info['name'].upper()} ---")

    url = "https://api.openweathermap.org/data/2.5/air_pollution/history"
    params = {
        "lat": city_info["lat"],
        "lon": city_info["lon"],
        "start": seven_days_ago,
        "end": now,
        "appid": ow_key,
    }

    resp = requests.get(url, params=params, timeout=60)
    if resp.status_code != 200:
        print(f"  ERROR: HTTP {resp.status_code}")
        continue

    data = resp.json()
    observations = data.get("list", [])
    print(f"  Raw observations: {len(observations)}")

    # Parse into DataFrame
    rows = []
    for obs in observations:
        dt = datetime.fromtimestamp(obs["dt"], tz=timezone.utc)
        components = obs.get("components", {})
        rows.append({
            "timestamp": dt.isoformat(),
            "location_id": city_id,
            "city_name": city_info["name"],
            "pm25": components.get("pm2_5"),
            "pm10": components.get("pm10"),
            "o3": components.get("o3"),
            "no2": components.get("no2"),
            "so2": components.get("so2"),
            "co": components.get("co"),
            "nh3": components.get("nh3"),
            "temperature": None,  # No weather data in pollution endpoint
            "humidity": None,
            "wind_speed": None,
            "pressure": None,
            "weather_condition": None,
            "data_type": "warmup_pollution",
            "collection_purpose": "nowcast_initialization",
        })

    city_df = pd.DataFrame(rows)

    # Validate
    city_df["timestamp"] = pd.to_datetime(city_df["timestamp"], utc=True)
    city_df = city_df.sort_values("timestamp").reset_index(drop=True)

    # Check for duplicates
    dups = city_df.duplicated(subset=["timestamp", "location_id"]).sum()
    print(f"  Duplicates: {dups}")

    # Check PM2.5 completeness
    pm25_valid = city_df["pm25"].notna().sum()
    pm25_total = len(city_df)
    print(f"  PM2.5 completeness: {pm25_valid}/{pm25_total} ({pm25_valid/pm25_total*100:.1f}%)")

    # Check PM10 completeness
    pm10_valid = city_df["pm10"].notna().sum()
    print(f"  PM10 completeness: {pm10_valid}/{pm25_total} ({pm10_valid/pm25_total*100:.1f}%)")

    # Check for negative concentrations
    neg_pm25 = (city_df["pm25"] < 0).sum() if city_df["pm25"].notna().any() else 0
    neg_pm10 = (city_df["pm10"] < 0).sum() if city_df["pm10"].notna().any() else 0
    print(f"  Negative PM2.5: {neg_pm25}")
    print(f"  Negative PM10: {neg_pm10}")

    # Check timestamp range
    print(f"  Time range: {city_df['timestamp'].min()} to {city_df['timestamp'].max()}")

    # Save warm-up
    city_file = output_dir / f"warmup_{city_id}.csv"
    city_df.to_csv(city_file, index=False)
    print(f"  Saved: {city_file}")

    all_warmup.append(city_df)

# Combine all cities
if all_warmup:
    combined = pd.concat(all_warmup, ignore_index=True)
    combined_file = output_dir / "warmup_all_cities.csv"
    combined.to_csv(combined_file, index=False)
    print(f"\nCombined warm-up: {len(combined)} rows")
    print(f"Saved: {combined_file}")

    # Summary
    print("\n" + "=" * 90)
    print("WARM-UP SUMMARY")
    print("=" * 90)
    for city_id in CITIES:
        city_data = combined[combined["location_id"] == city_id]
        print(f"  {city_id:<12} {len(city_data):>4} rows  "
              f"PM2.5 valid: {city_data['pm25'].notna().sum():>4}  "
              f"PM10 valid: {city_data['pm10'].notna().sum():>4}")

print(f"\nEnd: {datetime.now(timezone.utc).isoformat()}")
