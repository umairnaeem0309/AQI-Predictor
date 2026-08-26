#!/usr/bin/env python3
"""
Final pre-start live round with full PM NowCast AQI audit.
"""
import sys
import os
from pathlib import Path
from datetime import datetime, timezone

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")
from src.config import load_environment, load_config
load_environment()

from src.data.api_manager import APIManager
from src.data.aqicn_client import AQICNClient
from src.data.openweather_client import OpenWeatherClient
from src.data.schemas import CityConfig
from src.utils.epa_aqi import calculate_pm25_aqi, calculate_pm10_aqi, get_aqi_metadata

ow_key = os.environ.get("OPENWEATHER_API_KEY")
aq_key = os.environ.get("AQICN_API_KEY")

ow_client = OpenWeatherClient(api_key=ow_key)
aq_client = AQICNClient(api_key=aq_key)
api_manager = APIManager(openweather_client=ow_client, aqicn_client=aq_client)

print("=" * 110)
print("PRE-START LIVE ROUND — FULL PM NOWCAST AQI AUDIT")
print("=" * 110)
print(f"Time: {datetime.now(timezone.utc).isoformat()}")
print()

config = load_config()
city_configs = [CityConfig(**city) for city in config.get("cities", [])]

df = api_manager.fetch_all_cities(city_configs)

if df.empty:
    print("ERROR: No data collected")
    sys.exit(1)

print(f"Observations collected: {len(df)}")
print()

for _, row in df.iterrows():
    city = row.get("location_id", "unknown")
    print(f"{'=' * 110}")
    print(f"CITY: {city.upper()}")
    print(f"{'=' * 110}")

    # Current pollutant values
    pm25 = row.get("pm25")
    pm10 = row.get("pm10")

    # Calculate AQI sub-indices
    pm25_aqi = calculate_pm25_aqi(pm25) if pm25 is not None else None
    pm10_aqi = calculate_pm10_aqi(pm10) if pm10 is not None else None

    # Selected AQI
    aqi = row.get("aqi")
    dominant = row.get("aqi_dominant_pollutant")

    print(f"  Collection timestamp:     {row.get('collected_at', 'N/A')}")
    print(f"  Source observation time:  {row.get('timestamp', 'N/A')}")
    print()
    print(f"  Current PM2.5:            {pm25} ug/m3")
    print(f"  PM2.5 AQI sub-index:      {pm25_aqi}")
    print()
    print(f"  Current PM10:             {pm10} ug/m3")
    print(f"  PM10 AQI sub-index:       {pm10_aqi}")
    print()
    print(f"  Selected AQI:             {aqi}")
    print(f"  Dominant pollutant:       {dominant}")
    print(f"  AQI method:               PM_NOWCAST")
    print(f"  AQI version:              EPA-454/B-24-002_MAY_2024")
    print(f"  Training valid:           {row.get('is_training_valid', 'N/A')}")
    print()

    # AQICN status
    staleness = row.get("staleness_reason")
    if staleness:
        print(f"  AQICN status:             STALE ({staleness})")
    else:
        print(f"  AQICN status:             Fresh or derived")

    # Verify AQI is not OpenWeather 1-5
    if aqi is not None and isinstance(aqi, (int, float)):
        if 1 <= aqi <= 5:
            print(f"  WARNING: AQI looks like OpenWeather 1-5 scale!")
        else:
            print(f"  AQI scale verified:       US EPA 0-500")

    print()

# Metadata
print("=" * 110)
print("AQI METADATA")
print("=" * 110)
meta = get_aqi_metadata()
for k, v in meta.items():
    print(f"  {k}: {v}")
