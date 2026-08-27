#!/usr/bin/env python
"""Test live data fetching from Open-Meteo."""
import sys
sys.path.insert(0, ".")

from src.data.live_fetcher import (
    fetch_current_weather,
    fetch_current_pollution,
    fetch_current_aqi,
    CITIES,
)

print("=" * 60)
print("  LIVE DATA FETCH TEST")
print("=" * 60)

for city_id in ["karachi", "lahore", "islamabad"]:
    print(f"\n--- {city_id.upper()} ---")

    # Weather
    try:
        weather = fetch_current_weather(city_id)
        print(f"  Weather: temp={weather['temperature']}C, humidity={weather['humidity']}%, wind={weather['wind_speed']}m/s")
    except Exception as e:
        print(f"  Weather error: {e}")

    # Pollution
    try:
        pollution = fetch_current_pollution(city_id)
        print(f"  Pollution: PM2.5={pollution['pm25']}, PM10={pollution['pm10']}, O3={pollution['o3']}")
    except Exception as e:
        print(f"  Pollution error: {e}")

    # AQI
    try:
        aqi = fetch_current_aqi(city_id)
        print(f"  AQI: US AQI={aqi['us_aqi']}, PM2.5 AQI={aqi['us_aqi_pm25']}, PM10 AQI={aqi['us_aqi_pm10']}")
    except Exception as e:
        print(f"  AQI error: {e}")

print("\n" + "=" * 60)
print("  TEST COMPLETE")
print("=" * 60)
