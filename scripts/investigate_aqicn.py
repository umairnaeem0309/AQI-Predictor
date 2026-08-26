#!/usr/bin/env python3
"""Investigate AQICN station freshness for Karachi, Lahore, Islamabad."""
import os
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.config import load_environment
load_environment()

import requests

API_KEY = os.environ.get("AQICN_API_KEY")
CITIES = ["karachi", "lahore", "islamabad"]

def investigate_city(city):
    """Full investigation for one city."""
    print(f"\n{'='*60}")
    print(f"INVESTIGATION: {city.upper()}")
    print(f"{'='*60}")

    # 1. City-level feed (current method)
    print(f"\n--- City-level feed: /feed/{city}/ ---")
    r = requests.get(f"https://api.waqi.info/feed/{city}/?token={API_KEY}", timeout=15)
    data = r.json()
    if data.get("status") == "ok":
        d = data["data"]
        station = d.get("station", {})
        time_info = d.get("time", {})
        print(f"  Station name: {station.get('name')}")
        print(f"  Station geo: {station.get('geo')}")
        print(f"  AQI: {d.get('aqi')}")
        print(f"  Time.s (source timestamp): {time_info.get('s')}")
        print(f"  Time.iso: {time_info.get('iso')}")
        print(f"  Time.tz: {time_info.get('tz')}")
        print(f"  Time.v (unix): {time_info.get('v')}")
        # Check iaqi for individual pollutant timestamps
        iaqi = d.get("iaqi", {})
        print(f"  IAQI pollutants: {list(iaqi.keys())}")
    else:
        print(f"  FAILED: {data.get('data')}")

    # 2. Search for stations in this city
    print(f"\n--- Station search: /search/?keyword={city} ---")
    r2 = requests.get(f"https://api.waqi.info/search/?keyword={city}&token={API_KEY}", timeout=15)
    data2 = r2.json()
    if data2.get("status") == "ok":
        stations = data2.get("data", [])
        print(f"  Found {len(stations)} stations matching '{city}'")
        for i, s in enumerate(stations[:8]):
            station_info = s.get("station", {})
            time_info = s.get("time", {})
            aqi = s.get("aqi")
            uid = s.get("station", {}).get("name", "")
            station_name = station_info.get("name", "unknown")
            print(f"  [{i+1}] AQI={aqi} | Station: {station_name}")
            print(f"       Time.s: {time_info.get('s')} | Time.iso: {time_info.get('iso')}")
            # Check if AQI is a number (not '-')
            is_numeric = isinstance(aqi, (int, float))
            print(f"       Numeric AQI: {is_numeric}")
    else:
        print(f"  FAILED: {data2.get('data')}")

    # 3. Try bound station IDs for Pakistan cities
    # Known WAQI station IDs for Pakistani cities
    known_ids = {
        "karachi": ["@7393", "@7386", "@7445"],
        "lahore": ["@7432", "@7444"],
        "islamabad": ["@7433", "@7438"],
    }
    print(f"\n--- Testing known bound station IDs ---")
    for sid in known_ids.get(city, []):
        r3 = requests.get(f"https://api.waqi.info/feed/{sid}/?token={API_KEY}", timeout=15)
        data3 = r3.json()
        if data3.get("status") == "ok":
            d3 = data3["data"]
            time_info = d3.get("time", {})
            station = d3.get("station", {})
            print(f"  {sid}: AQI={d3.get('aqi')} | Station={station.get('name')} | Time.s={time_info.get('s')}")
        else:
            print(f"  {sid}: FAILED - {data3.get('data', 'unknown')}")


def main():
    for city in CITIES:
        investigate_city(city)

    print(f"\n{'='*60}")
    print("INVESTIGATION COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
