#!/usr/bin/env python3
"""Debug AQICN bound station response to check timestamps."""
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

# Test bound station @7393 (Karachi)
r = requests.get(f"https://api.waqi.info/feed/@7393/?token={API_KEY}", timeout=15)
data = r.json()
if data.get("status") == "ok":
    d = data["data"]
    print("Station @7393 (Karachi) response:")
    print(f"  AQI: {d.get('aqi')}")
    print(f"  Time: {json.dumps(d.get('time', {}), indent=4)}")
    print(f"  Station: {json.dumps(d.get('station', {}), indent=4)}")
    print(f"  IAQI keys: {list(d.get('iaqi', {}).keys())}")
    print(f"  Forecast: {'forecast' in d}")
else:
    print(f"FAILED: {data}")

# Test bound station @7432 (Lahore)
r2 = requests.get(f"https://api.waqi.info/feed/@7432/?token={API_KEY}", timeout=15)
data2 = r2.json()
if data2.get("status") == "ok":
    d2 = data2["data"]
    print("\nStation @7432 (Lahore) response:")
    print(f"  AQI: {d2.get('aqi')}")
    print(f"  Time: {json.dumps(d2.get('time', {}), indent=4)}")
