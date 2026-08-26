#!/usr/bin/env python3
"""Debug AQICN response parsing for bound station."""
import os
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.config import load_environment
load_environment()

import requests
from src.data.schemas import AQICNResponse

API_KEY = os.environ.get("AQICN_API_KEY")

# Test bound station @7393 (Karachi)
r = requests.get(f"https://api.waqi.info/feed/@7393/?token={API_KEY}", timeout=15)
raw = r.json()

# Parse with Pydantic model
parsed = AQICNResponse(**raw)
data = parsed.data
print(f"parsed.status: {parsed.status}")
print(f"data.aqi: {data.aqi}")
print(f"data.time: {data.time}")
print(f"data.time.iso: {data.time.iso if data.time else None}")
print(f"data.time.v: {data.time.v if data.time else None}")
print(f"data.iaqi keys: {list(data.iaqi.keys()) if data.iaqi else None}")

# Test timestamp parsing
from src.data.aqicn_client import _parse_aqicn_timestamp
if data.time:
    time_dict = {"iso": data.time.iso, "v": data.time.v}
    dt = _parse_aqicn_timestamp(time_dict)
    print(f"\nParsed timestamp: {dt}")
    print(f"Is fresh: {(dt is not None)}")
