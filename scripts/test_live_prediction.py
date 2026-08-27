#!/usr/bin/env python
"""Test full live prediction pipeline."""
import sys
import pickle
import json
import time
sys.path.insert(0, ".")

from src.data.live_fetcher import get_live_prediction, CITIES

print("=" * 60)
print("  LIVE PREDICTION PIPELINE TEST")
print("=" * 60)

# Load model
print("\n1. Loading model...")
with open("models/production/xgboost_model.pkl", "rb") as f:
    model = pickle.load(f)
print(f"   [OK] Model loaded: {type(model).__name__}")

# Test each city
print("\n2. Live predictions:")
for city_id in ["karachi", "lahore", "islamabad"]:
    print(f"\n   {city_id.upper()}:")
    try:
        t0 = time.time()
        result = get_live_prediction(city_id, model)
        elapsed = (time.time() - t0) * 1000

        print(f"     24h: AQI={result['aqi_24h']} ({result['category_24h']})")
        print(f"     48h: AQI={result['aqi_48h']} ({result['category_48h']})")
        print(f"     72h: AQI={result['aqi_72h']} ({result['category_72h']})")
        print(f"     Source: {result['source']}")
        print(f"     Time: {elapsed:.0f}ms")
        print(f"   [OK]")
    except Exception as e:
        print(f"   [ERROR] {e}")

print("\n" + "=" * 60)
print("  ALL TESTS COMPLETE")
print("=" * 60)
