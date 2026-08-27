#!/usr/bin/env python
"""Test FastAPI API with live Open-Meteo data."""
import sys
sys.path.insert(0, ".")

from fastapi.testclient import TestClient
from app.backend.main import app
from app.services.model_service import init_model_service
from app.services.feature_service import init_feature_service
from app.services.prediction_service import init_prediction_service
from src.monitoring.prediction_logger import PredictionLogger

# Initialize services
print("Initializing services...")
model_service = init_model_service(registry=None)
model_service.load_local_model()
feature_service = init_feature_service(fallback_enabled=False)
prediction_logger = PredictionLogger(log_dir="data/predictions", enable_security_checks=True)
prediction_service = init_prediction_service(
    model_service=model_service,
    feature_service=feature_service,
    prediction_logger=prediction_logger,
)
print(f"Model loaded: {model_service.is_loaded()}")

client = TestClient(app)

print("\n" + "=" * 60)
print("  LIVE DATA API TEST")
print("=" * 60)

# Test all 3 cities
print("\n1. POST /prediction for all cities (live data)...")
for city in ["karachi", "lahore", "islamabad"]:
    r = client.post("/prediction",
                    json={"city": city, "include_explanation": False},
                    headers={"X-API-Key": "test-key"})
    if r.status_code == 200:
        data = r.json()
        print(f"\n   {city.upper()}:")
        print(f"     24h: AQI={data.get('aqi_24h')} ({data.get('category_24h')})")
        print(f"     48h: AQI={data.get('aqi_48h')} ({data.get('category_48h')})")
        print(f"     72h: AQI={data.get('aqi_72h')} ({data.get('category_72h')})")
        print(f"   [OK]")
    else:
        print(f"   {city}: Status {r.status_code}")
        print(f"   [FAIL]")

print("\n" + "=" * 60)
print("  TEST COMPLETE")
print("=" * 60)
