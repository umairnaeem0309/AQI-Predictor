#!/usr/bin/env python
"""Test FastAPI API with live feature adapter."""
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
print("  LIVE API TEST")
print("=" * 60)

# 1. Health
print("\n1. GET /health...")
r = client.get("/health")
print(f"   Status: {r.status_code}")
print(f"   Response: {r.json()}")
print(f"   [OK]" if r.status_code == 200 else f"   [FAIL]")

# 2. Prediction for all 3 cities
print("\n2. POST /prediction for all cities...")
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
        print(f"     Model: {data.get('model_version', 'unknown')}")
        print(f"   [OK]")
    else:
        print(f"   {city}: Status {r.status_code} - {r.json()}")
        print(f"   [FAIL]")

# 3. Invalid city
print("\n3. POST /prediction (invalid city)...")
r = client.post("/prediction",
                json={"city": "london", "include_explanation": False},
                headers={"X-API-Key": "test-key"})
print(f"   Status: {r.status_code}")
print(f"   Response: {r.json()}")
print(f"   [OK] Invalid city rejected" if r.status_code == 400 else f"   [INFO]")

print("\n" + "=" * 60)
print("  ALL TESTS COMPLETE")
print("=" * 60)
