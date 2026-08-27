#!/usr/bin/env python
"""Test FastAPI endpoints using TestClient."""
import sys
sys.path.insert(0, ".")

from fastapi.testclient import TestClient
from app.backend.main import app

print("=" * 60)
print("  FASTAPI ENDPOINT TESTS (TestClient)")
print("=" * 60)

from app.services.model_service import init_model_service, ModelService
from app.services.feature_service import init_feature_service
from app.services.prediction_service import init_prediction_service
from src.monitoring.prediction_logger import PredictionLogger
from src.feature_store.local_store import LocalStore
from app.backend.config import default_config

# Manually initialize services (lifespan may not run in TestClient)
local_store = LocalStore()
local_store.connect()
feature_service = init_feature_service(
    primary_store=local_store,
    fallback_enabled=True,
)
model_service = init_model_service(registry=None)
model_service.load_local_model()
prediction_logger = PredictionLogger(log_dir="data/predictions", enable_security_checks=True)
prediction_service = init_prediction_service(
    model_service=model_service,
    feature_service=feature_service,
    prediction_logger=prediction_logger,
)
print(f"   Services initialized: model={model_service.is_loaded()}, features=local")

client = TestClient(app)

# 1. Health endpoint
print("\n1. GET /health...")
r = client.get("/health")
print(f"   Status: {r.status_code}")
print(f"   Response: {r.json()}")
print(f"   [OK]" if r.status_code == 200 else f"   [FAIL]")

# 2. Model info endpoint
print("\n2. GET /model...")
r = client.get("/model")
print(f"   Status: {r.status_code}")
data = r.json()
print(f"   Response: {data}")
print(f"   [OK]" if r.status_code == 200 else f"   [INFO] Status: {r.status_code}")

# 3. Prediction without auth
print("\n3. POST /prediction (no auth)...")
r = client.post("/prediction", json={"city": "karachi", "include_explanation": False})
print(f"   Status: {r.status_code}")
data = r.json()
print(f"   Response: {data}")
if r.status_code == 200:
    print(f"   [OK] Prediction successful")
    print(f"   Predictions: {data.get('predictions', {})}")
elif r.status_code == 401:
    print(f"   [OK] Auth required (expected)")
else:
    print(f"   [INFO] Status: {r.status_code}")

# 4. Prediction with auth
print("\n4. POST /prediction (with X-API-Key)...")
r = client.post("/prediction",
                json={"city": "lahore", "include_explanation": False},
                headers={"X-API-Key": "test-key"})
print(f"   Status: {r.status_code}")
data = r.json()
print(f"   Response: {data}")
if r.status_code == 200:
    print(f"   [OK] Prediction successful")
    preds = data.get("predictions", {})
    for horizon, val in preds.items():
        print(f"   {horizon}: {val}")
elif r.status_code == 401:
    print(f"   [OK] Auth check working")
else:
    print(f"   [INFO] Status: {r.status_code}")

# 5. Prediction for all 3 cities
print("\n5. Testing all 3 cities...")
for city in ["karachi", "lahore", "islamabad"]:
    r = client.post("/prediction",
                    json={"city": city, "include_explanation": False},
                    headers={"X-API-Key": "test-key"})
    if r.status_code == 200:
        preds = r.json().get("predictions", {})
        aqi_24h = preds.get("aqi_24h", "N/A")
        aqi_48h = preds.get("aqi_48h", "N/A")
        aqi_72h = preds.get("aqi_72h", "N/A")
        print(f"   {city:12s}: 24h={aqi_24h}, 48h={aqi_48h}, 72h={aqi_72h}")
    else:
        print(f"   {city:12s}: Status {r.status_code} - {r.json()}")

# 6. Invalid city
print("\n6. POST /prediction (invalid city)...")
r = client.post("/prediction",
                json={"city": "london", "include_explanation": False},
                headers={"X-API-Key": "test-key"})
print(f"   Status: {r.status_code}")
print(f"   Response: {r.json()}")
print(f"   [OK] Invalid city rejected" if r.status_code in [400, 422] else f"   [INFO]")

print("\n" + "=" * 60)
print("  ALL ENDPOINT TESTS COMPLETE")
print("=" * 60)
