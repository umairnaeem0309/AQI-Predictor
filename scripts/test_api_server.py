#!/usr/bin/env python
"""Start FastAPI server and test all endpoints."""
import sys
import time
import threading
import requests
from pathlib import Path

sys.path.insert(0, ".")

def run_server():
    """Run uvicorn in a thread."""
    import uvicorn
    from app.backend.main import app
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")

print("=" * 60)
print("  FASTAPI SERVER TEST")
print("=" * 60)

# Start server in background thread
print("\n1. Starting FastAPI server...")
server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()
time.sleep(3)  # Wait for server to start

BASE = "http://127.0.0.1:8000"

# Test 2. Health endpoint
print("\n2. Testing /health endpoint...")
try:
    r = requests.get(f"{BASE}/health", timeout=5)
    print(f"   Status: {r.status_code}")
    data = r.json()
    print(f"   Response: {data}")
    print(f"   [OK] Health endpoint works" if r.status_code == 200 else f"   [FAIL] Health returned {r.status_code}")
except Exception as e:
    print(f"   [FAIL] Health endpoint error: {e}")

# Test 3. Model info endpoint
print("\n3. Testing /model endpoint...")
try:
    r = requests.get(f"{BASE}/model", timeout=5)
    print(f"   Status: {r.status_code}")
    data = r.json()
    print(f"   Response: {data}")
    print(f"   [OK] Model endpoint works" if r.status_code == 200 else f"   [FAIL] Model returned {r.status_code}")
except Exception as e:
    print(f"   [FAIL] Model endpoint error: {e}")

# Test 4. Prediction endpoint (without API key first)
print("\n4. Testing /prediction endpoint (no auth)...")
try:
    r = requests.post(f"{BASE}/prediction",
                      json={"city": "karachi", "include_explanation": False},
                      timeout=10)
    print(f"   Status: {r.status_code}")
    data = r.json()
    print(f"   Response: {data}")
    if r.status_code == 401:
        print(f"   [OK] Auth required (expected)")
    elif r.status_code == 200:
        print(f"   [OK] Prediction returned")
    else:
        print(f"   [INFO] Response code: {r.status_code}")
except Exception as e:
    print(f"   [FAIL] Prediction endpoint error: {e}")

# Test 5. Prediction with API key header
print("\n5. Testing /prediction endpoint (with X-API-Key)...")
try:
    headers = {"X-API-Key": "test-key"}
    r = requests.post(f"{BASE}/prediction",
                      json={"city": "karachi", "include_explanation": False},
                      headers=headers,
                      timeout=10)
    print(f"   Status: {r.status_code}")
    data = r.json()
    if r.status_code == 200:
        print(f"   Prediction: {data.get('predictions', {})}")
        print(f"   [OK] Prediction successful")
    else:
        print(f"   Response: {data}")
        print(f"   [INFO] Status: {r.status_code}")
except Exception as e:
    print(f"   [FAIL] Prediction with auth error: {e}")

# Test 6. Invalid city
print("\n6. Testing /prediction with invalid city...")
try:
    headers = {"X-API-Key": "test-key"}
    r = requests.post(f"{BASE}/prediction",
                      json={"city": "invalid_city", "include_explanation": False},
                      headers=headers,
                      timeout=10)
    print(f"   Status: {r.status_code}")
    data = r.json()
    print(f"   Response: {data}")
    print(f"   [OK] Invalid city rejected" if r.status_code in [400, 422] else f"   [INFO] Status: {r.status_code}")
except Exception as e:
    print(f"   [FAIL] Invalid city test error: {e}")

print("\n" + "=" * 60)
print("  API SERVER TEST COMPLETE")
print("=" * 60)
