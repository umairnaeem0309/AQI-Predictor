#!/usr/bin/env python
"""Test Streamlit API client connection to FastAPI backend."""
import sys
sys.path.insert(0, ".")

from app.frontend.utils.api_client import APIClient

print("=" * 60)
print("  STREAMLIT API CLIENT TEST")
print("=" * 60)

# Create client
client = APIClient(base_url="http://localhost:8000", mock_mode=False)

print(f"\n1. Client config:")
print(f"   Base URL: {client.base_url}")
print(f"   Mock mode: {client.mock_mode}")
print(f"   API key: {client.api_key}")

# Test connection (will fail if server not running)
print(f"\n2. Testing connection...")
try:
    available = client.is_available()
    print(f"   Available: {available}")
except Exception as e:
    print(f"   Connection error: {e}")
    print(f"   (Expected if FastAPI server not running)")

# Test mock mode
print(f"\n3. Testing mock mode...")
mock_client = APIClient(base_url="http://localhost:8000", mock_mode=True)
try:
    prediction = mock_client.get_prediction("karachi")
    print(f"   Mock prediction: {prediction}")
    print(f"   [OK] Mock mode works")
except Exception as e:
    print(f"   Mock error: {e}")

print("\n" + "=" * 60)
print("  TEST COMPLETE")
print("=" * 60)
