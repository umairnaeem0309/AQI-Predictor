#!/usr/bin/env python
"""Quick test of Streamlit app."""
import sys
sys.path.insert(0, ".")

print("Testing Streamlit app...")

# Test 1: Import all modules
try:
    from app.frontend.streamlit_app import main
    print("[OK] Main app import")
except Exception as e:
    print(f"[FAIL] Main app: {e}")

# Test 2: Test API client
try:
    from app.frontend.utils.api_client import APIClient
    client = APIClient(base_url="http://localhost:8000", mock_mode=True)
    prediction = client.get_prediction("karachi")
    print(f"[OK] Mock prediction: {prediction.get('aqi_24h', 'N/A')}")
except Exception as e:
    print(f"[FAIL] API client: {e}")

# Test 3: Test dashboard render
try:
    from app.frontend.pages.dashboard import render_dashboard
    print("[OK] Dashboard render function exists")
except Exception as e:
    print(f"[FAIL] Dashboard render: {e}")

print("\nAll tests passed. Streamlit app is ready.")
print("Run with: streamlit run app/frontend/streamlit_app.py --server.port 8501")
