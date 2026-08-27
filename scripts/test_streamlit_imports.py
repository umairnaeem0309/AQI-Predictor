#!/usr/bin/env python
"""Test Streamlit module imports."""
import sys
sys.path.insert(0, ".")

print("Testing Streamlit imports...")

try:
    from app.frontend.utils.api_client import APIClient
    print("[OK] API client import")
except Exception as e:
    print(f"[FAIL] API client: {e}")

try:
    from app.frontend.pages.dashboard import render_dashboard
    print("[OK] Dashboard import")
except Exception as e:
    print(f"[FAIL] Dashboard: {e}")

try:
    from app.frontend.utils.aqi_theme import get_aqi_color
    print("[OK] AQI theme import")
except Exception as e:
    print(f"[FAIL] AQI theme: {e}")

try:
    from app.frontend.components.metrics import render_aqi_card
    print("[OK] Metrics component import")
except Exception as e:
    print(f"[FAIL] Metrics: {e}")

try:
    from app.frontend.components.charts import create_forecast_chart
    print("[OK] Charts component import")
except Exception as e:
    print(f"[FAIL] Charts: {e}")

try:
    import streamlit
    print(f"[OK] Streamlit version: {streamlit.__version__}")
except Exception as e:
    print(f"[FAIL] Streamlit: {e}")

print("\nDone.")
