"""
AQI Predictor Dashboard — Streamlit Cloud Entry Point

This file serves as the Streamlit Cloud entry point.
It redirects to the actual app in app/frontend/streamlit_app.py
"""
import os
import sys

# Ensure project root is in path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import and run the main app
from app.frontend.streamlit_app import main

if __name__ == "__main__":
    main()
