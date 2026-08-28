"""
AQI Predictor Dashboard — Streamlit Cloud Entry Point

This file is the root entry point for Streamlit Cloud deployment.
It imports and runs the main Streamlit app from app/frontend/.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

# Import and run the main app
from app.frontend.streamlit_app import main

if __name__ == "__main__":
    main()
