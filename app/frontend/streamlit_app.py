"""
AQI Predictor Dashboard

Main Streamlit application for AQI prediction visualization.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st

from app.frontend.utils.api_client import APIClient
from app.frontend.pages.dashboard import render_dashboard
from app.frontend.pages.analytics import render_analytics
from app.frontend.pages.explainability import render_explainability
from app.frontend.pages.system import render_system


def init_session_state():
    """Initialize Streamlit session state."""
    if "selected_city" not in st.session_state:
        st.session_state.selected_city = "Karachi"
    if "prediction_data" not in st.session_state:
        st.session_state.prediction_data = None
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = None


def main():
    """Main application entry point."""
    # Page configuration
    st.set_page_config(
        page_title="AQI Predictor",
        page_icon="🌬️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Initialize session state
    init_session_state()
    
    # Initialize API client
    api_client = APIClient.from_env()
    
    # Sidebar navigation
    st.sidebar.title("🌬️ AQI Predictor")
    
    page = st.sidebar.radio(
        "Navigation",
        ["Dashboard", "Analytics", "Explainability", "System"],
        index=0,
    )
    
    # Mock mode indicator
    if api_client.mock_mode:
        st.sidebar.warning("⚠️ Mock Mode Active")
        st.sidebar.caption("Using simulated data for development")
    
    # API connection status
    if not api_client.mock_mode:
        if api_client.is_available():
            st.sidebar.success("✅ API Connected")
        else:
            st.sidebar.error("❌ API Unavailable")
    
    # Render selected page
    if page == "Dashboard":
        render_dashboard(api_client)
    elif page == "Analytics":
        render_analytics(api_client)
    elif page == "Explainability":
        render_explainability(api_client)
    elif page == "System":
        render_system(api_client)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.caption("AQI Predictor v1.0.0")
    st.sidebar.caption("US EPA AQI Standards")


if __name__ == "__main__":
    main()
