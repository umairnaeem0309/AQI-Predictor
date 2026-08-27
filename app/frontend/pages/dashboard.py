import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
"""
Main Dashboard Page

Primary view with current AQI, 3-day forecast, and conditions.
"""

import streamlit as st
from typing import Optional

from app.frontend.utils.api_client import APIClient, APIClientError
from app.frontend.utils.formatters import format_timestamp, format_time_ago
from app.frontend.utils.aqi_theme import get_aqi_color, get_aqi_category, get_dashboard_css
from app.frontend.components.metrics import (
    render_aqi_card,
    render_loading_state,
    render_error_state,
    render_warning_state,
    render_unavailable_state,
)
from app.frontend.components.charts import create_forecast_chart, create_gauge_chart

# Valid cities
VALID_CITIES = ["Karachi", "Lahore", "Islamabad"]


def render_dashboard(api_client: APIClient):
    """
    Render main dashboard page.
    
    Args:
        api_client: API client instance
    """
    # Inject custom CSS
    st.markdown(get_dashboard_css(), unsafe_allow_html=True)
    
    # Header
    st.markdown('<p class="main-header">AQI Predictor Dashboard</p>', unsafe_allow_html=True)
    
    # City selector
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        selected_city = st.selectbox(
            "Select City",
            VALID_CITIES,
            key="city_selector",
        )
    
    with col2:
        if st.button("🔄 Refresh", key="refresh_btn"):
            st.cache_data.clear()
            st.rerun()
    
    with col3:
        st.markdown(f"**Last Updated:** {format_time_ago(st.session_state.get('last_refresh'))}")
    
    # Store selected city
    st.session_state.selected_city = selected_city
    
    # Fetch prediction
    with render_loading_state("Fetching prediction data..."):
        try:
            prediction = api_client.get_prediction(selected_city)
            st.session_state.prediction_data = prediction
            st.session_state.last_refresh = prediction.get("timestamp")
        except APIClientError as e:
            render_error_state("Failed to fetch prediction data", e)
            return
    
    if not prediction:
        render_warning_state("No prediction data available")
        return
    
    # AQI Cards Row
    st.subheader("AQI Forecasts")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        render_aqi_card(
            label="Current AQI",
            aqi_value=prediction.get("aqi_24h", 0),
            category=prediction.get("category_24h", "N/A"),
        )
    
    with col2:
        render_aqi_card(
            label="24h Forecast",
            aqi_value=prediction.get("aqi_24h", 0),
            category=prediction.get("category_24h", "N/A"),
        )
    
    with col3:
        render_aqi_card(
            label="48h Forecast",
            aqi_value=prediction.get("aqi_48h", 0),
            category=prediction.get("category_48h", "N/A"),
        )
    
    with col4:
        render_aqi_card(
            label="72h Forecast",
            aqi_value=prediction.get("aqi_72h", 0),
            category=prediction.get("category_72h", "N/A"),
        )
    
    # Forecast Chart
    st.subheader("3-Day Forecast")
    
    timestamps = [
        prediction.get("timestamp", ""),
        "24h",
        "48h",
        "72h",
    ]
    aqi_values = [
        prediction.get("aqi_24h", 0),
        prediction.get("aqi_24h", 0),
        prediction.get("aqi_48h", 0),
        prediction.get("aqi_72h", 0),
    ]
    
    fig = create_forecast_chart(timestamps, aqi_values, f"AQI Forecast - {selected_city}")
    st.plotly_chart(fig, use_container_width=True)
    
    # Model Info
    st.subheader("Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**Model Version:** {prediction.get('model_version', 'N/A')}")
    
    with col2:
        st.markdown(f"**Confidence:** {prediction.get('confidence', 'N/A')}")
    
    # Note about confidence
    if prediction.get("confidence") is None:
        st.caption("ℹ️ Confidence scores will be available when uncertainty quantification is implemented.")
