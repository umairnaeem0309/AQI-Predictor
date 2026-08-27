import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
"""
System Status Page

Model status, pipeline status, and data freshness.
All data from backend - no hardcoded values.
"""

import streamlit as st

from app.frontend.utils.api_client import APIClient, APIClientError
from app.frontend.utils.formatters import format_timestamp, format_time_ago
from app.frontend.utils.aqi_theme import get_dashboard_css
from app.frontend.components.metrics import (
    render_status_card,
    render_info_card,
    render_error_state,
    render_warning_state,
)


def render_system(api_client: APIClient):
    """
    Render system status page.
    
    Args:
        api_client: API client instance
    """
    st.markdown(get_dashboard_css(), unsafe_allow_html=True)
    
    st.header("⚙️ System Status")
    
    # Refresh button
    if st.button("🔄 Refresh Status", key="refresh_system"):
        st.rerun()
    
    # Health Check
    st.subheader("Service Health")
    
    try:
        health = api_client.get_health()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            status = health.get("status", "unknown")
            render_status_card(
                label="Service Status",
                value=status.title(),
                status="ok" if status == "healthy" else "error",
            )
        
        with col2:
            model_loaded = health.get("model_loaded", False)
            render_status_card(
                label="Model Status",
                value="Loaded" if model_loaded else "Not Loaded",
                status="ok" if model_loaded else "warning",
            )
        
        with col3:
            fs_connected = health.get("feature_store_connected", False)
            render_status_card(
                label="Feature Store",
                value="Connected" if fs_connected else "Disconnected",
                status="ok" if fs_connected else "error",
            )
        
        # Last Prediction
        last_pred = health.get("last_prediction")
        st.markdown(f"**Last Prediction:** {format_time_ago(last_pred)}")
        
    except APIClientError as e:
        render_error_state("Cannot fetch health status", e)
    
    # Model Information
    st.subheader("Model Information")
    
    try:
        model_info = api_client.get_model_info()
        
        col1, col2 = st.columns(2)
        
        with col1:
            render_info_card("Model Name", model_info.get("model_name", "N/A"))
            render_info_card("Model Version", model_info.get("model_version", "N/A"))
            render_info_card("Status", model_info.get("status", "N/A"))
        
        with col2:
            render_info_card("Approval Status", model_info.get("approval_status", "N/A"))
            render_info_card("Dataset Type", model_info.get("dataset_type", "N/A"))
            render_info_card("Feature Version", model_info.get("feature_version", "N/A"))
        
        # Training Date
        st.markdown(f"**Training Date:** {model_info.get('training_date', 'N/A')}")
        
        # Metrics
        metrics = model_info.get("metrics", {})
        if metrics:
            st.subheader("Model Metrics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                render_info_card("MAE", f"{metrics.get('mae', 'N/A'):.2f}" if metrics.get('mae') else "N/A")
            
            with col2:
                render_info_card("RMSE", f"{metrics.get('rmse', 'N/A'):.2f}" if metrics.get('rmse') else "N/A")
            
            with col3:
                render_info_card("R²", f"{metrics.get('r2', 'N/A'):.2f}" if metrics.get('r2') else "N/A")
        
    except APIClientError as e:
        render_error_state("Cannot fetch model information", e)
    
    # Pipeline Status
    st.subheader("Pipeline Status")
    
    # Note: Pipeline status requires additional backend endpoints
    st.info(
        "Detailed pipeline status (data collection, feature engineering, training) "
        "will be available when the backend provides pipeline monitoring endpoints."
    )
    
    # Data Freshness
    st.subheader("Data Freshness")
    
    last_pred = health.get("last_prediction") if 'health' in locals() else None
    st.markdown(f"**Last Prediction:** {format_time_ago(last_pred)}")
    
    st.caption("ℹ️ Data freshness information will be more detailed when monitoring endpoints are implemented.")
