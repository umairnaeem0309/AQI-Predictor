import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
"""
System Status Page

Model status, pipeline status, and data freshness.
All data from backend - no hardcoded values.
"""

import streamlit as st
import json
from pathlib import Path
from datetime import datetime

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

    health = {}
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
                status="ok" if fs_connected else "warning",
            )

    except APIClientError as e:
        render_error_state("Cannot fetch health status — is the API running?", str(e))

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
            render_info_card("Data Provider", model_info.get("data_provider", "N/A"))

        # Training Date
        training_date = model_info.get("training_date", "N/A")
        st.markdown(f"**Training Date:** {training_date}")

        # Metrics
        metrics = model_info.get("metrics", {})
        if metrics:
            st.subheader("Model Metrics")
            col1, col2, col3 = st.columns(3)

            with col1:
                mae = metrics.get("mae")
                render_info_card("MAE", f"{mae:.2f}" if mae else "N/A")

            with col2:
                rmse = metrics.get("rmse")
                render_info_card("RMSE", f"{rmse:.2f}" if rmse else "N/A")

            with col3:
                r2 = metrics.get("r2")
                render_info_card("R²", f"{r2:.4f}" if r2 else "N/A")

        # Feature info
        feature_cols = model_info.get("feature_columns", [])
        target_cols = model_info.get("target_columns", [])
        if feature_cols or target_cols:
            st.subheader("Dataset Schema")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Features", len(feature_cols))
            with col2:
                st.metric("Targets", len(target_cols))

    except APIClientError as e:
        render_error_state("Cannot fetch model information", str(e))

    # Pipeline Status
    st.subheader("Pipeline Status")

    # Check for local model files
    model_path = Path("models/production/xgboost_model.pkl")
    metadata_path = Path("models/production/model_metadata.json")
    dataset_path = Path("data/processed/final_dataset.csv")

    col1, col2, col3 = st.columns(3)

    with col1:
        model_exists = model_path.exists()
        render_status_card(
            label="Model Artifact",
            value="Available" if model_exists else "Missing",
            status="ok" if model_exists else "error",
        )

    with col2:
        metadata_exists = metadata_path.exists()
        render_status_card(
            label="Metadata",
            value="Available" if metadata_exists else "Missing",
            status="ok" if metadata_exists else "error",
        )

    with col3:
        dataset_exists = dataset_path.exists()
        render_status_card(
            label="Dataset",
            value="Available" if dataset_exists else "Missing",
            status="ok" if dataset_exists else "warning",
        )

    # Dataset stats
    if dataset_exists:
        try:
            import pandas as pd
            df = pd.read_csv(dataset_path)
            st.subheader("Dataset Statistics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Rows", f"{len(df):,}")
            with col2:
                cities = df["city"].nunique() if "city" in df.columns else "N/A"
                st.metric("Cities", cities)
            with col3:
                if "timestamp" in df.columns:
                    st.metric("Date Range", f"{df['timestamp'].min()[:10]} → {df['timestamp'].max()[:10]}")
                else:
                    st.metric("Date Range", "N/A")
            with col4:
                st.metric("Columns", len(df.columns))
        except Exception:
            st.info("Could not load dataset statistics.")

    # Data Freshness
    st.subheader("Data Freshness")

    last_pred = health.get("last_prediction") if health else None
    if last_pred:
        st.markdown(f"**Last Prediction:** {format_time_ago(last_pred)}")
    else:
        st.markdown("**Last Prediction:** No predictions yet")

    # Show model file modification time
    if model_exists:
        try:
            mtime = datetime.fromtimestamp(model_path.stat().st_mtime)
            st.markdown(f"**Model Last Modified:** {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
        except Exception:
            pass

    st.caption("ℹ️ Real-time monitoring will be available when the backend provides pipeline monitoring endpoints.")
