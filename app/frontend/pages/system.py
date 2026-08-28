import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
"""
System Status Page

Model status, pipeline status, data freshness, monitoring, and alerts.
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

    # ── Tabs ──
    tab1, tab2, tab3 = st.tabs(["🏥 Service Health", "🔍 Monitoring", "🚨 Alerts"])

    # ── Tab 1: Service Health ──
    with tab1:
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

        if model_exists:
            try:
                mtime = datetime.fromtimestamp(model_path.stat().st_mtime)
                st.markdown(f"**Model Last Modified:** {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
            except Exception:
                pass

    # ── Tab 2: Monitoring ──
    with tab2:
        st.subheader("🔍 Data Drift Monitoring")

        try:
            with st.spinner("Running drift detection..."):
                drift = api_client.get_drift_report()

            # Handle unavailable data gracefully
            if drift.get("status") == "unavailable" or not drift.get("features") and drift.get("total_features", 0) == 0:
                st.info(f"ℹ️ {drift.get('message', 'Training data not available for drift detection. This is expected in deployed environments.')}")
            else:
                drift_detected = drift.get("drift_detected", False)
                drifted_count = drift.get("drifted_count", 0)
                drift_pct = drift.get("drift_percentage", 0)
                total = drift.get("total_features", 0)

                col1, col2, col3 = st.columns(3)
                with col1:
                    render_status_card(
                        "Drift Status",
                        "Detected" if drift_detected else "No Drift",
                        "warning" if drift_detected else "ok",
                    )
                with col2:
                    st.metric("Drifted Features", f"{drifted_count}/{total}")
                with col3:
                    st.metric("Drift %", f"{drift_pct:.1f}%")

                if drift_detected:
                    st.warning(f"⚠️ Data drift detected in {drifted_count} features ({drift_pct:.1f}%)")
                    drifted_cols = drift.get("drifted_columns", [])
                    if drifted_cols:
                        st.markdown("**Drifted columns:**")
                        for col_info in drifted_cols:
                            st.markdown(f"- `{col_info.get('column', 'unknown')}`")
                else:
                    st.success("✅ No data drift detected. Feature distributions are stable.")

        except APIClientError as e:
            render_error_state("Cannot run drift detection", str(e))

        st.divider()

        # Performance Metrics
        st.subheader("📊 Model Performance")

        try:
            perf = api_client.get_performance()
            metrics = perf.get("training_metrics", {})

            if metrics:
                # Overall
                overall = metrics.get("overall", {})
                if overall:
                    st.markdown("**Overall Training Metrics:**")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("MAE", f"{overall.get('mae', 0):.2f}")
                    with col2:
                        st.metric("RMSE", f"{overall.get('rmse', 0):.2f}")
                    with col3:
                        st.metric("R²", f"{overall.get('r2', 0):.4f}")

                # Per-horizon
                for h in ["24h", "48h", "72h"]:
                    if h in metrics:
                        m = metrics[h]
                        st.markdown(f"**{h} Horizon:** MAE={m.get('mae', 0):.2f} | RMSE={m.get('rmse', 0):.2f} | R²={m.get('r2', 0):.4f}")
            else:
                st.info("No training metrics available.")

        except APIClientError as e:
            render_error_state("Cannot fetch performance metrics", str(e))

    # ── Tab 3: Alerts ──
    with tab3:
        st.subheader("🚨 AQI Hazard Alerts")

        try:
            with st.spinner("Checking current AQI levels..."):
                alerts_data = api_client.get_alerts()

            alerts = alerts_data.get("alerts", [])
            total_alerts = alerts_data.get("total_alerts", 0)

            if total_alerts == 0:
                st.success("✅ No active AQI alerts. Air quality is within safe levels.")
            else:
                st.warning(f"⚠️ {total_alerts} active alert(s)")

                for alert in alerts:
                    city = alert.get("city", "Unknown")
                    aqi = alert.get("aqi", 0)
                    category = alert.get("category", "Unknown")
                    level = alert.get("alert_level", "none")
                    recommendation = alert.get("recommendation", "")

                    # Color based on alert level
                    if level == "critical":
                        st.error(f"🔴 **{city}** — AQI {aqi} ({category})")
                    elif level == "warning":
                        st.warning(f"🟠 **{city}** — AQI {aqi} ({category})")
                    elif level == "caution":
                        st.info(f"🟡 **{city}** — AQI {aqi} ({category})")

                    if recommendation:
                        st.caption(f"📋 {recommendation}")

        except APIClientError as e:
            render_error_state("Cannot check AQI alerts", str(e))

        # AQI Categories Reference
        st.divider()
        st.subheader("📖 AQI Reference")

        import pandas as pd
        categories = pd.DataFrame([
            {"AQI Range": "0-50", "Category": "Good", "Color": "🟢", "Health": "Satisfactory"},
            {"AQI Range": "51-100", "Category": "Moderate", "Color": "🟡", "Health": "Acceptable"},
            {"AQI Range": "101-150", "Category": "USG", "Color": "🟠", "Health": "Sensitive groups affected"},
            {"AQI Range": "151-200", "Category": "Unhealthy", "Color": "🔴", "Health": "Everyone affected"},
            {"AQI Range": "201-300", "Category": "Very Unhealthy", "Color": "🟣", "Health": "Health alert"},
            {"AQI Range": "301-500", "Category": "Hazardous", "Color": "⚫", "Health": "Emergency"},
        ])
        st.dataframe(categories, hide_index=True, use_container_width=True)
