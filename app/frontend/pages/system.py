import os
import sys
import json
from datetime import datetime
from pathlib import Path
import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from app.frontend.components.metrics import (
    render_error_state,
    render_property,
    render_warning_state,
)
from app.frontend.utils.api_client import APIClient, APIClientError
from app.frontend.utils.aqi_theme import (
    AQI_COLORS,
    get_aqi_color,
    get_dashboard_css,
)
from app.frontend.utils.formatters import format_time_ago, format_timestamp


def render_system(api_client: APIClient):
    """Render system status page."""
    st.markdown(get_dashboard_css(), unsafe_allow_html=True)

    st.title("System Status")

    # Determine overall system health
    try:
        health = api_client.get_health()
        overall_ok = health.get("status", "unknown") == "healthy"
        model_ok   = health.get("model_loaded", False)
        fs_ok      = health.get("feature_store_connected", False)
        if overall_ok and model_ok:
            hero_label = "All Systems Operational"
        elif overall_ok:
            hero_label = "Partially Degraded"
        else:
            hero_label = "Service Incident"
    except Exception:
        health = {}
        hero_label = "API Unreachable"
        overall_ok = model_ok = fs_ok = False

    st.caption(f"Current Status: **{hero_label}**")

    if st.button("Refresh Status", key="refresh_system"):
        st.rerun()

    st.markdown("---")

    # Tabs — NO st.status inside to prevent double-click reset bug
    tab1, tab2, tab3 = st.tabs(["Service Health", "Monitoring", "Alerts"])

    # ── Tab 1: Service Health ─────────────────────────────────────────────────
    with tab1:
        st.subheader("Service Health")
        
        with st.container(border=True):
            h1, h2, h3 = st.columns(3)
            with h1:
                render_property(
                    "API Service", 
                    "Healthy" if overall_ok else "Unreachable",
                    value_color="#00C853" if overall_ok else "#D50000"
                )
            with h2:
                render_property(
                    "Model Inference", 
                    "Loaded" if model_ok else "Not Loaded",
                    value_color="#00C853" if model_ok else "#FF9800"
                )
            with h3:
                render_property(
                    "Feature Store", 
                    "Connected" if fs_ok else "Disconnected",
                    value_color="#00C853" if fs_ok else "#FF9800"
                )

        st.subheader("Model Information")

        try:
            model_info = api_client.get_model_info()

            with st.container(border=True):
                mi1, mi2, mi3 = st.columns(3)
                with mi1:
                    render_property("Model Name", model_info.get("model_name", "N/A"))
                    render_property("Model Version", model_info.get("model_version", "N/A"))
                with mi2:
                    render_property("Status", model_info.get("status", "N/A").title())
                    render_property("Approval Status", model_info.get("approval_status", "N/A").title())
                with mi3:
                    render_property("Dataset Type", model_info.get("dataset_type", "N/A"))
                    render_property("Data Provider", model_info.get("data_provider", "N/A"))
                
                st.markdown(
                    f'<div style="margin-top: 10px; font-size: 0.85rem; color: #64748B;">'
                    f'Training Date: <b>{model_info.get("training_date", "N/A")}</b></div>',
                    unsafe_allow_html=True,
                )

            metrics = model_info.get("metrics", {})
            if metrics:
                st.markdown("**Model Metrics**")
                m1, m2, m3 = st.columns(3)
                with m1: st.metric("MAE", f"{metrics.get('mae', 0):.2f}")
                with m2: st.metric("RMSE", f"{metrics.get('rmse', 0):.2f}")
                with m3: st.metric("R²", f"{metrics.get('r2', 0):.4f}")

        except APIClientError as e:
            render_error_state("Cannot fetch model information", str(e))

        st.subheader("Pipeline Artifacts")

        model_path    = Path("models/production/best_model.pkl")
        metadata_path = Path("models/production/model_metadata.json")

        with st.container(border=True):
            pp1, pp2 = st.columns(2)
            with pp1:
                model_exists = model_path.exists()
                render_property("Model Artifact (.pkl)", "Available" if model_exists else "Missing", 
                                value_color="#00C853" if model_exists else "#D50000")
            with pp2:
                metadata_exists = metadata_path.exists()
                render_property("Metadata (.json)", "Available" if metadata_exists else "Missing",
                                value_color="#00C853" if metadata_exists else "#D50000")

        try:
            if metadata_path.exists():
                with open(metadata_path) as _f:
                    _meta = json.load(_f)

                st.markdown("**Dataset Statistics**")
                ds1, ds2, ds3, ds4 = st.columns(4)
                with ds1: st.metric("Train Rows", f"{_meta.get('train_rows', 0):,}")
                with ds2: st.metric("Val Rows", f"{_meta.get('val_rows', 0):,}")
                with ds3: st.metric("Test Rows", f"{_meta.get('test_rows', 0):,}")
                with ds4: st.metric("Features", _meta.get("n_features", 0))
        except Exception:
            pass

        st.subheader("Data Freshness")

        last_pred = health.get("last_prediction") if health else None
        pred_ago  = format_time_ago(last_pred) if last_pred else "No predictions yet"

        with st.container(border=True):
            fr1, fr2 = st.columns(2)
            with fr1:
                render_property("Last Prediction Request", pred_ago)
            with fr2:
                if model_path.exists():
                    try:
                        mtime = datetime.fromtimestamp(model_path.stat().st_mtime)
                        render_property("Model Last Modified", mtime.strftime("%Y-%m-%d %H:%M:%S"))
                    except Exception:
                        render_property("Model Last Modified", "Unknown")

    # ── Tab 2: Monitoring ─────────────────────────────────────────────────────
    with tab2:
        st.subheader("Data Drift Monitoring")

        try:
            with st.spinner("Running drift detection..."):
                drift = api_client.get_drift_report()

            unavailable = (
                drift.get("status") == "unavailable"
                or (not drift.get("features") and drift.get("total_features", 0) == 0)
            )

            if unavailable:
                st.info(
                    drift.get(
                        "message",
                        "Training data not available for drift detection. "
                        "This is expected in deployed environments."
                    )
                )
            else:
                drift_detected = drift.get("drift_detected", False)
                drifted_count  = drift.get("drifted_count", 0)
                drift_pct      = drift.get("drift_percentage", 0)
                total_feats    = drift.get("total_features", 0)

                if drift_detected:
                    st.warning(f"Data Drift Detected: {drifted_count} of {total_feats} features drifted ({drift_pct:.1f}%). Consider retraining.")
                else:
                    st.success("No data drift detected. Feature distributions are stable.")

                with st.container(border=True):
                    dc1, dc2, dc3 = st.columns(3)
                    with dc1:
                        render_property(
                            "Drift Status", 
                            "Detected" if drift_detected else "Stable",
                            value_color="#FF9800" if drift_detected else "#00C853"
                        )
                    with dc2:
                        st.metric("Drifted Features", f"{drifted_count}/{total_feats}")
                    with dc3:
                        st.metric("Drift %", f"{drift_pct:.1f}%")

                drifted_cols = drift.get("drifted_columns", [])
                if drifted_cols:
                    st.markdown("**Drifted Columns**")
                    cols_list = [col_info.get("column", "unknown") for col_info in drifted_cols]
                    st.write(", ".join(cols_list))

        except APIClientError as e:
            render_error_state("Cannot run drift detection", str(e))

        st.divider()

        st.subheader("Model Performance")

        try:
            perf = api_client.get_performance()
            metrics = perf.get("training_metrics", {})

            if metrics:
                overall = metrics.get("overall", {})
                if overall:
                    st.markdown("**Overall Training Metrics**")
                    with st.container(border=True):
                        pm1, pm2, pm3 = st.columns(3)
                        with pm1: st.metric("MAE", f"{overall.get('mae', 0):.2f}")
                        with pm2: st.metric("RMSE", f"{overall.get('rmse', 0):.2f}")
                        with pm3: st.metric("R²", f"{overall.get('r2', 0):.4f}")

                horizon_rows = []
                for h in ["24h", "48h", "72h"]:
                    if h in metrics:
                        m = metrics[h]
                        horizon_rows.append({
                            "Horizon": h,
                            "MAE": round(m.get("mae", 0), 2),
                            "RMSE": round(m.get("rmse", 0), 2),
                            "R²": round(m.get("r2", 0), 4),
                        })

                if horizon_rows:
                    st.markdown("**Per-Horizon Metrics**")
                    st.dataframe(
                        pd.DataFrame(horizon_rows),
                        hide_index=True,
                        use_container_width=True,
                        column_config={
                            "R²": st.column_config.ProgressColumn("R²", min_value=0, max_value=1)
                        },
                    )
            else:
                st.info("No training metrics available.")

        except APIClientError as e:
            render_error_state("Cannot fetch performance metrics", str(e))

    # ── Tab 3: Alerts ─────────────────────────────────────────────────────────
    with tab3:
        st.subheader("Active AQI Hazard Alerts")

        try:
            with st.spinner("Checking current AQI levels..."):
                alerts_data = api_client.get_alerts()

            alerts       = alerts_data.get("alerts", [])
            total_alerts = alerts_data.get("total_alerts", 0)

            if total_alerts == 0:
                st.success("No active AQI alerts. Air quality is within safe levels across all cities.")
            else:
                st.error(f"{total_alerts} Active Alert{'s' if total_alerts != 1 else ''} found.")

                for alert in alerts:
                    city         = alert.get("city", "Unknown")
                    aqi          = alert.get("aqi", 0)
                    category     = alert.get("category", "Unknown")
                    recommendation = alert.get("recommendation", "")
                    aqi_color    = get_aqi_color(aqi)
                    
                    with st.container(border=True):
                        st.markdown(f"### {city}")
                        st.markdown(
                            f'<span style="background:{aqi_color}22;color:{aqi_color};'
                            f'border:1.5px solid {aqi_color};border-radius:20px;padding:2px 10px;'
                            f'font-size:0.75rem;font-weight:700;">AQI {aqi}</span> '
                            f'**{category}**',
                            unsafe_allow_html=True
                        )
                        if recommendation:
                            st.caption(recommendation)

        except APIClientError as e:
            render_error_state("Cannot check AQI alerts", str(e))

        st.divider()
        st.subheader("AQI Reference Guide")

        aqi_ref_rows = [
            ("0–50",   "Good",                    AQI_COLORS["good"],                  "#E8F5E9", "Satisfactory. Air quality poses little or no risk."),
            ("51–100", "Moderate",                AQI_COLORS["moderate"],              "#FFFDE7", "Acceptable. Sensitive individuals may experience minor effects."),
            ("101–150","Unhealthy for Sensitive", AQI_COLORS["unhealthy_sensitive"],   "#FFF3E0", "Members of sensitive groups may experience health effects."),
            ("151–200","Unhealthy",               AQI_COLORS["unhealthy"],             "#FFEBEE", "Everyone may begin to experience health effects."),
            ("201–300","Very Unhealthy",          AQI_COLORS["very_unhealthy"],        "#F3E5F5", "Health alert: everyone may experience serious effects."),
            ("301–500","Hazardous",               AQI_COLORS["hazardous"],             "#FCE4EC", "Emergency conditions. The entire population is likely to be affected."),
        ]

        for aqi_range, cat_name, color, bg, health_note in aqi_ref_rows:
            st.markdown(
                f"""
                <div style="display:flex;align-items:center;gap:14px;padding:10px 14px;
                     background:{bg};border-left:4px solid {color};border-radius:0 8px 8px 0;
                     margin-bottom:6px;">
                  <div style="min-width:70px;font-size:0.88rem;font-weight:700;color:{color};">
                    {aqi_range}
                  </div>
                  <div style="min-width:180px;font-size:0.85rem;font-weight:700;color:#1A1A2E;">
                    {cat_name}
                  </div>
                  <div style="font-size:0.78rem;color:#64748B;flex:1;">
                    {health_note}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
