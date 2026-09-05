import json
import os
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from app.frontend.components.charts import apply_chart_theme
from app.frontend.components.metrics import (
    render_error_state,
    render_info_card,
    render_status_card,
    render_warning_state,
)
from app.frontend.utils.api_client import APIClient, APIClientError
from app.frontend.utils.aqi_theme import (
    AQI_COLORS,
    get_aqi_category,
    get_aqi_color,
    get_dashboard_css,
)
from app.frontend.utils.formatters import format_time_ago, format_timestamp


def render_system(api_client: APIClient):
    """Render system status page."""
    st.markdown(get_dashboard_css(), unsafe_allow_html=True)

    # Determine overall system health
    try:
        _probe_health = api_client.get_health()
        _overall_ok = _probe_health.get("status", "unknown") == "healthy"
        _model_ok = _probe_health.get("model_loaded", False)
        _fs_ok = _probe_health.get("feature_store_connected", False)
        if _overall_ok and _model_ok:
            _hero_status = ("All Systems Operational", "#00C853", "#E8F5E9")
        elif _overall_ok:
            _hero_status = ("Partially Degraded", "#FF9800", "#FFF3E0")
        else:
            _hero_status = ("Service Incident", "#D50000", "#FFEBEE")
    except Exception:
        _hero_status = ("API Unreachable", "#D50000", "#FFEBEE")

    # Page hero
    st.title("System Status")
    st.caption(f"Current State: **{_hero_status[0]}**")

    if st.button("Refresh Status", key="refresh_system"):
        st.rerun()

    # Tabs — NO st.status inside to prevent double-click reset bug
    tab1, tab2, tab3 = st.tabs(["Service Health", "Monitoring", "Alerts"])

    # ── Tab 1: Service Health ─────────────────────────────────────────────────
    with tab1:
        st.subheader("Service Health")

        health = {}
        try:
            health = api_client.get_health()

            with st.container(border=True):
                h1, h2, h3 = st.columns(3)

                with h1:
                    status = health.get("status", "unknown")
                    render_status_card(
                        label="Service Status",
                        value=status.title(),
                        status="ok" if status == "healthy" else "error",
                    )

                with h2:
                    model_loaded = health.get("model_loaded", False)
                    render_status_card(
                        label="Model Status",
                        value="Loaded" if model_loaded else "Not Loaded",
                        status="ok" if model_loaded else "warning",
                    )

                with h3:
                    fs_connected = health.get("feature_store_connected", False)
                    render_status_card(
                        label="Feature Store",
                        value="Connected" if fs_connected else "Disconnected",
                        status="ok" if fs_connected else "warning",
                    )

        except APIClientError as e:
            render_error_state("Cannot fetch health status — is the API running?", str(e))

        st.subheader("Model Information")

        try:
            model_info = api_client.get_model_info()

            with st.container(border=True):
                mi1, mi2, mi3 = st.columns(3)

                with mi1:
                    render_info_card("Model Name", model_info.get("model_name", "N/A"))
                    render_info_card("Model Version", model_info.get("model_version", "N/A"))

                with mi2:
                    render_info_card("Status", model_info.get("status", "N/A").title())
                    render_info_card(
                        "Approval Status", model_info.get("approval_status", "N/A").title()
                    )

                with mi3:
                    render_info_card("Dataset Type", model_info.get("dataset_type", "N/A"))
                    render_info_card("Data Provider", model_info.get("data_provider", "N/A"))

            training_date = model_info.get("training_date", "N/A")
            st.markdown(
                f'<div class="info-strip">Training Date: <b>{training_date}</b></div>',
                unsafe_allow_html=True,
            )

            metrics = model_info.get("metrics", {})
            if metrics:
                st.subheader("Model Metrics")
                with st.container(border=True):
                    mm1, mm2, mm3 = st.columns(3)
                    with mm1:
                        mae = metrics.get("mae")
                        render_info_card("MAE", f"{mae:.2f}" if mae else "N/A")
                    with mm2:
                        rmse = metrics.get("rmse")
                        render_info_card("RMSE", f"{rmse:.2f}" if rmse else "N/A")
                    with mm3:
                        r2 = metrics.get("r2")
                        render_info_card("R²", f"{r2:.4f}" if r2 else "N/A")

            feature_cols = model_info.get("feature_columns", [])
            target_cols = model_info.get("target_columns", [])
            if feature_cols or target_cols:
                st.subheader("Dataset Schema")
                with st.container(border=True):
                    sc1, sc2 = st.columns(2)
                    with sc1:
                        st.metric("Input Features", len(feature_cols))
                    with sc2:
                        st.metric("Target Variables", len(target_cols))

        except APIClientError as e:
            render_error_state("Cannot fetch model information", str(e))

        st.subheader("Pipeline Status")

        model_path = Path("models/production/best_model.pkl")
        metadata_path = Path("models/production/model_metadata.json")

        with st.container(border=True):
            pp1, pp2, pp3 = st.columns(3)

            with pp1:
                model_exists = model_path.exists()
                render_status_card(
                    label="Model Artifact",
                    value="Available" if model_exists else "Missing",
                    status="ok" if model_exists else "error",
                )

            with pp2:
                metadata_exists = metadata_path.exists()
                render_status_card(
                    label="Metadata",
                    value="Available" if metadata_exists else "Missing",
                    status="ok" if metadata_exists else "error",
                )

            with pp3:
                fs_ok = health.get("feature_store_connected", False) if health else False
                render_status_card(
                    label="Feature Store",
                    value="Connected" if fs_ok else "Not Connected",
                    status="ok" if fs_ok else "warning",
                )

        try:
            if metadata_path.exists():
                with open(metadata_path) as _f:
                    _meta = json.load(_f)

                st.subheader("Dataset Statistics")
                st.caption(f"Source: `{_meta.get('data_source', 'hopsworks_feature_store')}`")

                with st.container(border=True):
                    ds1, ds2, ds3, ds4 = st.columns(4)
                    with ds1:
                        st.metric("Train Rows", f"{_meta.get('train_rows', 0):,}")
                    with ds2:
                        st.metric("Val Rows", f"{_meta.get('val_rows', 0):,}")
                    with ds3:
                        st.metric("Test Rows", f"{_meta.get('test_rows', 0):,}")
                    with ds4:
                        st.metric("Features", _meta.get("n_features", 0))
        except Exception:
            pass

        st.subheader("Data Freshness")

        last_pred = health.get("last_prediction") if health else None
        pred_ago = format_time_ago(last_pred) if last_pred else "No predictions yet"

        with st.container(border=True):
            fr1, fr2 = st.columns(2)
            with fr1:
                st.markdown(
                    f'<div class="health-card-label">Last Prediction</div>'
                    f'<div class="health-card-value" style="font-size:1.1rem;color:#1A1A2E;margin-top:4px;">'
                    f"{pred_ago}</div>",
                    unsafe_allow_html=True,
                )
            with fr2:
                if model_path.exists():
                    try:
                        mtime = datetime.fromtimestamp(model_path.stat().st_mtime)
                        st.markdown(
                            f'<div class="health-card-label">Model Last Modified</div>'
                            f'<div class="health-card-value" style="font-size:1.1rem;color:#1A1A2E;margin-top:4px;">'
                            f'{mtime.strftime("%Y-%m-%d %H:%M:%S")}</div>',
                            unsafe_allow_html=True,
                        )
                    except Exception:
                        pass

    # ── Tab 2: Monitoring ─────────────────────────────────────────────────────
    with tab2:
        st.subheader("Data Drift Monitoring")

        try:
            with st.spinner("Running drift detection..."):
                drift = api_client.get_drift_report()

            unavailable = drift.get("status") == "unavailable" or (
                not drift.get("features") and drift.get("total_features", 0) == 0
            )

            if unavailable:
                st.info(
                    drift.get(
                        "message",
                        "Training data not available for drift detection. "
                        "This is expected in deployed environments.",
                    )
                )
            else:
                drift_detected = drift.get("drift_detected", False)
                drifted_count = drift.get("drifted_count", 0)
                drift_pct = drift.get("drift_percentage", 0)
                total_feats = drift.get("total_features", 0)

                if drift_detected:
                    st.markdown(
                        f"""
                        <div class="alert-card alert-warning">
                          <div class="alert-card-body" style="padding-left:14px;">
                            <div class="alert-card-city">Data Drift Detected</div>
                            <div class="alert-card-aqi">{drifted_count} of {total_feats} features drifted ({drift_pct:.1f}%)</div>
                            <div class="alert-card-rec">Consider retraining the model with updated data.</div>
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.success("No data drift detected — feature distributions are stable.")

                with st.container(border=True):
                    dc1, dc2, dc3 = st.columns(3)
                    with dc1:
                        render_status_card(
                            "Drift Status",
                            "Detected" if drift_detected else "Stable",
                            "warning" if drift_detected else "ok",
                        )
                    with dc2:
                        st.metric("Drifted Features", f"{drifted_count}/{total_feats}")
                    with dc3:
                        st.metric("Drift %", f"{drift_pct:.1f}%")

                drifted_cols = drift.get("drifted_columns", [])
                if drifted_cols:
                    st.subheader("Drifted Columns")
                    pills_html = "".join(
                        f'<span class="drift-pill">{col_info.get("column", "unknown")}</span>'
                        for col_info in drifted_cols
                    )
                    st.markdown(pills_html, unsafe_allow_html=True)

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
                    with st.container(border=True):
                        st.markdown(
                            '<div style="font-size:0.78rem;font-weight:700;color:#64748B;'
                            'text-transform:uppercase;letter-spacing:0.6px;margin-bottom:10px;">'
                            "Overall Training Metrics</div>",
                            unsafe_allow_html=True,
                        )
                        pm1, pm2, pm3 = st.columns(3)
                        with pm1:
                            st.metric("MAE", f"{overall.get('mae', 0):.2f}")
                        with pm2:
                            st.metric("RMSE", f"{overall.get('rmse', 0):.2f}")
                        with pm3:
                            st.metric("R²", f"{overall.get('r2', 0):.4f}")

                horizon_rows = []
                for h in ["24h", "48h", "72h"]:
                    if h in metrics:
                        m = metrics[h]
                        horizon_rows.append(
                            {
                                "Horizon": h,
                                "MAE": round(m.get("mae", 0), 2),
                                "RMSE": round(m.get("rmse", 0), 2),
                                "R²": round(m.get("r2", 0), 4),
                            }
                        )

                if horizon_rows:
                    import pandas as pd

                    st.subheader("Per-Horizon Metrics")
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

            alerts = alerts_data.get("alerts", [])
            total_alerts = alerts_data.get("total_alerts", 0)

            if total_alerts == 0:
                st.success(
                    "No active AQI alerts. Air quality is within safe levels across all cities."
                )
            else:
                st.markdown(
                    f'<div class="status-pill status-pill-error" style="margin-bottom:14px;">'
                    f'{total_alerts} Active Alert{"s" if total_alerts != 1 else ""}</div>',
                    unsafe_allow_html=True,
                )

                for alert in alerts:
                    city = alert.get("city", "Unknown")
                    aqi = alert.get("aqi", 0)
                    category = alert.get("category", "Unknown")
                    level = alert.get("alert_level", "none")
                    recommendation = alert.get("recommendation", "")
                    aqi_color = get_aqi_color(aqi)

                    if level == "critical":
                        card_cls = "alert-critical"
                    elif level == "warning":
                        card_cls = "alert-warning"
                    else:
                        card_cls = "alert-caution"

                    badge_html = (
                        f'<span style="background:{aqi_color}22;color:{aqi_color};'
                        f"border:1.5px solid {aqi_color};border-radius:20px;padding:2px 10px;"
                        f'font-size:0.75rem;font-weight:700;">AQI {aqi}</span>'
                    )

                    st.markdown(
                        f"""
                        <div class="alert-card {card_cls}">
                          <div class="alert-card-body" style="padding-left:14px;">
                            <div class="alert-card-city">{city}</div>
                            <div class="alert-card-aqi">{badge_html} &nbsp; {category}</div>
                            {'<div class="alert-card-rec">' + recommendation + '</div>' if recommendation else ''}
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        except APIClientError as e:
            render_error_state("Cannot check AQI alerts", str(e))

        st.divider()
        st.subheader("AQI Reference Guide")

        aqi_ref_rows = [
            (
                "0–50",
                "Good",
                AQI_COLORS["good"],
                "#E8F5E9",
                "Satisfactory. Air quality poses little or no risk.",
            ),
            (
                "51–100",
                "Moderate",
                AQI_COLORS["moderate"],
                "#FFFDE7",
                "Acceptable. Sensitive individuals may experience minor effects.",
            ),
            (
                "101–150",
                "Unhealthy for Sensitive",
                AQI_COLORS["unhealthy_sensitive"],
                "#FFF3E0",
                "Members of sensitive groups may experience health effects.",
            ),
            (
                "151–200",
                "Unhealthy",
                AQI_COLORS["unhealthy"],
                "#FFEBEE",
                "Everyone may begin to experience health effects.",
            ),
            (
                "201–300",
                "Very Unhealthy",
                AQI_COLORS["very_unhealthy"],
                "#F3E5F5",
                "Health alert: everyone may experience serious effects.",
            ),
            (
                "301–500",
                "Hazardous",
                AQI_COLORS["hazardous"],
                "#FCE4EC",
                "Emergency conditions. The entire population is likely to be affected.",
            ),
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
