import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
"""
Explainability Page

Model explainability: XGBoost feature importance, SHAP global/per-prediction analysis,
and multi-model comparison. All API calls and data bindings preserved unchanged.
"""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.frontend.components.charts import apply_chart_theme
from app.frontend.components.metrics import render_error_state, render_info_card
from app.frontend.utils.api_client import APIClient, APIClientError
from app.frontend.utils.aqi_theme import (
    CHART_COLORS,
    get_aqi_color,
    get_city_color,
    get_dashboard_css,
)


def render_explainability(api_client: APIClient):
    """
    Render explainability page.

    Args:
        api_client: API client instance
    """
    st.markdown(get_dashboard_css(), unsafe_allow_html=True)

    # ── Page Hero ──────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="page-hero">
          <div class="page-hero-title">🔍 Model Explainability</div>
          <div class="page-hero-sub">
            XGBoost feature importance, SHAP value analysis, per-prediction explanations,
            and multi-model benchmark comparisons.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Tab Layout ─────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📊 Feature Importance",
            "🎯 Global SHAP",
            "🔬 Explain Prediction",
            "📈 Model Comparison",
        ]
    )

    # ══════════════════════════════════════════════════════════════════════════
    # Tab 1 — XGBoost Feature Importance
    # ══════════════════════════════════════════════════════════════════════════
    with tab1:
        try:
            with st.status("⏳ Loading model analysis…", expanded=False) as sw:
                feature_data = api_client.get_feature_importance(top_n=20)
                model_summary = api_client.get_model_summary()
                sw.update(label="✅ Model data loaded", state="complete", expanded=False)

            # ── Model Overview KPI Row ─────────────────────────────────────────
            st.markdown(
                '<div class="section-header">📋 Model Overview</div>',
                unsafe_allow_html=True,
            )

            with st.container(border=True):
                ov1, ov2, ov3, ov4 = st.columns(4)
                metrics = model_summary.get("metrics", {})
                with ov1:
                    render_info_card(
                        "Model Type",
                        model_summary.get("model_type", "XGBoost"),
                        help_text="The active production model architecture",
                    )
                with ov2:
                    render_info_card(
                        "MAE",
                        f"{metrics.get('mae', 0):.2f}",
                        help_text="Mean Absolute Error on the test set",
                    )
                with ov3:
                    render_info_card(
                        "R²",
                        f"{metrics.get('r2', 0):.4f}",
                        help_text="Coefficient of determination — closer to 1.0 is better",
                    )
                with ov4:
                    render_info_card(
                        "Features",
                        str(feature_data.get("total_features", 0)),
                        help_text="Number of input features used by the model",
                    )

            # ── Model Parameters Expander ──────────────────────────────────────
            params = model_summary.get("parameters", {})
            if params:
                with st.expander("🔧 Model Hyperparameters", expanded=False):
                    p_col1, p_col2 = st.columns(2)
                    items = list(params.items())
                    half = (len(items) + 1) // 2
                    for col, chunk in [(p_col1, items[:half]), (p_col2, items[half:])]:
                        with col:
                            for k, v in chunk:
                                st.markdown(
                                    f'<div style="display:flex;justify-content:space-between;'
                                    f'padding:4px 0;border-bottom:1px solid #F0F4F8;">'
                                    f'<span style="color:#64748B;font-size:0.8rem;">{k}</span>'
                                    f'<code style="font-size:0.8rem;">{v}</code></div>',
                                    unsafe_allow_html=True,
                                )

            # ── Feature Importance Chart ───────────────────────────────────────
            st.markdown(
                '<div class="section-header">📊 Top Feature Importance (XGBoost Gain)</div>',
                unsafe_allow_html=True,
            )

            features = feature_data.get("features", [])
            if features:
                with st.container(border=True):
                    # Gradient color from blue (low) to red (high) by rank
                    n = len(features)
                    bar_colors = [
                        f"rgba({int(30 + 183 * i / max(n - 1, 1))},"
                        f"{int(136 - 136 * i / max(n - 1, 1))},"
                        f"{int(229 - 229 * i / max(n - 1, 1))},0.85)"
                        for i in range(n)
                    ]

                    fi_fig = go.Figure()
                    fi_fig.add_trace(
                        go.Bar(
                            y=[f["feature"] for f in features],
                            x=[f["importance"] for f in features],
                            orientation="h",
                            marker_color=list(reversed(bar_colors)),
                            marker_line_width=0,
                            hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
                        )
                    )
                    fi_fig.update_yaxes(autorange="reversed")
                    apply_chart_theme(
                        fi_fig,
                        height=max(420, n * 26),
                        title=f"Top {n} Features by XGBoost Gain",
                        xaxis_title="Importance Score",
                        yaxis_title="",
                    )
                    st.plotly_chart(fi_fig, use_container_width=True)

                # Feature table with progress bars
                st.dataframe(
                    features,
                    column_config={
                        "feature": st.column_config.TextColumn("Feature Name"),
                        "importance": st.column_config.ProgressColumn(
                            "Importance",
                            help="XGBoost gain-based importance score",
                            min_value=0,
                            max_value=(max(f["importance"] for f in features) if features else 1),
                        ),
                    },
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.warning("No feature importance data available.", icon="⚠️")

            # ── Category Importance Donut ──────────────────────────────────────
            cat_importance = feature_data.get("category_importance", {})
            if cat_importance:
                st.markdown(
                    '<div class="section-header">📂 Importance by Feature Category</div>',
                    unsafe_allow_html=True,
                )
                with st.container(border=True):
                    donut_col, legend_col = st.columns([2, 1])
                    with donut_col:
                        donut_fig = px.pie(
                            names=list(cat_importance.keys()),
                            values=list(cat_importance.values()),
                            hole=0.55,
                            color_discrete_sequence=px.colors.qualitative.Set2,
                        )
                        donut_fig.update_traces(
                            textposition="outside",
                            textinfo="percent+label",
                            hovertemplate="<b>%{label}</b><br>%{percent}<extra></extra>",
                        )
                        apply_chart_theme(
                            donut_fig,
                            height=340,
                            title="Feature Category Distribution",
                        )
                        donut_fig.update_layout(showlegend=False)
                        st.plotly_chart(donut_fig, use_container_width=True)

                    with legend_col:
                        st.markdown(
                            "<div style='padding-top:60px;'></div>",
                            unsafe_allow_html=True,
                        )
                        sorted_cats = sorted(
                            cat_importance.items(), key=lambda x: x[1], reverse=True
                        )
                        for cat, imp in sorted_cats:
                            pct = imp / sum(cat_importance.values()) * 100
                            st.markdown(
                                f"<div style='margin-bottom:8px;'>"
                                f"<div style='font-size:0.75rem;font-weight:700;color:#64748B;"
                                f"text-transform:uppercase;letter-spacing:0.5px;'>{cat.title()}</div>"
                                f"<div style='font-size:0.9rem;font-weight:600;color:#1A1A2E;'>"
                                f"{imp:.4f} &nbsp;<span style='color:#94A3B8;font-size:0.75rem;'>"
                                f"({pct:.1f}%)</span></div>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )

            # ── Training Data Info ─────────────────────────────────────────────
            training = model_summary.get("training_data", {})
            if training:
                st.markdown(
                    '<div class="section-header">📝 Training Data</div>',
                    unsafe_allow_html=True,
                )
                with st.container(border=True):
                    td1, td2, td3, td4 = st.columns(4)
                    with td1:
                        render_info_card("Data Provider", training.get("provider", "N/A"))
                    with td2:
                        render_info_card("Date Range", training.get("date_range", "N/A"))
                    with td3:
                        render_info_card("Cities", ", ".join(training.get("cities", [])))
                    with td4:
                        render_info_card("Total Hours", f"{training.get('total_hours', 0):,}")

            # AQI method callout
            st.markdown(
                f'<div class="info-strip">🧮 AQI Method: <b>{model_summary.get("aqi_method", "US EPA PM NowCast AQI")}</b>'
                f'&nbsp;·&nbsp;📡 Source: <b>{model_summary.get("data_source", "Open-Meteo")}</b></div>',
                unsafe_allow_html=True,
            )

        except APIClientError as e:
            render_error_state("Cannot fetch model data — is the API running?", str(e))

    # ══════════════════════════════════════════════════════════════════════════
    # Tab 2 — Global SHAP Analysis
    # ══════════════════════════════════════════════════════════════════════════
    with tab2:
        try:
            with st.status("⏳ Computing global SHAP importance…", expanded=False) as sw:
                global_shap = api_client.get_global_shap(top_n=25)
                sw.update(label="✅ SHAP computed", state="complete", expanded=False)

            st.markdown(
                '<div class="section-header">🎯 Global SHAP Feature Importance</div>',
                unsafe_allow_html=True,
            )

            if global_shap.get("message"):
                st.info(f"{global_shap['message']}", icon="ℹ️")
            elif global_shap.get("n_samples", 0) == 0:
                st.info(
                    "Training data not available for SHAP computation. "
                    "Feature importance from XGBoost gain is available in the first tab.",
                    icon="ℹ️",
                )

            shap_features = global_shap.get("features", [])
            if shap_features:
                n_samples = global_shap.get("n_samples", 0)
                method = global_shap.get("method", "TreeExplainer")

                st.caption(
                    f"Method: **{method}** · Background samples: **{n_samples:,}**"
                )

                with st.container(border=True):
                    # Two-tone gradient bars: magnitude-colored
                    n_feats = len(shap_features)
                    max_shap = max(f["mean_abs_shap"] for f in shap_features)

                    shap_colors = [
                        f"rgba({int(255 * f['mean_abs_shap'] / max_shap)},"
                        f"{int(107 * (1 - f['mean_abs_shap'] / max_shap))},"
                        f"{int(107 * (1 - f['mean_abs_shap'] / max_shap))},0.82)"
                        for f in reversed(shap_features)
                    ]

                    shap_fig = go.Figure()
                    shap_fig.add_trace(
                        go.Bar(
                            y=[f["feature"] for f in reversed(shap_features)],
                            x=[f["mean_abs_shap"] for f in reversed(shap_features)],
                            orientation="h",
                            marker_color=shap_colors,
                            marker_line_width=0,
                            hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.4f}<extra></extra>",
                        )
                    )
                    apply_chart_theme(
                        shap_fig,
                        height=max(420, n_feats * 26),
                        title=f"Top {n_feats} Features by Mean |SHAP Value|",
                        xaxis_title="Mean |SHAP Value|",
                        yaxis_title="",
                    )
                    st.plotly_chart(shap_fig, use_container_width=True)

                st.dataframe(
                    [
                        {
                            "Rank": i + 1,
                            "Feature": f["feature"],
                            "Mean |SHAP|": f["mean_abs_shap"],
                        }
                        for i, f in enumerate(shap_features)
                    ],
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Mean |SHAP|": st.column_config.ProgressColumn(
                            "Mean |SHAP|",
                            min_value=0,
                            max_value=max(f["mean_abs_shap"] for f in shap_features),
                        )
                    },
                )

                # Interpretation callout
                st.markdown(
                    """
                    <div class="shap-callout">
                      <b>How to read this:</b> SHAP (SHapley Additive exPlanations) values show how much
                      each feature pushes predictions away from the average. A higher Mean |SHAP| means
                      the feature has a larger overall influence on model output. Unlike XGBoost's built-in
                      gain importance, SHAP provides <b>game-theoretic attribution</b> — each feature
                      receives credit proportional to its marginal contribution across all samples.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.warning("No SHAP data available.", icon="⚠️")

        except APIClientError as e:
            render_error_state("Cannot compute SHAP — is the API running?", str(e))

    # ══════════════════════════════════════════════════════════════════════════
    # Tab 3 — Per-Prediction SHAP Explanation
    # ══════════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown(
            '<div class="section-header">🔬 Explain a Prediction</div>',
            unsafe_allow_html=True,
        )
        st.caption("Enter feature values to see how each one contributes to the AQI prediction.")

        input_col, result_col = st.columns([1, 3])

        with input_col:
            with st.container(border=True):
                city = st.selectbox(
                    "🏙️ City", ["karachi", "lahore", "islamabad"], key="shap_city"
                )
                target = st.selectbox(
                    "🎯 Target",
                    ["target_aqi_24h", "target_aqi_48h", "target_aqi_72h"],
                    key="shap_target",
                )

                st.markdown(
                    '<div style="font-size:0.78rem;font-weight:700;color:#64748B;'
                    'text-transform:uppercase;letter-spacing:0.6px;margin:10px 0 6px;">Pollutant Inputs</div>',
                    unsafe_allow_html=True,
                )
                p1, p2 = st.columns(2)
                with p1:
                    pm25 = st.number_input("PM2.5 (µg/m³)", value=50.0, min_value=0.0, max_value=500.0, step=1.0, key="shap_pm25")
                    o3   = st.number_input("O₃ (µg/m³)",   value=40.0, min_value=0.0, max_value=200.0, step=1.0, key="shap_o3")
                    so2  = st.number_input("SO₂ (µg/m³)",  value=10.0, min_value=0.0, max_value=100.0, step=1.0, key="shap_so2")
                with p2:
                    pm10 = st.number_input("PM10 (µg/m³)", value=70.0, min_value=0.0, max_value=600.0, step=1.0, key="shap_pm10")
                    no2  = st.number_input("NO₂ (µg/m³)",  value=25.0, min_value=0.0, max_value=200.0, step=1.0, key="shap_no2")
                    co   = st.number_input("CO (µg/m³)",   value=1000.0, min_value=0.0, max_value=10000.0, step=10.0, key="shap_co")

                st.markdown(
                    '<div style="font-size:0.78rem;font-weight:700;color:#64748B;'
                    'text-transform:uppercase;letter-spacing:0.6px;margin:10px 0 6px;">Weather Inputs</div>',
                    unsafe_allow_html=True,
                )
                w1, w2 = st.columns(2)
                with w1:
                    temp     = st.number_input("Temp (°C)",    value=35.0, min_value=-10.0, max_value=55.0, step=1.0, key="shap_temp")
                    wind     = st.number_input("Wind (km/h)",  value=10.0, min_value=0.0, max_value=100.0, step=1.0, key="shap_wind")
                with w2:
                    humidity = st.number_input("Humidity (%)", value=60.0, min_value=0.0, max_value=100.0, step=1.0, key="shap_humidity")
                    pressure = st.number_input("Pressure (hPa)", value=1010.0, min_value=900.0, max_value=1100.0, step=1.0, key="shap_pressure")

                explain_btn = st.button(
                    "🧠 Explain Prediction", type="primary", key="shap_btn", use_container_width=True
                )

        with result_col:
            if explain_btn:
                features = {
                    "pm25": pm25, "pm10": pm10, "o3": o3, "no2": no2,
                    "so2": so2, "co": co, "temperature": temp,
                    "humidity": humidity, "wind_speed": wind, "pressure": pressure,
                    "pm25_aqi": pm25, "pm10_aqi": pm10,
                    "aqi": pm25, "aqi_derived": pm25,
                }

                try:
                    with st.status("⏳ Computing SHAP values…", expanded=False) as sw:
                        explanation = api_client.get_shap_explanation(features, target)
                        sw.update(label="✅ SHAP computed", state="complete", expanded=False)

                    pred = explanation.get("prediction", 0)
                    base = explanation.get("base_value", 0)
                    aqi_color = get_aqi_color(int(pred))

                    # Prediction KPI
                    st.markdown(
                        f"""
                        <div class="kpi-block" style="border-left:4px solid {aqi_color};margin-bottom:14px;">
                          <div class="kpi-label">SHAP Prediction — {target.replace('target_','').upper()}</div>
                          <div class="kpi-value" style="color:{aqi_color};">AQI {pred:.1f}</div>
                          <div class="kpi-delta">
                            {'▲' if pred > base else '▼'} {abs(pred - base):.1f} from baseline avg ({base:.1f})
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Waterfall bar chart
                    shap_vals = explanation.get("shap_values", [])
                    if shap_vals:
                        top_n = min(15, len(shap_vals))
                        top_shap = shap_vals[:top_n]

                        colors = [
                            "#D50000" if s["shap_value"] > 0 else "#00897B"
                            for s in reversed(top_shap)
                        ]

                        wf_fig = go.Figure(
                            go.Bar(
                                y=[s["feature"] for s in reversed(top_shap)],
                                x=[s["shap_value"] for s in reversed(top_shap)],
                                orientation="h",
                                marker_color=colors,
                                marker_line_width=0,
                                text=[f'{s["shap_value"]:+.2f}' for s in reversed(top_shap)],
                                textposition="outside",
                                hovertemplate=(
                                    "<b>%{y}</b><br>SHAP: %{x:+.3f}<extra></extra>"
                                ),
                            )
                        )
                        wf_fig.add_vline(
                            x=0,
                            line_dash="dash",
                            line_color="#94A3B8",
                            annotation_text="Baseline",
                            annotation_font_size=10,
                            annotation_font_color="#94A3B8",
                        )
                        apply_chart_theme(
                            wf_fig,
                            height=max(360, top_n * 28),
                            title=f"SHAP Contributions → {target}",
                            xaxis_title="SHAP Value (pushes AQI ↑ or ↓)",
                            yaxis_title="",
                        )
                        with st.container(border=True):
                            st.plotly_chart(wf_fig, use_container_width=True)

                        # Positive / negative split cards
                        pos = explanation.get("top_positive", [])
                        neg = explanation.get("top_negative", [])

                        pos_col, neg_col = st.columns(2)
                        with pos_col:
                            st.markdown(
                                '<div style="font-size:0.8rem;font-weight:700;color:#B71C1C;'
                                'text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px;">'
                                '🔴 Increases AQI (worse air)</div>',
                                unsafe_allow_html=True,
                            )
                            for p in pos[:5]:
                                st.markdown(
                                    f'<div style="background:#FFEBEE;border-left:3px solid #D50000;'
                                    f'border-radius:0 6px 6px 0;padding:6px 10px;margin-bottom:5px;">'
                                    f'<b>{p["feature"]}</b><br>'
                                    f'<span style="color:#D50000;font-weight:700;">+{p["shap_value"]:.2f}</span>'
                                    f'<span style="color:#94A3B8;font-size:0.75rem;"> · val={p["feature_value"]:.1f}</span>'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )
                        with neg_col:
                            st.markdown(
                                '<div style="font-size:0.8rem;font-weight:700;color:#004D40;'
                                'text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px;">'
                                '🟢 Decreases AQI (better air)</div>',
                                unsafe_allow_html=True,
                            )
                            for n in neg[:5]:
                                st.markdown(
                                    f'<div style="background:#E0F2F1;border-left:3px solid #00897B;'
                                    f'border-radius:0 6px 6px 0;padding:6px 10px;margin-bottom:5px;">'
                                    f'<b>{n["feature"]}</b><br>'
                                    f'<span style="color:#00897B;font-weight:700;">{n["shap_value"]:.2f}</span>'
                                    f'<span style="color:#94A3B8;font-size:0.75rem;"> · val={n["feature_value"]:.1f}</span>'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )
                    else:
                        st.warning("No SHAP values returned.", icon="⚠️")

                except APIClientError as e:
                    render_error_state("SHAP explanation failed", str(e))

            else:
                # Idle state — instructional callout
                st.markdown(
                    """
                    <div class="shap-callout" style="margin-top:24px;">
                      <b>👈 Enter feature values on the left and click Explain Prediction</b><br><br>
                      This tool will:
                      <ul style="margin:8px 0 0 0;padding-left:18px;">
                        <li>Run the inputs through the XGBoost model</li>
                        <li>Use SHAP TreeExplainer to decompose the prediction</li>
                        <li>Show which features push AQI higher (worse) or lower (better)</li>
                      </ul>
                      <br>This gives a <b>per-prediction</b> explanation, unlike the global importance
                      shown in other tabs.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # ══════════════════════════════════════════════════════════════════════════
    # Tab 4 — Model Comparison
    # ══════════════════════════════════════════════════════════════════════════
    with tab4:
        try:
            import json as _json
            import pandas as _pd

            st.markdown(
                '<div class="section-header">📈 Model Comparison</div>',
                unsafe_allow_html=True,
            )

            meta_path = os.path.join("models", "production", "model_metadata.json")

            models_data = {}
            best_model = "XGBoost"
            train_rows = val_rows = test_rows = n_features = 0

            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = _json.load(f)
                best_model = meta.get("model_name", "XGBoost")
                train_rows = meta.get("train_rows", 0)
                val_rows   = meta.get("val_rows", 0)
                test_rows  = meta.get("test_rows", 0)
                n_features = meta.get("n_features", 0)
                total      = train_rows + val_rows + test_rows

                mc = meta.get("model_comparison", {})
                for key, data in mc.items():
                    name = data.get("name", key)
                    models_data[name] = {
                        "test_metrics": {
                            "mae":  data.get("test_mae", 0),
                            "rmse": data.get("test_rmse", 0),
                            "r2":   data.get("test_r2", 0),
                        },
                        "val_metrics": {
                            "mae":  data.get("val_mae", 0),
                            "rmse": data.get("val_rmse", 0),
                            "r2":   data.get("val_r2", 0),
                        },
                        "train_time": data.get("train_time", 0),
                    }

            if models_data:
                total = train_rows + val_rows + test_rows

                # Dataset KPI strip
                with st.container(border=True):
                    ds1, ds2, ds3, ds4, ds5 = st.columns(5)
                    with ds1:
                        st.metric("Train Rows", f"{train_rows:,}")
                    with ds2:
                        st.metric("Val Rows", f"{val_rows:,}")
                    with ds3:
                        st.metric("Test Rows", f"{test_rows:,}")
                    with ds4:
                        st.metric("Total Rows", f"{total:,}")
                    with ds5:
                        st.metric("Features", n_features)

                st.caption(
                    f"Source: Hopsworks Feature Store · {len(models_data)} models evaluated"
                )

                # ── Performance Table ──────────────────────────────────────────
                st.markdown(
                    '<div class="section-header">📋 Test Set Performance</div>',
                    unsafe_allow_html=True,
                )

                table_data = []
                for name, data in models_data.items():
                    test = data.get("test_metrics", {})
                    is_best = name == best_model
                    table_data.append(
                        {
                            "Model": f"🏆 {name}" if is_best else name,
                            "MAE": round(test.get("mae", 0), 2),
                            "RMSE": round(test.get("rmse", 0), 2),
                            "R²": round(test.get("r2", 0), 4),
                            "Train Time (s)": round(data.get("train_time", 0), 1),
                        }
                    )

                best_mae = models_data.get(best_model, {}).get("test_metrics", {}).get("mae", 0)
                max_r2 = max(d["test_metrics"].get("r2", 0) for d in models_data.values())

                st.dataframe(
                    _pd.DataFrame(table_data),
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "MAE": st.column_config.NumberColumn(
                            "MAE ↓", format="%.2f", help="Lower is better"
                        ),
                        "R²": st.column_config.ProgressColumn(
                            "R² ↑",
                            help="Coefficient of determination — higher is better",
                            min_value=0,
                            max_value=1,
                        ),
                    },
                )

                # ── Bar Charts Side by Side ────────────────────────────────────
                st.markdown(
                    '<div class="section-header">📊 Test Set Visualisation</div>',
                    unsafe_allow_html=True,
                )

                model_names = list(models_data.keys())
                mae_values  = [models_data[m]["test_metrics"]["mae"]  for m in model_names]
                r2_values   = [models_data[m]["test_metrics"]["r2"]   for m in model_names]

                def _model_color(m: str) -> str:
                    return "#00C853" if m == best_model else "#1E88E5"

                bar1_col, bar2_col = st.columns(2)

                with bar1_col:
                    with st.container(border=True):
                        mae_fig = go.Figure(
                            go.Bar(
                                x=model_names,
                                y=mae_values,
                                marker_color=[_model_color(m) for m in model_names],
                                text=[f"{v:.2f}" for v in mae_values],
                                textposition="outside",
                                hovertemplate="<b>%{x}</b><br>MAE: %{y:.2f}<extra></extra>",
                            )
                        )
                        apply_chart_theme(
                            mae_fig, height=320,
                            title="MAE (Lower = Better)",
                            xaxis_title="Model", yaxis_title="MAE",
                        )
                        st.plotly_chart(mae_fig, use_container_width=True)

                with bar2_col:
                    with st.container(border=True):
                        r2_fig = go.Figure(
                            go.Bar(
                                x=model_names,
                                y=r2_values,
                                marker_color=[_model_color(m) for m in model_names],
                                text=[f"{v:.4f}" for v in r2_values],
                                textposition="outside",
                                hovertemplate="<b>%{x}</b><br>R²: %{y:.4f}<extra></extra>",
                            )
                        )
                        apply_chart_theme(
                            r2_fig, height=320,
                            title="R² (Higher = Better)",
                            xaxis_title="Model", yaxis_title="R²",
                        )
                        st.plotly_chart(r2_fig, use_container_width=True)

                # ── Per-Horizon Comparison ─────────────────────────────────────
                horizons = ["24h", "48h", "72h"]
                has_per_horizon = any(models_data[m].get("per_horizon") for m in model_names)

                if has_per_horizon:
                    st.markdown(
                        '<div class="section-header">⏱️ Per-Horizon Performance</div>',
                        unsafe_allow_html=True,
                    )
                    horizon_data = []
                    for name in model_names:
                        ph = models_data[name].get("per_horizon", {})
                        for h in horizons:
                            h_data = ph.get(h, {}).get("test", {})
                            if h_data:
                                horizon_data.append(
                                    {
                                        "Horizon": h,
                                        "Model": name,
                                        "MAE": h_data.get("mae", 0),
                                        "RMSE": h_data.get("rmse", 0),
                                        "R²": h_data.get("r2", 0),
                                    }
                                )

                    if horizon_data:
                        horizon_df = _pd.DataFrame(horizon_data)
                        h1_col, h2_col = st.columns(2)

                        with h1_col:
                            with st.container(border=True):
                                h_mae_fig = go.Figure()
                                for name in model_names:
                                    df_m = horizon_df[horizon_df["Model"] == name]
                                    if not df_m.empty:
                                        h_mae_fig.add_trace(
                                            go.Bar(
                                                name=name,
                                                x=df_m["Horizon"].tolist(),
                                                y=df_m["MAE"].tolist(),
                                                marker_color=_model_color(name),
                                                hovertemplate=f"<b>{name}</b><br>%{{x}}: MAE %{{y:.2f}}<extra></extra>",
                                            )
                                        )
                                apply_chart_theme(
                                    h_mae_fig, height=340,
                                    title="MAE by Forecast Horizon",
                                    xaxis_title="Horizon", yaxis_title="MAE",
                                )
                                h_mae_fig.update_layout(barmode="group")
                                st.plotly_chart(h_mae_fig, use_container_width=True)

                        with h2_col:
                            with st.container(border=True):
                                h_r2_fig = go.Figure()
                                for name in model_names:
                                    df_m = horizon_df[horizon_df["Model"] == name]
                                    if not df_m.empty:
                                        h_r2_fig.add_trace(
                                            go.Bar(
                                                name=name,
                                                x=df_m["Horizon"].tolist(),
                                                y=df_m["R²"].tolist(),
                                                marker_color=_model_color(name),
                                                hovertemplate=f"<b>{name}</b><br>%{{x}}: R² %{{y:.4f}}<extra></extra>",
                                            )
                                        )
                                apply_chart_theme(
                                    h_r2_fig, height=340,
                                    title="R² by Forecast Horizon",
                                    xaxis_title="Horizon", yaxis_title="R²",
                                )
                                h_r2_fig.update_layout(barmode="group")
                                st.plotly_chart(h_r2_fig, use_container_width=True)

                        st.dataframe(horizon_df, hide_index=True, use_container_width=True)

                # ── Winner Banner ──────────────────────────────────────────────
                st.success(
                    f"🏆 **Production Model: {best_model}** — Test MAE: **{best_mae:.2f}** "
                    f"| R²: **{models_data.get(best_model, {}).get('test_metrics', {}).get('r2', 0):.4f}**",
                    icon="🏆",
                )
            else:
                st.info(
                    "Model comparison data not found. Run the training pipeline first.",
                    icon="ℹ️",
                )

        except Exception as e:
            render_error_state("Error loading model comparison", str(e))
