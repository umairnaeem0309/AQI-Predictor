import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
"""
Dashboard Page

Primary view: city selector, 3-day AQI KPI cards, forecast chart,
confidence intervals, and model info.
"""

from typing import Optional

import plotly.graph_objects as go
import streamlit as st

from app.frontend.components.charts import apply_chart_theme, create_forecast_chart
from app.frontend.components.metrics import (
    render_aqi_card,
    render_error_state,
    render_warning_state,
)
from app.frontend.utils.api_client import APIClient, APIClientError
from app.frontend.utils.aqi_theme import (
    get_aqi_category,
    get_aqi_category_short,
    get_aqi_color,
    get_dashboard_css,
    render_aqi_badge,
)
from app.frontend.utils.formatters import format_time_ago, format_timestamp

VALID_CITIES = ["Karachi", "Lahore", "Islamabad"]

_CITY_FLAGS = {"Karachi": "🏙️", "Lahore": "🌳", "Islamabad": "🏔️"}


def render_dashboard(api_client: APIClient):
    """Render main dashboard page."""
    st.markdown(get_dashboard_css(), unsafe_allow_html=True)

    # Toolbar
    toolbar_col1, toolbar_col2, toolbar_col3 = st.columns([3, 1, 2])

    with toolbar_col1:
        selected_city = st.selectbox(
            "Select City",
            VALID_CITIES,
            key="city_selector",
            help="Choose a city to view its AQI forecast.",
        )

    with toolbar_col2:
        st.markdown("<div style='padding-top:26px;'>", unsafe_allow_html=True)
        refresh_clicked = st.button("Refresh", key="refresh_btn", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with toolbar_col3:
        last_upd = format_time_ago(st.session_state.get("last_refresh"))
        st.markdown(
            f"<div style='padding-top:30px;font-size:0.78rem;color:#64748B;'>"
            f"Last updated: <b>{last_upd}</b></div>",
            unsafe_allow_html=True,
        )

    if refresh_clicked:
        st.cache_data.clear()
        st.rerun()

    st.session_state.selected_city = selected_city

    # Fetch prediction
    try:
        with st.spinner("Fetching forecast data..."):
            prediction = api_client.get_prediction(selected_city)
            st.session_state.prediction_data = prediction
            st.session_state.last_refresh = prediction.get("timestamp")
    except APIClientError as e:
        render_error_state("Failed to fetch prediction data", e)
        return

    if not prediction:
        render_warning_state("No prediction data available")
        return

    aqi_24h = prediction.get("aqi_24h", 0)
    aqi_48h = prediction.get("aqi_48h", 0)
    aqi_72h = prediction.get("aqi_72h", 0)
    cat_24h = prediction.get("category_24h", get_aqi_category(aqi_24h))
    cat_48h = prediction.get("category_48h", get_aqi_category(aqi_48h))
    cat_72h = prediction.get("category_72h", get_aqi_category(aqi_72h))

    # Hero band
    badge_html = render_aqi_badge(aqi_24h, get_aqi_category_short(aqi_24h), size="md")
    city_flag = _CITY_FLAGS.get(selected_city, "")
    model_ver = prediction.get("model_version", "v1")

    st.markdown(
        f"""
        <div class="page-hero">
          <div class="page-hero-title">{city_flag} {selected_city} — Air Quality Forecast</div>
          <div class="page-hero-sub" style="display:flex;align-items:center;gap:12px;margin-top:8px;">
            Current 24h AQI: {badge_html}
            <span style="opacity:0.7;">Model {model_ver}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Confidence intervals (extract early for KPI cards)
    confidence = prediction.get("confidence")
    intervals = confidence.get("intervals") if confidence else None

    def _get_ci(horizon: str):
        if not intervals:
            return None, None
        iv = intervals.get(horizon)
        if iv:
            return int(iv.get("lower", 0)), int(iv.get("upper", 0))
        return None, None

    ci_24_lo, ci_24_hi = _get_ci("24h")
    ci_48_lo, ci_48_hi = _get_ci("48h")
    ci_72_lo, ci_72_hi = _get_ci("72h")

    # Forecast KPI cards
    st.subheader("Forecast Summary")

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
        render_aqi_card(
            label="Current / 24h",
            aqi_value=aqi_24h,
            category=cat_24h,
            ci_lower=ci_24_lo,
            ci_upper=ci_24_hi,
        )

    with kpi2:
        render_aqi_card(
            label="24h Forecast",
            aqi_value=aqi_24h,
            category=cat_24h,
            delta=None,
            ci_lower=ci_24_lo,
            ci_upper=ci_24_hi,
        )

    with kpi3:
        render_aqi_card(
            label="48h Forecast",
            aqi_value=aqi_48h,
            category=cat_48h,
            delta=aqi_48h - aqi_24h,
            ci_lower=ci_48_lo,
            ci_upper=ci_48_hi,
        )

    with kpi4:
        render_aqi_card(
            label="72h Forecast",
            aqi_value=aqi_72h,
            category=cat_72h,
            delta=aqi_72h - aqi_48h,
            ci_lower=ci_72_lo,
            ci_upper=ci_72_hi,
        )

    # Forecast chart
    st.subheader("3-Day Forecast Trend")

    timestamps = [
        prediction.get("timestamp", "Now"),
        "24h",
        "48h",
        "72h",
    ]
    aqi_values = [aqi_24h, aqi_24h, aqi_48h, aqi_72h]

    with st.container(border=True):
        fig = create_forecast_chart(timestamps, aqi_values, f"AQI Forecast — {selected_city}")
        st.plotly_chart(fig, use_container_width=True)

    # Confidence interval section
    if confidence and intervals:
        st.subheader("Prediction Confidence Intervals")

        ci_method = confidence.get("method", "N/A")
        ci_level = confidence.get("level", 90)

        st.caption(
            f"Method: **{ci_method}** · Confidence Level: **{ci_level}%**"
        )

        horizons, point_preds, lower_bounds, upper_bounds = [], [], [], []
        for h in ["24h", "48h", "72h"]:
            iv = intervals.get(h)
            if iv:
                horizons.append(h)
                point_preds.append(iv["point_prediction"])
                lower_bounds.append(iv["lower"])
                upper_bounds.append(iv["upper"])

        if horizons:
            with st.container(border=True):
                ci_fig = go.Figure()

                ci_fig.add_trace(
                    go.Scatter(
                        x=horizons + horizons[::-1],
                        y=upper_bounds + lower_bounds[::-1],
                        fill="toself",
                        fillcolor="rgba(30,136,229,0.12)",
                        line=dict(color="rgba(30,136,229,0)"),
                        name=f"{ci_level}% Confidence Band",
                        hoverinfo="skip",
                    )
                )

                ci_fig.add_trace(
                    go.Scatter(
                        x=horizons, y=upper_bounds,
                        mode="lines",
                        line=dict(color="rgba(30,136,229,0.4)", width=1, dash="dot"),
                        name="Upper Bound",
                        hovertemplate="Upper: <b>%{y}</b><extra></extra>",
                    )
                )
                ci_fig.add_trace(
                    go.Scatter(
                        x=horizons, y=lower_bounds,
                        mode="lines",
                        line=dict(color="rgba(30,136,229,0.4)", width=1, dash="dot"),
                        name="Lower Bound",
                        hovertemplate="Lower: <b>%{y}</b><extra></extra>",
                    )
                )

                ci_fig.add_trace(
                    go.Scatter(
                        x=horizons, y=point_preds,
                        mode="lines+markers+text",
                        name="Point Prediction",
                        line=dict(color="#1E88E5", width=3),
                        marker=dict(size=12, color="#1E88E5", line=dict(width=2, color="white")),
                        text=[f"AQI {int(p)}" for p in point_preds],
                        textposition="top center",
                        textfont={"size": 11, "weight": 700},
                        hovertemplate="Forecast: <b>AQI %{y}</b><extra></extra>",
                    )
                )

                apply_chart_theme(
                    ci_fig, height=360,
                    title=f"AQI Forecast with {ci_level}% Confidence Intervals",
                    xaxis_title="Forecast Horizon",
                    yaxis_title="AQI",
                )
                ci_fig.update_layout(
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(ci_fig, use_container_width=True)

                import pandas as pd

                iv_data = []
                for h in ["24h", "48h", "72h"]:
                    iv = intervals.get(h)
                    if iv:
                        iv_data.append({
                            "Horizon": h,
                            "Point Prediction": int(iv["point_prediction"]),
                            "Lower Bound": int(iv["lower"]),
                            "Upper Bound": int(iv["upper"]),
                            "Interval Width": int(iv["width"]),
                        })

                if iv_data:
                    max_width = max(r["Interval Width"] for r in iv_data)
                    st.dataframe(
                        pd.DataFrame(iv_data),
                        hide_index=True,
                        use_container_width=True,
                        column_config={
                            "Interval Width": st.column_config.ProgressColumn(
                                "Interval Width",
                                help="Wider = more uncertainty",
                                min_value=0,
                                max_value=max_width + 20,
                            )
                        },
                    )

    elif confidence is None:
        st.info("Confidence intervals will appear once residual statistics are computed.")

    # Model info strip
    level = confidence.get("level", "N/A") if confidence else "N/A"
    st.markdown(
        f'<div class="info-strip">Model: <b>{prediction.get("model_version", "N/A")}</b>'
        f'&nbsp;&nbsp;·&nbsp;&nbsp;Confidence Level: <b>{level}%</b>'
        f'&nbsp;&nbsp;·&nbsp;&nbsp;Source: US EPA PM NowCast AQI</div>',
        unsafe_allow_html=True,
    )
