import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
"""
Analytics Page

Historical trends, pollutant analysis, weather correlations, and city comparisons.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from app.frontend.components.charts import apply_chart_theme
from app.frontend.components.metrics import (
    render_error_state,
    render_info_card,
    render_warning_state,
)
from app.frontend.utils.api_client import APIClient, APIClientError
from app.frontend.utils.aqi_theme import (
    AQI_COLORS,
    CITY_COLORS,
    get_aqi_color,
    get_city_color,
    get_dashboard_css,
)

VALID_CITIES = ["Karachi", "Lahore", "Islamabad"]


def render_analytics(api_client: APIClient):
    """Render analytics page."""
    st.markdown(get_dashboard_css(), unsafe_allow_html=True)

    # Page header
    st.title("Analytics Dashboard")
    st.caption("Historical AQI trends, pollutant analysis, and city comparisons.")

    # Controls toolbar
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            selected_city = st.selectbox(
                "City", VALID_CITIES, key="analytics_city",
                help="Select a city to analyze",
            )
        with col2:
            start_date = st.date_input(
                "Start Date",
                value=pd.Timestamp("2026-08-01").date(),
                key="analytics_start",
            )
        with col3:
            end_date = st.date_input(
                "End Date",
                value=pd.Timestamp("2026-09-01").date(),
                key="analytics_end",
            )

    # Fetch data
    try:
        with st.spinner("Loading historical data..."):
            history = api_client.get_historical_data(
                city=selected_city.lower(),
                start_date=str(start_date),
                end_date=str(end_date),
                limit=2000,
            )
            stats = api_client.get_statistics(city=selected_city.lower())

        data = history.get("data", [])

        if not data:
            render_warning_state("No data available for the selected date range.")
            return

        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        # Summary KPIs
        st.subheader("Summary Statistics")

        with st.container(border=True):
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)

            with kpi1:
                st.metric("Data Points", f"{len(df):,}")
            with kpi2:
                if "aqi" in df.columns:
                    st.metric("Average AQI", f"{df['aqi'].mean():.0f}")
            with kpi3:
                if "pm25" in df.columns:
                    st.metric("Avg PM2.5", f"{df['pm25'].mean():.1f} µg/m³")
            with kpi4:
                if "temperature" in df.columns:
                    st.metric("Avg Temperature", f"{df['temperature'].mean():.1f}°C")

        # AQI timeline
        st.subheader("AQI Over Time")

        if "aqi" in df.columns:
            with st.container(border=True):
                fig = go.Figure()

                fig.add_trace(
                    go.Scatter(
                        x=df["timestamp"], y=df["aqi"],
                        fill="tozeroy",
                        fillcolor="rgba(30,136,229,0.08)",
                        line=dict(color="rgba(30,136,229,0)", width=0),
                        showlegend=False, hoverinfo="skip",
                    )
                )

                city_color = get_city_color(selected_city)
                fig.add_trace(
                    go.Scatter(
                        x=df["timestamp"], y=df["aqi"],
                        mode="lines", name=f"{selected_city} AQI",
                        line=dict(color=city_color, width=2),
                        hovertemplate="<b>%{x|%b %d, %Y %H:%M}</b><br>AQI: <b>%{y}</b><extra></extra>",
                    )
                )

                bands = [
                    (0, 50, AQI_COLORS["good"], "Good"),
                    (50, 100, AQI_COLORS["moderate"], "Moderate"),
                    (100, 150, AQI_COLORS["unhealthy_sensitive"], "USG"),
                    (150, 200, AQI_COLORS["unhealthy"], "Unhealthy"),
                ]
                for y0, y1, color, label in bands:
                    fig.add_hrect(y0=y0, y1=y1, fillcolor=color, opacity=0.07, line_width=0)

                apply_chart_theme(
                    fig, height=420,
                    title=f"AQI Trend — {selected_city}",
                    xaxis_title="Date", yaxis_title="AQI",
                )
                st.plotly_chart(fig, use_container_width=True)

        # Pollutant analysis (tabbed)
        st.subheader("Pollutant Analysis")

        pollutant_cols = ["pm25", "pm10", "co", "no2", "so2", "o3"]
        available_pollutants = [c for c in pollutant_cols if c in df.columns]

        if available_pollutants:
            pm_cols = [c for c in ["pm25", "pm10"] if c in available_pollutants]
            nox_cols = [c for c in ["no2", "so2"] if c in available_pollutants]
            other_cols = [c for c in ["co", "o3"] if c in available_pollutants]

            tab_labels, tab_groups = [], []
            if pm_cols:
                tab_labels.append("Particulate Matter")
                tab_groups.append(pm_cols)
            if nox_cols:
                tab_labels.append("NOx & SO₂")
                tab_groups.append(nox_cols)
            if other_cols:
                tab_labels.append("CO & O₃")
                tab_groups.append(other_cols)

            if tab_labels:
                pol_tabs = st.tabs(tab_labels)
                for tab, group in zip(pol_tabs, tab_groups):
                    with tab:
                        with st.container(border=True):
                            sub_cols = st.columns(len(group))
                            for idx, col_name in enumerate(group):
                                with sub_cols[idx]:
                                    pfig = go.Figure()
                                    pfig.add_trace(
                                        go.Scatter(
                                            x=df["timestamp"], y=df[col_name],
                                            mode="lines", name=col_name.upper(),
                                            line=dict(color=_pollutant_color(col_name), width=2),
                                            fill="tozeroy",
                                            fillcolor=f"rgba({_hex_to_rgb(_pollutant_color(col_name))},0.1)",
                                            hovertemplate=(
                                                f"<b>{col_name.upper()}</b><br>"
                                                "%{x|%b %d %H:%M}<br>%{y:.2f}<extra></extra>"
                                            ),
                                        )
                                    )
                                    apply_chart_theme(
                                        pfig, height=300,
                                        title=col_name.upper(),
                                        xaxis_title="Date", yaxis_title="Concentration",
                                    )
                                    st.plotly_chart(pfig, use_container_width=True)
        else:
            st.info("No pollutant data columns found in the dataset.")

        # Weather vs AQI
        st.subheader("Weather vs AQI Correlation")

        weather_cols = ["temperature", "humidity", "wind_speed", "pressure"]
        available_weather = [c for c in weather_cols if c in df.columns]

        if available_weather and "aqi" in df.columns:
            with st.container(border=True):
                w_col1, w_col2 = st.columns([1, 3])
                with w_col1:
                    selected_weather = st.selectbox(
                        "Compare AQI with", available_weather,
                        key="weather_compare",
                        format_func=lambda x: x.replace("_", " ").title(),
                    )
                with w_col2:
                    scatter_fig = px.scatter(
                        df, x=selected_weather, y="aqi",
                        opacity=0.45, trendline="ols",
                        color_discrete_sequence=[get_city_color(selected_city)],
                        labels={"aqi": "AQI", selected_weather: selected_weather.replace("_", " ").title()},
                    )
                    for trace in scatter_fig.data:
                        if hasattr(trace, "line"):
                            trace.line.color = "#D50000"
                            trace.line.width = 2

                    apply_chart_theme(
                        scatter_fig, height=360,
                        title=f"AQI vs {selected_weather.replace('_', ' ').title()} — {selected_city}",
                        xaxis_title=selected_weather.replace("_", " ").title(),
                        yaxis_title="AQI",
                    )
                    st.plotly_chart(scatter_fig, use_container_width=True)

        # City comparison
        st.subheader("City Comparison")

        compare_data = []
        for city in ["karachi", "lahore", "islamabad"]:
            try:
                city_stats = api_client.get_statistics(city=city)
                stats_data = city_stats.get("statistics", {})
                if "aqi" in stats_data:
                    compare_data.append({
                        "City": city.title(),
                        "Avg AQI": round(stats_data["aqi"]["mean"], 1),
                        "Max AQI": round(stats_data["aqi"]["max"], 1),
                        "Avg PM2.5": round(stats_data.get("pm25", {}).get("mean", 0), 1),
                    })
            except Exception:
                pass

        if compare_data:
            compare_df = pd.DataFrame(compare_data)

            # City KPI tiles
            with st.container(border=True):
                tile_cols = st.columns(len(compare_data))
                for i, row in compare_df.iterrows():
                    with tile_cols[i]:
                        city_name = row["City"]
                        avg_aqi = row["Avg AQI"]
                        city_col = CITY_COLORS.get(city_name, "#1E88E5")
                        st.markdown(
                            f"""
                            <div class="kpi-block" style="border-left:4px solid {city_col};">
                              <div class="kpi-label">{city_name}</div>
                              <div class="kpi-value" style="color:{get_aqi_color(int(avg_aqi))};">{avg_aqi:.0f}</div>
                              <div class="kpi-delta">Avg AQI · Max: {row['Max AQI']:.0f}</div>
                              <div class="kpi-ci">PM2.5: {row['Avg PM2.5']:.1f} µg/m³</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            # Grouped bar chart
            with st.container(border=True):
                bar_fig = go.Figure()
                for metric_key, metric_label in [("Avg AQI", "Average AQI"), ("Avg PM2.5", "Average PM2.5")]:
                    bar_fig.add_trace(
                        go.Bar(
                            x=compare_df["City"], y=compare_df[metric_key],
                            name=metric_label,
                            marker_color=[CITY_COLORS.get(c, "#1E88E5") for c in compare_df["City"]],
                            text=compare_df[metric_key].apply(lambda v: f"{v:.1f}"),
                            textposition="outside",
                        )
                    )
                apply_chart_theme(
                    bar_fig, height=380,
                    title="City Comparison — Average AQI & PM2.5",
                    xaxis_title="City", yaxis_title="Value",
                )
                bar_fig.update_layout(barmode="group")
                st.plotly_chart(bar_fig, use_container_width=True)

    except APIClientError as e:
        render_error_state("Cannot fetch historical data — is the API running?", str(e))


_POLLUTANT_COLORS = {
    "pm25": "#1E88E5", "pm10": "#039BE5",
    "no2": "#E53935", "so2": "#FB8C00",
    "co": "#6D4C41", "o3": "#00897B",
}

def _pollutant_color(name: str) -> str:
    return _POLLUTANT_COLORS.get(name, "#1E88E5")

def _hex_to_rgb(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    if len(h) == 6:
        return f"{int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)}"
    return "30,136,229"
