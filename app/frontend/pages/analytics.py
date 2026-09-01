import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
"""
Analytics Page

Historical trends and pollutant analysis.
Uses API client for data.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from app.frontend.components.metrics import (
    render_error_state,
    render_info_card,
    render_warning_state,
)
from app.frontend.utils.api_client import APIClient, APIClientError
from app.frontend.utils.aqi_theme import get_city_color, get_dashboard_css

# Valid cities
VALID_CITIES = ["Karachi", "Lahore", "Islamabad"]


def render_analytics(api_client: APIClient):
    """
    Render analytics page.

    Args:
        api_client: API client instance
    """
    st.markdown(get_dashboard_css(), unsafe_allow_html=True)

    st.header("📊 Analytics Dashboard")

    # Controls
    col1, col2, col3 = st.columns(3)

    with col1:
        selected_city = st.selectbox("Select City", VALID_CITIES, key="analytics_city")

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

    # Fetch historical data
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
            st.warning("No data available for the selected date range.")
            return

        # Convert to DataFrame
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        # Summary stats
        st.subheader("📊 Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Data Points", f"{len(df):,}")

        with col2:
            if "aqi" in df.columns:
                avg_aqi = df["aqi"].mean()
                st.metric("Average AQI", f"{avg_aqi:.0f}")

        with col3:
            if "pm25" in df.columns:
                avg_pm25 = df["pm25"].mean()
                st.metric("Avg PM2.5", f"{avg_pm25:.1f} µg/m³")

        with col4:
            if "temperature" in df.columns:
                avg_temp = df["temperature"].mean()
                st.metric("Avg Temperature", f"{avg_temp:.1f}°C")

        # AQI Timeline
        st.subheader("📈 AQI Over Time")

        if "aqi" in df.columns:
            fig = px.line(
                df,
                x="timestamp",
                y="aqi",
                title=f"AQI Trend - {selected_city}",
                labels={"aqi": "AQI", "timestamp": "Date"},
            )

            # Add AQI category bands
            fig.add_hrect(y0=0, y1=50, fillcolor="green", opacity=0.1, annotation_text="Good")
            fig.add_hrect(
                y0=51,
                y1=100,
                fillcolor="yellow",
                opacity=0.1,
                annotation_text="Moderate",
            )
            fig.add_hrect(y0=101, y1=150, fillcolor="orange", opacity=0.1, annotation_text="USG")
            fig.add_hrect(
                y0=151,
                y1=200,
                fillcolor="red",
                opacity=0.1,
                annotation_text="Unhealthy",
            )

            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        # Pollutant Analysis
        st.subheader("🧪 Pollutant Analysis")

        pollutant_cols = ["pm25", "pm10", "co", "no2", "so2", "o3"]
        available_pollutants = [c for c in pollutant_cols if c in df.columns]

        if available_pollutants:
            fig = make_subplots(
                rows=2,
                cols=3,
                subplot_titles=[p.upper() for p in available_pollutants[:6]],
            )

            for i, col in enumerate(available_pollutants[:6]):
                row = i // 3 + 1
                c = i % 3 + 1
                fig.add_trace(
                    go.Scatter(x=df["timestamp"], y=df[col], name=col.upper(), mode="lines"),
                    row=row,
                    col=c,
                )

            fig.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        # Weather vs AQI Correlation
        st.subheader("🌤️ Weather vs AQI")

        weather_cols = ["temperature", "humidity", "wind_speed", "pressure"]
        available_weather = [c for c in weather_cols if c in df.columns]

        if available_weather and "aqi" in df.columns:
            selected_weather = st.selectbox(
                "Compare with", available_weather, key="weather_compare"
            )

            fig = px.scatter(
                df,
                x=selected_weather,
                y="aqi",
                title=f"AQI vs {selected_weather.title()} - {selected_city}",
                opacity=0.5,
                trendline="ols",
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        # City comparison
        st.subheader("🏙️ City Comparison (Full Dataset)")

        compare_data = []
        for city in ["karachi", "lahore", "islamabad"]:
            try:
                city_stats = api_client.get_statistics(city=city)
                stats_data = city_stats.get("statistics", {})
                if "aqi" in stats_data:
                    compare_data.append(
                        {
                            "City": city.title(),
                            "Avg AQI": stats_data["aqi"]["mean"],
                            "Max AQI": stats_data["aqi"]["max"],
                            "Avg PM2.5": stats_data.get("pm25", {}).get("mean", 0),
                        }
                    )
            except Exception:
                pass

        if compare_data:
            compare_df = pd.DataFrame(compare_data)
            fig = px.bar(
                compare_df,
                x="City",
                y=["Avg AQI", "Avg PM2.5"],
                barmode="group",
                title="City Comparison - Average AQI & PM2.5",
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    except APIClientError as e:
        render_error_state("Cannot fetch historical data — is the API running?", str(e))
