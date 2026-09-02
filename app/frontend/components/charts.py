import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
"""
Chart Components

Reusable chart components for dashboard.
"""

from typing import Any, Dict, List, Optional

import plotly.graph_objects as go

from app.frontend.utils.aqi_theme import AQI_COLORS, CHART_COLORS, get_aqi_color, get_city_color


def create_forecast_chart(
    timestamps: List[str],
    aqi_values: List[int],
    title: str = "3-Day AQI Forecast",
) -> go.Figure:
    """
    Create AQI forecast line chart.

    Args:
        timestamps: List of timestamps
        aqi_values: List of AQI values
        title: Chart title

    Returns:
        Plotly figure
    """
    colors = [get_aqi_color(aqi) for aqi in aqi_values]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=aqi_values,
            mode="lines+markers",
            name="AQI",
            line=dict(color=CHART_COLORS["primary"], width=3),
            marker=dict(size=10, color=colors),
        )
    )

    # Add category zones
    fig.add_hrect(y0=0, y1=50, fillcolor=AQI_COLORS["good"], opacity=0.1, line_width=0)
    fig.add_hrect(y0=50, y1=100, fillcolor=AQI_COLORS["moderate"], opacity=0.1, line_width=0)
    fig.add_hrect(
        y0=100,
        y1=150,
        fillcolor=AQI_COLORS["unhealthy_sensitive"],
        opacity=0.1,
        line_width=0,
    )
    fig.add_hrect(y0=150, y1=200, fillcolor=AQI_COLORS["unhealthy"], opacity=0.1, line_width=0)

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="AQI",
        yaxis=dict(range=[0, 300]),
        template="plotly_white",
        showlegend=False,
    )

    return fig


def create_multi_city_chart(
    city_data: Dict[str, Dict[str, Any]],
    title: str = "AQI by City",
) -> go.Figure:
    """
    Create multi-city comparison chart.

    Args:
        city_data: Dictionary of city data
        title: Chart title

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    for city, data in city_data.items():
        color = get_city_color(city)
        fig.add_trace(
            go.Scatter(
                x=data.get("timestamps", []),
                y=data.get("aqi_values", []),
                mode="lines+markers",
                name=city,
                line=dict(color=color, width=2),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="AQI",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    return fig


def create_pollutant_bar_chart(
    pollutants: Dict[str, float],
    title: str = "Pollutant Levels",
) -> go.Figure:
    """
    Create pollutant bar chart.

    Args:
        pollutants: Dictionary of pollutant values
        title: Chart title

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=list(pollutants.keys()),
            y=list(pollutants.values()),
            marker_color=CHART_COLORS["primary"],
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Pollutant",
        yaxis_title="Concentration",
        template="plotly_white",
    )

    return fig


def create_gauge_chart(
    value: int,
    title: str = "Current AQI",
) -> go.Figure:
    """
    Create AQI gauge chart.

    Args:
        value: AQI value
        title: Chart title

    Returns:
        Plotly figure
    """
    color = get_aqi_color(value)

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": title},
            gauge={
                "axis": {"range": [0, 500]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 50], "color": AQI_COLORS["good"]},
                    {"range": [50, 100], "color": AQI_COLORS["moderate"]},
                    {"range": [100, 150], "color": AQI_COLORS["unhealthy_sensitive"]},
                    {"range": [150, 200], "color": AQI_COLORS["unhealthy"]},
                    {"range": [200, 300], "color": AQI_COLORS["very_unhealthy"]},
                    {"range": [300, 500], "color": AQI_COLORS["hazardous"]},
                ],
            },
        )
    )

    fig.update_layout(height=300)

    return fig
