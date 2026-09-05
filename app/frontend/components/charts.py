import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
"""
Chart Components

Reusable chart factory with consistent AirPulse theming applied to every figure.
"""

from typing import Any, Dict, List, Optional

import plotly.graph_objects as go

from app.frontend.utils.aqi_theme import (
    AQI_COLORS,
    CHART_COLORS,
    get_aqi_color,
    get_city_color,
    get_plotly_template,
)

# ── Theme Application Utility ─────────────────────────────────────────────────


def apply_chart_theme(
    fig: go.Figure,
    height: int = 400,
    title: Optional[str] = None,
    xaxis_title: Optional[str] = None,
    yaxis_title: Optional[str] = None,
) -> go.Figure:
    """
    Apply the AirPulse Plotly theme (fonts, colors, grid, hover) to any figure.

    Args:
        fig: Plotly figure to style
        height: Chart height in pixels
        title: Optional chart title override
        xaxis_title: Optional x-axis label
        yaxis_title: Optional y-axis label

    Returns:
        The same figure with layout updated in-place
    """
    layout_updates = {
        **get_plotly_template(),
        "height": height,
    }
    if title:
        layout_updates["title"] = {
            "text": title,
            "font": {"size": 15, "weight": 700, "color": "#1A1A2E"},
            "x": 0.0,
            "xanchor": "left",
        }
    if xaxis_title:
        layout_updates["xaxis"] = {**layout_updates.get("xaxis", {}), "title": xaxis_title}
    if yaxis_title:
        layout_updates["yaxis"] = {**layout_updates.get("yaxis", {}), "title": yaxis_title}

    fig.update_layout(**layout_updates)
    return fig


# ── AQI Zone Band Helper ──────────────────────────────────────────────────────


def _add_aqi_zone_bands(fig: go.Figure, annotate_right: bool = True) -> None:
    """Add EPA AQI color zone bands as background rectangles to a figure."""
    bands = [
        (0, 50, AQI_COLORS["good"], "Good"),
        (50, 100, AQI_COLORS["moderate"], "Moderate"),
        (100, 150, AQI_COLORS["unhealthy_sensitive"], "USG"),
        (150, 200, AQI_COLORS["unhealthy"], "Unhealthy"),
        (200, 300, AQI_COLORS["very_unhealthy"], "Very Unhealthy"),
    ]
    for y0, y1, color, label in bands:
        fig.add_hrect(
            y0=y0,
            y1=y1,
            fillcolor=color,
            opacity=0.07,
            line_width=0,
        )
        if annotate_right:
            fig.add_annotation(
                x=1.0,
                y=(y0 + y1) / 2,
                xref="paper",
                yref="y",
                text=label,
                showarrow=False,
                xanchor="left",
                font={"size": 9, "color": "#94A3B8"},
            )


# ── Public Chart Factories ────────────────────────────────────────────────────


def create_forecast_chart(
    timestamps: List[str],
    aqi_values: List[int],
    title: str = "3-Day AQI Forecast",
) -> go.Figure:
    """
    Create a polished AQI forecast line chart with zone bands and area fill.

    Args:
        timestamps: List of x-axis labels (horizon labels or timestamps)
        aqi_values: List of AQI values
        title: Chart title

    Returns:
        Themed Plotly figure
    """
    colors = [get_aqi_color(aqi) for aqi in aqi_values]

    fig = go.Figure()

    # Filled area beneath the line
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=aqi_values,
            fill="tozeroy",
            fillcolor="rgba(30,136,229,0.08)",
            line=dict(color=CHART_COLORS["primary"], width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Main line
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=aqi_values,
            mode="lines+markers+text",
            name="AQI Forecast",
            line=dict(color=CHART_COLORS["primary"], width=3, shape="spline", smoothing=0.6),
            marker=dict(
                size=12,
                color=colors,
                line=dict(width=2, color="white"),
            ),
            text=[str(v) for v in aqi_values],
            textposition="top center",
            textfont={"size": 11, "color": "#1A1A2E", "weight": 700},
            hovertemplate="<b>%{x}</b><br>AQI: <b>%{y}</b><extra></extra>",
        )
    )

    _add_aqi_zone_bands(fig, annotate_right=True)

    apply_chart_theme(
        fig,
        height=380,
        title=title,
        xaxis_title="Forecast Horizon",
        yaxis_title="AQI Value",
    )
    fig.update_yaxes(range=[0, max(max(aqi_values, default=100) + 50, 250)])
    fig.update_layout(showlegend=False)
    return fig


def create_multi_city_chart(
    city_data: Dict[str, Dict[str, Any]],
    title: str = "AQI by City",
) -> go.Figure:
    """
    Create a multi-city AQI comparison line chart.

    Args:
        city_data: {"CityName": {"timestamps": [...], "aqi_values": [...]}}
        title: Chart title

    Returns:
        Themed Plotly figure
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
                line=dict(color=color, width=2.5),
                marker=dict(size=6, color=color),
                hovertemplate=f"<b>{city}</b><br>AQI: <b>%{{y}}</b><br>%{{x}}<extra></extra>",
            )
        )

    _add_aqi_zone_bands(fig, annotate_right=False)
    apply_chart_theme(fig, height=400, title=title, xaxis_title="Time", yaxis_title="AQI")
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig


def create_pollutant_bar_chart(
    pollutants: Dict[str, float],
    title: str = "Pollutant Levels",
) -> go.Figure:
    """
    Create a styled pollutant concentration bar chart.

    Args:
        pollutants: {"pollutant_name": concentration_value}
        title: Chart title

    Returns:
        Themed Plotly figure
    """
    names = list(pollutants.keys())
    values = list(pollutants.values())
    # Gradient color by value magnitude
    max_v = max(values, default=1)
    bar_colors = [f"rgba(30,136,229,{0.4 + 0.6 * (v / max_v)})" for v in values]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=names,
            y=values,
            marker_color=bar_colors,
            marker_line_color="rgba(30,136,229,0.7)",
            marker_line_width=1.5,
            hovertemplate="<b>%{x}</b><br>%{y:.2f}<extra></extra>",
        )
    )
    apply_chart_theme(
        fig, height=360, title=title, xaxis_title="Pollutant", yaxis_title="Concentration"
    )
    return fig


def create_gauge_chart(
    value: int,
    title: str = "Current AQI",
) -> go.Figure:
    """
    Create a styled AQI gauge indicator chart.

    Args:
        value: AQI value (0-500)
        title: Chart title

    Returns:
        Themed Plotly figure
    """
    color = get_aqi_color(value)

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=value,
            title={"text": title, "font": {"size": 14, "color": "#1A1A2E"}},
            number={"font": {"size": 36, "color": color, "weight": 800}},
            gauge={
                "axis": {"range": [0, 500], "tickcolor": "#94A3B8", "tickwidth": 1},
                "bar": {"color": color, "thickness": 0.28},
                "bgcolor": "white",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 50], "color": "rgba(0,200,83,0.15)"},
                    {"range": [50, 100], "color": "rgba(255,214,0,0.15)"},
                    {"range": [100, 150], "color": "rgba(255,109,0,0.15)"},
                    {"range": [150, 200], "color": "rgba(213,0,0,0.15)"},
                    {"range": [200, 300], "color": "rgba(106,27,154,0.15)"},
                    {"range": [300, 500], "color": "rgba(74,0,16,0.15)"},
                ],
                "threshold": {
                    "line": {"color": color, "width": 3},
                    "thickness": 0.8,
                    "value": value,
                },
            },
        )
    )

    fig.update_layout(
        height=280,
        paper_bgcolor="rgba(0,0,0,0)",
        margin={"l": 20, "r": 20, "t": 40, "b": 10},
        font={"family": "Inter, system-ui, sans-serif"},
    )
    return fig
