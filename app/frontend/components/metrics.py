import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
"""
Metric Components

Reusable metric display components.
"""

import streamlit as st
from typing import Optional, Any

from app.frontend.utils.aqi_theme import get_aqi_color, get_aqi_category
from app.frontend.utils.formatters import format_aqi, format_time_ago


def render_aqi_card(
    label: str,
    aqi_value: int,
    category: str,
    delta: Optional[int] = None,
):
    """
    Render AQI metric card.
    
    Args:
        label: Card label (e.g., "24h Forecast")
        aqi_value: AQI value
        category: AQI category
        delta: Change from previous value
    """
    color = get_aqi_color(aqi_value)
    
    st.metric(
        label=label,
        value=format_aqi(aqi_value),
        delta=f"{delta:+d}" if delta else None,
        delta_color="inverse" if delta and delta > 0 else "normal",
    )
    
    # Category badge
    st.markdown(
        f'<p style="color:{color};font-weight:bold;">{category}</p>',
        unsafe_allow_html=True,
    )


def render_status_card(
    label: str,
    value: Any,
    status: str = "ok",
):
    """
    Render status metric card.
    
    Args:
        label: Card label
        value: Display value
        status: Status indicator (ok, warning, error)
    """
    status_colors = {
        "ok": "#43A047",
        "warning": "#FB8C00",
        "error": "#E53935",
    }
    
    color = status_colors.get(status, "#757575")
    
    st.markdown(
        f"""
        <div style="padding:10px;border-radius:5px;background:#f8f9fa;">
            <p style="margin:0;color:#666;">{label}</p>
            <p style="margin:0;font-size:1.2em;font-weight:bold;color:{color};">{value}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_info_card(
    label: str,
    value: Any,
    help_text: Optional[str] = None,
):
    """
    Render info metric card.
    
    Args:
        label: Card label
        value: Display value
        help_text: Optional help tooltip
    """
    st.metric(
        label=label,
        value=str(value) if value is not None else "N/A",
        help=help_text,
    )


def render_loading_state(message: str = "Loading..."):
    """Render loading spinner."""
    return st.spinner(message)


def render_error_state(message: str, error: Optional[Exception] = None):
    """
    Render error state.
    
    Args:
        message: Error message
        error: Optional exception details
    """
    st.error(message)
    if error:
        with st.expander("Error Details"):
            st.code(str(error))


def render_warning_state(message: str):
    """Render warning state."""
    st.warning(message)


def render_unavailable_state(feature: str):
    """
    Render feature unavailable state.
    
    Args:
        feature: Name of unavailable feature
    """
    st.info(f"ℹ️ {feature} is currently unavailable.")
    st.caption("This feature requires additional backend support.")
