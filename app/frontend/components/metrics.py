import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
"""
Metric Components

Reusable metric display components with AirPulse styling.
"""

from typing import Any, Optional

import streamlit as st

from app.frontend.utils.aqi_theme import (
    get_aqi_bg_color,
    get_aqi_category,
    get_aqi_category_short,
    get_aqi_color,
)
from app.frontend.utils.formatters import format_aqi, format_time_ago


# CSS class helpers
_STRIPE_CLASS = {
    "good": "kpi-stripe-good",
    "moderate": "kpi-stripe-moderate",
    "unhealthy_sensitive": "kpi-stripe-usg",
    "unhealthy": "kpi-stripe-unhealthy",
    "very_unhealthy": "kpi-stripe-very-unhealthy",
    "hazardous": "kpi-stripe-hazardous",
}

_BADGE_CLASS = {
    "good": "badge-good",
    "moderate": "badge-moderate",
    "unhealthy_sensitive": "badge-usg",
    "unhealthy": "badge-unhealthy",
    "very_unhealthy": "badge-very-unhealthy",
    "hazardous": "badge-hazardous",
}

_DOT_CLASS = {"ok": "dot-ok animated", "warning": "dot-warning animated", "error": "dot-error animated"}


def _get_category_key(aqi_value: int) -> str:
    if aqi_value <= 50:
        return "good"
    elif aqi_value <= 100:
        return "moderate"
    elif aqi_value <= 150:
        return "unhealthy_sensitive"
    elif aqi_value <= 200:
        return "unhealthy"
    elif aqi_value <= 300:
        return "very_unhealthy"
    else:
        return "hazardous"


def render_aqi_card(
    label: str,
    aqi_value: int,
    category: str,
    delta: Optional[int] = None,
    ci_lower: Optional[int] = None,
    ci_upper: Optional[int] = None,
):
    """
    Render a styled AQI KPI card with tier stripe and category badge.

    Args:
        label: Card label (e.g., "24h Forecast")
        aqi_value: Numeric AQI value
        category: Display category string
        delta: Change vs. previous period (positive = worse)
        ci_lower: Lower confidence bound
        ci_upper: Upper confidence bound
    """
    key = _get_category_key(aqi_value)
    stripe = _STRIPE_CLASS.get(key, "kpi-stripe-good")
    badge_cls = _BADGE_CLASS.get(key, "badge-good")
    color = get_aqi_color(aqi_value)

    delta_html = ""
    if delta is not None and delta != 0:
        sign = "+" if delta > 0 else ""
        d_color = "#D50000" if delta > 0 else "#00C853"
        delta_html = (
            f'<div class="kpi-delta" style="color:{d_color};">'
            f'{sign}{delta} vs prev</div>'
        )

    ci_html = ""
    if ci_lower is not None and ci_upper is not None:
        ci_html = f'<div class="kpi-ci">{ci_lower} – {ci_upper}</div>'

    short_cat = get_aqi_category_short(aqi_value)

    html = f"""
<div class="kpi-block {stripe}">
  <div class="kpi-label">{label}</div>
  <div class="kpi-value" style="color:{color};">{aqi_value}</div>
  <span class="kpi-badge {badge_cls}">{short_cat}</span>
  {delta_html}
  {ci_html}
</div>
"""
    st.markdown(html, unsafe_allow_html=True)


def render_status_card(
    label: str,
    value: Any,
    status: str = "ok",
    icon: Optional[str] = None,
):
    """
    Render a system status card with animated dot indicator.

    Args:
        label: Card label
        value: Display value string
        status: 'ok', 'warning', or 'error'
        icon: Optional emoji icon override
    """
    status_colors = {"ok": "#00C853", "warning": "#FF9800", "error": "#D50000"}
    text_color = status_colors.get(status, "#64748B")

    st.markdown(
        f"""
        <div style="padding: 6px 0; border-bottom: 1px solid #F1F5F9; margin-bottom: 4px;">
            <div style="font-size: 0.75rem; color: #64748B; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 2px;">
                {label}
            </div>
            <div style="font-size: 0.95rem; color: {text_color}; font-weight: 500;">
                {value}
            </div>
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
    Render an info metric card using st.metric.

    Args:
        label: Card label
        value: Display value
        help_text: Optional tooltip text
    """
    display_val = str(value) if value is not None else "N/A"
    st.markdown(
        f"""
        <div style="padding: 6px 0; border-bottom: 1px solid #F1F5F9; margin-bottom: 4px;">
            <div style="font-size: 0.75rem; color: #64748B; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 2px;">
                {label}
            </div>
            <div style="font-size: 0.95rem; color: #0F172A; font-weight: 500; word-wrap: break-word;">
                {display_val}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_loading_state(message: str = "Loading data..."):
    """Render loading spinner context manager."""
    return st.spinner(message)


def render_error_state(message: str, error: Optional[Any] = None):
    """
    Render error state with collapsible details.

    Args:
        message: User-facing error description
        error: Optional exception or string details
    """
    st.error(f"**{message}**")
    if error:
        with st.expander("Error Details", expanded=False):
            st.code(str(error), language="text")


def render_warning_state(message: str):
    """Render warning banner."""
    st.warning(f"**{message}**")


def render_unavailable_state(feature: str):
    """
    Render feature-unavailable info card.

    Args:
        feature: Name of unavailable feature
    """
    st.info(
        f"**{feature}** is currently unavailable — "
        "additional backend support is required."
    )


def render_aqi_badge_html(aqi_value: int, category: str, size: str = "md") -> str:
    """
    Return an HTML AQI severity badge string for embedding in st.markdown().

    Args:
        aqi_value: Numeric AQI value
        category: Display category name
        size: 'sm', 'md', or 'lg'
    """
    color = get_aqi_color(aqi_value)
    bg = get_aqi_bg_color(aqi_value)
    size_styles = {
        "sm": "font-size:0.65rem;padding:2px 8px;",
        "md": "font-size:0.78rem;padding:3px 11px;",
        "lg": "font-size:0.9rem;padding:5px 16px;",
    }
    style = size_styles.get(size, size_styles["md"])
    return (
        f'<span style="background:{bg};color:{color};border:1.5px solid {color};'
        f'border-radius:20px;{style}font-weight:700;display:inline-block;'
        f'white-space:nowrap;line-height:1.5;">'
        f'<b>{aqi_value}</b> · {category}</span>'
    )
