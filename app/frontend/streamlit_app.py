"""
AirPulse — AQI Forecaster

Main Streamlit application for AQI prediction visualization.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st

from app.frontend.pages.analytics import render_analytics
from app.frontend.pages.dashboard import render_dashboard
from app.frontend.pages.explainability import render_explainability
from app.frontend.pages.system import render_system
from app.frontend.utils.api_client import APIClient
from app.frontend.utils.aqi_theme import (
    CITY_COLORS,
    get_aqi_category_short,
    get_aqi_color,
    get_dashboard_css,
)


def init_session_state():
    """Initialize Streamlit session state."""
    if "selected_city" not in st.session_state:
        st.session_state.selected_city = "Karachi"
    if "prediction_data" not in st.session_state:
        st.session_state.prediction_data = None
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = None


def _render_sidebar_nav(api_client: APIClient) -> str:
    """Render the branded sidebar and return the selected page name."""
    st.sidebar.markdown(get_dashboard_css(), unsafe_allow_html=True)

    # Brand header
    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
          <div class="sidebar-brand-icon">🌬️</div>
          <div class="sidebar-brand-text">
            <span class="sidebar-brand-name">AirPulse</span>
            <span class="sidebar-brand-sub">Pakistan AQI Intelligence</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        ["Dashboard", "Analytics", "Explainability", "System"],
        label_visibility="collapsed",
    )
    return page


def _render_sidebar_footer():
    """Render the city status strip and footer at the bottom of the sidebar."""
    # City quick-status strip
    pred = st.session_state.get("prediction_data")
    if pred:
        selected = st.session_state.get("selected_city", "Karachi")
        aqi_now = pred.get("aqi_24h", 0)
        color_now = get_aqi_color(aqi_now)
        cat_now = get_aqi_category_short(aqi_now)

        city_rows = ""
        for city in ["Karachi", "Lahore", "Islamabad"]:
            if city.lower() == selected.lower():
                cat_label = f"<b>AQI {aqi_now}</b> · {cat_now}"
                city_style = f"font-weight: 600; color: {color_now};"
            else:
                cat_label = "<span style='color:#94A3B8;'>—</span>"
                city_style = "color: #94A3B8;"

            city_rows += (
                f'<div class="city-strip-item">'
                f'<span style="{city_style}">{city}</span>'
                f'<span style="font-size:0.72rem;">{cat_label}</span>'
                f'</div>'
            )

        st.sidebar.markdown(
            f'<div class="city-strip">{city_rows}</div>',
            unsafe_allow_html=True,
        )
    else:
        city_rows = "".join(
            f'<div class="city-strip-item">'
            f'<span style="color:#94A3B8;">{c}</span>'
            f'<span style="font-size:0.7rem;color:#94A3B8;">—</span>'
            f'</div>'
            for c in ["Karachi", "Lahore", "Islamabad"]
        )
        st.sidebar.markdown(
            f'<div class="city-strip">{city_rows}</div>',
            unsafe_allow_html=True,
        )

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.caption("**AirPulse** v1.0.0")
    st.sidebar.caption("US EPA AQI Standards")


def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="AirPulse — AQI Forecaster",
        page_icon="🌬️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()

    api_client = APIClient.from_env()

    page = _render_sidebar_nav(api_client)

    if page == "Dashboard":
        render_dashboard(api_client)
    elif page == "Analytics":
        render_analytics(api_client)
    elif page == "Explainability":
        render_explainability(api_client)
    elif page == "System":
        render_system(api_client)
        
    _render_sidebar_footer()


if __name__ == "__main__":
    main()
