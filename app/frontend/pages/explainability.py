import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
"""
Explainability Page

Model explainability with XGBoost feature importance.
Uses API client for data.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from app.frontend.utils.api_client import APIClient, APIClientError
from app.frontend.utils.aqi_theme import get_dashboard_css
from app.frontend.components.metrics import (
    render_error_state,
    render_info_card,
)


def render_explainability(api_client: APIClient):
    """
    Render explainability page.

    Args:
        api_client: API client instance
    """
    st.markdown(get_dashboard_css(), unsafe_allow_html=True)

    st.header("🔍 Model Explainability")

    # Fetch feature importance
    try:
        with st.spinner("Loading model analysis..."):
            feature_data = api_client.get_feature_importance(top_n=20)
            model_summary = api_client.get_model_summary()

        # Model Summary
        st.subheader("📋 Model Overview")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            render_info_card("Model Type", model_summary.get("model_type", "XGBoost"))
        with col2:
            metrics = model_summary.get("metrics", {})
            render_info_card("MAE", f"{metrics.get('mae', 0):.2f}")
        with col3:
            render_info_card("R²", f"{metrics.get('r2', 0):.4f}")
        with col4:
            render_info_card("Features", str(feature_data.get("total_features", 0)))

        # Parameters
        params = model_summary.get("parameters", {})
        if params:
            with st.expander("Model Parameters"):
                for k, v in params.items():
                    st.markdown(f"**{k}:** {v}")

        # Feature Importance
        st.subheader("📊 Top Feature Importance")

        features = feature_data.get("features", [])
        if features:
            # Bar chart
            fig = px.bar(
                x=[f["importance"] for f in features],
                y=[f["feature"] for f in features],
                orientation="h",
                title=f"Top {len(features)} Most Important Features",
                labels={"x": "Importance Score", "y": "Feature"},
            )
            fig.update_layout(height=max(400, len(features) * 25))
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)

            # Feature table
            st.dataframe(
                features,
                column_config={
                    "feature": "Feature Name",
                    "importance": st.column_config.ProgressColumn(
                        "Importance",
                        min_value=0,
                        max_value=max(f["importance"] for f in features) if features else 1,
                    ),
                },
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.warning("No feature importance data available.")

        # Category Importance
        st.subheader("📂 Importance by Feature Category")

        cat_importance = feature_data.get("category_importance", {})
        if cat_importance:
            fig = px.pie(
                names=list(cat_importance.keys()),
                values=list(cat_importance.values()),
                title="Feature Category Distribution",
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

            # Category details
            for cat, imp in cat_importance.items():
                st.markdown(f"**{cat.title()}:** {imp:.4f}")

        # Training Data Info
        st.subheader("📝 Training Data")

        training = model_summary.get("training_data", {})
        if training:
            col1, col2 = st.columns(2)
            with col1:
                render_info_card("Data Provider", training.get("provider", "N/A"))
                render_info_card("Date Range", training.get("date_range", "N/A"))
            with col2:
                render_info_card("Cities", ", ".join(training.get("cities", [])))
                render_info_card("Total Hours", f"{training.get('total_hours', 0):,}")

        # AQI Method
        st.subheader("🧮 AQI Calculation Method")
        st.info(model_summary.get("aqi_method", "US EPA PM NowCast AQI"))
        st.caption(f"Data Source: {model_summary.get('data_source', 'Open-Meteo')}")

    except APIClientError as e:
        render_error_state("Cannot fetch model data — is the API running?", str(e))
