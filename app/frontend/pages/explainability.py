"""
Explainability Page

Model explainability and feature importance.
Shows explanations only when backend provides them.
"""

import streamlit as st

from app.frontend.utils.api_client import APIClient, APIClientError
from app.frontend.utils.aqi_theme import get_dashboard_css
from app.frontend.components.metrics import (
    render_error_state,
    render_unavailable_state,
)


def render_explainability(api_client: APIClient):
    """
    Render explainability page.
    
    Args:
        api_client: API client instance
    """
    st.markdown(get_dashboard_css(), unsafe_allow_html=True)
    
    st.header("🔍 Model Explainability")
    
    # Check if explanations are available
    st.info(
        "Model explainability features require SHAP integration in the backend. "
        "This feature will be available when the prediction endpoint supports "
        "the `include_explanation` parameter with backend SHAP computation."
    )
    
    # Feature Importance Section
    st.subheader("Feature Importance")
    
    render_unavailable_state("Feature importance explanations")
    
    st.markdown("""
    **When available, this page will show:**
    - SHAP feature importance for each prediction
    - Global feature importance across all predictions
    - Feature interaction effects
    - Partial dependence plots
    """)
    
    # Prediction Explanation
    st.subheader("Prediction Explanation")
    
    city = st.selectbox(
        "Select City",
        ["Karachi", "Lahore", "Islamabad"],
        key="explain_city",
    )
    
    if st.button("Get Explanation", key="get_explanation"):
        render_unavailable_state("Prediction explanations")
        
        st.caption(
            "To enable explanations, the backend must implement SHAP computation "
            "and include explanation data in the prediction response."
        )
    
    # Technical Details
    st.subheader("Technical Details")
    
    with st.expander("About SHAP Explanations"):
        st.markdown("""
        **SHAP (SHapley Additive exPlanations)** is a game-theoretic approach to 
        explain the output of any machine learning model.
        
        **When implemented, you will see:**
        - **Feature importance**: Which features contributed most to the prediction
        - **Feature values**: Actual values of important features
        - **Direction**: Whether each feature pushed the prediction up or down
        
        **Requirements:**
        - Backend SHAP integration
        - Prediction endpoint with `include_explanation=true`
        - SHAP values in API response
        """)
