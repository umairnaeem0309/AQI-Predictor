import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
"""
Explainability Page

Model explainability with XGBoost feature importance and SHAP values.
Uses API client for data.
"""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.frontend.components.metrics import render_error_state, render_info_card
from app.frontend.utils.api_client import APIClient, APIClientError
from app.frontend.utils.aqi_theme import get_city_color, get_dashboard_css


def render_explainability(api_client: APIClient):
    """
    Render explainability page.

    Args:
        api_client: API client instance
    """
    st.markdown(get_dashboard_css(), unsafe_allow_html=True)

    st.header("🔍 Model Explainability")

    # Tab layout for different explanation types
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📊 Feature Importance",
            "🎯 SHAP Global Analysis",
            "🔬 SHAP Prediction Explanation",
            "📈 Model Comparison",
        ]
    )

    # ── Tab 1: XGBoost Feature Importance ──
    with tab1:
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

            # Feature Importance Bar Chart
            st.subheader("📊 Top Feature Importance (XGBoost Gain)")

            features = feature_data.get("features", [])
            if features:
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
                            max_value=(max(f["importance"] for f in features) if features else 1),
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

    # ── Tab 2: Global SHAP Analysis ──
    with tab2:
        try:
            with st.spinner("Computing global SHAP importance (may take a moment)..."):
                global_shap = api_client.get_global_shap(top_n=25)

            st.subheader("🎯 Global SHAP Feature Importance")

            # Check if data is available
            if global_shap.get("message"):
                st.info(f"ℹ️ {global_shap['message']}")
            elif global_shap.get("n_samples", 0) == 0:
                st.info(
                    "ℹ️ Training data not available for SHAP computation. Feature importance from XGBoost gain is still available in the first tab."
                )
            else:
                st.caption(
                    f"Method: {global_shap.get('method', 'TreeExplainer')} | "
                    f"Background samples: {global_shap.get('n_samples', 0)}"
                )

            shap_features = global_shap.get("features", [])
            if shap_features:
                # Waterfall-style horizontal bar chart
                fig = go.Figure()
                fig.add_trace(
                    go.Bar(
                        y=[f["feature"] for f in reversed(shap_features)],
                        x=[f["mean_abs_shap"] for f in reversed(shap_features)],
                        orientation="h",
                        marker_color="#FF6B6B",
                        name="Mean |SHAP|",
                    )
                )
                fig.update_layout(
                    title=f"Top {len(shap_features)} Features by Mean |SHAP Value|",
                    xaxis_title="Mean |SHAP Value|",
                    yaxis_title="Feature",
                    height=max(400, len(shap_features) * 28),
                )
                st.plotly_chart(fig, use_container_width=True)

                # Table
                st.dataframe(
                    [
                        {
                            "Rank": i + 1,
                            "Feature": f["feature"],
                            "Mean |SHAP|": f["mean_abs_shap"],
                        }
                        for i, f in enumerate(shap_features)
                    ],
                    hide_index=True,
                    use_container_width=True,
                )

                # Interpretation
                st.markdown(
                    """
                **How to read this:** SHAP (SHapley Additive exPlanations) values show how much each
                feature contributes to pushing predictions away from the average prediction.
                Higher |SHAP| = more influence on the model's output.

                Unlike XGBoost's built-in importance (which uses split gain), SHAP provides
                **game-theoretic** feature attribution — each feature gets credit proportional
                to its marginal contribution.
                """
                )
            else:
                st.warning("No SHAP data available.")

        except APIClientError as e:
            render_error_state("Cannot compute SHAP — is the API running?", str(e))

    # ── Tab 3: Per-Prediction SHAP Explanation ──
    with tab3:
        st.subheader("🔬 Explain a Prediction")
        st.caption("Enter feature values to see how each one contributes to the AQI prediction.")

        col1, col2 = st.columns([1, 3])

        with col1:
            city = st.selectbox("City", ["karachi", "lahore", "islamabad"], key="shap_city")
            target = st.selectbox(
                "Target",
                ["target_aqi_24h", "target_aqi_48h", "target_aqi_72h"],
                key="shap_target",
            )

            st.markdown("**Pollutant Inputs**")
            pm25 = st.number_input(
                "PM2.5 (μg/m³)",
                value=50.0,
                min_value=0.0,
                max_value=500.0,
                step=1.0,
                key="shap_pm25",
            )
            pm10 = st.number_input(
                "PM10 (μg/m³)",
                value=70.0,
                min_value=0.0,
                max_value=600.0,
                step=1.0,
                key="shap_pm10",
            )
            o3 = st.number_input(
                "O3 (μg/m³)",
                value=40.0,
                min_value=0.0,
                max_value=200.0,
                step=1.0,
                key="shap_o3",
            )
            no2 = st.number_input(
                "NO2 (μg/m³)",
                value=25.0,
                min_value=0.0,
                max_value=200.0,
                step=1.0,
                key="shap_no2",
            )
            so2 = st.number_input(
                "SO2 (μg/m³)",
                value=10.0,
                min_value=0.0,
                max_value=100.0,
                step=1.0,
                key="shap_so2",
            )
            co = st.number_input(
                "CO (μg/m³)",
                value=1000.0,
                min_value=0.0,
                max_value=10000.0,
                step=10.0,
                key="shap_co",
            )

            st.markdown("**Weather Inputs**")
            temp = st.number_input(
                "Temperature (°C)",
                value=35.0,
                min_value=-10.0,
                max_value=55.0,
                step=1.0,
                key="shap_temp",
            )
            humidity = st.number_input(
                "Humidity (%)",
                value=60.0,
                min_value=0.0,
                max_value=100.0,
                step=1.0,
                key="shap_humidity",
            )
            wind = st.number_input(
                "Wind Speed (km/h)",
                value=10.0,
                min_value=0.0,
                max_value=100.0,
                step=1.0,
                key="shap_wind",
            )
            pressure = st.number_input(
                "Pressure (hPa)",
                value=1010.0,
                min_value=900.0,
                max_value=1100.0,
                step=1.0,
                key="shap_pressure",
            )

            explain_btn = st.button("🧠 Explain Prediction", type="primary", key="shap_btn")

        with col2:
            if explain_btn:
                features = {
                    "pm25": pm25,
                    "pm10": pm10,
                    "o3": o3,
                    "no2": no2,
                    "so2": so2,
                    "co": co,
                    "temperature": temp,
                    "humidity": humidity,
                    "wind_speed": wind,
                    "pressure": pressure,
                    "pm25_aqi": pm25,
                    "pm10_aqi": pm10,
                    "aqi": pm25,
                    "aqi_derived": pm25,
                }

                try:
                    with st.spinner("Computing SHAP values..."):
                        explanation = api_client.get_shap_explanation(features, target)

                    # Show prediction
                    pred = explanation.get("prediction", 0)
                    base = explanation.get("base_value", 0)

                    st.metric(
                        "SHAP Prediction",
                        f"AQI {pred:.1f}",
                        delta=f"{pred - base:+.1f} from average ({base:.1f})",
                    )

                    # Waterfall chart
                    shap_vals = explanation.get("shap_values", [])
                    if shap_vals:
                        top_n = min(15, len(shap_vals))
                        top_shap = shap_vals[:top_n]

                        colors = [
                            "#FF6B6B" if s["shap_value"] > 0 else "#4ECDC4"
                            for s in reversed(top_shap)
                        ]

                        fig = go.Figure(
                            go.Bar(
                                y=[s["feature"] for s in reversed(top_shap)],
                                x=[s["shap_value"] for s in reversed(top_shap)],
                                orientation="h",
                                marker_color=colors,
                                text=[f'{s["shap_value"]:+.2f}' for s in reversed(top_shap)],
                                textposition="auto",
                            )
                        )
                        fig.update_layout(
                            title=f"SHAP Contributions to {target}",
                            xaxis_title="SHAP Value (pushes prediction ↑ or ↓)",
                            height=max(350, top_n * 28),
                        )
                        fig.add_vline(x=0, line_dash="dash", line_color="gray")
                        st.plotly_chart(fig, use_container_width=True)

                        # Positive / Negative split
                        pos = explanation.get("top_positive", [])
                        neg = explanation.get("top_negative", [])

                        col_pos, col_neg = st.columns(2)
                        with col_pos:
                            st.markdown("**🔴 Increases AQI (worse air)**")
                            for p in pos[:5]:
                                st.markdown(
                                    f"- **{p['feature']}**: +{p['shap_value']:.2f} (value={p['feature_value']:.1f})"
                                )
                        with col_neg:
                            st.markdown("**🟢 Decreases AQI (better air)**")
                            for n in neg[:5]:
                                st.markdown(
                                    f"- **{n['feature']}**: {n['shap_value']:.2f} (value={n['feature_value']:.1f})"
                                )
                    else:
                        st.warning("No SHAP values returned.")

                except APIClientError as e:
                    render_error_state("SHAP explanation failed", str(e))
            else:
                st.info(
                    "👆 Enter feature values and click **Explain Prediction** to see SHAP analysis."
                )
                st.markdown(
                    """
                **What this does:**
                - Takes the feature values you enter
                - Runs them through the XGBoost model
                - Uses SHAP TreeExplainer to decompose the prediction
                - Shows which features push AQI higher or lower

                This gives you a **per-prediction** explanation, unlike the global
                importance shown in the other tabs.
                """
                )

    # ── Tab 4: Model Comparison ──
    with tab4:
        try:
            st.subheader("📈 Model Comparison")

            import json as _json
            import pandas as _pd

            meta_path = os.path.join("models", "production", "model_metadata.json")

            models_data = {}
            best_model = "XGBoost"
            train_rows = 0
            val_rows = 0
            test_rows = 0
            n_features = 0

            # ALWAYS use model_metadata.json (authoritative from Hopsworks training)
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = _json.load(f)
                best_model = meta.get("model_name", "XGBoost")
                train_rows = meta.get("train_rows", 0)
                val_rows = meta.get("val_rows", 0)
                test_rows = meta.get("test_rows", 0)
                n_features = meta.get("n_features", 0)
                total = train_rows + val_rows + test_rows

                mc = meta.get("model_comparison", {})
                for key, data in mc.items():
                    name = data.get("name", key)
                    models_data[name] = {
                        "test_metrics": {
                            "mae": data.get("test_mae", 0),
                            "rmse": data.get("test_rmse", 0),
                            "r2": data.get("test_r2", 0),
                        },
                        "val_metrics": {
                            "mae": data.get("val_mae", 0),
                            "rmse": data.get("val_rmse", 0),
                            "r2": data.get("val_r2", 0),
                        },
                        "train_time": data.get("train_time", 0),
                    }

            if models_data:
                st.caption(
                    f"Source: Hopsworks Feature Store | {total:,} rows ({train_rows:,} train + {val_rows:,} val + {test_rows:,} test) | {n_features} features | {len(models_data)} models"
                )

                # Overall comparison table
                st.subheader("Overall Performance (Test Set)")
                table_data = []
                for name, data in models_data.items():
                    test = data.get("test_metrics", {})
                    table_data.append(
                        {
                            "Model": name,
                            "MAE": f"{test.get('mae', 0):.2f}",
                            "RMSE": f"{test.get('rmse', 0):.2f}",
                            "R²": f"{test.get('r2', 0):.4f}",
                            "Train Time": f"{data.get('train_time', 0):.1f}s",
                        }
                    )
                st.dataframe(table_data, hide_index=True, use_container_width=True)

                # Bar charts
                st.subheader("Test Set Comparison")
                model_names = list(models_data.keys())
                mae_values = [models_data[m]["test_metrics"]["mae"] for m in model_names]
                r2_values = [models_data[m]["test_metrics"]["r2"] for m in model_names]

                col1, col2 = st.columns(2)
                with col1:
                    fig = go.Figure(
                        data=[
                            go.Bar(
                                x=model_names,
                                y=mae_values,
                                marker_color=[
                                    "#2ecc71" if m == best_model else "#3498db" for m in model_names
                                ],
                            )
                        ]
                    )
                    fig.update_layout(title="MAE (Lower = Better)", yaxis_title="MAE", height=350)
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    fig = go.Figure(
                        data=[
                            go.Bar(
                                x=model_names,
                                y=r2_values,
                                marker_color=[
                                    "#2ecc71" if m == best_model else "#3498db" for m in model_names
                                ],
                            )
                        ]
                    )
                    fig.update_layout(title="R² (Higher = Better)", yaxis_title="R²", height=350)
                    st.plotly_chart(fig, use_container_width=True)

                # Per-horizon comparison
                horizons = ["24h", "48h", "72h"]
                has_per_horizon = any(models_data[m].get("per_horizon") for m in model_names)

                if has_per_horizon:
                    st.subheader("Per-Horizon Performance (Test Set)")
                    horizon_data = []
                    for name in model_names:
                        ph = models_data[name].get("per_horizon", {})
                        for h in horizons:
                            h_data = ph.get(h, {}).get("test", {})
                            if h_data:
                                horizon_data.append(
                                    {
                                        "Horizon": h,
                                        "Model": name,
                                        "MAE": h_data.get("mae", 0),
                                        "RMSE": h_data.get("rmse", 0),
                                        "R²": h_data.get("r2", 0),
                                    }
                                )

                    if horizon_data:
                        horizon_df = _pd.DataFrame(horizon_data)

                        fig = go.Figure()
                        for name in model_names:
                            df_m = horizon_df[horizon_df["Model"] == name]
                            if not df_m.empty:
                                fig.add_trace(
                                    go.Bar(
                                        name=name,
                                        x=df_m["Horizon"].tolist(),
                                        y=df_m["MAE"].tolist(),
                                    )
                                )
                        fig.update_layout(
                            barmode="group", title="MAE by Horizon", yaxis_title="MAE", height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        fig2 = go.Figure()
                        for name in model_names:
                            df_m = horizon_df[horizon_df["Model"] == name]
                            if not df_m.empty:
                                fig2.add_trace(
                                    go.Bar(
                                        name=name,
                                        x=df_m["Horizon"].tolist(),
                                        y=df_m["R²"].tolist(),
                                    )
                                )
                        fig2.update_layout(
                            barmode="group", title="R² by Horizon", yaxis_title="R²", height=400
                        )
                        st.plotly_chart(fig2, use_container_width=True)

                        st.dataframe(horizon_df, hide_index=True, use_container_width=True)

                # Winner annotation
                best_mae = models_data.get(best_model, {}).get("test_metrics", {}).get("mae", 0)
                st.success(f"🏆 **Production Model: {best_model}** — Test MAE: {best_mae:.2f}")
            else:
                st.info("Model comparison data not found. Run training first.")

        except Exception as e:
            st.error(f"Error loading comparison: {e}")
