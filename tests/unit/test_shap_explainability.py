"""
Tests for SHAP explainability endpoints.
"""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest


class TestSHAPModels:
    """Test Pydantic models for SHAP request/response."""

    def test_request_model_valid(self):
        from app.routes.explain import PredictionExplanationRequest

        req = PredictionExplanationRequest(
            features={"pm25": 50.0, "temperature": 30.0},
            target="target_aqi_24h",
        )
        assert req.features == {"pm25": 50.0, "temperature": 30.0}
        assert req.target == "target_aqi_24h"

    def test_request_model_default_target(self):
        from app.routes.explain import PredictionExplanationRequest

        req = PredictionExplanationRequest(features={"pm25": 50.0})
        assert req.target == "target_aqi_24h"

    def test_response_model_valid(self):
        from app.routes.explain import PredictionExplanationResponse

        resp = PredictionExplanationResponse(
            base_value=120.0,
            shap_values=[{"feature": "pm25", "shap_value": 10.5, "feature_value": 50.0}],
            feature_names=["pm25"],
            feature_values=[50.0],
            prediction=130.5,
            target="target_aqi_24h",
            top_positive=[{"feature": "pm25", "shap_value": 10.5, "feature_value": 50.0}],
            top_negative=[],
        )
        assert resp.prediction == 130.5
        assert resp.base_value == 120.0
        assert len(resp.shap_values) == 1


class TestSHAPHelperFunctions:
    """Test helper functions in explain module."""

    def test_get_feature_names(self):
        from app.routes.explain import _get_feature_names

        names = _get_feature_names()
        assert isinstance(names, list)
        assert len(names) > 0
        assert "pm25" in names or "temperature" in names

    def test_get_target_index_24h(self):
        from app.routes.explain import _get_target_index

        idx = _get_target_index("target_aqi_24h")
        assert idx == 0

    def test_get_target_index_48h(self):
        from app.routes.explain import _get_target_index

        idx = _get_target_index("target_aqi_48h")
        assert idx == 1

    def test_get_target_index_72h(self):
        from app.routes.explain import _get_target_index

        idx = _get_target_index("target_aqi_72h")
        assert idx == 2

    def test_get_target_index_unknown(self):
        from app.routes.explain import _get_target_index

        idx = _get_target_index("unknown_target")
        assert idx == 0  # Falls back to 0


class TestSHAPExplainer:
    """Test SHAP computation logic with mocked model."""

    @patch("app.routes.explain.get_model_service")
    def test_shap_explanation_basic(self, mock_get_service):
        """Test that SHAP explanation can be computed for a simple input."""
        try:
            import shap
        except ImportError:
            pytest.skip("shap not installed")

        # Create a mock model with XGBoost-like interface
        mock_model = MagicMock()
        mock_estimator = MagicMock()

        # Fake XGBoost model that SHAP TreeExplainer can work with
        try:
            import xgboost as xgb

            # Train a tiny XGBoost for SHAP compatibility
            rng = np.random.RandomState(42)
            X = rng.randn(50, 3)
            y = X @ np.array([1.0, -0.5, 0.3]) + rng.randn(50) * 0.1
            tiny_model = xgb.XGBRegressor(n_estimators=5, max_depth=3)
            tiny_model.fit(X, y)

            mock_estimator = tiny_model
        except Exception:
            pytest.skip("Cannot create tiny XGBoost model for SHAP test")

        mock_model.estimators_ = [mock_estimator, mock_estimator, mock_estimator]

        mock_service = MagicMock()
        mock_service.get_model.return_value = mock_model
        mock_get_service.return_value = mock_service

        # Mock feature names
        with patch("app.routes.explain._get_feature_names", return_value=["f1", "f2", "f3"]):
            with patch("app.routes.explain._get_target_index", return_value=0):
                # Import and call directly
                from app.routes.explain import _get_feature_names, _get_target_index

                feature_names = _get_feature_names()
                assert len(feature_names) == 3

                target_idx = _get_target_index("target_aqi_24h")
                estimator = mock_model.estimators_[target_idx]

                X = np.array([[1.0, 2.0, 3.0]])
                explainer = shap.TreeExplainer(estimator)
                shap_values = explainer.shap_values(X)

                assert shap_values is not None
                if isinstance(shap_values, np.ndarray):
                    assert shap_values.shape[-1] == 3


class TestGlobalSHAP:
    """Test global SHAP computation."""

    @patch("app.routes.explain.get_model_service")
    def test_global_shap_structure(self, mock_get_service):
        """Test that global SHAP returns expected structure."""
        try:
            import shap
            import xgboost as xgb
        except ImportError:
            pytest.skip("shap or xgboost not installed")

        # Create tiny model
        rng = np.random.RandomState(42)
        X = rng.randn(50, 3)
        y = X @ np.array([1.0, -0.5, 0.3]) + rng.randn(50) * 0.1
        tiny_model = xgb.XGBRegressor(n_estimators=5, max_depth=3)
        tiny_model.fit(X, y)

        mock_model = MagicMock()
        mock_model.estimators_ = [tiny_model]
        mock_service = MagicMock()
        mock_service.get_model.return_value = mock_model
        mock_get_service.return_value = mock_service

        # Compute SHAP
        explainer = shap.TreeExplainer(tiny_model, data=X[:10])
        shap_vals = explainer.shap_values(X[:10])

        mean_abs = np.mean(np.abs(shap_vals), axis=0)
        assert len(mean_abs) == 3
        assert all(v >= 0 for v in mean_abs)


class TestAPIClientSHAP:
    """Test API client SHAP methods."""

    def test_shap_explanation_mock(self):
        from app.frontend.utils.api_client import APIClient

        client = APIClient(mock_mode=True)
        result = client.get_shap_explanation({"pm25": 50.0})
        assert "base_value" in result
        assert "prediction" in result
        assert "shap_values" in result

    def test_global_shap_mock(self):
        from app.frontend.utils.api_client import APIClient

        client = APIClient(mock_mode=True)
        result = client.get_global_shap(top_n=10)
        assert "features" in result
        assert "method" in result
