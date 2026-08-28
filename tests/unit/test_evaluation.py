"""
Tests for model evaluation — MAE, RMSE, R² calculations.
"""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.models.evaluation import compare_models, compute_metrics, evaluate_model

# =============================================================================
# Test Compute Metrics
# =============================================================================


class TestComputeMetrics:
    """Tests for basic metric computation."""

    def test_perfect_predictions(self):
        """Perfect predictions yield MAE=0, RMSE=0, R²=1."""
        y_true = np.array([100.0, 110.0, 120.0, 130.0])
        y_pred = np.array([100.0, 110.0, 120.0, 130.0])
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["mae"] == 0.0
        assert metrics["rmse"] == 0.0
        assert metrics["r2"] == 1.0

    def test_imperfect_predictions(self):
        """Imperfect predictions have positive error."""
        y_true = np.array([100.0, 110.0, 120.0, 130.0])
        y_pred = np.array([105.0, 115.0, 115.0, 125.0])
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["mae"] > 0
        assert metrics["rmse"] > 0
        assert metrics["r2"] < 1.0

    def test_with_nan_values(self):
        """NaN values are filtered out."""
        y_true = np.array([100.0, np.nan, 120.0, 130.0])
        y_pred = np.array([100.0, 110.0, 120.0, 130.0])
        metrics = compute_metrics(y_true, y_pred)
        assert not np.isnan(metrics["mae"])

    def test_all_nan(self):
        """All NaN values return NaN metrics."""
        y_true = np.array([np.nan, np.nan])
        y_pred = np.array([np.nan, np.nan])
        metrics = compute_metrics(y_true, y_pred)
        assert np.isnan(metrics["mae"])


# =============================================================================
# Test Evaluate Model
# =============================================================================


class TestEvaluateModel:
    """Tests for model evaluation on validation data."""

    def test_evaluate_returns_all_horizons(self):
        """Evaluation returns metrics for all horizons."""
        model = MagicMock()
        model.predict.return_value = np.array([[100, 110, 120]] * 50)

        X_val = pd.DataFrame({"feat": range(50)})
        y_val = pd.DataFrame(
            {
                "target_aqi_24h": range(100, 150),
                "target_aqi_48h": range(110, 160),
                "target_aqi_72h": range(120, 170),
            }
        )

        metrics = evaluate_model(model, X_val, y_val)
        assert "mae_24h" in metrics
        assert "mae_48h" in metrics
        assert "mae_72h" in metrics
        assert "mae_avg" in metrics
        assert "rmse_avg" in metrics
        assert "r2_avg" in metrics

    def test_evaluate_with_nan_targets(self):
        """Evaluation handles NaN targets gracefully."""
        model = MagicMock()
        model.predict.return_value = np.array([[100, 110, 120]] * 50)

        X_val = pd.DataFrame({"feat": range(50)})
        y_val = pd.DataFrame(
            {
                "target_aqi_24h": [100] * 40 + [np.nan] * 10,
                "target_aqi_48h": [110] * 40 + [np.nan] * 10,
                "target_aqi_72h": [120] * 40 + [np.nan] * 10,
            }
        )

        metrics = evaluate_model(model, X_val, y_val)
        assert not np.isnan(metrics["mae_avg"])


# =============================================================================
# Test Compare Models
# =============================================================================


class TestCompareModels:
    """Tests for model comparison table."""

    def test_comparison_table_structure(self):
        """Comparison table has correct columns."""
        results = [
            {
                "model_name": "ridge",
                "metrics": {
                    "mae_24h": 5.0,
                    "rmse_24h": 7.0,
                    "r2_24h": 0.9,
                    "mae_avg": 5.0,
                    "rmse_avg": 7.0,
                    "r2_avg": 0.9,
                },
                "training_time": 0.1,
                "feature_columns": ["f1", "f2"],
                "is_reportable": True,
            },
        ]
        df = compare_models(results)
        assert "model" in df.columns
        assert "mae_avg" in df.columns
        assert "rmse_avg" in df.columns
        assert "r2_avg" in df.columns
        assert "is_reportable" in df.columns

    def test_comparison_excludes_errors(self):
        """Comparison excludes models with errors."""
        results = [
            {
                "model_name": "ridge",
                "metrics": {"mae_avg": 5.0},
                "training_time": 0.1,
                "feature_columns": [],
                "is_reportable": True,
            },
            {"model_name": "failed_model", "error": "training failed"},
        ]
        df = compare_models(results)
        assert len(df) == 1
        assert df["model"].iloc[0] == "ridge"
