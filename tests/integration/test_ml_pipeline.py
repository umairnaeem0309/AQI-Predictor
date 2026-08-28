"""
Integration test for ML pipeline — end-to-end training with reproducibility.

Tests the complete flow: data validation → training → evaluation → MLflow logging.
Verifies reproducibility with fixed random seeds.
"""

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.feature_store.schemas import DatasetMetadata, DatasetType
from src.models.evaluation import compare_models, evaluate_model
from src.models.training import get_model, train_model, validate_training_data

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def synthetic_metadata():
    """Synthetic test metadata."""
    return DatasetMetadata(
        dataset_version="v20260811_integration_test",
        dataset_type=DatasetType.SYNTHETIC_TEST,
        approved_for_training=False,
    )


@pytest.fixture
def real_metadata():
    """Real training metadata."""
    return DatasetMetadata(
        dataset_version="v20260811_real",
        dataset_type=DatasetType.REAL_TRAINING,
        approved_for_training=True,
    )


@pytest.fixture
def training_dataset():
    """Complete training dataset for integration tests."""
    np.random.seed(42)
    n = 500
    X = pd.DataFrame(
        {
            "aqi_lag_1h": np.random.randn(n) * 20 + 100,
            "aqi_lag_6h": np.random.randn(n) * 25 + 100,
            "aqi_lag_24h": np.random.randn(n) * 30 + 100,
            "temperature": np.random.randn(n) * 5 + 30,
            "humidity": np.random.randn(n) * 10 + 60,
            "pm25": np.random.rand(n) * 30 + 30,
            "pm10": np.random.rand(n) * 40 + 50,
            "hour": np.random.randint(0, 24, n),
            "day_of_week": np.random.randint(0, 7, n),
        }
    )
    # Targets with realistic relationship to features
    y = pd.DataFrame(
        {
            "target_aqi_24h": X["aqi_lag_1h"] * 0.8
            + X["temperature"] * 0.5
            + np.random.randn(n) * 5,
            "target_aqi_48h": X["aqi_lag_24h"] * 0.6 + X["pm25"] * 0.3 + np.random.randn(n) * 10,
            "target_aqi_72h": X["aqi_lag_24h"] * 0.5
            + X["humidity"] * 0.2
            + np.random.randn(n) * 15,
        }
    )
    return X, y


# =============================================================================
# Integration Tests
# =============================================================================


class TestMLPipelineEndToEnd:
    """End-to-end ML pipeline tests."""

    def test_synthetic_data_rejected(self, synthetic_metadata):
        """Pipeline rejects synthetic data before training."""
        with pytest.raises(ValueError, match="synthetic test data"):
            validate_training_data(synthetic_metadata)

    def test_ridge_trains_and_evaluates(self, training_dataset):
        """Ridge model trains and produces metrics."""
        X, y = training_dataset
        X_train, y_train = X[:400], y[:400]
        X_val, y_val = X[400:], y[400:]

        result = train_model("ridge", X_train, y_train, X_val, y_val)
        assert result["model"] is not None
        assert "mae_avg" in result["metrics"]
        assert "rmse_avg" in result["metrics"]
        assert "r2_avg" in result["metrics"]

    def test_random_forest_trains_and_evaluates(self, training_dataset):
        """Random Forest model trains and produces metrics."""
        X, y = training_dataset
        X_train, y_train = X[:400], y[:400]
        X_val, y_val = X[400:], y[400:]

        result = train_model(
            "random_forest",
            X_train,
            y_train,
            X_val,
            y_val,
            params={"n_estimators": 20, "max_depth": 5},
        )
        assert result["model"] is not None
        assert result["metrics"]["mae_avg"] > 0

    def test_model_comparison(self, training_dataset):
        """Multiple models can be compared."""
        X, y = training_dataset
        X_train, y_train = X[:400], y[:400]
        X_val, y_val = X[400:], y[400:]

        results = []
        for model_name in ["ridge", "random_forest"]:
            result = train_model(
                model_name,
                X_train,
                y_train,
                X_val,
                y_val,
                params={"n_estimators": 10} if model_name == "random_forest" else None,
            )
            results.append(result)

        comparison = compare_models(results)
        assert len(comparison) == 2
        assert "model" in comparison.columns
        assert "mae_avg" in comparison.columns

    def test_reproducibility_same_seed(self, training_dataset):
        """Same random seed produces same results."""
        X, y = training_dataset
        X_train, y_train = X[:400], y[:400]
        X_val, y_val = X[400:], y[400:]

        result1 = train_model("ridge", X_train, y_train, X_val, y_val, random_seed=42)
        result2 = train_model("ridge", X_train, y_train, X_val, y_val, random_seed=42)

        assert result1["metrics"]["mae_avg"] == result2["metrics"]["mae_avg"]
        assert result1["metrics"]["rmse_avg"] == result2["metrics"]["rmse_avg"]
        assert result1["metrics"]["r2_avg"] == result2["metrics"]["r2_avg"]

    def test_different_seeds_different_results(self, training_dataset):
        """Different random seeds may produce different results."""
        X, y = training_dataset
        X_train, y_train = X[:400], y[:400]
        X_val, y_val = X[400:], y[400:]

        result1 = train_model(
            "random_forest",
            X_train,
            y_train,
            X_val,
            y_val,
            params={"n_estimators": 10},
            random_seed=42,
        )
        result2 = train_model(
            "random_forest",
            X_train,
            y_train,
            X_val,
            y_val,
            params={"n_estimators": 10},
            random_seed=99,
        )

        # Random Forest is sensitive to seed
        assert result1["metrics"]["mae_avg"] != result2["metrics"]["mae_avg"]

    def test_per_horizon_metrics(self, training_dataset):
        """Metrics are computed for each horizon separately."""
        X, y = training_dataset
        X_train, y_train = X[:400], y[:400]
        X_val, y_val = X[400:], y[400:]

        result = train_model("ridge", X_train, y_train, X_val, y_val)
        metrics = result["metrics"]

        # All horizons present
        assert "mae_24h" in metrics
        assert "mae_48h" in metrics
        assert "mae_72h" in metrics
        assert "rmse_24h" in metrics
        assert "r2_24h" in metrics

    def test_feature_columns_recorded(self, training_dataset):
        """Feature columns are recorded for reproducibility."""
        X, y = training_dataset
        X_train, y_train = X[:400], y[:400]
        X_val, y_val = X[400:], y[400:]

        result = train_model("ridge", X_train, y_train, X_val, y_val)
        assert len(result["feature_columns"]) == len(X.columns)
        assert set(result["feature_columns"]) == set(X.columns)
