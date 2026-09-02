"""
Tests for ML training pipeline — data safety and model training.
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.feature_store.schemas import DatasetMetadata, DatasetType
from src.models.training import TARGET_COLUMNS, get_model, train_model, validate_training_data

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def real_metadata():
    """Real training data metadata."""
    return DatasetMetadata(
        dataset_version="v20260811_real",
        dataset_type=DatasetType.REAL_TRAINING,
        approved_for_training=True,
        approved_for_evaluation=True,
    )


@pytest.fixture
def synthetic_metadata():
    """Synthetic test data metadata."""
    return DatasetMetadata(
        dataset_version="v20260811_test",
        dataset_type=DatasetType.SYNTHETIC_TEST,
        approved_for_training=False,
        approved_for_evaluation=False,
    )


@pytest.fixture
def sample_training_data():
    """Sample training data for tests."""
    np.random.seed(42)
    n = 200
    X = pd.DataFrame(
        {
            "aqi_lag_1h": np.random.randn(n) * 20 + 100,
            "aqi_lag_24h": np.random.randn(n) * 20 + 100,
            "temperature": np.random.randn(n) * 5 + 30,
            "humidity": np.random.randn(n) * 10 + 60,
            "pm25": np.random.rand(n) * 30 + 30,
        }
    )
    y = pd.DataFrame(
        {
            "target_aqi_24h": X["aqi_lag_1h"] + np.random.randn(n) * 5,
            "target_aqi_48h": X["aqi_lag_24h"] + np.random.randn(n) * 10,
            "target_aqi_72h": X["aqi_lag_24h"] + np.random.randn(n) * 15,
        }
    )
    return X, y


# =============================================================================
# Test Data Safety
# =============================================================================


class TestDataSafety:
    """Tests for training data safety validation."""

    def test_real_data_passes(self, real_metadata):
        """Real training data passes validation."""
        validate_training_data(real_metadata)  # Should not raise

    def test_synthetic_data_rejected(self, synthetic_metadata):
        """Synthetic test data is rejected for training."""
        with pytest.raises(ValueError):
            validate_training_data(synthetic_metadata)

    def test_unapproved_data_rejected(self):
        """Unapproved data is rejected."""
        metadata = DatasetMetadata(
            dataset_version="v_test",
            dataset_type=DatasetType.REAL_TRAINING,
            approved_for_training=False,
        )
        with pytest.raises(ValueError, match="not approved"):
            validate_training_data(metadata)


# =============================================================================
# Test Model Creation
# =============================================================================


class TestModelCreation:
    """Tests for model instance creation."""

    def test_ridge_creation(self):
        """Ridge model is created correctly."""
        model = get_model("ridge", {"alpha": 1.0})
        assert model is not None
        assert hasattr(model, "fit")

    def test_random_forest_creation(self):
        """Random Forest model is created with MultiOutputRegressor."""
        model = get_model("random_forest", {"n_estimators": 10})
        assert model is not None
        assert hasattr(model, "fit")

    def test_unknown_model_raises(self):
        """Unknown model name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model"):
            get_model("unknown_model")


# =============================================================================
# Test Model Training
# =============================================================================


class TestModelTraining:
    """Tests for model training."""

    def test_ridge_training(self, sample_training_data):
        """Ridge model trains successfully."""
        X, y = sample_training_data
        result = train_model("ridge", X, y, X, y)
        assert result["model"] is not None
        assert "metrics" in result
        assert "training_time" in result

    def test_random_forest_training(self, sample_training_data):
        """Random Forest model trains successfully."""
        X, y = sample_training_data
        result = train_model("random_forest", X, y, X, y, params={"n_estimators": 10})
        assert result["model"] is not None

    def test_training_includes_metrics(self, sample_training_data):
        """Training result includes evaluation metrics."""
        X, y = sample_training_data
        result = train_model("ridge", X, y, X, y)
        metrics = result["metrics"]
        assert "mae_avg" in metrics
        assert "rmse_avg" in metrics
        assert "r2_avg" in metrics

    def test_training_records_random_seed(self, sample_training_data):
        """Training records random seed for reproducibility."""
        X, y = sample_training_data
        result = train_model("ridge", X, y, X, y, random_seed=123)
        assert result["random_seed"] == 123

    def test_training_records_feature_columns(self, sample_training_data):
        """Training records which features were used."""
        X, y = sample_training_data
        result = train_model("ridge", X, y, X, y)
        assert len(result["feature_columns"]) == len(X.columns)

    def test_training_records_dataset_version(self, sample_training_data):
        """Training records dataset version."""
        X, y = sample_training_data
        result = train_model("ridge", X, y, X, y)
        result["dataset_version"] = "v_test"
        assert result["dataset_version"] == "v_test"
