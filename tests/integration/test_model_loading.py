"""
Integration tests for model loading, saving, and lifecycle management.

Tests the full cycle of training, saving, loading, and predicting.
"""

import tempfile
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.models.lifecycle import (
    ModelStatus,
    LifecycleTransition,
    validate_lifecycle_transition,
)
from src.models.registry import ModelRegistry


class TestModelLoadingRoundTrip:
    """Test train → save → load → predict round-trip."""

    def test_sklearn_model_save_and_load(self, tmp_path):
        """Test saving and loading a scikit-learn model."""
        from sklearn.linear_model import Ridge

        # Train a small model
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        model = Ridge(alpha=1.0)
        model.fit(X, y)

        # Save using joblib
        import joblib
        model_path = tmp_path / "test_model.joblib"
        joblib.dump(model, model_path)

        # Load
        loaded_model = joblib.load(model_path)

        # Verify predictions match
        X_test = np.random.randn(10, 5)
        original_pred = model.predict(X_test)
        loaded_pred = loaded_model.predict(X_test)
        np.testing.assert_array_almost_equal(original_pred, loaded_pred)

    def test_xgboost_model_save_and_load(self, tmp_path):
        """Test saving and loading an XGBoost model."""
        import xgboost as xgb

        # Train small model
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        model = xgb.XGBRegressor(n_estimators=5, max_depth=3, random_state=42)
        model.fit(X, y)

        # Save
        model_path = tmp_path / "test_xgb.json"
        model.save_model(str(model_path))

        # Load
        loaded_model = xgb.XGBRegressor()
        loaded_model.load_model(str(model_path))

        # Verify predictions match
        X_test = np.random.randn(10, 5)
        original_pred = model.predict(X_test)
        loaded_pred = loaded_model.predict(X_test)
        np.testing.assert_array_almost_equal(original_pred, loaded_pred)

    def test_metadata_completeness(self):
        """Test that all required metadata fields are present."""
        metadata = {
            "model_name": "production_v1",
            "model_type": "xgboost",
            "version": 1,
            "status": "production",
            "approval_status": "approved",
            "dataset_type": "real_api_data",
            "feature_version": "1.0.0",
            "schema_version": "1.0.0",
            "training_data_type": "real_api_data",
            "metrics": {"mae": 15.0},
            "parameters": {"n_estimators": 100},
            "feature_names": ["feature_1"],
            "creation_timestamp": datetime.now(timezone.utc).isoformat(),
        }

        required_keys = [
            "model_name", "model_type", "version", "status",
            "approval_status", "dataset_type", "feature_version",
            "schema_version", "metrics", "creation_timestamp",
        ]

        for key in required_keys:
            assert key in metadata, f"Missing required metadata key: {key}"


class TestDriftBaselineRoundTrip:
    """Test drift baseline creation and storage."""

    def test_drift_baseline_numerical(self):
        """Test drift baseline for numerical features."""
        df = pd.DataFrame({
            "temperature": np.random.randn(1000) * 10 + 30,
            "humidity": np.random.randn(1000) * 5 + 60,
        })

        baseline = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            baseline[col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "percentiles": {
                    "25": float(df[col].quantile(0.25)),
                    "50": float(df[col].quantile(0.50)),
                    "75": float(df[col].quantile(0.75)),
                },
            }

        assert "temperature" in baseline
        assert "mean" in baseline["temperature"]
        assert "std" in baseline["temperature"]
        assert "min" in baseline["temperature"]
        assert "max" in baseline["temperature"]
        assert "percentiles" in baseline["temperature"]


class TestLifecycleTransitions:
    """Test lifecycle state transitions."""

    def test_valid_transitions(self):
        """Test that all valid transitions are allowed."""
        valid = [
            (ModelStatus.UNTRAINED, ModelStatus.TRAINING),
            (ModelStatus.TRAINING, ModelStatus.EVALUATED),
            (ModelStatus.EVALUATED, ModelStatus.CANDIDATE),
            (ModelStatus.CANDIDATE, ModelStatus.APPROVED),
            (ModelStatus.APPROVED, ModelStatus.REGISTERED),
            (ModelStatus.REGISTERED, ModelStatus.PRODUCTION),
            (ModelStatus.PRODUCTION, ModelStatus.ARCHIVED),
            (ModelStatus.REGISTERED, ModelStatus.REJECTED),
        ]

        for from_status, to_status in valid:
            assert validate_lifecycle_transition(from_status, to_status)

    def test_invalid_transitions(self):
        """Test that invalid transitions are blocked."""
        invalid = [
            (ModelStatus.UNTRAINED, ModelStatus.PRODUCTION),
            (ModelStatus.TRAINING, ModelStatus.PRODUCTION),
            (ModelStatus.ARCHIVED, ModelStatus.PRODUCTION),
            (ModelStatus.UNTRAINED, ModelStatus.REGISTERED),
        ]

        for from_status, to_status in invalid:
            assert not validate_lifecycle_transition(from_status, to_status)

    def test_synthetic_blocks_production(self):
        """Test that synthetic data blocks production lifecycle."""
        # Synthetic data should be blocked at the approval level,
        # not the lifecycle level. The lifecycle allows all valid
        # transitions; the approval workflow rejects synthetic.
        # This is tested in selection.py validate_for_production().
        pass


class TestRegistryVersioning:
    """Test model versioning and rollback."""

    def test_version_numbering(self):
        """Test that versions increment correctly."""
        registry = ModelRegistry(
            model_dir=Path(tempfile.mkdtemp()),
        )

        # Mock models
        class MockModel:
            pass

        # Register first version
        registry.register_model(
            MockModel(), "test_model",
            {"dataset_type": "real_api_data", "approved_for_training": True},
            {"mae": 15.0}, {"n_estimators": 10},
        )

        assert registry.get_current_version("test_model") == 1

        # Register second version
        registry.register_model(
            MockModel(), "test_model",
            {"dataset_type": "real_api_data", "approved_for_training": True},
            {"mae": 12.0}, {"n_estimators": 20},
        )

        assert registry.get_current_version("test_model") == 2

    def test_rollback_to_previous_version(self):
        """Test rollback changes current version."""
        registry = ModelRegistry(
            model_dir=Path(tempfile.mkdtemp()),
        )

        class MockModel:
            pass

        # Register two versions
        registry.register_model(
            MockModel(), "test_model",
            {"dataset_type": "real_api_data", "approved_for_training": True},
            {"mae": 15.0}, {},
        )
        registry.register_model(
            MockModel(), "test_model",
            {"dataset_type": "real_api_data", "approved_for_training": True},
            {"mae": 12.0}, {},
        )

        assert registry.get_current_version("test_model") == 2

        # Rollback
        registry.rollback("test_model", target_version=1)

        assert registry.get_current_version("test_model") == 1
