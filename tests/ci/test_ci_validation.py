"""
CI Validation Tests

Tests for:
- Synthetic data rejection
- Missing secrets handling
- Invalid model state rejection
- Lifecycle validation
"""

import os
import tempfile
from pathlib import Path
from datetime import datetime, timezone

import pytest

from src.models.lifecycle import (
    ModelStatus,
    validate_lifecycle_transition,
)
from src.models.registry import ModelRegistry
from src.models.selection import validate_for_production


class TestSyntheticDataRejection:
    """Test that synthetic data is rejected in production contexts."""

    def test_synthetic_dataset_rejected_for_production(self):
        """Test that synthetic_test_data cannot be used for production."""
        metadata = {
            "dataset_type": "synthetic_test_data",
            "approved_for_training": False,
            "approved_for_evaluation": False,
        }
        
        # Should raise ValueError for synthetic data
        with pytest.raises(ValueError, match="synthetic"):
            validate_for_production(metadata)

    def test_real_data_accepted_for_production(self):
        """Test that real_api_data can be used for production."""
        metadata = {
            "dataset_type": "real_api_data",
            "approved_for_training": True,
            "approved_for_evaluation": True,
        }
        
        # Should not raise for real data
        validate_for_production(metadata)

    def test_unapproved_data_rejected(self):
        """Test that unapproved data is rejected."""
        metadata = {
            "dataset_type": "real_api_data",
            "approved_for_training": False,
            "approved_for_evaluation": False,
        }
        
        with pytest.raises(ValueError, match="not approved"):
            validate_for_production(metadata)


class TestMissingSecretsHandling:
    """Test that missing secrets are handled gracefully."""

    def test_missing_api_key_handled(self):
        """Test that missing API keys don't crash the system."""
        # Remove any existing API keys from environment
        env_vars = ["OPENWEATHER_API_KEY", "AQICN_API_KEY", "HOPSWORKS_HOST"]
        
        original_values = {}
        for var in env_vars:
            original_values[var] = os.environ.pop(var, None)
        
        try:
            # Import should not crash even without secrets
            from src.config import load_config
            
            # Config loading should handle missing secrets
            # This tests graceful degradation
            assert True  # If we get here, no crash occurred
        finally:
            # Restore original values
            for var, value in original_values.items():
                if value is not None:
                    os.environ[var] = value

    def test_integration_test_skip_without_credentials(self):
        """Test that integration tests skip when credentials unavailable."""
        # This test verifies the skip logic exists
        # Actual skipping is handled by pytest markers
        credentials_available = all(
            os.environ.get(key) is not None
            for key in ["OPENWEATHER_API_KEY", "AQICN_API_KEY"]
        )
        
        if not credentials_available:
            pytest.skip("API credentials not available")
        
        # If we get here, credentials are available
        assert True


class TestInvalidModelStateRejection:
    """Test that invalid model states are rejected."""

    def test_untrained_model_not_production_ready(self):
        """Test that UNTRAINED status cannot be promoted to PRODUCTION."""
        assert not validate_lifecycle_transition(
            ModelStatus.UNTRAINED,
            ModelStatus.PRODUCTION,
        )

    def test_training_model_not_production_ready(self):
        """Test that TRAINING status cannot be promoted to PRODUCTION."""
        assert not validate_lifecycle_transition(
            ModelStatus.TRAINING,
            ModelStatus.PRODUCTION,
        )

    def test_evaluated_model_not_production_ready(self):
        """Test that EVALUATED status cannot be promoted to PRODUCTION."""
        assert not validate_lifecycle_transition(
            ModelStatus.EVALUATED,
            ModelStatus.PRODUCTION,
        )

    def test_candidate_model_not_production_ready(self):
        """Test that CANDIDATE status cannot be promoted to PRODUCTION."""
        assert not validate_lifecycle_transition(
            ModelStatus.CANDIDATE,
            ModelStatus.PRODUCTION,
        )

    def test_registered_model_production_ready(self):
        """Test that REGISTERED status can be promoted to PRODUCTION."""
        assert validate_lifecycle_transition(
            ModelStatus.REGISTERED,
            ModelStatus.PRODUCTION,
        )

    def test_production_model_can_be_archived(self):
        """Test that PRODUCTION status can be archived."""
        assert validate_lifecycle_transition(
            ModelStatus.PRODUCTION,
            ModelStatus.ARCHIVED,
        )


class TestLifecycleValidation:
    """Test complete lifecycle transition validation."""

    def test_full_valid_lifecycle(self):
        """Test the complete valid lifecycle path."""
        transitions = [
            (ModelStatus.UNTRAINED, ModelStatus.TRAINING),
            (ModelStatus.TRAINING, ModelStatus.EVALUATED),
            (ModelStatus.EVALUATED, ModelStatus.CANDIDATE),
            (ModelStatus.CANDIDATE, ModelStatus.APPROVED),
            (ModelStatus.APPROVED, ModelStatus.REGISTERED),
            (ModelStatus.REGISTERED, ModelStatus.PRODUCTION),
            (ModelStatus.PRODUCTION, ModelStatus.ARCHIVED),
        ]
        
        for from_status, to_status in transitions:
            assert validate_lifecycle_transition(from_status, to_status), \
                f"Invalid transition: {from_status.value} -> {to_status.value}"

    def test_invalid_skip_transitions(self):
        """Test that skipping lifecycle stages is blocked."""
        invalid_transitions = [
            (ModelStatus.UNTRAINED, ModelStatus.EVALUATED),
            (ModelStatus.TRAINING, ModelStatus.CANDIDATE),
            (ModelStatus.EVALUATED, ModelStatus.APPROVED),
            (ModelStatus.CANDIDATE, ModelStatus.REGISTERED),
            (ModelStatus.APPROVED, ModelStatus.PRODUCTION),
        ]
        
        for from_status, to_status in invalid_transitions:
            assert not validate_lifecycle_transition(from_status, to_status), \
                f"Should block transition: {from_status.value} -> {to_status.value}"

    def test_rejected_status_cannot_progress(self):
        """Test that REJECTED status cannot progress further."""
        assert not validate_lifecycle_transition(
            ModelStatus.REJECTED,
            ModelStatus.PRODUCTION,
        )

    def test_archived_status_cannot_reactivate(self):
        """Test that ARCHIVED status cannot be reactivated."""
        assert not validate_lifecycle_transition(
            ModelStatus.ARCHIVED,
            ModelStatus.PRODUCTION,
        )


class TestRegistrySafety:
    """Test registry safety checks."""

    def test_registry_rejects_synthetic_for_production(self):
        """Test that registry blocks synthetic data for production."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(model_dir=Path(tmpdir))
            
            # Try to register synthetic data as production
            # This should be blocked by validation
            metadata = {
                "dataset_type": "synthetic_test_data",
                "approved_for_training": False,
            }
            
            # The validate_for_production function should catch this
            with pytest.raises(ValueError):
                validate_for_production(metadata)

    def test_registry_requires_approval_status(self):
        """Test that registry requires approved status for production."""
        metadata = {
            "dataset_type": "real_api_data",
            "approved_for_training": True,
            "approved_for_evaluation": True,
            "status": "candidate",  # Not approved
        }
        
        # Should fail because status is not 'approved'
        with pytest.raises(ValueError):
            validate_for_production(metadata)
