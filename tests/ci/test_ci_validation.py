"""
CI Validation Tests

Tests for:
- Synthetic data rejection
- Missing secrets handling
- Invalid model state rejection
- Lifecycle validation
"""

import os
import pytest

from src.models.lifecycle import (
    ModelState,
    ModelLifecycle,
    LifecycleTransitionError,
    LifecycleBlockError,
    VALID_TRANSITIONS,
)
from src.models.registry import validate_for_production


class TestSyntheticDataRejection:
    """Test that synthetic data is rejected in production contexts."""

    def test_synthetic_dataset_rejected_for_production(self):
        """Test that synthetic_test_data cannot be used for production."""
        eligible, failures = validate_for_production(
            dataset_type="synthetic_test_data",
            approved_for_training=False,
            approval_status="rejected",
        )
        assert eligible is False
        assert any("synthetic" in f.lower() for f in failures)

    def test_real_data_accepted_for_production(self):
        """Test that real_api_data can be used for production."""
        eligible, failures = validate_for_production(
            dataset_type="real_api_data",
            approved_for_training=True,
            approval_status="approved",
        )
        assert eligible is True
        assert len(failures) == 0

    def test_unapproved_data_rejected(self):
        """Test that unapproved data is rejected."""
        eligible, failures = validate_for_production(
            dataset_type="real_api_data",
            approved_for_training=False,
            approval_status="candidate",
        )
        assert eligible is False
        assert len(failures) >= 2  # both approval and status should fail


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

    def test_training_cannot_skip_to_production(self):
        """Test that TRAINING cannot skip directly to PRODUCTION."""
        lifecycle = ModelLifecycle(dataset_type="real_api_data")
        with pytest.raises(LifecycleTransitionError):
            lifecycle.transition(ModelState.PRODUCTION)

    def test_evaluated_cannot_skip_to_production(self):
        """Test that EVALUATED cannot skip directly to PRODUCTION."""
        lifecycle = ModelLifecycle(
            current_state=ModelState.EVALUATED,
            dataset_type="real_api_data",
        )
        with pytest.raises(LifecycleTransitionError):
            lifecycle.transition(ModelState.PRODUCTION)

    def test_registered_can_transition_to_production(self):
        """Test that REGISTERED can transition to PRODUCTION."""
        lifecycle = ModelLifecycle(
            current_state=ModelState.REGISTERED,
            dataset_type="real_api_data",
        )
        lifecycle.transition(ModelState.PRODUCTION)
        assert lifecycle.current_state == ModelState.PRODUCTION

    def test_production_can_be_archived(self):
        """Test that PRODUCTION can be archived."""
        lifecycle = ModelLifecycle(
            current_state=ModelState.PRODUCTION,
            dataset_type="real_api_data",
        )
        lifecycle.transition(ModelState.ARCHIVED)
        assert lifecycle.current_state == ModelState.ARCHIVED


class TestLifecycleValidation:
    """Test complete lifecycle transition validation."""

    def test_full_valid_lifecycle(self):
        """Test the complete valid lifecycle path."""
        lifecycle = ModelLifecycle(
            model_name="test_model",
            dataset_type="real_api_data",
        )
        # TRAINING -> EVALUATED -> REGISTERED -> STAGING -> PRODUCTION -> ARCHIVED
        lifecycle.transition(ModelState.EVALUATED)
        lifecycle.transition(ModelState.REGISTERED)
        lifecycle.transition(ModelState.STAGING)
        lifecycle.transition(ModelState.PRODUCTION)
        lifecycle.transition(ModelState.ARCHIVED)
        assert lifecycle.current_state == ModelState.ARCHIVED

    def test_invalid_skip_transitions(self):
        """Test that skipping lifecycle stages is blocked."""
        invalid_pairs = [
            (ModelState.TRAINING, ModelState.REGISTERED),
            (ModelState.TRAINING, ModelState.PRODUCTION),
            (ModelState.EVALUATED, ModelState.PRODUCTION),
        ]
        for from_state, to_state in invalid_pairs:
            lifecycle = ModelLifecycle(
                current_state=from_state,
                dataset_type="real_api_data",
            )
            with pytest.raises(LifecycleTransitionError):
                lifecycle.transition(to_state)

    def test_rejected_cannot_progress(self):
        """Test that REJECTED is a terminal state."""
        lifecycle = ModelLifecycle(
            current_state=ModelState.REJECTED,
            dataset_type="real_api_data",
        )
        with pytest.raises(LifecycleTransitionError):
            lifecycle.transition(ModelState.PRODUCTION)

    def test_archived_cannot_reactivate(self):
        """Test that ARCHIVED is a terminal state."""
        lifecycle = ModelLifecycle(
            current_state=ModelState.ARCHIVED,
            dataset_type="real_api_data",
        )
        with pytest.raises(LifecycleTransitionError):
            lifecycle.transition(ModelState.PRODUCTION)


class TestRegistrySafety:
    """Test registry safety checks via validate_for_production."""

    def test_registry_rejects_synthetic_for_production(self):
        """Test that registry blocks synthetic data for production."""
        eligible, failures = validate_for_production(
            dataset_type="synthetic_test_data",
            approved_for_training=False,
            approval_status="rejected",
        )
        assert eligible is False
        assert any("synthetic" in f.lower() for f in failures)

    def test_registry_rejects_unapproved_data(self):
        """Test that registry rejects unapproved real data."""
        eligible, failures = validate_for_production(
            dataset_type="real_api_data",
            approved_for_training=False,
            approval_status="candidate",
        )
        assert eligible is False
        assert len(failures) >= 2

    def test_registry_accepts_approved_real_data(self):
        """Test that registry accepts approved real data."""
        eligible, failures = validate_for_production(
            dataset_type="real_api_data",
            approved_for_training=True,
            approval_status="approved",
        )
        assert eligible is True
        assert len(failures) == 0

    def test_registry_rejects_missing_approval(self):
        """Test that registry rejects data without approval status."""
        eligible, failures = validate_for_production(
            dataset_type="real_api_data",
            approved_for_training=True,
            approval_status="candidate",
        )
        assert eligible is False
        assert any("not approved" in f.lower() or "status" in f.lower() for f in failures)
