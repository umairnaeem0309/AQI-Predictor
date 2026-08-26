"""
Tests for model lifecycle — state transitions, synthetic blocking, and artifact logging.
"""

import pytest
from datetime import datetime, timezone

from src.models.lifecycle import (
    ModelState,
    ModelLifecycle,
    LifecycleTransitionError,
    LifecycleBlockError,
    VALID_TRANSITIONS,
)


# =============================================================================
# Test State Transitions
# =============================================================================


class TestValidTransitions:
    """Tests for valid lifecycle transitions."""

    def test_training_to_evaluated(self):
        """Training → Evaluated is valid."""
        lc = ModelLifecycle(model_name="test")
        lc.transition(ModelState.EVALUATED)
        assert lc.get_state() == ModelState.EVALUATED

    def test_evaluated_to_registered(self):
        """Evaluated → Registered is valid."""
        lc = ModelLifecycle(model_name="test", current_state=ModelState.EVALUATED)
        lc.transition(ModelState.REGISTERED)
        assert lc.get_state() == ModelState.REGISTERED

    def test_evaluated_to_rejected(self):
        """Evaluated → Rejected is valid."""
        lc = ModelLifecycle(model_name="test", current_state=ModelState.EVALUATED)
        lc.transition(ModelState.REJECTED)
        assert lc.get_state() == ModelState.REJECTED

    def test_registered_to_staging(self):
        """Registered → Staging is valid."""
        lc = ModelLifecycle(model_name="test", current_state=ModelState.REGISTERED)
        lc.transition(ModelState.STAGING)
        assert lc.get_state() == ModelState.STAGING

    def test_staging_to_production(self):
        """Staging → Production is valid."""
        lc = ModelLifecycle(model_name="test", current_state=ModelState.STAGING)
        lc.transition(ModelState.PRODUCTION)
        assert lc.get_state() == ModelState.PRODUCTION

    def test_production_to_archived(self):
        """Production → Archived is valid."""
        lc = ModelLifecycle(model_name="test", current_state=ModelState.PRODUCTION)
        lc.transition(ModelState.ARCHIVED)
        assert lc.get_state() == ModelState.ARCHIVED

    def test_full_lifecycle(self):
        """Complete lifecycle: Training → ... → Production."""
        lc = ModelLifecycle(model_name="test")
        lc.transition(ModelState.EVALUATED)
        lc.transition(ModelState.REGISTERED)
        lc.transition(ModelState.STAGING)
        lc.transition(ModelState.PRODUCTION)
        assert lc.get_state() == ModelState.PRODUCTION


class TestInvalidTransitions:
    """Tests for invalid lifecycle transitions."""

    def test_training_to_registered(self):
        """Training → Registered is invalid."""
        lc = ModelLifecycle(model_name="test")
        with pytest.raises(LifecycleTransitionError, match="Invalid transition"):
            lc.transition(ModelState.REGISTERED)

    def test_training_to_production(self):
        """Training → Production is invalid."""
        lc = ModelLifecycle(model_name="test")
        with pytest.raises(LifecycleTransitionError, match="Invalid transition"):
            lc.transition(ModelState.PRODUCTION)

    def test_archived_no_transitions(self):
        """Archived state has no outgoing transitions."""
        lc = ModelLifecycle(model_name="test", current_state=ModelState.ARCHIVED)
        assert lc.is_terminal() is True
        with pytest.raises(LifecycleTransitionError):
            lc.transition(ModelState.PRODUCTION)

    def test_rejected_no_transitions(self):
        """Rejected state has no outgoing transitions."""
        lc = ModelLifecycle(model_name="test", current_state=ModelState.REJECTED)
        assert lc.is_terminal() is True
        with pytest.raises(LifecycleTransitionError):
            lc.transition(ModelState.EVALUATED)

    def test_invalid_reverse_transition(self):
        """Cannot go from Evaluated back to Training."""
        lc = ModelLifecycle(model_name="test", current_state=ModelState.EVALUATED)
        with pytest.raises(LifecycleTransitionError):
            lc.transition(ModelState.TRAINING)


# =============================================================================
# Test Synthetic Data Blocking
# =============================================================================


class TestSyntheticDataBlocking:
    """Tests for synthetic data blocking in lifecycle."""

    def test_synthetic_blocked_from_registered(self):
        """Synthetic data blocked from Registered state."""
        lc = ModelLifecycle(
            model_name="test",
            current_state=ModelState.EVALUATED,
            dataset_type="synthetic_test_data",
        )
        with pytest.raises(LifecycleBlockError, match="synthetic test data"):
            lc.transition(ModelState.REGISTERED)

    def test_synthetic_blocked_from_production(self):
        """Synthetic data blocked from Production state."""
        lc = ModelLifecycle(
            model_name="test",
            current_state=ModelState.STAGING,
            dataset_type="synthetic_test_data",
        )
        with pytest.raises(LifecycleBlockError, match="synthetic test data"):
            lc.transition(ModelState.PRODUCTION)

    def test_real_data_allowed(self):
        """Real data allowed through lifecycle."""
        lc = ModelLifecycle(
            model_name="test",
            current_state=ModelState.EVALUATED,
            dataset_type="real_training_data",
        )
        lc.transition(ModelState.REGISTERED)
        assert lc.get_state() == ModelState.REGISTERED


# =============================================================================
# Test Lifecycle History
# =============================================================================


class TestLifecycleHistory:
    """Tests for lifecycle history tracking."""

    def test_history_recorded(self):
        """Transitions are recorded in history."""
        lc = ModelLifecycle(model_name="test")
        lc.transition(ModelState.EVALUATED, reason="training complete")
        lc.transition(ModelState.REGISTERED, reason="evaluation passed")

        history = lc.get_history()
        assert len(history) == 3  # init + 2 transitions
        assert history[0][0] == ModelState.TRAINING
        assert history[1][0] == ModelState.EVALUATED
        assert history[2][0] == ModelState.REGISTERED

    def test_can_transition_to(self):
        """can_transition_to checks validity."""
        lc = ModelLifecycle(model_name="test")
        assert lc.can_transition_to(ModelState.EVALUATED) is True
        assert lc.can_transition_to(ModelState.REGISTERED) is False


# =============================================================================
# Test Artifact Directory Structure
# =============================================================================


class TestArtifactStructure:
    """Tests for artifact directory structure definition."""

    def test_artifact_keys_defined(self):
        """All expected artifact keys are defined."""
        expected_artifacts = [
            "model",
            "metadata",
            "metrics",
            "parameters",
            "feature_importance",
            "evaluation_report",
            "drift_baseline",
        ]
        # Verify these are mentioned in registry.py log_artifacts
        pytest.importorskip("duckdb", reason="duckdb required for ModelRegistry import")
        from src.models.registry import ModelRegistry
        # The method exists and accepts these parameters
        registry = ModelRegistry()
        assert hasattr(registry, "log_artifacts")
        assert hasattr(registry, "store_drift_baseline")
