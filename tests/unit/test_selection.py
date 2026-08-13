"""
Tests for model selection — ranking, approval, registry safety, and rollback.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock

from src.models.selection import (
    SelectionWeights,
    ModelApprovalStatus,
    ModelEvaluation,
    normalize_score,
    compute_performance_score,
    compute_engineering_score,
    compute_combined_score,
    rank_models,
    check_minimum_thresholds,
    generate_tradeoff_documentation,
)
from src.models.registry import (
    generate_model_name,
    validate_for_production,
    ModelRegistry,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def good_evaluation():
    """Model evaluation with good metrics."""
    return ModelEvaluation(
        model_name="xgboost",
        overall_metrics={"mae_avg": 15.0, "rmse_avg": 20.0, "r2_avg": 0.85},
        training_time=5.0,
        inference_time_ms=10.0,
        feature_count=37,
        dataset_version="v_real",
        is_reportable=True,
        approval_status=ModelApprovalStatus.CANDIDATE,
    )


@pytest.fixture
def poor_evaluation():
    """Model evaluation with poor metrics."""
    return ModelEvaluation(
        model_name="ridge",
        overall_metrics={"mae_avg": 50.0, "rmse_avg": 70.0, "r2_avg": 0.3},
        training_time=0.5,
        inference_time_ms=1.0,
        feature_count=37,
        dataset_version="v_real",
        is_reportable=True,
        approval_status=ModelApprovalStatus.CANDIDATE,
    )


@pytest.fixture
def synthetic_evaluation():
    """Model evaluation from synthetic data."""
    return ModelEvaluation(
        model_name="random_forest",
        overall_metrics={"mae_avg": 20.0, "rmse_avg": 25.0, "r2_avg": 0.7},
        dataset_version="v_synthetic",
        is_reportable=False,
        approval_status=ModelApprovalStatus.CANDIDATE,
    )


# =============================================================================
# Test Configurable Weights
# =============================================================================


class TestConfigurableWeights:
    """Tests for configurable selection weights."""

    def test_default_weights(self):
        """Default weights are 70/30."""
        w = SelectionWeights()
        assert w.performance == 0.7
        assert w.engineering == 0.3

    def test_custom_weights(self):
        """Custom weights are accepted."""
        w = SelectionWeights(performance=0.6, engineering=0.4)
        assert w.performance == 0.6
        assert w.engineering == 0.4

    def test_invalid_weights_raises(self):
        """Invalid weights that don't sum to 1.0 raise AssertionError."""
        with pytest.raises(AssertionError):
            SelectionWeights(performance=0.8, engineering=0.3)


# =============================================================================
# Test Scoring
# =============================================================================


class TestScoring:
    """Tests for scoring functions."""

    def test_performance_score_range(self, good_evaluation):
        """Performance score is in [0, 1] range."""
        score = compute_performance_score(good_evaluation, SelectionWeights())
        assert 0 <= score <= 1

    def test_engineering_score_range(self, good_evaluation):
        """Engineering score is in [0, 1] range."""
        score = compute_engineering_score(good_evaluation, SelectionWeights())
        assert 0 <= score <= 1

    def test_combined_score_range(self, good_evaluation):
        """Combined score is in [0, 1] range."""
        score = compute_combined_score(good_evaluation, SelectionWeights())
        assert 0 <= score <= 1

    def test_better_model_higher_score(self, good_evaluation, poor_evaluation):
        """Better model gets higher combined score."""
        score_good = compute_combined_score(good_evaluation, SelectionWeights())
        score_poor = compute_combined_score(poor_evaluation, SelectionWeights())
        assert score_good > score_poor


# =============================================================================
# Test Ranking
# =============================================================================


class TestRanking:
    """Tests for model ranking."""

    def test_ranking_order(self, good_evaluation, poor_evaluation):
        """Better model is ranked first."""
        ranked = rank_models([poor_evaluation, good_evaluation])
        assert ranked[0][0].model_name == "xgboost"
        assert ranked[1][0].model_name == "ridge"

    def test_ranking_with_custom_weights(self, good_evaluation, poor_evaluation):
        """Custom weights affect ranking."""
        # Heavily weight engineering (ridge is simpler)
        w = SelectionWeights(performance=0.3, engineering=0.7)
        ranked = rank_models([poor_evaluation, good_evaluation], weights=w)
        # Ridge should score higher with heavy engineering weight
        assert len(ranked) == 2


# =============================================================================
# Test City-Level Evaluation
# =============================================================================


class TestCityLevelEvaluation:
    """Tests for city-level evaluation tracking."""

    def test_city_metrics_recorded(self):
        """City-level metrics are recorded in evaluation."""
        eval = ModelEvaluation(
            model_name="xgboost",
            city_metrics={
                "karachi": {"mae_avg": 12.0, "rmse_avg": 18.0, "r2_avg": 0.88},
                "lahore": {"mae_avg": 18.0, "rmse_avg": 25.0, "r2_avg": 0.80},
                "islamabad": {"mae_avg": 10.0, "rmse_avg": 15.0, "r2_avg": 0.90},
            },
        )
        assert len(eval.city_metrics) == 3
        assert "karachi" in eval.city_metrics
        assert "lahore" in eval.city_metrics
        assert "islamabad" in eval.city_metrics


# =============================================================================
# Test Horizon-Level Comparison
# =============================================================================


class TestHorizonLevelComparison:
    """Tests for horizon-level comparison."""

    def test_horizon_metrics_recorded(self):
        """Horizon-level metrics are recorded."""
        eval = ModelEvaluation(
            model_name="xgboost",
            horizon_metrics={
                "24h": {"mae": 10.0, "rmse": 15.0, "r2": 0.90},
                "48h": {"mae": 15.0, "rmse": 22.0, "r2": 0.82},
                "72h": {"mae": 25.0, "rmse": 35.0, "r2": 0.70},
            },
        )
        assert "24h" in eval.horizon_metrics
        assert "48h" in eval.horizon_metrics
        assert "72h" in eval.horizon_metrics


# =============================================================================
# Test Threshold Validation
# =============================================================================


class TestThresholdValidation:
    """Tests for minimum threshold checks."""

    def test_good_model_passes(self, good_evaluation):
        """Good model passes thresholds."""
        passes, reasons = check_minimum_thresholds(good_evaluation)
        assert passes is True

    def test_poor_r2_fails(self):
        """Model with R² <= 0 fails."""
        eval = ModelEvaluation(
            model_name="test",
            overall_metrics={"mae_avg": 50.0, "rmse_avg": 70.0, "r2_avg": -0.1},
        )
        passes, reasons = check_minimum_thresholds(eval)
        assert passes is False
        assert any("R²" in r for r in reasons)


# =============================================================================
# Test Registry Safety
# =============================================================================


class TestRegistrySafety:
    """Tests for registry safety checks."""

    def test_synthetic_rejected(self):
        """Synthetic data rejected from production."""
        eligible, failures = validate_for_production(
            "synthetic_test_data", True, "approved"
        )
        assert eligible is False
        assert any("synthetic" in f.lower() for f in failures)

    def test_unapproved_rejected(self):
        """Unapproved data rejected from production."""
        eligible, failures = validate_for_production(
            "real_training_data", False, "approved"
        )
        assert eligible is False
        assert any("approved" in f.lower() for f in failures)

    def test_not_approved_status_rejected(self):
        """Non-approved status rejected from production."""
        eligible, failures = validate_for_production(
            "real_training_data", True, "candidate"
        )
        assert eligible is False

    def test_real_approved_accepted(self):
        """Real approved data accepted for production."""
        eligible, failures = validate_for_production(
            "real_training_data", True, "approved"
        )
        assert eligible is True
        assert len(failures) == 0


# =============================================================================
# Test Model Naming
# =============================================================================


class TestModelNaming:
    """Tests for model naming conventions."""

    def test_naming_convention(self):
        """Model name follows convention."""
        name = generate_model_name("xgboost", 1, "20260813")
        assert name == "xgboost_v1_20260813"

    def test_naming_with_default_date(self):
        """Model name uses today's date when not specified."""
        name = generate_model_name("ridge", 1)
        assert name.startswith("ridge_v1_")


# =============================================================================
# Test Tradeoff Documentation
# =============================================================================


class TestTradeoffDocumentation:
    """Tests for tradeoff documentation generation."""

    def test_documentation_generation(self, good_evaluation, poor_evaluation):
        """Tradeoff documentation is generated."""
        doc = generate_tradeoff_documentation(
            good_evaluation,
            [poor_evaluation],
            SelectionWeights(),
        )
        assert "Model Selection Decision" in doc
        assert "xgboost" in doc
        assert "ridge" in doc
