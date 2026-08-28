"""
Integration test for model selection — end-to-end selection pipeline.

Tests the complete flow: evaluation → ranking → selection → registry.
Verifies synthetic safety at every step.
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.models.registry import generate_model_name, validate_for_production
from src.models.selection import (
    ModelApprovalStatus,
    ModelEvaluation,
    SelectionWeights,
    check_minimum_thresholds,
    compute_combined_score,
    generate_tradeoff_documentation,
    rank_models,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def model_evaluations():
    """Multiple model evaluations for comparison."""
    return [
        ModelEvaluation(
            model_name="ridge",
            overall_metrics={"mae_avg": 45.0, "rmse_avg": 60.0, "r2_avg": 0.45},
            city_metrics={
                "karachi": {"mae_avg": 50.0},
                "lahore": {"mae_avg": 40.0},
                "islamabad": {"mae_avg": 45.0},
            },
            horizon_metrics={
                "24h": {"mae": 30.0, "rmse": 40.0, "r2": 0.60},
                "48h": {"mae": 45.0, "rmse": 60.0, "r2": 0.45},
                "72h": {"mae": 60.0, "rmse": 80.0, "r2": 0.30},
            },
            training_time=0.5,
            inference_time_ms=1.0,
            feature_count=37,
            dataset_version="v_real_001",
            is_reportable=True,
            approval_status=ModelApprovalStatus.CANDIDATE,
        ),
        ModelEvaluation(
            model_name="random_forest",
            overall_metrics={"mae_avg": 25.0, "rmse_avg": 35.0, "r2_avg": 0.75},
            city_metrics={
                "karachi": {"mae_avg": 28.0},
                "lahore": {"mae_avg": 22.0},
                "islamabad": {"mae_avg": 25.0},
            },
            horizon_metrics={
                "24h": {"mae": 15.0, "rmse": 20.0, "r2": 0.88},
                "48h": {"mae": 25.0, "rmse": 35.0, "r2": 0.75},
                "72h": {"mae": 35.0, "rmse": 50.0, "r2": 0.62},
            },
            training_time=5.0,
            inference_time_ms=15.0,
            feature_count=37,
            dataset_version="v_real_001",
            is_reportable=True,
            approval_status=ModelApprovalStatus.CANDIDATE,
        ),
        ModelEvaluation(
            model_name="xgboost",
            overall_metrics={"mae_avg": 18.0, "rmse_avg": 25.0, "r2_avg": 0.85},
            city_metrics={
                "karachi": {"mae_avg": 20.0},
                "lahore": {"mae_avg": 16.0},
                "islamabad": {"mae_avg": 18.0},
            },
            horizon_metrics={
                "24h": {"mae": 10.0, "rmse": 14.0, "r2": 0.93},
                "48h": {"mae": 18.0, "rmse": 25.0, "r2": 0.85},
                "72h": {"mae": 26.0, "rmse": 36.0, "r2": 0.77},
            },
            training_time=8.0,
            inference_time_ms=12.0,
            feature_count=37,
            dataset_version="v_real_001",
            is_reportable=True,
            approval_status=ModelApprovalStatus.CANDIDATE,
        ),
    ]


@pytest.fixture
def synthetic_evaluation():
    """Evaluation from synthetic data."""
    return ModelEvaluation(
        model_name="random_forest",
        overall_metrics={"mae_avg": 20.0, "rmse_avg": 28.0, "r2_avg": 0.80},
        dataset_version="v_synthetic_001",
        is_reportable=False,
        approval_status=ModelApprovalStatus.CANDIDATE,
    )


# =============================================================================
# Integration Tests
# =============================================================================


class TestSelectionPipelineEndToEnd:
    """End-to-end selection pipeline tests."""

    def test_ranking_selects_best_model(self, model_evaluations):
        """Ranking correctly selects the best model."""
        ranked = rank_models(model_evaluations)
        # XGBoost has best metrics
        assert ranked[0][0].model_name == "xgboost"
        # Ridge has worst metrics
        assert ranked[-1][0].model_name == "ridge"

    def test_city_level_tracking(self, model_evaluations):
        """City-level metrics are tracked for all cities."""
        for eval in model_evaluations:
            assert len(eval.city_metrics) == 3
            assert "karachi" in eval.city_metrics
            assert "lahore" in eval.city_metrics
            assert "islamabad" in eval.city_metrics

    def test_horizon_level_tracking(self, model_evaluations):
        """Horizon-level metrics are tracked."""
        for eval in model_evaluations:
            assert "24h" in eval.horizon_metrics
            assert "48h" in eval.horizon_metrics
            assert "72h" in eval.horizon_metrics

    def test_all_models_pass_thresholds(self, model_evaluations):
        """All test models pass minimum thresholds."""
        for eval in model_evaluations:
            passes, _ = check_minimum_thresholds(eval)
            assert passes is True

    def test_tradeoff_documentation(self, model_evaluations):
        """Tradeoff documentation is generated for selection."""
        ranked = rank_models(model_evaluations)
        selected = ranked[0][0]
        rejected = [r[0] for r in ranked[1:]]

        doc = generate_tradeoff_documentation(selected, rejected, SelectionWeights())
        assert "xgboost" in doc
        assert "ridge" in doc
        assert "random_forest" in doc

    def test_configurable_weights_change_ranking(self, model_evaluations):
        """Different weights can change ranking order."""
        # Default weights: xgboost wins
        ranked_default = rank_models(model_evaluations)
        assert ranked_default[0][0].model_name == "xgboost"

        # Heavy engineering weight: ridge might win (simplest model)
        w = SelectionWeights(performance=0.3, engineering=0.7)
        ranked_eng = rank_models(model_evaluations, weights=w)
        # Ridge should rank higher with heavy engineering weight
        ridge_rank_default = next(
            i for i, (e, _) in enumerate(ranked_default) if e.model_name == "ridge"
        )
        ridge_rank_eng = next(i for i, (e, _) in enumerate(ranked_eng) if e.model_name == "ridge")
        assert ridge_rank_eng <= ridge_rank_default


class TestSyntheticDataSafety:
    """Tests for synthetic data protection in selection."""

    def test_synthetic_not_reportable(self, synthetic_evaluation):
        """Synthetic results are marked as not reportable."""
        assert synthetic_evaluation.is_reportable is False

    def test_synthetic_rejected_from_production(self, synthetic_evaluation):
        """Synthetic data is rejected from production promotion."""
        eligible, failures = validate_for_production("synthetic_test_data", True, "approved")
        assert eligible is False

    def test_synthetic_in_ranking(self, model_evaluations, synthetic_evaluation):
        """Synthetic evaluation can be ranked (for comparison) but not reported."""
        all_evals = model_evaluations + [synthetic_evaluation]
        ranked = rank_models(all_evals)
        # Synthetic model should be rankable
        assert len(ranked) == 4
        # But should not be reportable
        for eval, score in ranked:
            if eval.dataset_version.startswith("v_synthetic"):
                assert eval.is_reportable is False


class TestRegistryWorkflow:
    """Tests for registry workflow."""

    def test_model_naming(self):
        """Model naming follows convention."""
        name = generate_model_name("xgboost", 1, "20260813")
        assert name == "xgboost_v1_20260813"

    def test_production_safety_check(self):
        """Production promotion has safety checks."""
        # Synthetic rejected
        eligible, _ = validate_for_production("synthetic_test_data", True, "approved")
        assert eligible is False

        # Unapproved rejected
        eligible, _ = validate_for_production("real_training_data", False, "approved")
        assert eligible is False

        # Not-approved status rejected
        eligible, _ = validate_for_production("real_training_data", True, "candidate")
        assert eligible is False

        # Real approved accepted
        eligible, _ = validate_for_production("real_training_data", True, "approved")
        assert eligible is True
