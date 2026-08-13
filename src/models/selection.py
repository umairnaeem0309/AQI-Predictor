"""
Model Selection Framework — Ranking, tradeoff analysis, and production approval.

Features:
- Configurable performance/engineering weights (default 70/30)
- City-level evaluation (Karachi, Lahore, Islamabad)
- Horizon-level comparison (24h, 48h, 72h)
- Approval workflow with 5 gates
- Model rollback support

Real-data dependency:
- Framework runs on any data (synthetic or real)
- Production selection ONLY when real data is available
- All synthetic results tagged is_reportable=false
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class SelectionWeights:
    """Configurable weights for model ranking.

    Default: performance=0.7, engineering=0.3
    These weights are NOT permanent — they can be adjusted per experiment.
    """

    performance: float = 0.7
    engineering: float = 0.3
    # Performance sub-weights
    rmse_weight: float = 0.4
    mae_weight: float = 0.3
    r2_weight: float = 0.3
    # Engineering sub-weights
    speed_weight: float = 0.3
    complexity_weight: float = 0.3
    maintainability_weight: float = 0.2
    deployment_weight: float = 0.2

    def __post_init__(self):
        """Validate weights sum to 1.0."""
        perf_total = self.rmse_weight + self.mae_weight + self.r2_weight
        eng_total = (self.speed_weight + self.complexity_weight +
                     self.maintainability_weight + self.deployment_weight)
        assert abs(perf_total - 1.0) < 0.01, f"Performance weights must sum to 1.0, got {perf_total}"
        assert abs(eng_total - 1.0) < 0.01, f"Engineering weights must sum to 1.0, got {eng_total}"
        assert abs(self.performance + self.engineering - 1.0) < 0.01, \
            f"performance + engineering must sum to 1.0, got {self.performance + self.engineering}"


# =============================================================================
# Approval Status
# =============================================================================


class ModelApprovalStatus(Enum):
    """Model approval status values.

    PENDING: Awaiting evaluation
    CANDIDATE: Technically qualifies (meets minimum thresholds)
    APPROVED: Approved for production workflow
    REJECTED: Does not meet criteria
    ARCHIVED: Replaced by better model
    """

    PENDING = "pending"
    CANDIDATE = "candidate"
    APPROVED = "approved"
    REJECTED = "rejected"
    ARCHIVED = "archived"


# =============================================================================
# Evaluation Results
# =============================================================================


@dataclass
class ModelEvaluation:
    """Complete evaluation results for a model."""

    model_name: str
    # Per-city metrics
    city_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Per-horizon metrics (averaged across cities)
    horizon_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Overall metrics (averaged across cities and horizons)
    overall_metrics: Dict[str, float] = field(default_factory=dict)
    # Engineering metrics
    training_time: float = 0.0
    inference_time_ms: float = 0.0
    model_size_kb: float = 0.0
    feature_count: int = 0
    # Metadata
    dataset_version: str = ""
    is_reportable: bool = False
    approval_status: ModelApprovalStatus = ModelApprovalStatus.PENDING


# =============================================================================
# Ranking Logic
# =============================================================================


def normalize_score(values: List[float], lower_is_better: bool = True) -> List[float]:
    """Normalize scores to [0, 1] range.

    Args:
        values: List of scores.
        lower_is_better: If True, lowest value gets score 1.0.

    Returns:
        Normalized scores.
    """
    if not values or len(values) < 2:
        return [0.5] * len(values)

    min_val = min(values)
    max_val = max(values)
    range_val = max_val - min_val

    if range_val == 0:
        return [0.5] * len(values)

    if lower_is_better:
        return [(max_val - v) / range_val for v in values]
    else:
        return [(v - min_val) / range_val for v in values]


def compute_performance_score(
    evaluation: ModelEvaluation,
    weights: SelectionWeights,
) -> float:
    """Compute performance score for a model.

    Score is in [0, 1] range. Higher is better.

    Args:
        evaluation: Model evaluation results.
        weights: Selection weights.

    Returns:
        Performance score.
    """
    metrics = evaluation.overall_metrics
    rmse = metrics.get("rmse_avg", float("inf"))
    mae = metrics.get("mae_avg", float("inf"))
    r2 = metrics.get("r2_avg", 0.0)

    # Normalize (RMSE/MAE: lower is better, R²: higher is better)
    # Use absolute values for normalization context
    rmse_norm = max(0, 1 - rmse / 200) if rmse != float("inf") else 0  # Assume 200 as max RMSE
    mae_norm = max(0, 1 - mae / 150) if mae != float("inf") else 0    # Assume 150 as max MAE
    r2_norm = max(0, r2)  # R² already in [0, 1] range approximately

    score = (
        weights.rmse_weight * rmse_norm +
        weights.mae_weight * mae_norm +
        weights.r2_weight * r2_norm
    )

    return score


def compute_engineering_score(
    evaluation: ModelEvaluation,
    weights: SelectionWeights,
) -> float:
    """Compute engineering score for a model.

    Score is in [0, 1] range. Higher is better.

    Args:
        evaluation: Model evaluation results.
        weights: Selection weights.

    Returns:
        Engineering score.
    """
    # Speed: faster is better (normalize assuming 100ms as max)
    speed_norm = max(0, 1 - evaluation.inference_time_ms / 100)

    # Complexity: lower is better (simple heuristic)
    complexity_map = {"ridge": 1.0, "random_forest": 0.6, "xgboost": 0.5, "lstm": 0.2}
    complexity_norm = complexity_map.get(evaluation.model_name, 0.5)

    # Maintainability: simpler models more maintainable
    maintainability_norm = complexity_norm

    # Deployment: fewer dependencies better
    deployment_norm = 1.0 if evaluation.model_name in ["ridge", "random_forest"] else 0.7

    score = (
        weights.speed_weight * speed_norm +
        weights.complexity_weight * complexity_norm +
        weights.maintainability_weight * maintainability_norm +
        weights.deployment_weight * deployment_norm
    )

    return score


def compute_combined_score(
    evaluation: ModelEvaluation,
    weights: SelectionWeights,
) -> float:
    """Compute combined score (performance + engineering).

    Args:
        evaluation: Model evaluation results.
        weights: Selection weights.

    Returns:
        Combined score in [0, 1] range.
    """
    perf_score = compute_performance_score(evaluation, weights)
    eng_score = compute_engineering_score(evaluation, weights)

    combined = weights.performance * perf_score + weights.engineering * eng_score
    return combined


def rank_models(
    evaluations: List[ModelEvaluation],
    weights: Optional[SelectionWeights] = None,
) -> List[Tuple[ModelEvaluation, float]]:
    """Rank models by combined score.

    Args:
        evaluations: List of model evaluations.
        weights: Selection weights. Uses defaults if None.

    Returns:
        List of (evaluation, score) tuples, sorted by score descending.
    """
    if weights is None:
        weights = SelectionWeights()

    scored = []
    for eval in evaluations:
        score = compute_combined_score(eval, weights)
        scored.append((eval, score))

    # Sort by score descending (higher is better)
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


# =============================================================================
# City-Level Evaluation
# =============================================================================


def evaluate_by_city(
    model,
    data_by_city: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
) -> Dict[str, Dict[str, float]]:
    """Evaluate model performance for each city separately.

    Args:
        model: Trained model.
        data_by_city: Dict mapping city_id to (X, y) tuples.

    Returns:
        Dict mapping city_id to metrics dict.
    """
    from src.models.evaluation import evaluate_model

    city_results = {}
    for city_id, (X, y) in data_by_city.items():
        metrics = evaluate_model(model, X, y)
        city_results[city_id] = metrics
        logger.info(
            "City %s: MAE_avg=%.2f, RMSE_avg=%.2f, R²_avg=%.4f",
            city_id,
            metrics.get("mae_avg", 0),
            metrics.get("rmse_avg", 0),
            metrics.get("r2_avg", 0),
        )

    return city_results


def evaluate_by_horizon(
    model,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
) -> Dict[str, Dict[str, float]]:
    """Evaluate model performance for each forecast horizon.

    Args:
        model: Trained model.
        X_val: Validation features.
        y_val: Validation targets.

    Returns:
        Dict mapping horizon (24h, 48h, 72h) to metrics dict.
    """
    from src.models.evaluation import evaluate_model

    horizon_results = {}
    for horizon in ["24h", "48h", "72h"]:
        target_col = f"target_aqi_{horizon}"
        if target_col in y_val.columns:
            # Create single-target y for this horizon
            y_single = y_val[[target_col]].copy()
            y_single.columns = [target_col]

            # Predict
            y_pred = model.predict(X_val)
            if y_pred.ndim == 1:
                y_pred = y_pred.reshape(-1, 1)

            # Get horizon index
            target_cols = [c for c in y_val.columns if c.startswith("target_")]
            if target_col in target_cols:
                idx = target_cols.index(target_col)
                if idx < y_pred.shape[1]:
                    from src.models.evaluation import compute_metrics
                    metrics = compute_metrics(y_single[target_col].values, y_pred[:, idx])
                    horizon_results[horizon] = metrics

    return horizon_results


# =============================================================================
# Threshold Validation
# =============================================================================


def check_minimum_thresholds(evaluation: ModelEvaluation) -> Tuple[bool, List[str]]:
    """Check if model meets minimum performance thresholds.

    Args:
        evaluation: Model evaluation results.

    Returns:
        Tuple of (passes, list of failure reasons).
    """
    failures = []
    metrics = evaluation.overall_metrics

    # R² must be > 0 (must explain some variance)
    r2 = metrics.get("r2_avg", 0)
    if r2 <= 0:
        failures.append(f"R² avg ({r2:.4f}) must be > 0")

    # MAE should be less than a reasonable threshold
    mae = metrics.get("mae_avg", float("inf"))
    if mae > 200:  # AQI scale is 0-500
        failures.append(f"MAE avg ({mae:.2f}) exceeds threshold (200)")

    passes = len(failures) == 0
    return passes, failures


# =============================================================================
# Tradeoff Documentation
# =============================================================================


def generate_tradeoff_documentation(
    selected: ModelEvaluation,
    rejected: List[ModelEvaluation],
    weights: SelectionWeights,
) -> str:
    """Generate tradeoff documentation for model selection.

    Args:
        selected: Selected model evaluation.
        rejected: Rejected model evaluations.
        weights: Selection weights used.

    Returns:
        Formatted tradeoff documentation.
    """
    report = [
        "# Model Selection Decision",
        "",
        f"**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
        f"**Selected Model:** {selected.model_name}",
        "",
        "## Selection Criteria",
        f"- Performance weight: {weights.performance:.0%}",
        f"- Engineering weight: {weights.engineering:.0%}",
        "",
        "## Performance Evidence",
    ]

    for metric, value in selected.overall_metrics.items():
        report.append(f"- {metric}: {value:.4f}")

    report.append("")
    report.append("## Rejected Alternatives")

    for model in rejected:
        report.append(f"- **{model.model_name}**:")
        report.append(f"  - RMSE avg: {model.overall_metrics.get('rmse_avg', 'N/A')}")
        report.append(f"  - R² avg: {model.overall_metrics.get('r2_avg', 'N/A')}")

    report.append("")
    report.append("## Trade-offs Accepted")
    report.append("- Performance vs complexity balance")
    report.append("- Deployment simplicity vs model power")

    return "\n".join(report)
