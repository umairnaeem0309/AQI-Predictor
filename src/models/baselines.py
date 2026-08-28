"""
Model Baselines — Ridge regression baseline and comparison framework.

The Ridge baseline provides a comparison point for model complexity.
Complex models do NOT have to beat Ridge — selection considers both
performance and complexity trade-offs.

Phase 7 establishes baselines.
Phase 8 makes the production model decision.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.models.evaluation import compare_models, evaluate_model
from src.models.training import get_model, train_model

logger = logging.getLogger(__name__)


def train_baseline(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    alpha: float = 1.0,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Train Ridge regression baseline.

    Args:
        X_train: Training features.
        y_train: Training targets.
        X_val: Validation features.
        y_val: Validation targets.
        alpha: Ridge regularization parameter.
        random_seed: Random seed.

    Returns:
        Training result dictionary.
    """
    return train_model(
        "ridge",
        X_train,
        y_train,
        X_val,
        y_val,
        params={"alpha": alpha},
        random_seed=random_seed,
    )


def train_all_baselines(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    random_seed: int = 42,
) -> List[Dict[str, Any]]:
    """Train Ridge baseline with different alpha values.

    Args:
        X_train: Training features.
        y_train: Training targets.
        X_val: Validation features.
        y_val: Validation targets.
        random_seed: Random seed.

    Returns:
        List of baseline results.
    """
    results = []

    for alpha in [0.1, 1.0, 10.0, 100.0]:
        result = train_baseline(
            X_train,
            y_train,
            X_val,
            y_val,
            alpha=alpha,
            random_seed=random_seed,
        )
        result["model_name"] = f"ridge_alpha_{alpha}"
        results.append(result)

    return results


def find_best_baseline(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Find the best Ridge baseline by RMSE average.

    Args:
        results: List of baseline results.

    Returns:
        Best baseline result.
    """
    valid_results = [r for r in results if "error" not in r and "metrics" in r]
    if not valid_results:
        return {"error": "No valid baseline results"}

    best = min(valid_results, key=lambda r: r["metrics"].get("rmse_avg", float("inf")))
    return best


def create_comparison_table(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create a comparison table for all trained models.

    Args:
        results: List of training results.

    Returns:
        DataFrame with comparison metrics.
    """
    return compare_models(results)
