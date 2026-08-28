"""
Confidence Interval Estimation

Provides prediction intervals for XGBoost predictions using
training residual analysis. XGBoost doesn't natively support
prediction intervals, so we use the empirical residual distribution
from the training set to estimate uncertainty.

Method:
1. During training, compute residuals (actual - predicted) on validation set
2. Fit a quantile regression or use empirical percentiles
3. At prediction time, apply the residual distribution to point predictions
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Cache for loaded residual stats
_residual_cache: Optional[Dict[str, Any]] = None


def compute_residual_stats(
    y_actual: np.ndarray,
    y_predicted: np.ndarray,
    confidence_levels: list = None,
) -> Dict[str, Any]:
    """
    Compute residual statistics for confidence intervals.

    Args:
        y_actual: Actual target values
        y_predicted: Predicted values
        confidence_levels: List of confidence levels (e.g., [0.80, 0.95])

    Returns:
        Dictionary with residual statistics
    """
    if confidence_levels is None:
        confidence_levels = [0.80, 0.95]

    residuals = y_actual - y_predicted
    abs_residuals = np.abs(residuals)

    stats = {
        "mean_residual": float(np.mean(residuals)),
        "std_residual": float(np.std(residuals)),
        "mae": float(np.mean(abs_residuals)),
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "median_abs_error": float(np.median(abs_residuals)),
        "n_samples": len(residuals),
    }

    # Compute quantile-based intervals
    for level in confidence_levels:
        alpha = 1 - level
        lower_q = alpha / 2
        upper_q = 1 - alpha / 2

        stats[f"interval_{int(level*100)}_lower"] = float(np.percentile(residuals, lower_q * 100))
        stats[f"interval_{int(level*100)}_upper"] = float(np.percentile(residuals, upper_q * 100))
        stats[f"interval_{int(level*100)}_width"] = (
            stats[f"interval_{int(level*100)}_upper"] - stats[f"interval_{int(level*100)}_lower"]
        )

    # Per-horizon stats (if 2D)
    if y_actual.ndim > 1 or (y_actual.ndim == 1 and len(y_actual.shape) > 0):
        stats["per_horizon"] = {}
        horizons = ["24h", "48h", "72h"]
        for i, h in enumerate(horizons):
            if i < y_actual.shape[-1] if y_actual.ndim > 1 else 0:
                continue
            if y_actual.ndim > 1 and i < y_actual.shape[1]:
                h_resid = y_actual[:, i] - y_predicted[:, i]
            elif y_actual.ndim == 1 and i == 0:
                h_resid = residuals
            else:
                continue

            h_stats = {
                "std": float(np.std(h_resid)),
                "mae": float(np.mean(np.abs(h_resid))),
            }
            for level in confidence_levels:
                alpha = 1 - level
                h_stats[f"interval_{int(level*100)}_lower"] = float(
                    np.percentile(h_resid, alpha / 2 * 100)
                )
                h_stats[f"interval_{int(level*100)}_upper"] = float(
                    np.percentile(h_resid, (1 - alpha / 2) * 100)
                )

            stats["per_horizon"][h] = h_stats

    return stats


def save_residual_stats(stats: Dict[str, Any], path: str = None):
    """Save residual statistics to file."""
    if path is None:
        path = os.path.join("models", "production", "residual_stats.json")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    logger.info("Saved residual stats to %s", path)


def load_residual_stats(path: str = None) -> Optional[Dict[str, Any]]:
    """Load residual statistics from file."""
    global _residual_cache

    if _residual_cache is not None:
        return _residual_cache

    if path is None:
        path = os.path.join("models", "production", "residual_stats.json")

    if os.path.exists(path):
        with open(path) as f:
            _residual_cache = json.load(f)
        return _residual_cache
    return None


def predict_with_confidence(
    point_predictions: np.ndarray,
    horizon: str = "24h",
    confidence_level: int = 90,
) -> Dict[str, Any]:
    """
    Add confidence intervals to point predictions.

    Args:
        point_predictions: Array of point predictions [n_samples, 3] or [3]
        horizon: Which horizon to use for interval ("24h", "48h", "72h")
        confidence_level: Confidence level (80 or 95)

    Returns:
        Dictionary with point predictions and confidence bounds
    """
    stats = load_residual_stats()

    if stats is None:
        # Fallback: use a rough estimate based on model MAE
        logger.warning("No residual stats found, using MAE-based estimate")
        mae = 21.32  # Default from model metadata
        interval_width = mae * 1.65 if confidence_level == 90 else mae * 1.96

        if point_predictions.ndim == 1:
            point_predictions = point_predictions.reshape(1, -1)

        result = {
            "predictions": point_predictions.tolist(),
            "confidence_level": confidence_level,
            "interval_method": "mae_estimate",
            "intervals": [],
        }

        for i, h in enumerate(["24h", "48h", "72h"]):
            if i < point_predictions.shape[1]:
                pred_val = float(point_predictions[0, i])
                result["intervals"].append(
                    {
                        "horizon": h,
                        "point_prediction": round(pred_val, 1),
                        "lower": round(max(0, pred_val - interval_width), 1),
                        "upper": round(pred_val + interval_width, 1),
                        "width": round(interval_width * 2, 1),
                    }
                )

        return result

    # Use actual residual stats
    level_key = f"interval_{confidence_level}_lower"
    upper_key = f"interval_{confidence_level}_upper"

    # Try per-horizon stats first
    per_horizon = stats.get("per_horizon", {})

    if point_predictions.ndim == 1:
        point_predictions = point_predictions.reshape(1, -1)

    result = {
        "predictions": point_predictions.tolist(),
        "confidence_level": confidence_level,
        "interval_method": "residual_quantile",
        "intervals": [],
    }

    for i, h in enumerate(["24h", "48h", "72h"]):
        if i < point_predictions.shape[1]:
            pred_val = float(point_predictions[0, i])

            if h in per_horizon and level_key in per_horizon[h]:
                lower_offset = per_horizon[h][level_key]
                upper_offset = per_horizon[h][upper_key]
            elif level_key in stats:
                lower_offset = stats[level_key]
                upper_offset = stats[upper_key]
            else:
                lower_offset = -stats.get("mae", 21.0) * 1.65
                upper_offset = stats.get("mae", 21.0) * 1.65

            result["intervals"].append(
                {
                    "horizon": h,
                    "point_prediction": round(pred_val, 1),
                    "lower": round(max(0, pred_val + lower_offset), 1),
                    "upper": round(pred_val + upper_offset, 1),
                    "width": round(upper_offset - lower_offset, 1),
                }
            )

    return result
