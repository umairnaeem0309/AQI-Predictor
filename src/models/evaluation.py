"""
Model Evaluation — Multi-horizon evaluation metrics.

Evaluates forecasting models on:
- MAE (Mean Absolute Error) — per horizon and averaged
- RMSE (Root Mean Squared Error) — per horizon and averaged
- R² (Coefficient of Determination) — per horizon and averaged

All metrics computed for:
- target_aqi_24h
- target_aqi_48h
- target_aqi_72h
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)

TARGET_COLUMNS = ["target_aqi_24h", "target_aqi_48h", "target_aqi_72h"]
HORIZON_LABELS = {
    "target_aqi_24h": "24h",
    "target_aqi_48h": "48h",
    "target_aqi_72h": "72h",
}


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute MAE, RMSE, R² for a single output.

    Args:
        y_true: True values.
        y_pred: Predicted values.

    Returns:
        Dictionary with mae, rmse, r2.
    """
    # Filter out NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]

    if len(y_true_clean) == 0:
        return {"mae": np.nan, "rmse": np.nan, "r2": np.nan}

    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    r2 = r2_score(y_true_clean, y_pred_clean)

    return {"mae": mae, "rmse": rmse, "r2": r2}


def evaluate_model(
    model: BaseEstimator,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
) -> Dict[str, Any]:
    """Evaluate a trained model on validation data.

    Computes metrics for each horizon (24h, 48h, 72h) and overall average.

    Args:
        model: Trained model.
        X_val: Validation features.
        y_val: Validation targets.

    Returns:
        Dictionary with per-horizon and average metrics.
    """
    # Predict
    y_pred = model.predict(X_val)

    # If y_pred is 1D (single output), reshape
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    results = {}

    # Per-horizon metrics
    available_targets = [c for c in TARGET_COLUMNS if c in y_val.columns]

    for i, target in enumerate(available_targets):
        horizon = HORIZON_LABELS.get(target, target)
        y_true = y_val[target].values

        if i < y_pred.shape[1]:
            y_pred_horizon = y_pred[:, i]
        else:
            y_pred_horizon = y_pred[:, 0] if y_pred.ndim > 1 else y_pred

        metrics = compute_metrics(y_true, y_pred_horizon)
        for key, value in metrics.items():
            results[f"{key}_{horizon}"] = value

    # Average metrics across horizons
    for metric in ["mae", "rmse", "r2"]:
        horizon_values = [results.get(f"{metric}_{h}") for h in ["24h", "48h", "72h"]]
        valid_values = [v for v in horizon_values if not np.isnan(v)]
        if valid_values:
            results[f"{metric}_avg"] = np.mean(valid_values)
        else:
            results[f"{metric}_avg"] = np.nan

    logger.info(
        "Evaluation: MAE_avg=%.2f, RMSE_avg=%.2f, R²_avg=%.4f",
        results.get("mae_avg", 0),
        results.get("rmse_avg", 0),
        results.get("r2_avg", 0),
    )

    return results


def compare_models(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Compare evaluation results across multiple models.

    Args:
        results: List of training result dictionaries.

    Returns:
        DataFrame with comparison table.
    """
    rows = []
    for result in results:
        if "error" in result:
            continue
        metrics = result.get("metrics", {})
        row = {
            "model": result.get("model_name", "unknown"),
            "mae_24h": metrics.get("mae_24h", np.nan),
            "mae_48h": metrics.get("mae_48h", np.nan),
            "mae_72h": metrics.get("mae_72h", np.nan),
            "mae_avg": metrics.get("mae_avg", np.nan),
            "rmse_24h": metrics.get("rmse_24h", np.nan),
            "rmse_48h": metrics.get("rmse_48h", np.nan),
            "rmse_72h": metrics.get("rmse_72h", np.nan),
            "rmse_avg": metrics.get("rmse_avg", np.nan),
            "r2_24h": metrics.get("r2_24h", np.nan),
            "r2_48h": metrics.get("r2_48h", np.nan),
            "r2_72h": metrics.get("r2_72h", np.nan),
            "r2_avg": metrics.get("r2_avg", np.nan),
            "training_time": result.get("training_time", np.nan),
            "feature_count": len(result.get("feature_columns", [])),
            "is_reportable": result.get("is_reportable", False),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def generate_evaluation_report(
    comparison_df: pd.DataFrame,
    output_path: Optional[str] = None,
) -> str:
    """Generate a text evaluation report.

    Args:
        comparison_df: Comparison DataFrame from compare_models().
        output_path: Optional path to save the report.

    Returns:
        Report as string.
    """
    report_lines = [
        "=" * 70,
        "AQI PREDICTOR — MODEL COMPARISON REPORT",
        "=" * 70,
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Models evaluated: {len(comparison_df)}",
        "",
        "-" * 70,
        "RESULTS SUMMARY",
        "-" * 70,
    ]

    for _, row in comparison_df.iterrows():
        report_lines.extend(
            [
                "",
                f"Model: {row['model']}",
                f"  MAE  — 24h: {row['mae_24h']:.2f}, 48h: {row['mae_48h']:.2f}, 72h: {row['mae_72h']:.2f}, Avg: {row['mae_avg']:.2f}",
                f"  RMSE — 24h: {row['rmse_24h']:.2f}, 48h: {row['rmse_48h']:.2f}, 72h: {row['rmse_72h']:.2f}, Avg: {row['rmse_avg']:.2f}",
                f"  R²   — 24h: {row['r2_24h']:.4f}, 48h: {row['r2_48h']:.4f}, 72h: {row['r2_72h']:.4f}, Avg: {row['r2_avg']:.4f}",
                f"  Training time: {row['training_time']:.2f}s",
                f"  Features: {row['feature_count']}",
                f"  Reportable: {row['is_reportable']}",
            ]
        )

    report_lines.extend(
        [
            "",
            "-" * 70,
            "NOTE: Production model selection happens in Phase 8.",
            "This report is for comparison only.",
            "-" * 70,
        ]
    )

    report = "\n".join(report_lines)

    if output_path:
        with open(output_path, "w") as f:
            f.write(report)
        logger.info("Evaluation report saved to %s", output_path)

    return report
