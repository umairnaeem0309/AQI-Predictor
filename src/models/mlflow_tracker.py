"""
MLflow Experiment Tracking Wrapper.

Provides a clean interface for logging experiments to MLflow:
- Parameters, metrics, artifacts
- Model registration
- Dataset metadata and lineage
- Synthetic data tagging

All runs are tagged with:
- dataset_version
- feature_version
- training_data_type (synthetic/real)
- is_reportable (false for synthetic)
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Default MLflow tracking URI (local file-based)
DEFAULT_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "")


def log_experiment(
    run_name: str,
    model_name: str,
    params: Dict[str, Any],
    metrics: Dict[str, float],
    feature_columns: list,
    dataset_version: str,
    feature_version: str,
    training_data_type: str = "real_training_data",
    is_reportable: bool = True,
    random_seed: int = 42,
    model_artifact: Any = None,
    feature_importance: Optional[Dict[str, float]] = None,
    training_time: float = 0.0,
    tracking_uri: Optional[str] = None,
) -> str:
    """Log an experiment run to MLflow.

    Args:
        run_name: Name for this run.
        model_name: Model type (ridge, random_forest, xgboost, lstm).
        params: Model hyperparameters.
        metrics: Evaluation metrics (mae_24h, rmse_24h, r2_24h, etc.).
        feature_columns: List of feature names used.
        dataset_version: Dataset version identifier.
        feature_version: Feature definition version.
        training_data_type: Type of training data.
        is_reportable: Whether results can be reported.
        random_seed: Random seed for reproducibility.
        model_artifact: Trained model object to log.
        feature_importance: Feature importance dictionary.
        training_time: Training duration in seconds.
        tracking_uri: MLflow tracking URI override.

    Returns:
        MLflow run ID.
    """
    try:
        import mlflow
        import mlflow.sklearn
    except ImportError:
        logger.warning("MLflow not installed — skipping experiment logging")
        return "mlflow_not_available"

    uri = tracking_uri or DEFAULT_TRACKING_URI
    if uri:
        mlflow.set_tracking_uri(uri)

    try:
        with mlflow.start_run(run_name=run_name) as run:
            # Tags
            mlflow.set_tag("model_name", model_name)
            mlflow.set_tag("dataset_version", dataset_version)
            mlflow.set_tag("feature_version", feature_version)
            mlflow.set_tag("training_data_type", training_data_type)
            mlflow.set_tag("is_reportable", str(is_reportable))
            mlflow.set_tag("random_seed", str(random_seed))
            mlflow.set_tag("training_timestamp", datetime.now(timezone.utc).isoformat())

            # Parameters
            mlflow.log_params(params)
            mlflow.log_param("feature_count", len(feature_columns))
            mlflow.log_param("training_time", training_time)

            # Metrics
            for key, value in metrics.items():
                if not (isinstance(value, float) and (value != value)):  # Skip NaN
                    mlflow.log_metric(key, value)

            # Feature importance artifact
            if feature_importance:
                importance_path = Path("feature_importance.json")
                with open(importance_path, "w") as f:
                    json.dump(feature_importance, f, indent=2)
                mlflow.log_artifact(str(importance_path))
                importance_path.unlink()

            # Feature list artifact
            features_path = Path("feature_list.json")
            with open(features_path, "w") as f:
                json.dump({"features": feature_columns, "count": len(feature_columns)}, f, indent=2)
            mlflow.log_artifact(str(features_path))
            features_path.unlink()

            # Model artifact
            if model_artifact is not None:
                try:
                    mlflow.sklearn.log_model(model_artifact, "model")
                except Exception as e:
                    logger.warning("Failed to log model artifact: %s", str(e))

            run_id = run.info.run_id
            logger.info("MLflow run logged: %s (id=%s)", run_name, run_id)
            return run_id

    except Exception as e:
        logger.error("MLflow logging failed: %s", str(e))
        return "logging_failed"


def log_model_comparison(
    comparison_results: list,
    dataset_version: str,
    is_reportable: bool = True,
) -> str:
    """Log a model comparison summary to MLflow.

    Args:
        comparison_results: List of training result dictionaries.
        dataset_version: Dataset version.
        is_reportable: Whether results can be reported.

    Returns:
        MLflow run ID.
    """
    try:
        import mlflow
    except ImportError:
        return "mlflow_not_available"

    try:
        with mlflow.start_run(run_name="model_comparison") as run:
            mlflow.set_tag("dataset_version", dataset_version)
            mlflow.set_tag("is_reportable", str(is_reportable))
            mlflow.set_tag("run_type", "comparison")

            for result in comparison_results:
                if "error" in result:
                    continue
                model_name = result.get("model_name", "unknown")
                metrics = result.get("metrics", {})
                for key, value in metrics.items():
                    if not (isinstance(value, float) and (value != value)):
                        mlflow.log_metric(f"{model_name}_{key}", value)

            return run.info.run_id

    except Exception as e:
        logger.error("MLflow comparison logging failed: %s", str(e))
        return "logging_failed"
