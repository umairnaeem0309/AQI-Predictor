"""
MLflow Model Registry — Model lifecycle management.

Features:
- Model registration with metadata
- Production promotion with safety checks
- Model rollback workflow
- Naming conventions
- Synthetic data protection

Naming convention: {model_name}_v{version}_{date}
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.feature_store.schemas import DatasetType
from src.models.selection import ModelApprovalStatus

logger = logging.getLogger(__name__)


# =============================================================================
# Model Naming Conventions
# =============================================================================


def generate_model_name(
    model_type: str,
    version: int,
    date_str: Optional[str] = None,
) -> str:
    """Generate standardized model name.

    Convention: {model_type}_v{version}_{date}
    Example: xgboost_v1_20260813

    Args:
        model_type: Model type (ridge, random_forest, xgboost, lstm).
        version: Model version number.
        date_str: Date string (YYYYMMDD). Defaults to today.

    Returns:
        Standardized model name.
    """
    if date_str is None:
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"{model_type}_v{version}_{date_str}"


# =============================================================================
# Registry Safety Checks
# =============================================================================


def validate_for_production(
    dataset_type: str,
    approved_for_training: bool,
    approval_status: str,
) -> Tuple[bool, List[str]]:
    """Validate model is eligible for production promotion.

    Safety checks:
    - dataset_type must NOT be synthetic_test_data
    - approved_for_training must be True
    - approval_status must be 'approved'

    Args:
        dataset_type: Type of training dataset.
        approved_for_training: Whether dataset is approved.
        approval_status: Current approval status.

    Returns:
        Tuple of (eligible, list of failure reasons).
    """
    failures = []

    if dataset_type == DatasetType.SYNTHETIC_TEST.value:
        failures.append(
            f"Cannot promote synthetic test data to production. "
            f"Dataset type: {dataset_type}"
        )

    if not approved_for_training:
        failures.append(
            f"Dataset not approved for training. "
            f"approved_for_training={approved_for_training}"
        )

    if approval_status != ModelApprovalStatus.APPROVED.value:
        failures.append(
            f"Model not approved for production. "
            f"Status: {approval_status}"
        )

    eligible = len(failures) == 0
    return eligible, failures


# =============================================================================
# Model Registry
# =============================================================================


class ModelRegistry:
    """MLflow model registry wrapper.

    Manages model versions, stages, and metadata.

    Usage:
        registry = ModelRegistry()
        registry.register_model(model, metadata)
        registry.promote_to_production(model_name, version)
        registry.rollback(version)
    """

    def __init__(self, experiment_name: str = "aqi_predictor"):
        """Initialize registry.

        Args:
            experiment_name: MLflow experiment name.
        """
        self.experiment_name = experiment_name
        self._client = None

    def _get_client(self):
        """Get or create MLflow client."""
        if self._client is None:
            try:
                import mlflow
                self._client = mlflow.tracking.MlflowClient()
            except ImportError:
                logger.warning("MLflow not installed")
                return None
        return self._client

    def register_model(
        self,
        model_name: str,
        model,
        metrics: Dict[str, float],
        params: Dict[str, Any],
        dataset_metadata: Dict[str, Any],
        feature_columns: List[str],
    ) -> Optional[str]:
        """Register a model in MLflow.

        Args:
            model_name: Standardized model name.
            model: Trained model object.
            metrics: Evaluation metrics.
            params: Model parameters.
            dataset_metadata: Dataset metadata (version, type, approval).
            feature_columns: List of feature names.

        Returns:
            MLflow run ID or None.
        """
        try:
            import mlflow
            import mlflow.sklearn
        except ImportError:
            logger.warning("MLflow not installed — skipping registration")
            return None

        try:
            with mlflow.start_run(run_name=model_name) as run:
                # Tags
                mlflow.set_tag("model_name", model_name)
                mlflow.set_tag("dataset_version", dataset_metadata.get("version", ""))
                mlflow.set_tag("dataset_type", dataset_metadata.get("type", ""))
                mlflow.set_tag("approved_for_training", str(dataset_metadata.get("approved", False)))
                mlflow.set_tag("approval_status", ModelApprovalStatus.CANDIDATE.value)
                mlflow.set_tag("registration_timestamp", datetime.now(timezone.utc).isoformat())

                # Parameters
                mlflow.log_params(params)
                mlflow.log_param("feature_count", len(feature_columns))

                # Metrics
                for key, value in metrics.items():
                    if not (isinstance(value, float) and (value != value)):
                        mlflow.log_metric(key, value)

                # Feature list
                features_data = {"features": feature_columns}
                mlflow.log_dict(features_data, "feature_list.json")

                # Model
                mlflow.sklearn.log_model(model, "model")

                run_id = run.info.run_id
                logger.info("Model registered: %s (run=%s)", model_name, run_id)
                return run_id

        except Exception as e:
            logger.error("Model registration failed: %s", str(e))
            return None

    def promote_to_production(
        self,
        model_name: str,
        version: int,
        dataset_type: str,
        approved_for_training: bool,
        approval_status: str,
    ) -> bool:
        """Promote a model version to production.

        Safety checks:
        - Rejects synthetic_test_data
        - Requires approved_for_training=true
        - Requires approval_status='approved'

        Args:
            model_name: Model name.
            version: Model version.
            dataset_type: Training dataset type.
            approved_for_training: Whether dataset is approved.
            approval_status: Current approval status.

        Returns:
            True if promotion succeeded.
        """
        eligible, failures = validate_for_production(
            dataset_type, approved_for_training, approval_status
        )

        if not eligible:
            for failure in failures:
                logger.error("Production promotion rejected: %s", failure)
            return False

        try:
            import mlflow
            client = self._get_client()
            if client is None:
                return False

            # Transition model version to Production stage
            client.transition_model_version_stage(
                name=self.experiment_name,
                version=version,
                stage="Production",
            )

            logger.info(
                "Model %s v%d promoted to Production",
                model_name,
                version,
            )
            return True

        except Exception as e:
            logger.error("Production promotion failed: %s", str(e))
            return False

    def rollback(
        self,
        target_version: int,
        reason: str = "Manual rollback",
    ) -> bool:
        """Rollback to a previous model version.

        Workflow:
        1. Demote current production model to Archived
        2. Promote target version to Production
        3. Log rollback event

        Args:
            target_version: Version to rollback to.
            reason: Reason for rollback.

        Returns:
            True if rollback succeeded.
        """
        try:
            client = self._get_client()
            if client is None:
                return False

            # Get current production model
            versions = client.get_latest_versions(
                self.experiment_name, stages=["Production"]
            )

            if versions:
                current_version = versions[0]
                # Demote current to Archived
                client.transition_model_version_stage(
                    name=self.experiment_name,
                    version=current_version.version,
                    stage="Archived",
                )
                logger.info(
                    "Demoted v%d to Archived: %s",
                    current_version.version,
                    reason,
                )

            # Promote target to Production
            client.transition_model_version_stage(
                name=self.experiment_name,
                version=target_version,
                stage="Production",
            )

            logger.info(
                "Rolled back to v%d: %s",
                target_version,
                reason,
            )
            return True

        except Exception as e:
            logger.error("Rollback failed: %s", str(e))
            return False

    def get_production_model(self) -> Optional[Dict[str, Any]]:
        """Get the current production model.

        Returns:
            Dictionary with model info or None.
        """
        try:
            client = self._get_client()
            if client is None:
                return None

            versions = client.get_latest_versions(
                self.experiment_name, stages=["Production"]
            )

            if versions:
                v = versions[0]
                return {
                    "name": v.name,
                    "version": v.version,
                    "stage": v.current_stage,
                    "run_id": v.run_id,
                }

            return None

        except Exception as e:
            logger.error("Failed to get production model: %s", str(e))
            return None

    def list_models(self, stage: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all model versions.

        Args:
            stage: Filter by stage (Production, Staging, Archived, None for all).

        Returns:
            List of model version info dictionaries.
        """
        try:
            client = self._get_client()
            if client is None:
                return []

            if stage:
                versions = client.get_latest_versions(self.experiment_name, stages=[stage])
            else:
                versions = client.search_model_versions(f"name='{self.experiment_name}'")

            return [
                {
                    "name": v.name,
                    "version": v.version,
                    "stage": v.current_stage,
                    "run_id": v.run_id,
                }
                for v in versions
            ]

        except Exception as e:
            logger.error("Failed to list models: %s", str(e))
            return []

    # =========================================================================
    # Artifact Logging
    # =========================================================================

    def log_artifacts(
        self,
        run_id: str,
        model=None,
        metrics: Optional[Dict[str, float]] = None,
        params: Optional[Dict[str, Any]] = None,
        feature_importance: Optional[Dict[str, float]] = None,
        feature_columns: Optional[List[str]] = None,
        evaluation_report: Optional[str] = None,
    ) -> bool:
        """Log all model artifacts to MLflow.

        Artifact structure:
        ├── model/              (sklearn model)
        ├── metadata.json       (version, date, dataset, features)
        ├── metrics.json        (evaluation metrics)
        ├── parameters.json     (hyperparameters)
        ├── feature_importance.json
        ├── feature_list.json   (feature columns)
        └── evaluation_report.txt

        Args:
            run_id: MLflow run ID.
            model: Trained model object.
            metrics: Evaluation metrics.
            params: Model parameters.
            feature_importance: Feature importance dict.
            feature_columns: List of feature names.
            evaluation_report: Evaluation report text.

        Returns:
            True if logging succeeded.
        """
        try:
            import mlflow
            import mlflow.sklearn
        except ImportError:
            logger.warning("MLflow not installed")
            return False

        try:
            with mlflow.start_run(run_id=run_id):
                # Model
                if model is not None:
                    mlflow.sklearn.log_model(model, "model")

                # Metrics
                if metrics:
                    mlflow.log_dict(metrics, "metrics.json")
                    for k, v in metrics.items():
                        if not (isinstance(v, float) and (v != v)):
                            mlflow.log_metric(k, v)

                # Parameters
                if params:
                    mlflow.log_dict(params, "parameters.json")

                # Feature importance
                if feature_importance:
                    mlflow.log_dict(feature_importance, "feature_importance.json")

                # Feature list
                if feature_columns:
                    mlflow.log_dict({"features": feature_columns}, "feature_list.json")

                # Evaluation report
                if evaluation_report:
                    report_path = Path("evaluation_report.txt")
                    report_path.write_text(evaluation_report)
                    mlflow.log_artifact(str(report_path))
                    report_path.unlink()

                logger.info("Artifacts logged for run %s", run_id)
                return True

        except Exception as e:
            logger.error("Artifact logging failed: %s", str(e))
            return False

    # =========================================================================
    # Version Metadata
    # =========================================================================

    def store_version_metadata(
        self,
        run_id: str,
        model_name: str,
        version: int,
        dataset_version: str,
        feature_version: str,
        schema_version: str,
        training_date: str,
        metrics: Dict[str, float],
    ) -> bool:
        """Store complete version metadata for a model.

        Metadata includes:
        - model name, version
        - training date
        - dataset version, feature version, schema version
        - evaluation metrics
        """
        try:
            import mlflow
        except ImportError:
            return False

        metadata = {
            "model_name": model_name,
            "version": version,
            "training_date": training_date,
            "dataset_version": dataset_version,
            "feature_version": feature_version,
            "schema_version": schema_version,
            "metrics": metrics,
            "stored_at": datetime.now(timezone.utc).isoformat(),
        }

        try:
            with mlflow.start_run(run_id=run_id):
                mlflow.log_dict(metadata, "version_metadata.json")
                logger.info("Version metadata stored for %s v%d", model_name, version)
                return True
        except Exception as e:
            logger.error("Failed to store metadata: %s", str(e))
            return False

    # =========================================================================
    # Model Loading
    # =========================================================================

    def load_production_model(self):
        """Load the current production model.

        Validates metadata before loading:
        - status must be Production
        - dataset_type must not be synthetic
        - approved_for_training must be true

        Returns:
            Loaded model object or None.
        """
        try:
            import mlflow
            import mlflow.pyfunc
        except ImportError:
            logger.warning("MLflow not installed")
            return None

        try:
            # Get production model version
            client = self._get_client()
            if client is None:
                return None

            versions = client.get_latest_versions(
                self.experiment_name, stages=["Production"]
            )

            if not versions:
                logger.warning("No production model found")
                return None

            prod_version = versions[0]

            # Validate metadata
            run = client.get_run(prod_version.run_id)
            tags = run.data.tags

            dataset_type = tags.get("dataset_type", "unknown")
            if dataset_type == "synthetic_test_data":
                logger.error("Cannot load synthetic test data as production model")
                return None

            approved = tags.get("approved_for_training", "false")
            if approved != "true":
                logger.error("Production model not approved for training")
                return None

            # Load model
            model_uri = f"runs:/{prod_version.run_id}/model"
            model = mlflow.pyfunc.load_model(model_uri)
            logger.info("Loaded production model: %s v%s", prod_version.name, prod_version.version)
            return model

        except Exception as e:
            logger.error("Failed to load production model: %s", str(e))
            return None

    def load_model_version(self, version: int):
        """Load a specific model version.

        Args:
            version: Model version number.

        Returns:
            Loaded model object or None.
        """
        try:
            import mlflow
        except ImportError:
            return None

        try:
            model_uri = f"models:/{self.experiment_name}/{version}"
            model = mlflow.pyfunc.load_model(model_uri)
            logger.info("Loaded model %s v%d", self.experiment_name, version)
            return model
        except Exception as e:
            logger.error("Failed to load model v%d: %s", version, str(e))
            return None

    # =========================================================================
    # Drift Baseline
    # =========================================================================

    def store_drift_baseline(
        self,
        run_id: str,
        feature_data,
        feature_columns: List[str],
    ) -> bool:
        """Store drift baseline statistics for monitoring.

        Drift baseline contains:
        Numerical: mean, std, min, max, percentiles (25, 50, 75)
        Categorical: frequency distribution

        Args:
            run_id: MLflow run ID.
            feature_data: DataFrame with features.
            feature_columns: Columns to compute baseline for.

        Returns:
            True if baseline stored.
        """
        try:
            import mlflow
            import numpy as np
        except ImportError:
            return False

        baseline = {}
        for col in feature_columns:
            if col not in feature_data.columns:
                continue

            series = feature_data[col]

            if series.dtype in ["float64", "int64", "float32", "int32"]:
                # Numerical baseline
                baseline[col] = {
                    "type": "numerical",
                    "mean": float(series.mean()),
                    "std": float(series.std()),
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "percentile_25": float(series.quantile(0.25)),
                    "percentile_50": float(series.quantile(0.50)),
                    "percentile_75": float(series.quantile(0.75)),
                }
            else:
                # Categorical baseline
                freq = series.value_counts(normalize=True).to_dict()
                baseline[col] = {
                    "type": "categorical",
                    "frequency_distribution": {str(k): float(v) for k, v in freq.items()},
                }

        try:
            with mlflow.start_run(run_id=run_id):
                mlflow.log_dict(baseline, "drift_baseline.json")
                logger.info("Drift baseline stored for %d features", len(baseline))
                return True
        except Exception as e:
            logger.error("Drift baseline storage failed: %s", str(e))
            return False

    def get_drift_baseline(self, run_id: str) -> Optional[Dict]:
        """Load drift baseline from MLflow.

        Args:
            run_id: MLflow run ID.

        Returns:
            Drift baseline dictionary or None.
        """
        try:
            import mlflow
            import tempfile

            client = self._get_client()
            if client is None:
                return None

            artifact_path = "drift_baseline.json"
            local_path = client.download_artifacts(run_id, artifact_path)

            with open(local_path, "r") as f:
                return json.load(f)

        except Exception as e:
            logger.error("Failed to load drift baseline: %s", str(e))
            return None
