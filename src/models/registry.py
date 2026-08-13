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
from typing import Any, Dict, List, Optional

from src.feature_store.schemas import DatasetType

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
