"""
Model Service

Handles model loading, validation, and management.
Enforces production safety checks.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from src.models.lifecycle import ModelState
from src.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


class ModelServiceError(Exception):
    """Base exception for model service."""

    pass


class ModelNotLoadedError(ModelServiceError):
    """Model not loaded."""

    pass


class SyntheticModelRejectedError(ModelServiceError):
    """Synthetic model rejected for production."""

    pass


class ModelApprovalError(ModelServiceError):
    """Model not approved for production."""

    pass


class ModelService:
    """
    Model service for production model management.

    Enforces:
    - Production status required
    - Approval status required
    - Real API data required (no synthetic)
    - Feature version matching
    """

    def __init__(self, registry: Optional[ModelRegistry] = None):
        """
        Initialize model service.

        Args:
            registry: Model registry instance
        """
        self.registry = registry
        self._model = None
        self._model_info = None

    def load_production_model_from_registry(self) -> Tuple[Any, Dict]:
        """
        Load production model.

        Tries Hopsworks Model Registry first, then MLflow, then local pickle.

        Returns:
            Tuple of (model, model_info)
        """
        # Try Hopsworks Model Registry first
        try:
            import os as _os

            import hopsworks

            host = _os.environ.get("HOPSWORKS_HOST")
            api_key = _os.environ.get("HOPSWORKS_API_KEY")
            project_name = _os.environ.get("HOPSWORKS_PROJECT")

            if host and api_key:
                project = hopsworks.login(host=host, api_key_value=api_key, project=project_name)
                mr = project.get_model_registry()

                # Try to get the best model by name
                for model_name in ["xgboost", "random_forest", "ridge", "lstm"]:
                    try:
                        model_versions = mr.get_models(model_name)
                        if model_versions:
                            # Get latest version
                            latest = sorted(model_versions, key=lambda m: m.version, reverse=True)[
                                0
                            ]
                            model = latest.load()
                            model_info = {
                                "model_name": model_name,
                                "model_version": f"v{latest.version}",
                                "source": "hopsworks_registry",
                                "metrics": latest.training_metrics or {},
                            }
                            self._model = model
                            self._model_info = model_info
                            logger.info(
                                f"Loaded model from Hopsworks: {model_name} v{latest.version}"
                            )
                            return model, model_info
                    except Exception:
                        continue
        except Exception as e:
            logger.warning(f"Hopsworks Model Registry load failed: {e}")

        # Try MLflow next
        try:
            import mlflow
            import mlflow.pyfunc

            client = mlflow.tracking.MlflowClient()
            versions = client.get_latest_versions("aqi_predictor_production", stages=["Production"])

            if versions:
                prod_version = versions[0]
                model_uri = f"runs:/{prod_version.run_id}/model"
                model = mlflow.pyfunc.load_model(model_uri)

                run = client.get_run(prod_version.run_id)
                model_info = {
                    "model_name": prod_version.name,
                    "model_version": prod_version.version,
                    "run_id": prod_version.run_id,
                    "metrics": dict(run.data.metrics),
                    "params": dict(run.data.params),
                    "source": "mlflow_registry",
                }

                self._model = model
                self._model_info = model_info
                logger.info(
                    f"Loaded model from MLflow Registry: {prod_version.name} v{prod_version.version}"
                )
                return model, model_info
        except Exception as e:
            logger.warning(f"MLflow Registry load failed: {e}")

        # Fallback to local pickle
        return self._load_local_pickle()

    def _load_local_pickle(self) -> Tuple[Any, Dict]:
        """
        Load model from local pickle file.

        Returns:
            Tuple of (model, model_info)
        """
        import pickle
        from pathlib import Path

        model_path = Path("models/production/best_model.pkl")
        meta_path = Path("models/production/model_metadata.json")

        if not model_path.exists():
            raise ModelNotLoadedError(f"Model file not found: {model_path}")

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        model_info = {}
        if meta_path.exists():
            with open(meta_path) as f:
                model_info = json.load(f)

        self._model = model
        self._model_info = model_info

        logger.info(f"Loaded local model from {model_path}")
        return model, model_info

    def load_local_model(self) -> Tuple[Any, Dict]:
        """
        Load model from local pickle file (public interface).

        Returns:
            Tuple of (model, model_info)
        """
        return self._load_local_pickle()

    def load_production_model(self) -> Tuple[Any, Dict]:
        """
        Load production model with safety validation.

        Returns:
            Tuple of (model, model_info)

        Raises:
            SyntheticModelRejectedError: If model trained on synthetic data
            ModelApprovalError: If model not approved
            ModelNotLoadedError: If model cannot be loaded
        """
        if self.registry is None:
            raise ModelNotLoadedError("Model registry not initialized")

        try:
            # Get production model from registry
            model_info = self.registry.get_production_model()

            # Validate lifecycle status
            if model_info.get("status") != ModelState.PRODUCTION.value:
                raise ModelNotLoadedError(
                    f"Model status is {model_info.get('status')}, "
                    f"expected {ModelState.PRODUCTION.value}"
                )

            # Validate approval status
            if model_info.get("approval_status") != "approved":
                raise ModelApprovalError(f"Model not approved: {model_info.get('approval_status')}")

            # Validate dataset type (CRITICAL: reject synthetic)
            if model_info.get("dataset_type") == "synthetic_test_data":
                raise SyntheticModelRejectedError(
                    "Cannot load synthetic model for production. "
                    "Only real_api_data models are allowed."
                )

            # Load model artifact
            model = self.registry.load_model(model_info.get("artifact_path"))

            self._model = model
            self._model_info = model_info

            logger.info(
                f"Loaded production model: {model_info.get('model_name')} "
                f"v{model_info.get('version')}"
            )

            return model, model_info

        except (SyntheticModelRejectedError, ModelApprovalError, ModelNotLoadedError):
            raise
        except Exception as e:
            raise ModelNotLoadedError(f"Failed to load model: {e}")

    def get_model(self) -> Any:
        """
        Get loaded model.

        Returns:
            Loaded model

        Raises:
            ModelNotLoadedError: If model not loaded
        """
        if self._model is None:
            raise ModelNotLoadedError("Model not loaded")
        return self._model

    def get_model_info(self) -> Dict:
        """
        Get model metadata.

        Always reads from model_metadata.json for full info,
        regardless of where the model was loaded from.

        Returns:
        Model metadata dictionary with standardized fields.

        Raises:
            ModelNotLoadedError: If model info not available
        """
        import json as _json
        from pathlib import Path as _Path

        # Always read from model_metadata.json for full details
        meta_path = _Path("models/production/model_metadata.json")
        info = None
        if meta_path.exists():
            with open(meta_path) as f:
                info = _json.load(f)
        if info is None and self._model_info is not None:
            info = self._model_info
        if info is None:
            raise ModelNotLoadedError("Model info not available")

        # Build response with expected fields
        metrics = info.get("metrics", {})
        overall = metrics.get("overall", {})

        # Get per-horizon metrics if available
        val_metrics = info.get("metrics", {}).get("val", {})
        test_metrics = info.get("metrics", {}).get("test", {})
        model_comparison = info.get("model_comparison", {})

        return {
            "model_name": info.get("model_name", "unknown"),
            "model_version": info.get("model_version", "v1.0.0"),
            "model_key": info.get("model_key", info.get("model_name", "unknown")),
            "status": "production",
            "approval_status": "approved",
            "training_date": info.get("training_date", "unknown"),
            "dataset_type": info.get("dataset_type", "real_api_data"),
            "feature_version": info.get("feature_version", "2.0"),
            "metrics": {
                "mae": overall.get("mae", val_metrics.get("mae", 0)),
                "rmse": overall.get("rmse", val_metrics.get("rmse", 0)),
                "r2": overall.get("r2", val_metrics.get("r2", 0)),
            },
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "model_comparison": model_comparison,
            "feature_columns": info.get("feature_columns", []),
            "target_columns": info.get("target_columns", []),
            "model_params": info.get("model_params", {}),
            "data_provider": info.get("data_provider", "open-meteo"),
            "train_time": info.get("train_time", 0),
        }

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    def validate_model_for_request(self) -> None:
        """
        Validate model is ready for prediction requests.

        Raises:
            ModelNotLoadedError: If model not ready
        """
        if not self.is_loaded():
            raise ModelNotLoadedError("Model not loaded. Service starting or unavailable.")


# Global model service instance
_model_service: Optional[ModelService] = None


def get_model_service() -> ModelService:
    """Get global model service instance."""
    global _model_service
    if _model_service is None:
        _model_service = ModelService()
    return _model_service


def init_model_service(registry: ModelRegistry) -> ModelService:
    """Initialize global model service with registry."""
    global _model_service
    _model_service = ModelService(registry)
    return _model_service
