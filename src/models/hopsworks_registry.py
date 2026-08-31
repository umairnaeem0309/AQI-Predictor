"""
Hopsworks Model Registry

Stores and retrieves trained models using Hopsworks Model Registry.
Replaces local MLflow for model versioning.

Features:
- Store model artifacts in Hopsworks
- Version tracking with metrics
- Production model selection
- Model comparison storage
"""

import json
import logging
import os
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class HopsworksModelRegistry:
    """Model registry using Hopsworks for model storage and versioning."""

    def __init__(self):
        """Initialize Hopsworks connection."""
        self._connection = None
        self._project = None

    def connect(self):
        """Connect to Hopsworks."""
        try:
            import hopsworks

            host = os.environ.get("HOPSWORKS_HOST")
            api_key = os.environ.get("HOPSWORKS_API_KEY")
            project_name = os.environ.get("HOPSWORKS_PROJECT", "project_umairnaeem0309")

            if not host or not api_key:
                logger.warning("Hopsworks credentials not configured")
                return False

            self._connection = hopsworks.login(
                host=host,
                api_key_value=api_key,
                project=project_name,
            )
            # In Hopsworks, the connection object IS the project
            self._project = self._connection
            logger.info(f"Connected to Hopsworks project: {project_name}")
            return True

        except Exception as e:
            logger.error(f"Hopsworks connection failed: {e}")
            return False

    def store_model(
        self,
        model_name: str,
        model: Any,
        metrics: Dict[str, float],
        metadata: Dict[str, Any],
        model_version: Optional[str] = None,
    ) -> bool:
        """
        Store a trained model in Hopsworks.

        Args:
            model_name: Name of the model (e.g., 'xgboost', 'random_forest')
            model: Trained model object
            metrics: Evaluation metrics
            metadata: Additional metadata
            model_version: Version string (default: timestamp)

        Returns:
            True if stored successfully
        """
        try:
            if not self._connection:
                if not self.connect():
                    return False

            # Create version number (Hopsworks expects small integer)
            if not model_version:
                # Use simple incrementing version
                try:
                    mr = self._connection.get_model_registry()
                    existing = mr.get_models(model_name)
                    model_version = len(existing) + 1
                except Exception:
                    model_version = 1

            # Save model to local temp file
            temp_dir = Path("temp_models")
            temp_dir.mkdir(exist_ok=True)
            model_path = temp_dir / f"{model_name}_{model_version}.pkl"

            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            # Get the model registry
            mr = self._connection.get_model_registry()

            # Create model in registry
            model_config = mr.python.create_model(
                name=model_name,
                version=model_version,
                description=f"AQI prediction model: {model_name}",
                metrics=metrics,
                input_example=pd.DataFrame(
                    [[0] * 10],  # Dummy input
                    columns=[f"feature_{i}" for i in range(10)],
                ),
            )

            # Save model artifact
            model_config.save(str(model_path))

            # Clean up temp file
            model_path.unlink()

            logger.info(f"Stored model {model_name} v{model_version} in Hopsworks")
            return True

        except Exception as e:
            logger.error(f"Failed to store model: {e}")
            return False

    def get_latest_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest version of a model.

        Args:
            model_name: Name of the model

        Returns:
            Dictionary with model info or None
        """
        try:
            if not self._connection:
                if not self.connect():
                    return None

            mr = self._connection.get_model_registry()

            # Get all versions of the model
            models = mr.get_models(model_name)

            if not models:
                logger.warning(f"No models found with name: {model_name}")
                return None

            # Sort by version (latest first)
            latest = max(models, key=lambda m: m.version)

            return {
                "name": latest.name,
                "version": latest.version,
                "metrics": latest.metrics,
                "description": latest.description,
                "model": latest,
            }

        except Exception as e:
            logger.error(f"Failed to get latest model: {e}")
            return None

    def get_best_model(
        self,
        metric: str = "mae",
        lower_is_better: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Get the best model based on a metric.

        Args:
            metric: Metric name to compare
            lower_is_better: Whether lower metric is better

        Returns:
            Dictionary with best model info
        """
        try:
            if not self._connection:
                if not self.connect():
                    return None

            mr = self._connection.get_model_registry()

            # Get all models
            all_models = mr.get_models()

            if not all_models:
                return None

            # Filter by metric and find best
            best_model = None
            best_value = float("inf") if lower_is_better else float("-inf")

            for model in all_models:
                if metric in model.metrics:
                    value = model.metrics[metric]
                    if lower_is_better and value < best_value:
                        best_value = value
                        best_model = model
                    elif not lower_is_better and value > best_value:
                        best_value = value
                        best_model = model

            if best_model:
                return {
                    "name": best_model.name,
                    "version": best_model.version,
                    "metrics": best_model.metrics,
                    "description": best_model.description,
                    "model": best_model,
                }

            return None

        except Exception as e:
            logger.error(f"Failed to get best model: {e}")
            return None

    def load_model(self, model_name: str, version: Optional[str] = None) -> Optional[Any]:
        """
        Load a model from the registry.

        Args:
            model_name: Name of the model
            version: Specific version (default: latest)

        Returns:
            Loaded model object
        """
        try:
            if not self._connection:
                if not self.connect():
                    return None

            mr = self._connection.get_model_registry()

            if version:
                model = mr.get_model(model_name, version=version)
            else:
                models = mr.get_models(model_name)
                if not models:
                    return None
                model = max(models, key=lambda m: m.version)

            # Load the model
            loaded_model = model.load()

            logger.info(f"Loaded model {model_name} v{model.version}")
            return loaded_model

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None

    def compare_models(
        self,
        model_results: Dict[str, Dict[str, float]],
    ) -> str:
        """
        Compare multiple models and return the best one.

        Args:
            model_results: Dictionary of model_name -> metrics

        Returns:
            Name of the best model
        """
        if not model_results:
            return ""

        # Use test MAE as primary metric
        best_model = min(model_results.keys(), key=lambda m: model_results[m].get("mae", float("inf")))

        return best_model


# Global instance
_registry: Optional[HopsworksModelRegistry] = None


def get_model_registry() -> HopsworksModelRegistry:
    """Get or create the global model registry instance."""
    global _registry
    if _registry is None:
        _registry = HopsworksModelRegistry()
    return _registry
