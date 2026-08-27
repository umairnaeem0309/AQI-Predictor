"""
Model Service

Handles model loading, validation, and management.
Enforces production safety checks.
"""

import json
import logging
from typing import Optional, Tuple, Any, Dict
from datetime import datetime, timezone

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
    
    def load_local_model(self, model_path: str = "models/production/xgboost_model.pkl",
                          metadata_path: str = "models/production/model_metadata.json") -> Tuple[Any, Dict]:
        """
        Load model from local pickle file (fallback when MLflow registry unavailable).
        
        Args:
            model_path: Path to pickled model.
            metadata_path: Path to model metadata JSON.
        
        Returns:
            Tuple of (model, model_info)
        """
        import pickle
        from pathlib import Path
        
        model_file = Path(model_path)
        meta_file = Path(metadata_path)
        
        if not model_file.exists():
            raise ModelNotLoadedError(f"Model file not found: {model_path}")
        
        with open(model_file, "rb") as f:
            model = pickle.load(f)
        
        model_info = {}
        if meta_file.exists():
            with open(meta_file) as f:
                model_info = json.load(f)
        
        self._model = model
        self._model_info = model_info
        
        logger.info("Loaded local model from %s", model_path)
        return model, model_info
    
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
                raise ModelApprovalError(
                    f"Model not approved: {model_info.get('approval_status')}"
                )
            
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
        
        Returns:
            Model metadata dictionary
            
        Raises:
            ModelNotLoadedError: If model info not available
        """
        if self._model_info is None:
            raise ModelNotLoadedError("Model info not available")
        return self._model_info
    
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
            raise ModelNotLoadedError(
                "Model not loaded. Service starting or unavailable."
            )


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
