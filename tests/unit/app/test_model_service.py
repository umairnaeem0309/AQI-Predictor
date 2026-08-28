"""
Unit tests for model service.
"""

from unittest.mock import MagicMock, Mock

import pytest

from app.services.model_service import (
    ModelApprovalError,
    ModelNotLoadedError,
    ModelService,
    SyntheticModelRejectedError,
)


class TestModelService:
    """Tests for ModelService class."""

    def test_initialization(self):
        """Test model service initialization."""
        service = ModelService()
        assert service._model is None
        assert service._model_info is None

    def test_is_loaded_false_initially(self):
        """Test model not loaded initially."""
        service = ModelService()
        assert service.is_loaded() is False

    def test_get_model_not_loaded(self):
        """Test getting model when not loaded."""
        service = ModelService()
        with pytest.raises(ModelNotLoadedError):
            service.get_model()

    def test_get_model_info_not_loaded(self):
        """Test getting model info when not loaded."""
        service = ModelService()
        with pytest.raises(ModelNotLoadedError):
            service.get_model_info()

    def test_validate_model_for_request_not_loaded(self):
        """Test validation fails when model not loaded."""
        service = ModelService()
        with pytest.raises(ModelNotLoadedError):
            service.validate_model_for_request()

    def test_load_synthetic_model_rejected(self):
        """Test that synthetic models are rejected."""
        mock_registry = Mock()
        mock_registry.get_production_model.return_value = {
            "status": "production",
            "approval_status": "approved",
            "dataset_type": "synthetic_test_data",  # This should be rejected
            "version": "1.0.0",
        }

        service = ModelService(registry=mock_registry)

        with pytest.raises(SyntheticModelRejectedError, match="synthetic"):
            service.load_production_model()

    def test_load_unapproved_model_rejected(self):
        """Test that unapproved models are rejected."""
        mock_registry = Mock()
        mock_registry.get_production_model.return_value = {
            "status": "production",
            "approval_status": "not_approved",  # This should be rejected
            "dataset_type": "real_api_data",
            "version": "1.0.0",
        }

        service = ModelService(registry=mock_registry)

        with pytest.raises(ModelApprovalError, match="not approved"):
            service.load_production_model()

    def test_load_non_production_status_rejected(self):
        """Test that non-production status is rejected."""
        mock_registry = Mock()
        mock_registry.get_production_model.return_value = {
            "status": "candidate",  # Not production
            "approval_status": "approved",
            "dataset_type": "real_api_data",
            "version": "1.0.0",
        }

        service = ModelService(registry=mock_registry)

        with pytest.raises(ModelNotLoadedError, match="status is candidate"):
            service.load_production_model()

    def test_load_production_model_success(self):
        """Test successful production model loading."""
        mock_registry = Mock()
        mock_registry.get_production_model.return_value = {
            "status": "production",
            "approval_status": "approved",
            "dataset_type": "real_api_data",
            "version": "1.0.0",
            "model_name": "xgboost_v1",
            "artifact_path": "/path/to/model",
        }
        mock_registry.load_model.return_value = Mock()  # Mock model

        service = ModelService(registry=mock_registry)
        model, info = service.load_production_model()

        assert model is not None
        assert info["dataset_type"] == "real_api_data"
        assert service.is_loaded() is True

    def test_validate_model_for_request_success(self):
        """Test validation passes when model is loaded."""
        service = ModelService()
        service._model = Mock()  # Simulate loaded model

        # Should not raise
        service.validate_model_for_request()
