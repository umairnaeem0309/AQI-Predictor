"""
Unit tests for prediction routes.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock

from app.backend.main import app
from app.services.model_service import ModelNotLoadedError, SyntheticModelRejectedError
from app.services.prediction_service import PredictionError


class TestPredictionRoutes:
    """Tests for prediction endpoints."""
    
    def test_prediction_endpoint_exists(self):
        """Test that prediction endpoint exists."""
        client = TestClient(app)
        response = client.post("/prediction", json={"city": "Karachi"})
        # Should get 401 or 503, not 404
        assert response.status_code != 404
    
    def test_prediction_invalid_city(self):
        """Test prediction with invalid city."""
        client = TestClient(app)
        response = client.post("/prediction", json={"city": "InvalidCity"})
        # Should get 400 Bad Request
        assert response.status_code == 400
    
    def test_prediction_missing_city(self):
        """Test prediction with missing city field."""
        client = TestClient(app)
        response = client.post("/prediction", json={})
        # Should get 422 Unprocessable Entity (Pydantic validation)
        assert response.status_code == 422
    
    def test_prediction_valid_cities(self):
        """Test that all valid cities are accepted."""
        client = TestClient(app)
        valid_cities = ["Karachi", "Lahore", "Islamabad"]
        
        for city in valid_cities:
            response = client.post("/prediction", json={"city": city})
            # Should not get 400 (might get 401 or 503)
            assert response.status_code != 400, f"City {city} should be valid"
    
    def test_prediction_model_not_loaded(self):
        """Test prediction when model not loaded."""
        client = TestClient(app)
        
        # Mock model service to raise error
        with patch("app.routes.prediction.get_prediction_service") as mock:
            mock_service = Mock()
            mock_service.predict.side_effect = ModelNotLoadedError("Model not loaded")
            mock.return_value = mock_service
            
            response = client.post(
                "/prediction",
                json={"city": "Karachi"},
                headers={"X-API-Key": "test-key"},
            )
            # Should get 503
            assert response.status_code == 503
    
    def test_prediction_synthetic_model_rejected(self):
        """Test prediction with synthetic model."""
        client = TestClient(app)
        
        with patch("app.routes.prediction.get_prediction_service") as mock:
            mock_service = Mock()
            mock_service.predict.side_effect = PredictionError(
                "Cannot load synthetic model"
            )
            mock.return_value = mock_service
            
            response = client.post(
                "/prediction",
                json={"city": "Karachi"},
                headers={"X-API-Key": "test-key"},
            )
            # Should get 500
            assert response.status_code == 500
    
    def test_prediction_rate_limit(self):
        """Test rate limiting on prediction endpoint."""
        client = TestClient(app)
        
        # Make multiple requests
        for i in range(5):
            response = client.post(
                "/prediction",
                json={"city": "Karachi"},
            )
            # Should succeed (rate limit not exceeded)
            assert response.status_code in [401, 503, 400]
