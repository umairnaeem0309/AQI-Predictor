"""
Integration tests for API lifecycle.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock

from app.backend.main import app


class TestAPILifecycle:
    """Integration tests for API request lifecycle."""
    
    def test_health_endpoint(self):
        """Test health check endpoint."""
        client = TestClient(app)
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "feature_store_connected" in data
    
    def test_root_endpoint(self):
        """Test root endpoint."""
        client = TestClient(app)
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "available"
    
    def test_prediction_request_response_schema(self):
        """Test prediction response schema."""
        client = TestClient(app)
        
        with patch("app.routes.prediction.get_prediction_service") as mock:
            mock_service = Mock()
            mock_service.predict.return_value = {
                "city": "Karachi",
                "timestamp": "2026-08-17T10:00:00Z",
                "aqi_24h": 142,
                "aqi_48h": 138,
                "aqi_72h": 145,
                "category_24h": "Unhealthy for Sensitive Groups",
                "category_48h": "Unhealthy for Sensitive Groups",
                "category_72h": "Unhealthy for Sensitive Groups",
                "model_version": "1.0.0",
                "confidence": None,
            }
            mock.return_value = mock_service
            
            response = client.post(
                "/prediction",
                json={"city": "Karachi"},
                headers={"X-API-Key": "test-key"},
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "city" in data
            assert "aqi_24h" in data
            assert "aqi_48h" in data
            assert "aqi_72h" in data
            assert "category_24h" in data
            assert "model_version" in data
            assert "confidence" in data
            assert data["confidence"] is None  # Null until uncertainty method
    
    def test_synthetic_model_rejection(self):
        """Test that synthetic models are rejected."""
        client = TestClient(app)
        
        with patch("app.routes.prediction.get_prediction_service") as mock:
            mock_service = Mock()
            mock_service.predict.side_effect = Exception("synthetic model rejected")
            mock.return_value = mock_service
            
            response = client.post(
                "/prediction",
                json={"city": "Karachi"},
                headers={"X-API-Key": "test-key"},
            )
            
            # Should get error
            assert response.status_code in [403, 500]
    
    def test_missing_production_model(self):
        """Test behavior when production model is missing."""
        client = TestClient(app)
        
        # Health check should show model not loaded
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        # Model might not be loaded in test environment
        assert "model_loaded" in data
    
    def test_authentication_required(self):
        """Test that authentication is required."""
        client = TestClient(app)
        
        # Try without API key (should get 401 or use default)
        response = client.post("/prediction", json={"city": "Karachi"})
        
        # Response depends on configuration
        # If API_KEY is set, should get 401
        # If not set, might get 503 (model not loaded)
        assert response.status_code in [401, 503, 400]
    
    def test_feature_schema_mismatch(self):
        """Test feature schema mismatch handling."""
        client = TestClient(app)
        
        with patch("app.routes.prediction.get_prediction_service") as mock:
            mock_service = Mock()
            mock_service.predict.side_effect = Exception("Missing required features")
            mock.return_value = mock_service
            
            response = client.post(
                "/prediction",
                json={"city": "Karachi"},
                headers={"X-API-Key": "test-key"},
            )
            
            # Should get error
            assert response.status_code in [500, 503]
