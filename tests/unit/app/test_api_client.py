"""
Unit tests for API client.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from app.frontend.utils.api_client import APIClient, APIClientError, APIConnectionError


class TestAPIClient:
    """Tests for API client."""

    def test_initialization(self):
        """Test client initialization."""
        client = APIClient(
            base_url="http://localhost:8000",
            api_key="test-key",
            mock_mode=False,
        )
        assert client.base_url == "http://localhost:8000"
        assert client.api_key == "test-key"
        assert client.mock_mode is False

    def test_mock_mode_enabled(self):
        """Test mock mode returns mock data."""
        client = APIClient(mock_mode=True)

        prediction = client.get_prediction("Karachi")
        assert "aqi_24h" in prediction
        assert prediction["city"] == "Karachi"

    def test_mock_health(self):
        """Test mock health response."""
        client = APIClient(mock_mode=True)

        health = client.get_health()
        assert health["status"] == "healthy"
        assert health["model_loaded"] is True

    def test_mock_model_info(self):
        """Test mock model info response."""
        client = APIClient(mock_mode=True)

        model_info = client.get_model_info()
        assert "model_name" in model_info
        assert model_info["status"] == "production"

    def test_production_connection_error(self):
        """Test connection error in production mode."""
        client = APIClient(
            base_url="http://invalid-url:9999",
            mock_mode=False,
        )

        with pytest.raises(APIConnectionError):
            client.get_prediction("Karachi")

    def test_is_available_mock_mode(self):
        """Test availability check in mock mode."""
        client = APIClient(mock_mode=True)
        assert client.is_available() is True

    def test_is_available_production(self):
        """Test availability check in production mode."""
        client = APIClient(
            base_url="http://invalid-url:9999",
            mock_mode=False,
        )
        assert client.is_available() is False

    def test_mock_prediction_values(self):
        """Test mock prediction returns reasonable values."""
        client = APIClient(mock_mode=True)

        prediction = client.get_prediction("Karachi")
        assert 50 <= prediction["aqi_24h"] <= 300
        assert prediction["confidence"] is None  # Null until implemented
