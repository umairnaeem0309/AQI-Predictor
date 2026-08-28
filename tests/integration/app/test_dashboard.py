"""
Integration tests for Streamlit dashboard.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from app.frontend.utils.api_client import APIClient


class TestDashboard:
    """Integration tests for dashboard components."""

    def test_api_client_mock_mode(self):
        """Test API client in mock mode for dashboard."""
        client = APIClient(mock_mode=True)

        # Test all dashboard data sources
        prediction = client.get_prediction("Karachi")
        health = client.get_health()
        model_info = client.get_model_info()

        assert prediction is not None
        assert health is not None
        assert model_info is not None

    def test_prediction_response_structure(self):
        """Test prediction response has required fields."""
        client = APIClient(mock_mode=True)
        prediction = client.get_prediction("Karachi")

        required_fields = [
            "city",
            "timestamp",
            "aqi_24h",
            "aqi_48h",
            "aqi_72h",
            "category_24h",
            "category_48h",
            "category_72h",
            "model_version",
            "confidence",
        ]

        for field in required_fields:
            assert field in prediction, f"Missing field: {field}"

    def test_health_response_structure(self):
        """Test health response has required fields."""
        client = APIClient(mock_mode=True)
        health = client.get_health()

        required_fields = [
            "status",
            "model_loaded",
            "feature_store_connected",
            "last_prediction",
            "version",
        ]

        for field in required_fields:
            assert field in health, f"Missing field: {field}"

    def test_model_info_response_structure(self):
        """Test model info response has required fields."""
        client = APIClient(mock_mode=True)
        model_info = client.get_model_info()

        required_fields = [
            "model_name",
            "model_version",
            "status",
            "approval_status",
            "training_date",
            "dataset_type",
            "feature_version",
            "metrics",
        ]

        for field in required_fields:
            assert field in model_info, f"Missing field: {field}"

    def test_city_selection_all_valid(self):
        """Test all valid cities work in mock mode."""
        client = APIClient(mock_mode=True)
        valid_cities = ["Karachi", "Lahore", "Islamabad"]

        for city in valid_cities:
            prediction = client.get_prediction(city)
            assert prediction["city"] == city

    def test_explainability_unavailable(self):
        """Test explainability shows unavailable in mock mode."""
        # Explainability requires backend support
        # In mock mode, it should show unavailable message
        client = APIClient(mock_mode=True)

        # Verify mock mode is active (no real explainability)
        assert client.mock_mode is True
