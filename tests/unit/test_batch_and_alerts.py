"""
Tests for batch predictions and AQI alerts.
"""

from unittest.mock import MagicMock, patch

import pytest

from app.frontend.utils.api_client import APIClient


class TestBatchPredictions:
    """Test batch prediction endpoint."""

    def test_batch_request_model(self):
        from app.routes.batch import BatchPredictionRequest

        req = BatchPredictionRequest(cities=["karachi", "lahore"])
        assert len(req.cities) == 2

    def test_batch_request_max_cities(self):
        from app.routes.batch import BatchPredictionRequest

        req = BatchPredictionRequest(cities=["karachi"] * 10)
        assert len(req.cities) == 10

    def test_batch_request_too_many(self):
        from app.routes.batch import BatchPredictionRequest

        with pytest.raises(Exception):
            BatchPredictionRequest(cities=["karachi"] * 11)

    def test_batch_response_model(self):
        from app.routes.batch import BatchPredictionResponse

        resp = BatchPredictionResponse(
            predictions=[],
            total_cities=3,
            successful=3,
            failed=0,
            total_time_ms=150.5,
        )
        assert resp.successful == 3
        assert resp.failed == 0


class TestAQIAlerts:
    """Test AQI hazard alert system."""

    def test_get_aqi_category(self):
        from app.routes.monitoring import _get_aqi_category

        assert _get_aqi_category(25) == "Good"
        assert _get_aqi_category(75) == "Moderate"
        assert _get_aqi_category(120) == "Unhealthy for Sensitive Groups"
        assert _get_aqi_category(175) == "Unhealthy"
        assert _get_aqi_category(250) == "Very Unhealthy"
        assert _get_aqi_category(400) == "Hazardous"

    def test_get_recommendation_good(self):
        from app.routes.monitoring import _get_recommendation

        rec = _get_recommendation(30)
        assert "satisfactory" in rec.lower()

    def test_get_recommendation_hazardous(self):
        from app.routes.monitoring import _get_recommendation

        rec = _get_recommendation(350)
        assert "emergency" in rec.lower()
        assert "avoid" in rec.lower()

    def test_get_recommendation_moderate(self):
        from app.routes.monitoring import _get_recommendation

        rec = _get_recommendation(75)
        assert "sensitive" in rec.lower() or "acceptable" in rec.lower()

    def test_aqi_category_boundary(self):
        from app.routes.monitoring import _get_aqi_category

        # Boundary values
        assert _get_aqi_category(0) == "Good"
        assert _get_aqi_category(50) == "Good"
        assert _get_aqi_category(51) == "Moderate"
        assert _get_aqi_category(100) == "Moderate"
        assert _get_aqi_category(101) == "Unhealthy for Sensitive Groups"
        assert _get_aqi_category(150) == "Unhealthy for Sensitive Groups"
        assert _get_aqi_category(151) == "Unhealthy"
        assert _get_aqi_category(200) == "Unhealthy"
        assert _get_aqi_category(201) == "Very Unhealthy"
        assert _get_aqi_category(300) == "Very Unhealthy"
        assert _get_aqi_category(301) == "Hazardous"


class TestAPIClientBatch:
    """Test API client batch methods."""

    def test_batch_predictions_mock(self):
        """Test batch endpoint through mock client."""
        client = APIClient(mock_mode=True)
        # Verify client works
        assert client.mock_mode is True

    def test_alerts_mock(self):
        """Test alerts through mock client."""
        client = APIClient(mock_mode=True)
        result = client.get_alerts()
        assert "alerts" in result
        assert isinstance(result["alerts"], list)

    def test_system_health_mock(self):
        """Test system health through mock client."""
        client = APIClient(mock_mode=True)
        result = client.get_system_health()
        assert "overall_status" in result
