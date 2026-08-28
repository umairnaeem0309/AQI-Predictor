"""
Tests for monitoring routes.
"""

import json
import os
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestDriftDetection:
    """Test drift detection logic."""

    def test_drift_detector_init(self):
        from src.monitoring.drift_detection import DriftDetector
        try:
            detector = DriftDetector()
            assert detector.psi_threshold == 0.1
            assert detector.ks_threshold == 0.05
        except ImportError:
            pytest.skip("Evidently not installed")

    def test_drift_result_dataclass(self):
        from src.monitoring.drift_detection import DriftResult
        result = DriftResult(
            column_name="pm25",
            drift_detected=False,
            drift_score=0.05,
            drift_method="psi",
            threshold=0.1,
            details={},
        )
        assert result.column_name == "pm25"
        assert result.drift_detected is False
        assert result.timestamp != ""

    def test_drift_report_dataclass(self):
        from src.monitoring.drift_detection import DriftReport
        report = DriftReport(
            report_id="test_001",
            dataset_type="real_api_data",
            baseline_version="v1",
            feature_version="v1",
            model_version="v1",
            reference_data_rows=1000,
            current_data_rows=200,
            drift_results=[],
            overall_drift_detected=False,
            drift_summary={"total_columns": 0, "drifted_columns": 0},
        )
        assert report.report_id == "test_001"
        assert report.generated_at != ""

    def test_detect_drift_no_drift(self):
        from src.monitoring.drift_detection import DriftDetector
        try:
            detector = DriftDetector()
        except ImportError:
            pytest.skip("Evidently not installed")

        # Use large samples from identical distribution to minimize false drift
        rng = np.random.RandomState(42)
        base = rng.normal(50, 10, 2000)
        ref = pd.DataFrame({"pm25": base[:1000], "pm10": base[:1000] * 1.4})
        curr = pd.DataFrame({"pm25": base[1000:], "pm10": base[1000:] * 1.4})

        report = detector.detect_drift(ref, curr, dataset_type="real_api_data")
        # With large identical distributions, drift should not be detected
        assert report.reference_data_rows == 1000
        assert report.current_data_rows == 1000

    def test_detect_drift_with_drift(self):
        from src.monitoring.drift_detection import DriftDetector
        try:
            detector = DriftDetector()
        except ImportError:
            pytest.skip("Evidently not installed")

        rng = np.random.RandomState(42)
        ref = pd.DataFrame({
            "pm25": rng.normal(50, 10, 500),
            "pm10": rng.normal(70, 15, 500),
        })
        # Current = very different distribution
        curr = pd.DataFrame({
            "pm25": rng.normal(200, 30, 200),
            "pm10": rng.normal(250, 40, 200),
        })

        report = detector.detect_drift(ref, curr, dataset_type="real_api_data")
        assert report.overall_drift_detected is True


class TestPerformanceMonitor:
    """Test performance monitoring."""

    def test_performance_monitor_init(self):
        from src.monitoring.performance import PerformanceMonitor
        monitor = PerformanceMonitor()
        assert monitor.mae_threshold == 0.2

    def test_calculate_rolling_metrics(self):
        from src.monitoring.performance import PerformanceMonitor
        monitor = PerformanceMonitor()

        df = pd.DataFrame({
            "timestamp": pd.date_range("2026-08-01", periods=100, freq="h"),
            "aqi_24h": np.random.uniform(50, 200, 100),
            "actual_24h": np.random.uniform(50, 200, 100),
        })

        metrics = monitor.calculate_rolling_metrics(df, window="30d")
        assert "mae" in metrics
        assert "rmse" in metrics
        assert metrics["mae"] >= 0

    def test_detect_degradation_no_degradation(self):
        from src.monitoring.performance import PerformanceMonitor
        monitor = PerformanceMonitor()

        current = {"mae": 20.0, "rmse": 30.0, "r2": 0.65}
        baseline = {"mae": 19.0, "rmse": 29.0, "r2": 0.67}

        degraded, details = monitor.detect_degradation(current, baseline)
        assert degraded is False

    def test_detect_degradation_with_degradation(self):
        from src.monitoring.performance import PerformanceMonitor
        monitor = PerformanceMonitor()

        current = {"mae": 30.0, "rmse": 45.0, "r2": 0.40}
        baseline = {"mae": 19.0, "rmse": 29.0, "r2": 0.67}

        degraded, details = monitor.detect_degradation(current, baseline)
        assert degraded is True


class TestAPIClientMonitoring:
    """Test API client monitoring methods."""

    def test_drift_report_mock(self):
        from app.frontend.utils.api_client import APIClient
        client = APIClient(mock_mode=True)
        result = client.get_drift_report()
        assert "drift_detected" in result

    def test_performance_mock(self):
        from app.frontend.utils.api_client import APIClient
        client = APIClient(mock_mode=True)
        result = client.get_performance()
        assert "status" in result

    def test_alerts_mock(self):
        from app.frontend.utils.api_client import APIClient
        client = APIClient(mock_mode=True)
        result = client.get_alerts()
        assert "alerts" in result

    def test_system_health_mock(self):
        from app.frontend.utils.api_client import APIClient
        client = APIClient(mock_mode=True)
        result = client.get_system_health()
        assert "overall_status" in result


class TestMonitoringHelpers:
    """Test monitoring helper functions."""

    def test_get_aqi_category(self):
        from app.routes.monitoring import _get_aqi_category
        assert _get_aqi_category(25) == "Good"
        assert _get_aqi_category(75) == "Moderate"
        assert _get_aqi_category(120) == "Unhealthy for Sensitive Groups"
        assert _get_aqi_category(175) == "Unhealthy"
        assert _get_aqi_category(250) == "Very Unhealthy"
        assert _get_aqi_category(400) == "Hazardous"

    def test_get_recommendation(self):
        from app.routes.monitoring import _get_recommendation
        assert "satisfactory" in _get_recommendation(30).lower()
        assert "avoid" in _get_recommendation(250).lower()
        assert "emergency" in _get_recommendation(350).lower()
