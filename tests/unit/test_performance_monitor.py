"""
Unit tests for performance monitoring module.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.monitoring.performance import (
    PerformanceMetric,
    PerformanceMonitor,
    PerformanceReport,
)


class TestPerformanceMonitor:
    """Tests for PerformanceMonitor class."""

    def test_initialization(self):
        """Test performance monitor initialization."""
        monitor = PerformanceMonitor(
            mae_threshold=0.2,
            rmse_threshold=0.2,
            r2_threshold=0.1,
        )
        assert monitor.mae_threshold == 0.2
        assert monitor.rmse_threshold == 0.2
        assert monitor.r2_threshold == 0.1

    def test_initialization_default_thresholds(self):
        """Test default threshold values."""
        monitor = PerformanceMonitor()
        assert monitor.mae_threshold == PerformanceMonitor.DEFAULT_MAE_INCREASE_THRESHOLD
        assert monitor.rmse_threshold == PerformanceMonitor.DEFAULT_RMSE_INCREASE_THRESHOLD
        assert monitor.r2_threshold == PerformanceMonitor.DEFAULT_R2_DECREASE_THRESHOLD

    def test_calculate_rolling_metrics(self):
        """Test rolling metrics calculation."""
        monitor = PerformanceMonitor()

        # Create sample predictions
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-08-01", periods=100, freq="h"),
                "aqi_24h": np.random.randint(50, 150, 100),
                "actual_24h": np.random.randint(50, 150, 100),
                "aqi_48h": np.random.randint(50, 150, 100),
                "actual_48h": np.random.randint(50, 150, 100),
                "aqi_72h": np.random.randint(50, 150, 100),
                "actual_72h": np.random.randint(50, 150, 100),
            }
        )

        metrics = monitor.calculate_rolling_metrics(df, window="24h")

        assert "mae" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics
        assert metrics["mae"] >= 0
        assert metrics["rmse"] >= 0

    def test_calculate_rolling_metrics_empty(self):
        """Test rolling metrics with empty data."""
        monitor = PerformanceMonitor()

        df = pd.DataFrame()
        metrics = monitor.calculate_rolling_metrics(df)

        assert metrics["mae"] == 0.0
        assert metrics["rmse"] == 0.0
        assert metrics["r2"] == 0.0

    def test_detect_degradation_no_degradation(self):
        """Test degradation detection with stable performance."""
        monitor = PerformanceMonitor()

        current = {"mae": 10.0, "rmse": 15.0, "r2": 0.8}
        baseline = {"mae": 10.0, "rmse": 15.0, "r2": 0.8}

        degradation, details = monitor.detect_degradation(current, baseline)

        assert not degradation
        assert "mae_change" in details
        assert details["mae_change"] == 0.0

    def test_detect_degradation_with_degradation(self):
        """Test degradation detection with performance drop."""
        monitor = PerformanceMonitor(mae_threshold=0.2)

        current = {"mae": 15.0, "rmse": 20.0, "r2": 0.6}
        baseline = {"mae": 10.0, "rmse": 15.0, "r2": 0.8}

        degradation, details = monitor.detect_degradation(current, baseline)

        assert degradation
        assert details["mae_change"] == 0.5  # 50% increase
        assert details["mae_degraded"] is True

    def test_performance_metric_creation(self):
        """Test PerformanceMetric dataclass creation."""
        metric = PerformanceMetric(
            metric_name="mae",
            value=10.5,
            window="24h",
            horizon="24h",
            city="Karachi",
        )

        assert metric.metric_name == "mae"
        assert metric.value == 10.5
        assert metric.timestamp != ""

    def test_performance_report_creation(self):
        """Test PerformanceReport dataclass creation."""
        report = PerformanceReport(
            report_id="test_report",
            dataset_type="real_api_data",
            baseline_version="1.0",
            feature_version="1.0",
            model_version="1.0",
            metrics=[],
            degradation_detected=False,
            degradation_summary={},
        )

        assert report.report_id == "test_report"
        assert report.generated_at != ""

    def test_generate_report(self):
        """Test full report generation."""
        monitor = PerformanceMonitor()

        np.random.seed(42)
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-08-01", periods=100, freq="h"),
                "aqi_24h": np.random.randint(50, 150, 100),
                "actual_24h": np.random.randint(50, 150, 100),
            }
        )

        baseline = {"mae": 20.0, "rmse": 25.0, "r2": 0.7}

        report = monitor.generate_report(
            df,
            baseline,
            dataset_type="real_api_data",
            city="Karachi",
        )

        assert isinstance(report, PerformanceReport)
        assert len(report.metrics) > 0

    def test_save_report(self):
        """Test saving performance report."""
        monitor = PerformanceMonitor()

        report = PerformanceReport(
            report_id="test_save",
            dataset_type="real_api_data",
            baseline_version="1.0",
            feature_version="1.0",
            model_version="1.0",
            metrics=[],
            degradation_detected=False,
            degradation_summary={},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            saved_path = monitor.save_report(report, output_dir)

            assert saved_path.exists()

            # Verify file content
            with open(saved_path, "r") as f:
                data = json.load(f)

            assert data["report_id"] == "test_save"
