"""
Unit tests for drift detection module.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift_detection import DriftDetector, DriftReport, DriftResult


class TestDriftDetector:
    """Tests for DriftDetector class."""

    def test_initialization(self):
        """Test drift detector initialization."""
        detector = DriftDetector(psi_threshold=0.1, ks_threshold=0.05)
        assert detector.psi_threshold == 0.1
        assert detector.ks_threshold == 0.05

    def test_initialization_default_thresholds(self):
        """Test default threshold values."""
        detector = DriftDetector()
        assert detector.psi_threshold == DriftDetector.DEFAULT_PSI_THRESHOLD
        assert detector.ks_threshold == DriftDetector.DEFAULT_KS_THRESHOLD

    def test_detect_drift_no_drift(self):
        """Test drift detection with similar distributions."""
        detector = DriftDetector()

        # Create similar datasets
        np.random.seed(42)
        reference = pd.DataFrame(
            {
                "temperature": np.random.randn(100) * 10 + 30,
                "humidity": np.random.randn(100) * 5 + 60,
            }
        )
        current = reference.copy()

        report = detector.detect_drift(reference, current)

        assert isinstance(report, DriftReport)
        assert not report.overall_drift_detected
        assert len(report.drift_results) > 0

    def test_detect_drift_with_drift(self):
        """Test drift detection with different distributions."""
        detector = DriftDetector()

        # Create datasets with different distributions
        np.random.seed(42)
        reference = pd.DataFrame(
            {
                "temperature": np.random.randn(100) * 10 + 30,
            }
        )
        current = pd.DataFrame(
            {
                "temperature": np.random.randn(100) * 10 + 50,  # Shifted mean
            }
        )

        report = detector.detect_drift(reference, current)

        assert isinstance(report, DriftReport)
        assert len(report.drift_results) > 0

    def test_detect_drift_empty_data(self):
        """Test drift detection with empty data."""
        detector = DriftDetector()

        reference = pd.DataFrame({"temperature": []})
        current = pd.DataFrame({"temperature": []})

        with pytest.raises(ValueError, match="cannot be empty"):
            detector.detect_drift(reference, current)

    def test_drift_result_creation(self):
        """Test DriftResult dataclass creation."""
        result = DriftResult(
            column_name="temperature",
            drift_detected=True,
            drift_score=0.15,
            drift_method="psi",
            threshold=0.1,
            details={"stattest": "psi"},
        )

        assert result.column_name == "temperature"
        assert result.drift_detected is True
        assert result.drift_score == 0.15
        assert result.timestamp != ""

    def test_drift_report_creation(self):
        """Test DriftReport dataclass creation."""
        report = DriftReport(
            report_id="test_report",
            dataset_type="real_api_data",
            baseline_version="1.0",
            feature_version="1.0",
            model_version="1.0",
            reference_data_rows=100,
            current_data_rows=100,
            drift_results=[],
            overall_drift_detected=False,
            drift_summary={"drifted_columns": 0},
        )

        assert report.report_id == "test_report"
        assert report.dataset_type == "real_api_data"
        assert report.generated_at != ""

    def test_save_and_load_report(self):
        """Test saving and loading drift report."""
        detector = DriftDetector()

        # Create report
        report = DriftReport(
            report_id="test_save",
            dataset_type="real_api_data",
            baseline_version="1.0",
            feature_version="1.0",
            model_version="1.0",
            reference_data_rows=100,
            current_data_rows=100,
            drift_results=[
                DriftResult(
                    column_name="temp",
                    drift_detected=False,
                    drift_score=0.05,
                    drift_method="psi",
                    threshold=0.1,
                    details={},
                )
            ],
            overall_drift_detected=False,
            drift_summary={},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            output_dir = Path(tmpdir)
            saved_path = detector.save_report(report, output_dir)

            assert saved_path.exists()

            # Load
            loaded_report = detector.load_report(saved_path)

            assert loaded_report.report_id == report.report_id
            assert loaded_report.dataset_type == report.dataset_type
            assert len(loaded_report.drift_results) == 1
