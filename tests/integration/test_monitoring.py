"""
Integration tests for monitoring pipeline.
"""

import json
import tempfile
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift_detection import DriftDetector
from src.monitoring.performance import PerformanceMonitor
from src.monitoring.alerting import AlertManager, AlertLevel, AlertType
from src.monitoring.notification import LogNotifier, ConsoleNotifier
from src.monitoring.baseline_manager import BaselineManager


class TestMonitoringPipeline:
    """Integration tests for monitoring pipeline."""
    
    def test_drift_detection_pipeline(self):
        """Test complete drift detection pipeline."""
        detector = DriftDetector()
        
        # Create reference and current data
        np.random.seed(42)
        reference = pd.DataFrame({
            "temperature": np.random.randn(100) * 10 + 30,
            "humidity": np.random.randn(100) * 5 + 60,
            "aqi": np.random.randint(0, 200, 100),
        })
        current = reference.copy()
        
        # Run drift detection
        report = detector.detect_drift(
            reference, current,
            dataset_type="real_api_data",
            baseline_version="1.0",
            feature_version="1.0",
            model_version="1.0",
        )
        
        # Verify report
        assert report.dataset_type == "real_api_data"
        assert report.baseline_version == "1.0"
        assert len(report.drift_results) > 0
        
        # Save and reload
        with tempfile.TemporaryDirectory() as tmpdir:
            saved_path = detector.save_report(report, Path(tmpdir))
            loaded_report = detector.load_report(saved_path)
            
            assert loaded_report.report_id == report.report_id
    
    def test_performance_monitoring_pipeline(self):
        """Test complete performance monitoring pipeline."""
        monitor = PerformanceMonitor()
        
        # Create sample predictions
        np.random.seed(42)
        df = pd.DataFrame({
            "timestamp": pd.date_range("2026-08-01", periods=100, freq="h"),
            "aqi_24h": np.random.randint(50, 150, 100),
            "actual_24h": np.random.randint(50, 150, 100),
            "aqi_48h": np.random.randint(50, 150, 100),
            "actual_48h": np.random.randint(50, 150, 100),
        })
        
        baseline = {"mae": 20.0, "rmse": 25.0, "r2": 0.7}
        
        # Generate report
        report = monitor.generate_report(
            df, baseline,
            dataset_type="real_api_data",
            feature_version="1.0",
            model_version="1.0",
            city="Karachi",
        )
        
        # Verify report
        assert report.dataset_type == "real_api_data"
        assert len(report.metrics) > 0
        
        # Save report
        with tempfile.TemporaryDirectory() as tmpdir:
            saved_path = monitor.save_report(report, Path(tmpdir))
            assert saved_path.exists()
    
    def test_alerting_pipeline(self):
        """Test complete alerting pipeline with notification."""
        manager = AlertManager(default_cooldown_minutes=0)  # No cooldown for test
        
        notifications = []
        
        def test_notifier(alert):
            notifications.append(alert)
        
        manager.add_notifier(test_notifier)
        
        # Fire alerts
        for i in range(3):
            manager.fire_alert(
                alert_type=AlertType.DATA_DRIFT,
                level=AlertLevel.WARNING,
                message=f"Drift alert {i}",
                details={"drift_score": 0.1 + i * 0.05},
                city="Karachi",
                force=True,
            )
        
        # Verify notifications
        assert len(notifications) == 3
        
        # Aggregate alerts
        aggregates = manager.aggregate_alerts(notifications)
        assert len(aggregates) == 1
        assert aggregates[0].count == 3
    
    def test_baseline_manager_synthetic_rejection(self):
        """Test that synthetic data is rejected for baselines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = BaselineManager(Path(tmpdir))
            
            data = pd.DataFrame({
                "temperature": np.random.randn(100) * 10 + 30,
            })
            
            # Should reject synthetic data
            with pytest.raises(ValueError, match="synthetic"):
                manager.create_baseline(
                    data,
                    baseline_type="training",
                    dataset_type="synthetic_test_data",
                )
            
            # Should accept real data
            metadata = manager.create_baseline(
                data,
                baseline_type="training",
                dataset_type="real_api_data",
            )
            
            assert metadata.dataset_type == "real_api_data"
    
    def test_baseline_manager_load_synthetic_rejection(self):
        """Test that synthetic baselines cannot be loaded for monitoring."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = BaselineManager(Path(tmpdir))
            
            data = pd.DataFrame({
                "temperature": np.random.randn(100) * 10 + 30,
            })
            
            # Create baseline directly with synthetic type (bypass check for test)
            metadata = manager.create_baseline(
                data,
                baseline_type="training",
                dataset_type="real_api_data",
            )
            
            # Manually modify metadata to simulate synthetic
            metadata_path = Path(tmpdir) / metadata.baseline_id / "metadata.json"
            with open(metadata_path, "r") as f:
                meta_dict = json.load(f)
            meta_dict["dataset_type"] = "synthetic_test_data"
            with open(metadata_path, "w") as f:
                json.dump(meta_dict, f)
            
            # Should reject loading synthetic baseline
            with pytest.raises(ValueError, match="synthetic"):
                manager.load_baseline(metadata.baseline_id, reject_synthetic=True)
    
    def test_monitoring_metadata_validation(self):
        """Test that monitoring reports include required metadata."""
        detector = DriftDetector()
        
        np.random.seed(42)
        reference = pd.DataFrame({"temperature": np.random.randn(100)})
        current = pd.DataFrame({"temperature": np.random.randn(100)})
        
        report = detector.detect_drift(
            reference, current,
            dataset_type="real_api_data",
            baseline_version="1.0",
            feature_version="1.0",
            model_version="1.0",
        )
        
        # Verify metadata fields
        assert report.dataset_type == "real_api_data"
        assert report.baseline_version == "1.0"
        assert report.feature_version == "1.0"
        assert report.model_version == "1.0"
        assert report.generated_at != ""
    
    def test_notification_abstraction(self):
        """Test notification abstraction layer."""
        # Test log notifier
        log_notifier = LogNotifier()
        assert log_notifier.logger is not None
        
        # Test console notifier
        console_notifier = ConsoleNotifier(use_colors=False)
        assert console_notifier.use_colors is False
    
    def test_full_monitoring_workflow(self):
        """Test complete monitoring workflow."""
        # Initialize components
        detector = DriftDetector()
        monitor = PerformanceMonitor()
        alert_manager = AlertManager(default_cooldown_minutes=0)
        baseline_manager = BaselineManager(Path(tempfile.mkdtemp()))
        
        # Add notification
        notifications = []
        alert_manager.add_notifier(lambda a: notifications.append(a))
        
        # Create baseline
        np.random.seed(42)
        baseline_data = pd.DataFrame({
            "temperature": np.random.randn(100) * 10 + 30,
            "humidity": np.random.randn(100) * 5 + 60,
        })
        
        baseline_metadata = baseline_manager.create_baseline(
            baseline_data,
            baseline_type="training",
            dataset_type="real_api_data",
        )
        
        # Load baseline
        loaded_metadata, loaded_data = baseline_manager.load_baseline(
            baseline_metadata.baseline_id
        )
        
        assert loaded_metadata.dataset_type == "real_api_data"
        
        # Run drift detection
        current_data = baseline_data.copy()
        drift_report = detector.detect_drift(
            baseline_data, current_data,
            dataset_type="real_api_data",
        )
        
        # Check for drift and fire alerts if needed
        if drift_report.overall_drift_detected:
            alert_manager.fire_alert(
                alert_type=AlertType.DATA_DRIFT,
                level=AlertLevel.WARNING,
                message="Data drift detected",
                details={"drift_summary": drift_report.drift_summary},
                city="all",
            )
        
        # Verify workflow completed
        assert drift_report.dataset_type == "real_api_data"
        assert len(notifications) == 0  # No drift, no alerts
