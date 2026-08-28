"""
Unit tests for alerting module.
"""

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.monitoring.alerting import (
    Alert,
    AlertAggregate,
    AlertLevel,
    AlertManager,
    AlertRule,
    AlertType,
)


class TestAlertManager:
    """Tests for AlertManager class."""

    def test_initialization(self):
        """Test alert manager initialization."""
        manager = AlertManager(
            default_cooldown_minutes=60,
            aggregation_window_minutes=15,
        )
        assert manager.default_cooldown_minutes == 60
        assert manager.aggregation_window_minutes == 15

    def test_fire_alert(self):
        """Test firing an alert."""
        manager = AlertManager()

        alert = manager.fire_alert(
            alert_type=AlertType.DATA_DRIFT,
            level=AlertLevel.WARNING,
            message="Drift detected in temperature",
            details={"column": "temperature", "drift_score": 0.15},
            city="Karachi",
        )

        assert alert is not None
        assert alert.alert_type == AlertType.DATA_DRIFT
        assert alert.level == AlertLevel.WARNING
        assert alert.city == "Karachi"
        assert len(manager.active_alerts) == 1

    def test_cooldown_prevents_duplicate_alerts(self):
        """Test that cooldown prevents duplicate alerts."""
        manager = AlertManager(default_cooldown_minutes=60)

        # Fire first alert
        alert1 = manager.fire_alert(
            alert_type=AlertType.DATA_DRIFT,
            level=AlertLevel.WARNING,
            message="First alert",
            details={},
            city="Karachi",
        )
        assert alert1 is not None

        # Try to fire same alert immediately (should be blocked)
        alert2 = manager.fire_alert(
            alert_type=AlertType.DATA_DRIFT,
            level=AlertLevel.WARNING,
            message="Second alert",
            details={},
            city="Karachi",
        )
        assert alert2 is None  # Blocked by cooldown

        # Fire different alert (should work)
        alert3 = manager.fire_alert(
            alert_type=AlertType.MODEL_PERFORMANCE,
            level=AlertLevel.WARNING,
            message="Different alert type",
            details={},
            city="Karachi",
        )
        assert alert3 is not None

    def test_force_alert_bypasses_cooldown(self):
        """Test that force=True bypasses cooldown."""
        manager = AlertManager(default_cooldown_minutes=60)

        # Fire first alert
        manager.fire_alert(
            alert_type=AlertType.DATA_DRIFT,
            level=AlertLevel.WARNING,
            message="First alert",
            details={},
            city="Karachi",
        )

        # Force second alert
        alert = manager.fire_alert(
            alert_type=AlertType.DATA_DRIFT,
            level=AlertLevel.WARNING,
            message="Forced alert",
            details={},
            city="Karachi",
            force=True,
        )

        assert alert is not None

    def test_different_cities_bypass_cooldown(self):
        """Test that different cities bypass cooldown."""
        manager = AlertManager(default_cooldown_minutes=60)

        # Fire alert for Karachi
        manager.fire_alert(
            alert_type=AlertType.DATA_DRIFT,
            level=AlertLevel.WARNING,
            message="Karachi alert",
            details={},
            city="Karachi",
        )

        # Fire alert for Lahore (should work)
        alert = manager.fire_alert(
            alert_type=AlertType.DATA_DRIFT,
            level=AlertLevel.WARNING,
            message="Lahore alert",
            details={},
            city="Lahore",
        )

        assert alert is not None

    def test_aggregate_alerts(self):
        """Test alert aggregation."""
        manager = AlertManager()

        # Create multiple alerts
        alerts = []
        for i in range(5):
            alert = Alert(
                alert_id=f"alert_{i}",
                alert_type=AlertType.DATA_DRIFT,
                level=AlertLevel.WARNING,
                message=f"Alert {i}",
                details={},
                city="Karachi",
            )
            alerts.append(alert)

        # Aggregate
        aggregates = manager.aggregate_alerts(alerts)

        assert len(aggregates) == 1
        assert aggregates[0].count == 5
        assert "Karachi" in aggregates[0].cities

    def test_aggregate_different_types(self):
        """Test aggregation with different alert types."""
        manager = AlertManager()

        alerts = [
            Alert(
                alert_id="alert_1",
                alert_type=AlertType.DATA_DRIFT,
                level=AlertLevel.WARNING,
                message="Drift alert",
                details={},
                city="Karachi",
            ),
            Alert(
                alert_id="alert_2",
                alert_type=AlertType.MODEL_PERFORMANCE,
                level=AlertLevel.WARNING,
                message="Performance alert",
                details={},
                city="Karachi",
            ),
        ]

        aggregates = manager.aggregate_alerts(alerts)

        assert len(aggregates) == 2  # Different types, not aggregated

    def test_acknowledge_alert(self):
        """Test acknowledging an alert."""
        manager = AlertManager()

        alert = manager.fire_alert(
            alert_type=AlertType.DATA_DRIFT,
            level=AlertLevel.WARNING,
            message="Test alert",
            details={},
            city="Karachi",
        )

        assert not alert.acknowledged

        result = manager.acknowledge_alert(alert.alert_id)

        assert result is True
        assert alert.acknowledged is True

    def test_resolve_alert(self):
        """Test resolving an alert."""
        manager = AlertManager()

        alert = manager.fire_alert(
            alert_type=AlertType.DATA_DRIFT,
            level=AlertLevel.WARNING,
            message="Test alert",
            details={},
            city="Karachi",
        )

        assert len(manager.active_alerts) == 1

        result = manager.resolve_alert(alert.alert_id)

        assert result is True
        assert len(manager.active_alerts) == 0
        assert alert.resolved is True

    def test_get_active_alerts_filtered(self):
        """Test filtering active alerts."""
        manager = AlertManager()

        # Create alerts of different types
        manager.fire_alert(
            alert_type=AlertType.DATA_DRIFT,
            level=AlertLevel.WARNING,
            message="Drift alert",
            details={},
            city="Karachi",
        )
        manager.fire_alert(
            alert_type=AlertType.MODEL_PERFORMANCE,
            level=AlertLevel.CRITICAL,
            message="Performance alert",
            details={},
            city="Lahore",
        )

        # Filter by type
        drift_alerts = manager.get_active_alerts(alert_type=AlertType.DATA_DRIFT)
        assert len(drift_alerts) == 1

        # Filter by city
        karachi_alerts = manager.get_active_alerts(city="Karachi")
        assert len(karachi_alerts) == 1

        # Filter by level
        critical_alerts = manager.get_active_alerts(level=AlertLevel.CRITICAL)
        assert len(critical_alerts) == 1

    def test_notifier_called(self):
        """Test that notifiers are called."""
        manager = AlertManager()

        notified_alerts = []

        def test_notifier(alert):
            notified_alerts.append(alert)

        manager.add_notifier(test_notifier)

        manager.fire_alert(
            alert_type=AlertType.DATA_DRIFT,
            level=AlertLevel.WARNING,
            message="Test alert",
            details={},
            city="Karachi",
        )

        assert len(notified_alerts) == 1

    def test_save_and_load_alert_history(self):
        """Test saving and loading alert history."""
        manager = AlertManager()

        # Create some alerts
        for i in range(3):
            manager.fire_alert(
                alert_type=AlertType.DATA_DRIFT,
                level=AlertLevel.WARNING,
                message=f"Alert {i}",
                details={},
                city="Karachi",
                force=True,
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            saved_path = manager.save_alert_history(output_dir)

            assert saved_path.exists()

            # Load history
            new_manager = AlertManager()
            loaded_alerts = new_manager.load_alert_history(saved_path)

            assert len(loaded_alerts) == 3

    def test_alert_rule_creation(self):
        """Test AlertRule dataclass creation."""
        rule = AlertRule(
            rule_id="rule_1",
            alert_type=AlertType.DATA_DRIFT,
            condition="psi > threshold",
            threshold=0.1,
            level=AlertLevel.WARNING,
            cooldown_minutes=60,
        )

        assert rule.rule_id == "rule_1"
        assert rule.cooldown_minutes == 60

    def test_alert_levels(self):
        """Test all alert levels are valid."""
        for level in AlertLevel:
            assert level.value in ["info", "warning", "critical", "emergency"]

    def test_alert_types(self):
        """Test all alert types are valid."""
        for alert_type in AlertType:
            assert alert_type.value in [
                "data_drift",
                "model_performance",
                "data_quality",
                "system",
            ]
