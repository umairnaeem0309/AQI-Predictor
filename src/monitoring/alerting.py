"""
Alerting Module

Provides alert management with cooldown, aggregation, and severity levels.
"""

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertType(Enum):
    """Types of alerts."""
    DATA_DRIFT = "data_drift"
    MODEL_PERFORMANCE = "model_performance"
    DATA_QUALITY = "data_quality"
    SYSTEM = "system"


@dataclass
class Alert:
    """Single alert instance."""
    alert_id: str
    alert_type: AlertType
    level: AlertLevel
    message: str
    details: Dict[str, Any]
    city: str
    created_at: str = ""
    acknowledged: bool = False
    resolved: bool = False

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


@dataclass
class AlertRule:
    """Alert rule definition."""
    rule_id: str
    alert_type: AlertType
    condition: str
    threshold: float
    level: AlertLevel
    cooldown_minutes: int = 60
    aggregation_window_minutes: int = 15


@dataclass
class AlertAggregate:
    """Aggregated alerts within a time window."""
    aggregate_id: str
    alert_type: AlertType
    level: AlertLevel
    count: int
    first_alert_at: str
    last_alert_at: str
    cities: List[str]
    details: Dict[str, Any]


class AlertManager:
    """
    Alert management with cooldown and aggregation.
    
    Features:
    - Alert cooldown to prevent flooding
    - Alert aggregation for similar alerts
    - Multiple notification targets
    - Alert history tracking
    """
    
    def __init__(
        self,
        default_cooldown_minutes: int = 60,
        aggregation_window_minutes: int = 15,
    ):
        """
        Initialize alert manager.
        
        Args:
            default_cooldown_minutes: Default cooldown between same alerts
            aggregation_window_minutes: Window for aggregating similar alerts
        """
        self.default_cooldown_minutes = default_cooldown_minutes
        self.aggregation_window_minutes = aggregation_window_minutes
        self.rules: Dict[str, AlertRule] = {}
        self.active_alerts: List[Alert] = []
        self.alert_history: List[Alert] = []
        self.last_alert_times: Dict[str, datetime] = {}
        self.notifiers: List[Callable[[Alert], None]] = []
    
    def add_rule(self, rule: AlertRule):
        """Add an alert rule."""
        self.rules[rule.rule_id] = rule
    
    def add_notifier(self, notifier: Callable[[Alert], None]):
        """Add a notification target."""
        self.notifiers.append(notifier)
    
    def check_cooldown(self, alert_key: str, cooldown_minutes: int) -> bool:
        """
        Check if alert is in cooldown period.
        
        Args:
            alert_key: Unique key for the alert
            cooldown_minutes: Cooldown period
            
        Returns:
            True if alert can fire, False if in cooldown
        """
        now = datetime.now(timezone.utc)
        
        if alert_key in self.last_alert_times:
            last_alert = self.last_alert_times[alert_key]
            time_since_last = (now - last_alert).total_seconds() / 60
            
            if time_since_last < cooldown_minutes:
                return False
        
        return True
    
    def fire_alert(
        self,
        alert_type: AlertType,
        level: AlertLevel,
        message: str,
        details: Dict[str, Any],
        city: str = "all",
        force: bool = False,
    ) -> Optional[Alert]:
        """
        Fire an alert if rules allow.
        
        Args:
            alert_type: Type of alert
            level: Severity level
            message: Alert message
            details: Additional details
            city: Affected city
            force: Force alert even if in cooldown
            
        Returns:
            Alert if fired, None if blocked by cooldown
        """
        # Generate alert key for cooldown
        alert_key = f"{alert_type.value}_{level.value}_{city}"
        
        # Check cooldown
        if not force and not self.check_cooldown(alert_key, self.default_cooldown_minutes):
            return None
        
        # Create alert
        alert = Alert(
            alert_id=f"alert_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}",
            alert_type=alert_type,
            level=level,
            message=message,
            details=details,
            city=city,
        )
        
        # Update cooldown
        self.last_alert_times[alert_key] = datetime.now(timezone.utc)
        
        # Store alert
        self.active_alerts.append(alert)
        self.alert_history.append(alert)
        
        # Notify
        for notifier in self.notifiers:
            try:
                notifier(alert)
            except Exception as e:
                print(f"Notification failed: {e}")
        
        return alert
    
    def aggregate_alerts(
        self,
        alerts: List[Alert],
        window_minutes: Optional[int] = None,
    ) -> List[AlertAggregate]:
        """
        Aggregate similar alerts within a time window.
        
        Args:
            alerts: List of alerts to aggregate
            window_minutes: Aggregation window (uses default if None)
            
        Returns:
            List of aggregated alerts
        """
        if window_minutes is None:
            window_minutes = self.aggregation_window_minutes
        
        if not alerts:
            return []
        
        # Group by type, level, and city
        groups: Dict[str, List[Alert]] = {}
        for alert in alerts:
            key = f"{alert.alert_type.value}_{alert.level.value}_{alert.city}"
            if key not in groups:
                groups[key] = []
            groups[key].append(alert)
        
        # Aggregate each group
        aggregates = []
        for key, group_alerts in groups.items():
            if len(group_alerts) == 1:
                # Single alert, no aggregation needed
                alert = group_alerts[0]
                aggregates.append(AlertAggregate(
                    aggregate_id=f"agg_{alert.alert_id}",
                    alert_type=alert.alert_type,
                    level=alert.level,
                    count=1,
                    first_alert_at=alert.created_at,
                    last_alert_at=alert.created_at,
                    cities=[alert.city],
                    details=alert.details,
                ))
            else:
                # Multiple alerts, aggregate
                timestamps = [a.created_at for a in group_alerts]
                cities = list(set(a.city for a in group_alerts))
                
                # Merge details
                merged_details = {}
                for alert in group_alerts:
                    merged_details.update(alert.details)
                
                aggregates.append(AlertAggregate(
                    aggregate_id=f"agg_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
                    alert_type=group_alerts[0].alert_type,
                    level=group_alerts[0].level,
                    count=len(group_alerts),
                    first_alert_at=min(timestamps),
                    last_alert_at=max(timestamps),
                    cities=cities,
                    details=merged_details,
                ))
        
        return aggregates
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        for alert in self.active_alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        for alert in self.active_alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                self.active_alerts.remove(alert)
                return True
        return False
    
    def get_active_alerts(
        self,
        alert_type: Optional[AlertType] = None,
        level: Optional[AlertLevel] = None,
        city: Optional[str] = None,
    ) -> List[Alert]:
        """Get active alerts with optional filtering."""
        filtered = self.active_alerts
        
        if alert_type:
            filtered = [a for a in filtered if a.alert_type == alert_type]
        if level:
            filtered = [a for a in filtered if a.level == level]
        if city:
            filtered = [a for a in filtered if a.city == city]
        
        return filtered
    
    def save_alert_history(self, output_dir: Path) -> Path:
        """Save alert history to file."""
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "alert_history.json"
        
        with open(json_path, "w") as f:
            json.dump(
                [asdict(a) for a in self.alert_history],
                f,
                indent=2,
                default=str,
            )
        
        return json_path
    
    def load_alert_history(self, json_path: Path) -> List[Alert]:
        """Load alert history from file."""
        with open(json_path, "r") as f:
            data = json.load(f)
        
        alerts = []
        for item in data:
            alerts.append(Alert(
                alert_id=item["alert_id"],
                alert_type=AlertType(item["alert_type"]),
                level=AlertLevel(item["level"]),
                message=item["message"],
                details=item["details"],
                city=item["city"],
                created_at=item.get("created_at", ""),
                acknowledged=item.get("acknowledged", False),
                resolved=item.get("resolved", False),
            ))
        
        self.alert_history = alerts
        return alerts
