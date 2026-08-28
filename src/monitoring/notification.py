"""
Notification Module

Provides notification abstraction for alerts.
Currently supports: Log notifier, Console notifier.
Future: Email, Slack integrations.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from src.monitoring.alerting import Alert, AlertLevel


class LogNotifier:
    """
    Log-based notifier.

    Writes alerts to Python logging system.
    """

    def __init__(self, logger_name: str = "aqi_predictor.alerts"):
        """
        Initialize log notifier.

        Args:
            logger_name: Name of the logger to use
        """
        self.logger = logging.getLogger(logger_name)

    def __call__(self, alert: Alert):
        """Notify via logging."""
        log_message = self._format_alert(alert)

        # Log at appropriate level
        if alert.level == AlertLevel.INFO:
            self.logger.info(log_message)
        elif alert.level == AlertLevel.WARNING:
            self.logger.warning(log_message)
        elif alert.level == AlertLevel.CRITICAL:
            self.logger.error(log_message)
        elif alert.level == AlertLevel.EMERGENCY:
            self.logger.critical(log_message)

    def _format_alert(self, alert: Alert) -> str:
        """Format alert for logging."""
        return (
            f"[{alert.level.value.upper()}] "
            f"{alert.alert_type.value}: "
            f"{alert.message} "
            f"(City: {alert.city}, "
            f"Time: {alert.created_at})"
        )


class ConsoleNotifier:
    """
    Console-based notifier.

    Prints alerts to console with color coding.
    """

    # ANSI color codes
    COLORS = {
        AlertLevel.INFO: "\033[34m",  # Blue
        AlertLevel.WARNING: "\033[33m",  # Yellow
        AlertLevel.CRITICAL: "\033[31m",  # Red
        AlertLevel.EMERGENCY: "\033[35m",  # Magenta
    }
    RESET_COLOR = "\033[0m"

    def __init__(self, use_colors: bool = True):
        """
        Initialize console notifier.

        Args:
            use_colors: Whether to use ANSI colors
        """
        self.use_colors = use_colors

    def __call__(self, alert: Alert):
        """Notify via console print."""
        message = self._format_alert(alert)
        print(message)

    def _format_alert(self, alert: Alert) -> str:
        """Format alert for console output."""
        if self.use_colors:
            color = self.COLORS.get(alert.level, "")
            return (
                f"{color}[{alert.level.value.upper()}] "
                f"{alert.alert_type.value}: "
                f"{alert.message} "
                f"(City: {alert.city}){self.RESET_COLOR}"
            )
        else:
            return (
                f"[{alert.level.value.upper()}] "
                f"{alert.alert_type.value}: "
                f"{alert.message} "
                f"(City: {alert.city})"
            )


class WebhookNotifier:
    """
    Webhook-based notifier (placeholder).

    Future implementation for Slack, Discord, etc.
    """

    def __init__(self, webhook_url: Optional[str] = None):
        """
        Initialize webhook notifier.

        Args:
            webhook_url: Webhook URL (not implemented yet)
        """
        self.webhook_url = webhook_url

    def __call__(self, alert: Alert):
        """Notify via webhook (not implemented)."""
        if self.webhook_url:
            # Future: POST to webhook_url
            pass
