"""
Data Formatters

Formatting utilities for dates, numbers, and display.
"""

from datetime import datetime, timezone
from typing import Optional


def format_timestamp(timestamp_str: Optional[str]) -> str:
    """
    Format timestamp for display.

    Args:
        timestamp_str: ISO timestamp string

    Returns:
        Formatted timestamp
    """
    if not timestamp_str:
        return "N/A"

    try:
        dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except (ValueError, AttributeError):
        return "N/A"


def format_time_ago(timestamp_str: Optional[str]) -> str:
    """
    Format timestamp as time ago.

    Args:
        timestamp_str: ISO timestamp string

    Returns:
        Time ago string
    """
    if not timestamp_str:
        return "Never"

    try:
        dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        diff = now - dt

        seconds = int(diff.total_seconds())
        if seconds < 60:
            return f"{seconds} seconds ago"
        elif seconds < 3600:
            minutes = seconds // 60
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        elif seconds < 86400:
            hours = seconds // 3600
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        else:
            days = seconds // 86400
            return f"{days} day{'s' if days != 1 else ''} ago"
    except (ValueError, AttributeError):
        return "N/A"


def format_aqi(aqi_value: int) -> str:
    """
    Format AQI value for display.

    Args:
        aqi_value: AQI value

    Returns:
        Formatted AQI string
    """
    return str(aqi_value)


def format_metric(value: Optional[float], decimals: int = 2) -> str:
    """
    Format metric value.

    Args:
        value: Metric value
        decimals: Number of decimal places

    Returns:
        Formatted metric string
    """
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}"


def format_percentage(value: Optional[float]) -> str:
    """
    Format percentage value.

    Args:
        value: Percentage value (0-100)

    Returns:
        Formatted percentage string
    """
    if value is None:
        return "N/A"
    return f"{value:.1f}%"
