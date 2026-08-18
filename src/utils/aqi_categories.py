"""
AQI Category Utility

US EPA AQI breakpoints and category classification.
Domain utility module - not API service.
"""

from typing import Tuple


# US EPA AQI Breakpoints for PM2.5 (µg/m³, 24-hour average)
# Source: https://www.airnow.gov/aqi/aqi-basics/
US_EPA_AQI_BREAKPOINTS = {
    "good": {"min": 0, "max": 50, "aqi_min": 0, "aqi_max": 50},
    "moderate": {"min": 51, "max": 100, "aqi_min": 51, "aqi_max": 100},
    "unhealthy_sensitive": {"min": 101, "max": 150, "aqi_min": 101, "aqi_max": 150},
    "unhealthy": {"min": 151, "max": 200, "aqi_min": 151, "aqi_max": 200},
    "very_unhealthy": {"min": 201, "max": 300, "aqi_min": 201, "aqi_max": 300},
    "hazardous": {"min": 301, "max": 500, "aqi_min": 301, "aqi_max": 500},
}

# Category display names
AQI_CATEGORY_NAMES = {
    "good": "Good",
    "moderate": "Moderate",
    "unhealthy_sensitive": "Unhealthy for Sensitive Groups",
    "unhealthy": "Unhealthy",
    "very_unhealthy": "Very Unhealthy",
    "hazardous": "Hazardous",
}

# Category colors for dashboard
AQI_CATEGORY_COLORS = {
    "good": "#00E400",
    "moderate": "#FFFF00",
    "unhealthy_sensitive": "#FF7E00",
    "unhealthy": "#FF0000",
    "very_unhealthy": "#8F3F97",
    "hazardous": "#7E0023",
}


def get_aqi_category(aqi_value: int) -> Tuple[str, str]:
    """
    Get AQI category from AQI value.
    
    Args:
        aqi_value: AQI value (0-500)
        
    Returns:
        Tuple of (category_key, category_name)
        
    Raises:
        ValueError: If AQI value is out of range
    """
    if aqi_value < 0 or aqi_value > 500:
        raise ValueError(f"AQI value must be between 0 and 500, got {aqi_value}")
    
    for category_key, bounds in US_EPA_AQI_BREAKPOINTS.items():
        if bounds["aqi_min"] <= aqi_value <= bounds["aqi_max"]:
            return category_key, AQI_CATEGORY_NAMES[category_key]
    
    # Fallback for values > 500
    return "hazardous", AQI_CATEGORY_NAMES["hazardous"]


def get_aqi_color(aqi_value: int) -> str:
    """
    Get AQI color from AQI value.
    
    Args:
        aqi_value: AQI value (0-500)
        
    Returns:
        Hex color string
    """
    category_key, _ = get_aqi_category(aqi_value)
    return AQI_CATEGORY_COLORS[category_key]


def get_aqi_category_range(category_key: str) -> Tuple[int, int]:
    """
    Get AQI range for a category.
    
    Args:
        category_key: Category key (good, moderate, etc.)
        
    Returns:
        Tuple of (min_aqi, max_aqi)
    """
    if category_key not in US_EPA_AQI_BREAKPOINTS:
        raise ValueError(f"Invalid category: {category_key}")
    
    bounds = US_EPA_AQI_BREAKPOINTS[category_key]
    return bounds["aqi_min"], bounds["aqi_max"]
