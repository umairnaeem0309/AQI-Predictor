"""
AQI Theme and Color Utility

Centralized color scheme and styling for dashboard.
"""

from typing import Dict, Tuple


# US EPA AQI Colors
AQI_COLORS = {
    "good": "#00E400",
    "moderate": "#FFFF00",
    "unhealthy_sensitive": "#FF7E00",
    "unhealthy": "#FF0000",
    "very_unhealthy": "#8F3F97",
    "hazardous": "#7E0023",
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

# Chart color palette
CHART_COLORS = {
    "primary": "#1E88E5",
    "secondary": "#43A047",
    "tertiary": "#FB8C00",
    "quaternary": "#E53935",
    "background": "#FFFFFF",
    "text": "#212121",
    "grid": "#E0E0E0",
}

# City colors for multi-city charts
CITY_COLORS = {
    "Karachi": "#1E88E5",
    "Lahore": "#43A047",
    "Islamabad": "#FB8C00",
}


def get_aqi_color(aqi_value: int) -> str:
    """
    Get AQI color from value.
    
    Args:
        aqi_value: AQI value (0-500)
        
    Returns:
        Hex color string
    """
    if aqi_value <= 50:
        return AQI_COLORS["good"]
    elif aqi_value <= 100:
        return AQI_COLORS["moderate"]
    elif aqi_value <= 150:
        return AQI_COLORS["unhealthy_sensitive"]
    elif aqi_value <= 200:
        return AQI_COLORS["unhealthy"]
    elif aqi_value <= 300:
        return AQI_COLORS["very_unhealthy"]
    else:
        return AQI_COLORS["hazardous"]


def get_aqi_category(aqi_value: int) -> str:
    """
    Get AQI category name from value.
    
    Args:
        aqi_value: AQI value (0-500)
        
    Returns:
        Category name
    """
    if aqi_value <= 50:
        return AQI_CATEGORY_NAMES["good"]
    elif aqi_value <= 100:
        return AQI_CATEGORY_NAMES["moderate"]
    elif aqi_value <= 150:
        return AQI_CATEGORY_NAMES["unhealthy_sensitive"]
    elif aqi_value <= 200:
        return AQI_CATEGORY_NAMES["unhealthy"]
    elif aqi_value <= 300:
        return AQI_CATEGORY_NAMES["very_unhealthy"]
    else:
        return AQI_CATEGORY_NAMES["hazardous"]


def get_aqi_category_key(aqi_value: int) -> str:
    """
    Get AQI category key from value.
    
    Args:
        aqi_value: AQI value (0-500)
        
    Returns:
        Category key
    """
    if aqi_value <= 50:
        return "good"
    elif aqi_value <= 100:
        return "moderate"
    elif aqi_value <= 150:
        return "unhealthy_sensitive"
    elif aqi_value <= 200:
        return "unhealthy"
    elif aqi_value <= 300:
        return "very_unhealthy"
    else:
        return "hazardous"


def get_city_color(city: str) -> str:
    """
    Get color for a city.
    
    Args:
        city: City name
        
    Returns:
        Hex color string
    """
    return CITY_COLORS.get(city, CHART_COLORS["primary"])


def get_dashboard_css() -> str:
    """Get custom CSS for dashboard."""
    return """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .aqi-good { color: #00E400; }
    .aqi-moderate { color: #FFFF00; background: #333; }
    .aqi-unhealthy-sensitive { color: #FF7E00; }
    .aqi-unhealthy { color: #FF0000; }
    .aqi-very-unhealthy { color: #8F3F97; }
    .aqi-hazardous { color: #7E0023; }
    </style>
    """
