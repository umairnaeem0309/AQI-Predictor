"""
Data Providers — Pluggable data source abstraction for historical and real-time data.

Each provider encapsulates:
- API endpoint configuration
- Response parsing
- Rate limiting awareness
- Error handling
- Standardized output format (pandas DataFrame)

Providers do NOT contain:
- AQI calculation (delegated to src/utils/epa_aqi.py)
- Feature engineering (delegated to src/features/)
- Data validation (delegated to src/data/validators.py)
"""

from src.data.providers.base_provider import BaseHistoricalProvider
from src.data.providers.open_meteo_weather import OpenMeteoWeatherProvider
from src.data.providers.open_meteo_air_quality import OpenMeteoAirQualityProvider

__all__ = [
    "BaseHistoricalProvider",
    "OpenMeteoWeatherProvider",
    "OpenMeteoAirQualityProvider",
]
