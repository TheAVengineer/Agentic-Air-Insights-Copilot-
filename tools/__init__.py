# Tools package
"""
External API clients for fetching weather, air quality, and NASA data.

All clients implement:
- Async HTTP calls with httpx
- Retry logic with exponential backoff
- Proper error handling
- Data validation
"""

from .weather_client import WeatherClient, AirQualityData, WeatherData
from .nasa_client import NASAClient, APODData

__all__ = [
    "WeatherClient",
    "AirQualityData",
    "WeatherData",
    "NASAClient",
    "APODData",
]
