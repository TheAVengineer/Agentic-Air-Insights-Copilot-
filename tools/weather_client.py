"""
Open-Meteo API client for weather and air quality data.

Open-Meteo provides free weather and air quality APIs without authentication.
- Weather API: https://api.open-meteo.com/v1/forecast
- Air Quality API: https://air-quality-api.open-meteo.com/v1/air-quality

Attribution: "Weather data by Open-Meteo.com" is required.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import httpx

from policies import SAFETY_RULES

logger = logging.getLogger(__name__)


@dataclass
class AirQualityData:
    """Air quality data from Open-Meteo."""
    
    pm25: list[float] = field(default_factory=list)
    pm10: list[float] = field(default_factory=list)
    timestamps: list[str] = field(default_factory=list)
    
    @property
    def pm25_avg(self) -> float:
        """Calculate average PM2.5, excluding None values."""
        valid = [v for v in self.pm25 if v is not None]
        return sum(valid) / len(valid) if valid else 0.0
    
    @property
    def pm10_avg(self) -> float:
        """Calculate average PM10, excluding None values."""
        valid = [v for v in self.pm10 if v is not None]
        return sum(valid) / len(valid) if valid else 0.0
    
    @property
    def data_quality(self) -> float:
        """Calculate data quality as percentage of non-null values."""
        total = len(self.pm25) + len(self.pm10)
        if total == 0:
            return 0.0
        valid = sum(1 for v in self.pm25 if v is not None)
        valid += sum(1 for v in self.pm10 if v is not None)
        return valid / total


@dataclass
class WeatherData:
    """Weather data from Open-Meteo."""
    
    temperature: list[float] = field(default_factory=list)
    timestamps: list[str] = field(default_factory=list)
    
    @property
    def temp_avg(self) -> float:
        """Calculate average temperature, excluding None values."""
        valid = [v for v in self.temperature if v is not None]
        return sum(valid) / len(valid) if valid else 0.0
    
    @property
    def temp_min(self) -> float:
        """Get minimum temperature."""
        valid = [v for v in self.temperature if v is not None]
        return min(valid) if valid else 0.0
    
    @property
    def temp_max(self) -> float:
        """Get maximum temperature."""
        valid = [v for v in self.temperature if v is not None]
        return max(valid) if valid else 0.0
    
    @property
    def data_quality(self) -> float:
        """Calculate data quality as percentage of non-null values."""
        if not self.temperature:
            return 0.0
        valid = sum(1 for v in self.temperature if v is not None)
        return valid / len(self.temperature)


@dataclass
class DailyForecast:
    """Daily weather forecast data."""
    
    dates: list[str] = field(default_factory=list)
    temp_max: list[float] = field(default_factory=list)
    temp_min: list[float] = field(default_factory=list)
    precipitation_sum: list[float] = field(default_factory=list)
    precipitation_probability: list[int] = field(default_factory=list)
    weather_code: list[int] = field(default_factory=list)
    sunrise: list[str] = field(default_factory=list)
    sunset: list[str] = field(default_factory=list)
    
    @staticmethod
    def weather_code_to_description(code: int) -> tuple[str, str]:
        """
        Convert WMO weather code to description and emoji.
        
        Returns: (description, emoji)
        """
        codes = {
            0: ("Clear sky", "☀️"),
            1: ("Mainly clear", "🌤️"),
            2: ("Partly cloudy", "⛅"),
            3: ("Overcast", "☁️"),
            45: ("Foggy", "🌫️"),
            48: ("Depositing rime fog", "🌫️"),
            51: ("Light drizzle", "🌧️"),
            53: ("Moderate drizzle", "🌧️"),
            55: ("Dense drizzle", "🌧️"),
            56: ("Light freezing drizzle", "🌨️"),
            57: ("Dense freezing drizzle", "🌨️"),
            61: ("Slight rain", "🌧️"),
            63: ("Moderate rain", "🌧️"),
            65: ("Heavy rain", "🌧️"),
            66: ("Light freezing rain", "🌨️"),
            67: ("Heavy freezing rain", "🌨️"),
            71: ("Slight snow", "🌨️"),
            73: ("Moderate snow", "🌨️"),
            75: ("Heavy snow", "❄️"),
            77: ("Snow grains", "🌨️"),
            80: ("Slight rain showers", "🌦️"),
            81: ("Moderate rain showers", "🌦️"),
            82: ("Violent rain showers", "⛈️"),
            85: ("Slight snow showers", "🌨️"),
            86: ("Heavy snow showers", "🌨️"),
            95: ("Thunderstorm", "⛈️"),
            96: ("Thunderstorm with slight hail", "⛈️"),
            99: ("Thunderstorm with heavy hail", "⛈️"),
        }
        return codes.get(code, ("Unknown", "❓"))
    
    def get_day_summary(self, index: int) -> dict:
        """Get a summary for a specific day."""
        if index >= len(self.dates):
            return {}
        
        code = self.weather_code[index] if index < len(self.weather_code) else 0
        desc, emoji = self.weather_code_to_description(code)
        
        return {
            "date": self.dates[index],
            "temp_max": self.temp_max[index] if index < len(self.temp_max) else None,
            "temp_min": self.temp_min[index] if index < len(self.temp_min) else None,
            "precipitation": self.precipitation_sum[index] if index < len(self.precipitation_sum) else 0,
            "precip_probability": self.precipitation_probability[index] if index < len(self.precipitation_probability) else 0,
            "weather": desc,
            "emoji": emoji,
        }


class WeatherClient:
    """
    Async client for Open-Meteo Weather and Air Quality APIs.
    
    Implements:
    - Separate endpoints for weather and air quality
    - Retry logic with exponential backoff
    - Data validation and error handling
    - Configurable timeouts
    """
    
    WEATHER_BASE_URL = "https://api.open-meteo.com/v1/forecast"
    AIR_QUALITY_BASE_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
    
    def __init__(self):
        """Initialize the weather client with configuration from policies."""
        agent_config = SAFETY_RULES.get("agent", {})
        self.max_retries = agent_config.get("max_retries", 3)
        self.backoff_base = agent_config.get("retry_backoff_base", 1.0)
        self.backoff_max = agent_config.get("retry_backoff_max", 10.0)
        self.timeout = agent_config.get("request_timeout_seconds", 10)
        
    async def _request_with_retry(
        self,
        url: str,
        params: dict,
        operation_name: str
    ) -> dict:
        """
        Make HTTP request with exponential backoff retry logic.
        
        Args:
            url: API endpoint URL
            params: Query parameters
            operation_name: Name for logging purposes
            
        Returns:
            JSON response as dict
            
        Raises:
            httpx.HTTPError: After all retries exhausted
        """
        last_exception = None
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for attempt in range(self.max_retries):
                try:
                    logger.info(
                        f"{operation_name}: Attempt {attempt + 1}/{self.max_retries}"
                    )
                    response = await client.get(url, params=params)
                    response.raise_for_status()
                    
                    logger.info(f"{operation_name}: Success")
                    return response.json()
                    
                except httpx.HTTPStatusError as e:
                    last_exception = e
                    logger.warning(
                        f"{operation_name}: HTTP {e.response.status_code} - "
                        f"{e.response.text[:200]}"
                    )
                    # Don't retry client errors (4xx)
                    if 400 <= e.response.status_code < 500:
                        raise
                        
                except httpx.RequestError as e:
                    last_exception = e
                    logger.warning(f"{operation_name}: Request error - {str(e)}")
                
                # Calculate backoff time
                if attempt < self.max_retries - 1:
                    backoff = min(
                        self.backoff_base * (2 ** attempt),
                        self.backoff_max
                    )
                    logger.info(f"{operation_name}: Retrying in {backoff}s...")
                    await asyncio.sleep(backoff)
        
        # All retries exhausted
        logger.error(f"{operation_name}: All {self.max_retries} attempts failed")
        raise last_exception or httpx.RequestError("All retries failed")
    
    async def get_air_quality(
        self,
        latitude: float,
        longitude: float,
        hours: int = 6,
        past_days: int = 0
    ) -> AirQualityData:
        """
        Fetch air quality data from Open-Meteo.
        
        Args:
            latitude: Location latitude (-90 to 90)
            longitude: Location longitude (-180 to 180)
            hours: Number of forecast hours (1-120, capped at API limit)
            past_days: Number of past days to include (0-92)
            
        Returns:
            AirQualityData with PM2.5 and PM10 values
        """
        # Open-Meteo Air Quality API supports max 120 forecast hours, 92 past days
        capped_hours = min(hours, 120)
        capped_past_days = min(past_days, 92)
        
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": "pm2_5,pm10",
        }
        
        # Add forecast hours if looking at future data
        if capped_hours > 0 and capped_past_days == 0:
            params["forecast_hours"] = capped_hours
        
        # Add past_days for historical data
        if capped_past_days > 0:
            params["past_days"] = capped_past_days
            # When looking at past data, limit forecast to minimal
            params["forecast_hours"] = min(capped_hours, 24) if capped_hours > 0 else 1
        
        try:
            data = await self._request_with_retry(
                self.AIR_QUALITY_BASE_URL,
                params,
                "AirQuality"
            )
            
            hourly = data.get("hourly", {})
            return AirQualityData(
                pm25=hourly.get("pm2_5", []),
                pm10=hourly.get("pm10", []),
                timestamps=hourly.get("time", [])
            )
            
        except Exception as e:
            logger.error(f"Failed to fetch air quality: {str(e)}")
            raise
    
    async def get_weather(
        self,
        latitude: float,
        longitude: float,
        hours: int = 6,
        past_days: int = 0
    ) -> WeatherData:
        """
        Fetch weather forecast from Open-Meteo.
        
        Args:
            latitude: Location latitude (-90 to 90)
            longitude: Location longitude (-180 to 180)
            hours: Number of forecast hours (1-384, up to 16 days)
            past_days: Number of past days to include (0-92)
            
        Returns:
            WeatherData with temperature values
        """
        # Open-Meteo Weather API supports up to 16 days (384 hours), 92 past days
        capped_hours = min(hours, 384)
        capped_past_days = min(past_days, 92)
        
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": "temperature_2m",
        }
        
        # Add forecast hours if looking at future data
        if capped_hours > 0 and capped_past_days == 0:
            params["forecast_hours"] = capped_hours
        
        # Add past_days for historical data
        if capped_past_days > 0:
            params["past_days"] = capped_past_days
            # When looking at past data, limit forecast to minimal
            params["forecast_hours"] = min(capped_hours, 24) if capped_hours > 0 else 1
        
        try:
            data = await self._request_with_retry(
                self.WEATHER_BASE_URL,
                params,
                "Weather"
            )
            
            hourly = data.get("hourly", {})
            return WeatherData(
                temperature=hourly.get("temperature_2m", []),
                timestamps=hourly.get("time", [])
            )
            
        except Exception as e:
            logger.error(f"Failed to fetch weather: {str(e)}")
            raise
    
    async def get_combined_data(
        self,
        latitude: float,
        longitude: float,
        hours: int = 6,
        past_days: int = 0
    ) -> tuple[AirQualityData, WeatherData]:
        """
        Fetch both air quality and weather data concurrently.
        
        This is more efficient than sequential calls as both APIs
        are independent and can be called in parallel.
        
        Args:
            latitude: Location latitude
            longitude: Location longitude  
            hours: Number of forecast hours
            past_days: Number of past days to include (0-92)
            
        Returns:
            Tuple of (AirQualityData, WeatherData)
        """
        air_quality_task = self.get_air_quality(latitude, longitude, hours, past_days)
        weather_task = self.get_weather(latitude, longitude, hours, past_days)
        
        air_quality, weather = await asyncio.gather(
            air_quality_task,
            weather_task
        )
        
        return air_quality, weather

    async def get_weekly_forecast(
        self,
        latitude: float,
        longitude: float,
        days: int = 7
    ) -> DailyForecast:
        """
        Fetch daily weather forecast for up to 16 days.
        
        This is useful for questions like "when will be the warmest day?"
        Uses Open-Meteo's daily aggregations for cleaner data.
        
        Args:
            latitude: Location latitude
            longitude: Location longitude  
            days: Number of forecast days (1-16, default 7)
            
        Returns:
            DailyForecast with daily weather data
        """
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max,weather_code,sunrise,sunset",
            "forecast_days": min(days, 16),
            "timezone": "auto",
        }
        
        try:
            data = await self._request_with_retry(
                self.WEATHER_BASE_URL,
                params,
                "DailyForecast"
            )
            daily = data.get("daily", {})
            
            return DailyForecast(
                dates=daily.get("time", []),
                temp_max=daily.get("temperature_2m_max", []),
                temp_min=daily.get("temperature_2m_min", []),
                precipitation_sum=daily.get("precipitation_sum", []),
                precipitation_probability=daily.get("precipitation_probability_max", []),
                weather_code=daily.get("weather_code", []),
                sunrise=daily.get("sunrise", []),
                sunset=daily.get("sunset", []),
            )
        except Exception as e:
            logger.error(f"Failed to fetch daily forecast: {str(e)}")
            raise

    async def get_historical_daily(
        self,
        latitude: float,
        longitude: float,
        past_days: int = 7
    ) -> DailyForecast:
        """
        Fetch historical daily weather data for up to 92 days in the past.
        
        Uses Open-Meteo's daily aggregations for cleaner historical data.
        
        Args:
            latitude: Location latitude
            longitude: Location longitude  
            past_days: Number of past days (1-92)
            
        Returns:
            DailyForecast with historical daily weather data
        """
        capped_past_days = min(past_days, 92)
        
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code,sunrise,sunset",
            "past_days": capped_past_days,
            "forecast_days": 0,  # No future forecast, only past
            "timezone": "auto",
        }
        
        try:
            data = await self._request_with_retry(
                self.WEATHER_BASE_URL,
                params,
                "HistoricalDaily"
            )
            daily = data.get("daily", {})
            
            return DailyForecast(
                dates=daily.get("time", []),
                temp_max=daily.get("temperature_2m_max", []),
                temp_min=daily.get("temperature_2m_min", []),
                precipitation_sum=daily.get("precipitation_sum", []),
                precipitation_probability=[],  # Not available for historical
                weather_code=daily.get("weather_code", []),
                sunrise=daily.get("sunrise", []),
                sunset=daily.get("sunset", []),
            )
        except Exception as e:
            logger.error(f"Failed to fetch historical daily: {str(e)}")
            raise
    
    async def get_extended_forecast(
        self,
        latitude: float,
        longitude: float,
        hours: int = 72
    ) -> tuple[AirQualityData, WeatherData, DailyForecast]:
        """
        Fetch comprehensive forecast data - hourly + daily.
        
        For periods > 72 hours, uses daily data instead.
        Combines air quality, hourly weather, and daily forecast.
        
        Args:
            latitude: Location latitude
            longitude: Location longitude  
            hours: Requested hours (will use daily for >72h)
            
        Returns:
            Tuple of (AirQualityData, WeatherData, DailyForecast)
        """
        # Calculate days needed
        days = max(7, (hours + 23) // 24)  # At least 7 dasys
        days = min(days, 16)  # Max 16 days
        
        # Hourly data is limited to 72 hours for air quality
        hourly_hours = min(hours, 72)
        
        # Fetch all data concurrently
        air_quality_task = self.get_air_quality(latitude, longitude, hourly_hours)
        weather_task = self.get_weather(latitude, longitude, hourly_hours)
        daily_task = self.get_weekly_forecast(latitude, longitude, days)
        
        air_quality, weather, daily = await asyncio.gather(
            air_quality_task,
            weather_task,
            daily_task
        )
        
        return air_quality, weather, daily
