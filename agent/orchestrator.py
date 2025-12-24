"""
Main agent orchestrator - the brain of Air & Insights.

The orchestrator coordinates:
1. Planning: Deciding which tools to call
2. Execution: Running the plan steps
3. Validation: Checking data quality and results
4. Reasoning: Calling LLM for guidance generation
5. Caching: Storing and retrieving results

This implements the agentic flow:
Plan → Fetch → Validate → Cache → Reason → Respond
"""

import logging
from datetime import datetime
from typing import Optional

from agent.memory import AgentMemory
from agent.planner import AgentPlanner, ToolType, ExecutionPlan
from api.models import (
    AnalyzeResponse,
    APODResponse,
    SafetyLevel,
    DataQuality,
)
from llm.client import LLMClient
from llm.prompts import AirQualityPrompts, APODPrompts
from policies import SAFETY_RULES
from policies.validation import (
    validator,
    validate_data_quality,
    get_data_quality_level,
)
from tools.weather_client import WeatherClient, AirQualityData, WeatherData
from tools.nasa_client import NASAClient, APODData

logger = logging.getLogger(__name__)


class AirInsightsAgent:
    """
    Main orchestrator for the Air & Insights Agent.
    
    Coordinates all components:
    - WeatherClient: Open-Meteo API for weather and air quality
    - NASAClient: NASA APOD API
    - LLMClient: GitHub Models for reasoning
    - AgentMemory: Caching layer
    - AgentPlanner: Execution planning
    
    Usage:
        agent = AirInsightsAgent()
        result = await agent.analyze(42.6977, 23.3219, 6)
    """
    
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        weather_client: Optional[WeatherClient] = None,
        nasa_client: Optional[NASAClient] = None,
        memory: Optional[AgentMemory] = None,
    ):
        """
        Initialize the agent with all required components.
        
        Args:
            llm_client: Optional custom LLM client
            weather_client: Optional custom weather client
            nasa_client: Optional custom NASA client
            memory: Optional custom memory/cache
        """
        self.llm = llm_client or LLMClient()
        self.weather = weather_client or WeatherClient()
        self.nasa = nasa_client or NASAClient()
        self.memory = memory or AgentMemory()
        self.planner = AgentPlanner()
        
        # Load thresholds from policy
        self.thresholds = SAFETY_RULES.get("air_quality_thresholds", {})
        self.temp_thresholds = SAFETY_RULES.get("temperature_thresholds", {})
        self.data_quality_threshold = SAFETY_RULES.get("agent", {}).get(
            "data_quality_threshold", 0.8
        )
        
        logger.info("AirInsightsAgent initialized")
    
    def _determine_safety_level(
        self,
        pm25_avg: float,
        pm10_avg: float,
        temp_avg: float,
    ) -> SafetyLevel:
        """
        Determine overall safety level based on policy thresholds.
        
        Uses the worst condition to determine overall safety.
        
        Args:
            pm25_avg: Average PM2.5 in µg/m³
            pm10_avg: Average PM10 in µg/m³
            temp_avg: Average temperature in °C
            
        Returns:
            SafetyLevel enum value
        """
        pm25_thresholds = self.thresholds.get("pm25", {})
        pm10_thresholds = self.thresholds.get("pm10", {})
        
        # Check PM2.5 level
        if pm25_avg >= pm25_thresholds.get("hazardous", 200):
            return SafetyLevel.HAZARDOUS
        elif pm25_avg >= pm25_thresholds.get("very_unhealthy", 150):
            return SafetyLevel.VERY_UNHEALTHY
        elif pm25_avg >= pm25_thresholds.get("unhealthy", 100):
            return SafetyLevel.UNHEALTHY
        elif pm25_avg >= pm25_thresholds.get("unhealthy_sensitive", 75):
            return SafetyLevel.UNHEALTHY_SENSITIVE
        elif pm25_avg >= pm25_thresholds.get("moderate", 50):
            return SafetyLevel.MODERATE
        
        # Check PM10 level
        if pm10_avg >= pm10_thresholds.get("hazardous", 400):
            return SafetyLevel.HAZARDOUS
        elif pm10_avg >= pm10_thresholds.get("very_unhealthy", 300):
            return SafetyLevel.VERY_UNHEALTHY
        elif pm10_avg >= pm10_thresholds.get("unhealthy", 200):
            return SafetyLevel.UNHEALTHY
        elif pm10_avg >= pm10_thresholds.get("unhealthy_sensitive", 150):
            return SafetyLevel.UNHEALTHY_SENSITIVE
        elif pm10_avg >= pm10_thresholds.get("moderate", 100):
            return SafetyLevel.MODERATE
        
        # Check temperature extremes
        if temp_avg < self.temp_thresholds.get("too_cold_exercise", 0):
            return SafetyLevel.UNHEALTHY
        if temp_avg > self.temp_thresholds.get("too_hot_exercise", 35):
            return SafetyLevel.UNHEALTHY
        
        return SafetyLevel.SAFE
    
    async def analyze(
        self,
        latitude: float,
        longitude: float,
        hours: int = 6,
        past_days: int = 0,
    ) -> AnalyzeResponse:
        """
        Analyze air quality and generate exercise guidance.
        
        This is the main entry point for air quality analysis.
        Implements the full agentic flow:
        Plan → Fetch → Validate → Cache → Reason → Respond
        
        Args:
            latitude: Location latitude (-90 to 90)
            longitude: Location longitude (-180 to 180)
            hours: Forecast hours (1-384 for future, 0-24 for historical)
            past_days: Historical days to fetch (0-92)
            
        Returns:
            AnalyzeResponse with data and guidance
            
        Raises:
            ValueError: If validation fails
        """
        logger.info(f"Analyzing: lat={latitude}, lon={longitude}, hours={hours}, past_days={past_days}")
        
        # Step 1: Validate inputs
        validation = validator.validate_analyze_request(latitude, longitude, hours)
        if not validation.is_valid:
            raise ValueError(f"Validation failed: {'; '.join(validation.errors)}")
        
        # Step 2: Check cache
        cache_key_params = {
            "latitude": round(latitude, 4),
            "longitude": round(longitude, 4),
            "hours": hours,
            "past_days": past_days,
        }
        
        cached_result = self.memory.get("analysis", **cache_key_params)
        if cached_result:
            logger.info("Cache HIT - returning cached result")
            cached_result["cached"] = True
            return AnalyzeResponse(**cached_result)
        
        logger.info("Cache MISS - fetching fresh data")
        
        # Step 3: Fetch data from APIs (in parallel)
        try:
            air_quality, weather = await self.weather.get_combined_data(
                latitude, longitude, hours, past_days
            )
        except Exception as e:
            logger.error(f"Failed to fetch data: {str(e)}")
            raise
        
        # Step 4: Validate data quality
        air_quality_score = air_quality.data_quality
        weather_quality_score = weather.data_quality
        overall_quality = (air_quality_score + weather_quality_score) / 2
        
        data_quality = DataQuality(get_data_quality_level(overall_quality))
        
        # Step 5: Calculate averages
        pm25_avg = air_quality.pm25_avg
        pm10_avg = air_quality.pm10_avg
        temp_avg = weather.temp_avg
        temp_min = weather.temp_min
        temp_max = weather.temp_max
        
        # Step 6: Determine safety level
        safety_level = self._determine_safety_level(pm25_avg, pm10_avg, temp_avg)
        
        # Step 7: Generate LLM guidance
        try:
            user_prompt = AirQualityPrompts.format_user_prompt(
                latitude=latitude,
                longitude=longitude,
                hours=hours,
                pm25_avg=pm25_avg,
                pm10_avg=pm10_avg,
                temp_avg=temp_avg,
                temp_min=temp_min,
                temp_max=temp_max,
                air_quality_score=air_quality_score,
                weather_quality_score=weather_quality_score,
            )
            
            guidance_text = await self.llm.generate_guidance(
                system_prompt=AirQualityPrompts.SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
        except Exception as e:
            logger.warning(f"LLM failed, using fallback: {str(e)}")
            guidance_text = self._generate_fallback_guidance(
                pm25_avg, pm10_avg, temp_avg, safety_level
            )
        
        # Step 8: Build response
        result = {
            "pm25_avg": round(pm25_avg, 2),
            "pm10_avg": round(pm10_avg, 2),
            "temp_avg": round(temp_avg, 2),
            "guidance_text": guidance_text,
            "safety_level": safety_level,
            "data_quality": data_quality,
            "forecast_hours": hours,
            "attribution": SAFETY_RULES["attribution"]["weather_data"],
            "cached": False,
            "timestamp": datetime.utcnow(),
        }
        
        # Step 9: Store in cache
        self.memory.set("analysis", result, **cache_key_params)
        
        return AnalyzeResponse(**result)
    
    def _generate_fallback_guidance(
        self,
        pm25_avg: float,
        pm10_avg: float,
        temp_avg: float,
        safety_level: SafetyLevel,
    ) -> str:
        """
        Generate fallback guidance when LLM is unavailable.
        
        Args:
            pm25_avg: PM2.5 average
            pm10_avg: PM10 average
            temp_avg: Temperature average
            safety_level: Determined safety level
            
        Returns:
            Basic guidance text
        """
        recommendations = SAFETY_RULES.get("exercise_recommendations", {})
        rec = recommendations.get(safety_level.value.lower(), {})
        
        icon = rec.get("icon", "ℹ️")
        rec_text = rec.get(
            "recommendation",
            "Unable to generate detailed guidance. Check current conditions."
        )
        
        return (
            f"{icon} PM2.5: {pm25_avg:.1f} µg/m³, PM10: {pm10_avg:.1f} µg/m³, "
            f"Temp: {temp_avg:.1f}°C. {rec_text}"
        )
    
    async def get_apod(self) -> APODResponse:
        """
        Get NASA Astronomy Picture of the Day with summary.
        
        Returns:
            APODResponse with image info and 2-line summary
        """
        logger.info("Fetching NASA APOD")
        
        # Check cache (APOD changes daily)
        cached = self.memory.get("apod", date=datetime.utcnow().date().isoformat())
        if cached:
            logger.info("APOD cache HIT")
            return APODResponse(**cached)
        
        logger.info("APOD cache MISS - fetching from NASA")
        
        # Fetch from NASA
        try:
            apod_data = await self.nasa.get_today()
        except Exception as e:
            logger.error(f"Failed to fetch APOD: {str(e)}")
            raise
        
        # Generate summary with LLM
        try:
            user_prompt = APODPrompts.format_user_prompt(
                title=apod_data.title,
                date=apod_data.date,
                explanation=apod_data.explanation,
            )
            
            summary = await self.llm.generate_guidance(
                system_prompt=APODPrompts.SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
        except Exception as e:
            logger.warning(f"LLM summary failed: {str(e)}")
            # Fallback: first two sentences of explanation
            sentences = apod_data.explanation.split(". ")[:2]
            summary = ". ".join(sentences) + "."
        
        # Build response
        result = {
            "title": apod_data.title,
            "url": apod_data.url,
            "explanation": apod_data.explanation,
            "summary": summary,
            "date": apod_data.date,
            "media_type": apod_data.media_type,
            "hdurl": apod_data.hdurl,
            "attribution": SAFETY_RULES["attribution"]["nasa_apod"],
        }
        
        # Cache for the day
        self.memory.set(
            "apod",
            result,
            date=datetime.utcnow().date().isoformat(),
            ttl_seconds=86400,  # 24 hours
        )
        
        return APODResponse(**result)
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics for monitoring."""
        return self.memory.stats
