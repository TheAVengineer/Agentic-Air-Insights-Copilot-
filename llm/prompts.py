"""
Prompt templates for LLM reasoning.

This module provides structured prompts for:
- Air quality analysis and exercise guidance
- NASA APOD summarization
- Natural language query parsing

Prompts are designed to:
- Be specific and actionable
- Include relevant data context
- Request structured reasoning
- Handle uncertainty gracefully
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PromptTemplate:
    """A prompt template with system and user components."""
    
    name: str
    system_prompt: str
    user_template: str
    description: str


class AirQualityPrompts:
    """Prompts for air quality analysis and exercise guidance."""
    
    SYSTEM_PROMPT = """You are an Air Quality & Exercise Safety Advisor. Your role is to analyze weather and air quality data and provide clear, actionable guidance about outdoor exercise safety.

Key responsibilities:
1. Assess PM2.5 and PM10 levels against WHO guidelines
2. Consider temperature for exercise comfort and safety
3. Provide specific, actionable recommendations
4. Explain your reasoning briefly
5. Note any data quality concerns

WHO Air Quality Guidelines:
- PM2.5: Safe <25 µg/m³, Moderate 25-50, Unhealthy >50
- PM10: Safe <50 µg/m³, Moderate 50-100, Unhealthy >100

Temperature Guidelines for Exercise:
- Optimal: 10-25°C
- Caution (cold): <10°C
- Caution (hot): >30°C
- Dangerous: <0°C or >35°C

Response format based on time period:
- Short periods (≤12h): Give specific recommendation for that window
- Medium periods (12-48h): Mention best times if conditions vary
- Long periods (>48h): Give overview and suggest checking closer to activity time

Always include:
- A clear safety verdict (✅ Safe, ⚠️ Caution, 🔴 Avoid)
- Key numbers (PM2.5, PM10, Temperature range)
- 1-2 sentence justification
- For longer periods: suggest best time windows if possible"""

    USER_TEMPLATE = """Analyze the following air quality and weather data for outdoor exercise safety:

Location: {latitude}°N, {longitude}°E
Forecast period: Next {hours} hours {period_context}

Air Quality Data:
- PM2.5 Average: {pm25_avg:.1f} µg/m³
- PM10 Average: {pm10_avg:.1f} µg/m³
- Data Quality: {air_quality_score:.0%} of readings available

Weather Data:
- Temperature Average: {temp_avg:.1f}°C
- Temperature Range: {temp_min:.1f}°C to {temp_max:.1f}°C
- Data Quality: {weather_quality_score:.0%} of readings available

{data_warnings}

Question: Is it safe to exercise outdoors (running, cycling, etc.) during this period?

Provide a clear, actionable response in 3-4 sentences maximum."""

    LOW_QUALITY_WARNING = """⚠️ DATA QUALITY WARNING: Some data points are missing ({quality_score:.0%} available). 
The guidance below should be treated with caution. Consider checking current conditions before exercising."""

    @classmethod
    def format_user_prompt(
        cls,
        latitude: float,
        longitude: float,
        hours: int,
        pm25_avg: float,
        pm10_avg: float,
        temp_avg: float,
        temp_min: float,
        temp_max: float,
        air_quality_score: float,
        weather_quality_score: float,
    ) -> str:
        """Format the user prompt with actual data."""
        
        # Check for data quality issues
        data_warnings = ""
        overall_quality = (air_quality_score + weather_quality_score) / 2
        
        if overall_quality < 0.8:
            data_warnings = cls.LOW_QUALITY_WARNING.format(
                quality_score=overall_quality
            )
        
        # Add context about the time period
        if hours <= 6:
            period_context = "(short-term forecast)"
        elif hours <= 24:
            period_context = "(~1 day)"
        elif hours <= 48:
            period_context = f"(~{hours // 24} days)"
        else:
            period_context = f"(~{hours // 24} days - consider rechecking closer to activity)"
        
        return cls.USER_TEMPLATE.format(
            latitude=latitude,
            longitude=longitude,
            hours=hours,
            period_context=period_context,
            pm25_avg=pm25_avg,
            pm10_avg=pm10_avg,
            temp_avg=temp_avg,
            temp_min=temp_min,
            temp_max=temp_max,
            air_quality_score=air_quality_score,
            weather_quality_score=weather_quality_score,
            data_warnings=data_warnings,
        )


class APODPrompts:
    """Prompts for NASA APOD summarization."""
    
    SYSTEM_PROMPT = """You are an astronomy educator who explains NASA's Astronomy Picture of the Day to a general audience. Your summaries should be:

1. Accessible - avoid jargon, explain technical terms
2. Engaging - capture the wonder of space
3. Accurate - stick to the facts in the explanation
4. Concise - 2 lines maximum

Format: Two complete sentences that cover what the image shows and why it's significant."""

    USER_TEMPLATE = """Summarize this NASA Astronomy Picture of the Day in exactly 2 lines:

Title: {title}
Date: {date}

Original Explanation:
{explanation}

Provide a 2-line summary that a general audience would find engaging and informative."""

    @classmethod
    def format_user_prompt(
        cls,
        title: str,
        date: str,
        explanation: str,
    ) -> str:
        """Format the APOD summarization prompt."""
        # Truncate very long explanations
        if len(explanation) > 1500:
            explanation = explanation[:1500] + "..."
        
        return cls.USER_TEMPLATE.format(
            title=title,
            date=date,
            explanation=explanation,
        )


class LocationExtractionPrompts:
    """Prompts for extracting location information from natural language."""
    
    SYSTEM_PROMPT = """You are a location extraction assistant. Your job is to identify and extract location information from user queries and provide accurate latitude/longitude coordinates.

You have knowledge of cities, countries, landmarks, and regions worldwide. When a user mentions a place name, provide its approximate geographic center coordinates.

IMPORTANT:
- Be accurate with coordinates - use well-known reference points
- If multiple places have the same name, choose the most famous/populous one
- Return coordinates to 4 decimal places
- If you cannot confidently identify the location, return null coordinates
- ALWAYS indicate if the location is a COUNTRY (not a city) - this is important!

Always respond in valid JSON format only."""

    USER_TEMPLATE = """Extract location information from this query:

"{query}"

If a location (city, country, landmark, region, or coordinates) is mentioned, extract:
- The location name as understood
- Latitude and longitude coordinates
- Whether this is a COUNTRY or a city/place

Respond ONLY with valid JSON (no markdown, no code blocks):
{{
    "location_found": true/false,
    "location_name": "extracted location name or null",
    "is_country": true/false,
    "latitude": float or null,
    "longitude": float or null,
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}

Examples:
- "air quality in Sofia" → Sofia, Bulgaria → 42.6977, 23.3219, is_country: false
- "what about Norway?" → Norway → 60.472, 8.4689, is_country: true
- "Serbia?" → Serbia → 44.0165, 21.0059, is_country: true
- "weather at 40.7128, -74.0060" → Coordinates given → 40.7128, -74.0060, is_country: false
- "check Tokyo air" → Tokyo, Japan → 35.6824, 139.769, is_country: false"""

    @classmethod
    def format_user_prompt(cls, query: str) -> str:
        """Format the location extraction prompt."""
        return cls.USER_TEMPLATE.format(query=query)


class IntentParsingPrompts:
    """Prompts for parsing natural language queries into structured intents."""
    
    SYSTEM_PROMPT = """You are a query parser for an Air Quality & Weather assistant. 
Parse user queries and extract structured information.

Supported intents:
1. "analyze" - Air quality/weather analysis for a location
2. "apod" - NASA Astronomy Picture of the Day request
3. "help" - User needs help or has a question about the system
4. "unknown" - Query doesn't match any supported intent

For "analyze" intent, extract:
- latitude (float, -90 to 90)
- longitude (float, -180 to 180) 
- hours (int, 1-384, default 6, supports up to 16 days)

Respond in JSON format only."""

    USER_TEMPLATE = """Parse this user query:

"{query}"

Respond with valid JSON:
{{
    "intent": "analyze|apod|help|unknown",
    "confidence": 0.0-1.0,
    "extracted": {{
        "latitude": float or null,
        "longitude": float or null,
        "hours": int or null
    }},
    "clarification_needed": "string or null"
}}"""

    @classmethod
    def format_user_prompt(cls, query: str) -> str:
        """Format the intent parsing prompt."""
        return cls.USER_TEMPLATE.format(query=query)


class PromptLibrary:
    """Central access point for all prompt templates."""
    
    air_quality = AirQualityPrompts
    apod = APODPrompts
    intent_parsing = IntentParsingPrompts
    location_extraction = LocationExtractionPrompts
    
    @classmethod
    def get_air_quality_prompts(cls) -> tuple[str, str]:
        """Get system and user template for air quality."""
        return (
            cls.air_quality.SYSTEM_PROMPT,
            cls.air_quality.USER_TEMPLATE,
        )
    
    @classmethod
    def get_apod_prompts(cls) -> tuple[str, str]:
        """Get system and user template for APOD."""
        return (
            cls.apod.SYSTEM_PROMPT,
            cls.apod.USER_TEMPLATE,
        )
    
    @classmethod
    def get_location_extraction_prompts(cls) -> tuple[str, str]:
        """Get system and user template for location extraction."""
        return (
            cls.location_extraction.SYSTEM_PROMPT,
            cls.location_extraction.USER_TEMPLATE,
        )
