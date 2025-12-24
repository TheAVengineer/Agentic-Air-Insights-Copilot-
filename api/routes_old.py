"""
FastAPI routes for the Air & Insights API.

Endpoints:
- POST /analyze: Air quality analysis with exercise guidance
- GET /apod/today: NASA Astronomy Picture of the Day
- GET /health: Health check
- POST /chat: Natural language chat interface

All endpoints are designed to be imported into Copilot Studio
via the generated OpenAPI specification.
"""

import logging
import json
import re
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from agent.orchestrator import AirInsightsAgent
from api.models import (
    AnalyzeRequest,
    AnalyzeResponse,
    APODResponse,
    HealthResponse,
    ErrorResponse,
    ChatRequest,
    ChatResponse,
)
from llm.client import LLMClient
import httpx

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Global agent instance (initialized in main.py)
_agent: Optional[AirInsightsAgent] = None

# Global LLM client instance for location extraction
_llm_client: Optional[LLMClient] = None

# Conversation context memory (simple state for follow-up queries)
_conversation_context: dict = {
    "last_intent": None,      # "analyze", "forecast", or "apod"
    "last_hours": 6,          # Remember hours setting
    "last_location": None,    # Remember last location name (for follow-ups)
    "last_coords": None,      # Remember last coordinates (lat, lon)
    "awaiting_location": False,  # True when we asked for a location
}


def get_agent() -> AirInsightsAgent:
    """Get the global agent instance."""
    global _agent
    if _agent is None:
        _agent = AirInsightsAgent()
    return _agent


def get_llm_client() -> LLMClient:
    """Get the global LLM client instance."""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


def get_conversation_context() -> dict:
    """Get conversation context."""
    return _conversation_context


def update_conversation_context(
    intent: str = None, 
    hours: int = None, 
    location: str = None, 
    coords: tuple = None,
    awaiting_location: bool = False
) -> None:
    """Update conversation context after a successful query."""
    global _conversation_context
    if intent:
        _conversation_context["last_intent"] = intent
    if hours:
        _conversation_context["last_hours"] = hours
    if location:
        _conversation_context["last_location"] = location
    if coords:
        _conversation_context["last_coords"] = coords
    _conversation_context["awaiting_location"] = awaiting_location


def set_agent(agent: AirInsightsAgent) -> None:
    """Set the global agent instance (for testing)."""
    global _agent
    _agent = agent


# =============================================================================
# Unified LLM Query Parser
# =============================================================================

async def parse_query_with_llm(query: str, context: dict) -> dict:
    """
    Use LLM to intelligently parse user query in one call.
    
    Extracts: intent, location, time_period, is_followup
    Uses conversation context for follow-up understanding.
    
    Returns dict with:
    - intent: "analyze" | "forecast" | "apod" | "help" | "greeting" | "unknown"
    - location: str or None (city/place name)
    - hours: int (time period in hours, default 6)
    - is_followup: bool (uses context from previous query)
    - needs_location: bool (query needs location but none found)
    """
    llm = get_llm_client()
    
    # Build context info for LLM
    context_info = ""
    if context.get("last_location"):
        context_info = f"""
Previous conversation context:
- Last location discussed: {context["last_location"]}
- Last intent: {context["last_intent"]}
- Last time period: {context.get("last_hours", 6)} hours
"""
    
    system_prompt = """You are a query parser for an Air & Insights agent. Parse the user's query and extract structured information.

The agent can help with:
1. AIR QUALITY ANALYSIS - checking if it's safe to exercise outdoors (PM2.5, PM10, temperature)
2. WEATHER FORECAST - multi-day weather forecasts (up to 16 days)
3. NASA APOD - astronomy picture of the day

IMPORTANT RULES:
- "weather" alone = forecast intent
- "air quality" or "safe to run/exercise" = analyze intent
- If user mentions a TIME PERIOD (today, tomorrow, next 2 days, this week, etc.) without a new location, use the previous location from context
- "what about X" where X is a location = same intent as before, new location
- "what about X" where X is a time period = same intent as before, same location, new time
- Greetings (hi, hello, hey, heyo) = greeting intent
- Questions about capabilities = help intent

TIME PERIOD MAPPING (convert to hours):
- "today" = 12 hours
- "tomorrow" = 24 hours  
- "next X hours" = X hours
- "next X days" = X * 24 hours
- "this week" / "next week" / "a week" = 168 hours (7 days)
- "two weeks" / "2 weeks" = 336 hours (14 days)
- "16 days" = 384 hours (max)
- If no time mentioned and intent is forecast, default to 168 (week)
- If no time mentioned and intent is analyze, default to 6

Respond ONLY with valid JSON (no markdown):
{
  "intent": "analyze|forecast|apod|help|greeting|unknown",
  "location": "city name or null if not mentioned and no context",
  "hours": <number>,
  "is_followup": true/false,
  "needs_location": true/false
}"""

    user_prompt = f"""{context_info}
User query: "{query}"

Parse this query and return JSON."""

    try:
        response = await llm.chat([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ], temperature=0.1, max_tokens=200)
        
        # Parse JSON response
        # Clean up response (remove markdown code blocks if present)
        response = response.strip()
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        response = response.strip()
        
        result = json.loads(response)
        
        # Apply context for follow-ups
        if result.get("is_followup") and not result.get("location") and context.get("last_location"):
            result["location"] = context["last_location"]
            result["needs_location"] = False
        
        # Ensure hours is int
        result["hours"] = int(result.get("hours", 6))
        
        logger.info(f"LLM parsed query: {result}")
        return result
        
    except Exception as e:
        logger.warning(f"LLM query parsing failed: {e}, falling back to basic parsing")
        # Minimal fallback - just detect obvious patterns
        query_lower = query.lower()
        if any(kw in query_lower for kw in ["apod", "astronomy", "nasa", "space picture"]):
            return {"intent": "apod", "location": None, "hours": 6, "is_followup": False, "needs_location": False}
        if any(kw in query_lower for kw in ["help", "what can you"]):
            return {"intent": "help", "location": None, "hours": 6, "is_followup": False, "needs_location": False}
        if any(kw in query_lower for kw in ["hi", "hello", "hey"]):
            return {"intent": "greeting", "location": None, "hours": 6, "is_followup": False, "needs_location": False}
        return {"intent": "unknown", "location": None, "hours": 6, "is_followup": False, "needs_location": True}


# =============================================================================
# Health Endpoint
# =============================================================================

@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health Check",
    description="Check if the service is running and healthy.",
)
async def health_check() -> HealthResponse:
    """Return service health status."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.utcnow(),
    )


# =============================================================================
# Air Quality Analysis Endpoint
# =============================================================================

@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Validation Error"},
        500: {"model": ErrorResponse, "description": "Internal Error"},
    },
    tags=["Air Quality"],
    summary="Analyze Air Quality",
    description="""
Analyze air quality and weather data for a location and determine if it's safe for outdoor exercise.

**Required Parameters:**
- `latitude`: Location latitude (-90 to 90)
- `longitude`: Location longitude (-180 to 180)
- `hours`: Forecast period (1-72, default: 6)

**Returns:**
- PM2.5 and PM10 averages
- Temperature average
- Safety level (SAFE, MODERATE, UNHEALTHY, etc.)
- LLM-generated actionable guidance

**Attribution:** Weather data by Open-Meteo.com
""",
)
async def analyze_air_quality(request: AnalyzeRequest) -> AnalyzeResponse:
    """
    Analyze air quality and generate exercise guidance.
    
    This endpoint:
    1. Fetches weather and air quality data from Open-Meteo
    2. Validates data quality
    3. Determines safety level based on policy thresholds
    4. Generates actionable guidance using LLM
    5. Caches results for 10 minutes
    """
    logger.info(
        f"Analyze request: lat={request.latitude}, "
        f"lon={request.longitude}, hours={request.hours}"
    )
    
    try:
        agent = get_agent()
        result = await agent.analyze(
            latitude=request.latitude,
            longitude=request.longitude,
            hours=request.hours,
        )
        return result
        
    except ValueError as e:
        logger.warning(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "VALIDATION_ERROR",
                "message": str(e),
            }
        )
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "ANALYSIS_FAILED",
                "message": "Failed to analyze air quality. Please try again.",
            }
        )


# =============================================================================
# NASA APOD Endpoint
# =============================================================================

@router.get(
    "/apod/today",
    response_model=APODResponse,
    responses={
        500: {"model": ErrorResponse, "description": "Internal Error"},
    },
    tags=["NASA APOD"],
    summary="Get Today's APOD",
    description="""
Get NASA's Astronomy Picture of the Day with a 2-line LLM-generated summary.

**Returns:**
- Image/video title and URL
- Original NASA explanation
- 2-line accessible summary
- HD image URL (if available)

**Attribution:** Image from NASA Astronomy Picture of the Day
""",
)
async def get_apod_today() -> APODResponse:
    """
    Get today's NASA Astronomy Picture of the Day.
    
    Includes LLM-generated 2-line summary for accessibility.
    Results are cached for 24 hours.
    """
    logger.info("APOD request")
    
    try:
        agent = get_agent()
        result = await agent.get_apod()
        return result
        
    except Exception as e:
        logger.error(f"APOD fetch failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "APOD_FAILED",
                "message": "Failed to fetch APOD. Please try again.",
            }
        )


# =============================================================================
# Chat Endpoint (for Web UI)
# =============================================================================

@router.post(
    "/chat",
    response_model=ChatResponse,
    tags=["Chat"],
    summary="Natural Language Chat",
    description="""
Process a natural language query about air quality, weather, or NASA APOD.

**Supported queries:**
- Air quality questions (e.g., "Is it safe to run in Sofia?")
- Weather forecasts (e.g., "What's the weather this week in Paris?")
- NASA APOD requests (e.g., "Show me today's astronomy picture")
- Follow-up questions (e.g., "What about tomorrow?" or "What about London?")

The system uses LLM to understand queries intelligently.
""",
)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Handle natural language queries with LLM-based understanding.
    
    Uses a unified LLM parser to extract intent, location, and time period.
    Maintains conversation context for natural follow-ups.
    """
    query = request.message.strip()
    logger.info(f"Chat request: {query[:100]}...")
    
    # Get current context for follow-up understanding
    context = get_conversation_context()
    
    # Use LLM to parse the query intelligently
    parsed = await parse_query_with_llm(query, context)
    intent = parsed["intent"]
    location_name = parsed.get("location")
    hours = parsed.get("hours", 6)
    needs_location = parsed.get("needs_location", False)
    
    logger.info(f"Parsed: intent={intent}, location={location_name}, hours={hours}")
    
    # Handle APOD
    if intent == "apod":
        try:
            agent = get_agent()
            apod = await agent.get_apod()
            response = (
                f"🌟 **{apod.title}**\n\n"
                f"{apod.summary}\n\n"
                f"[View Image]({apod.url})\n\n"
                f"_{apod.attribution}_"
            )
            update_conversation_context("apod")
            return ChatResponse(response=response, intent="apod", data={"title": apod.title, "url": apod.url})
        except Exception as e:
            return ChatResponse(response=f"Sorry, I couldn't fetch today's APOD: {str(e)}", intent="apod")
    
    # Handle air quality analysis or weather forecast
    if intent in ["analyze", "forecast"]:
        coords = None
        
        # Try to geocode the location from LLM parsing
        if location_name:
            geocode_result = await geocode_location(location_name)
            if geocode_result:
                if geocode_result.get("is_country"):
                    # Country detected - ask for specific city
                    country = geocode_result.get("country", location_name)
                    cities, info = await get_country_cities_from_llm(country)
                    cities_str = ", ".join(cities[:5]) if cities else "major cities"
                    response = (
                        f"🌍 **{country}** is a country. For accurate data, please specify a city.\n\n"
                        f"**Try:** {cities_str}"
                    )
                    update_conversation_context(intent, awaiting_location=True)
                    return ChatResponse(response=response, intent=intent, data={"needs": "city", "country": country})
                else:
                    coords = geocode_result.get("coords")
                    location_name = geocode_result.get("location_name", location_name)
        
        # Also check for explicit coordinates in query
        if not coords:
            coords = extract_coordinates_from_pattern(query)
        
        # If we have coordinates, proceed with the request
        if coords:
            lat, lon = coords
            try:
                if intent == "analyze":
                    # Air quality analysis
                    agent = get_agent()
                    result = await agent.analyze(lat, lon, hours)
                    
                    # Format time period naturally
                    if hours <= 12:
                        time_str = f"next {hours} hours"
                    elif hours <= 24:
                        time_str = "today"
                    elif hours <= 48:
                        time_str = f"next {hours // 24} days"
                    else:
                        time_str = f"next {hours // 24} days"
                    
                    response = (
                        f"📍 **Air Quality Analysis for {location_name}** ({time_str})\n\n"
                        f"{result.guidance_text}\n\n"
                        f"**Data:**\n"
                        f"- PM2.5: {result.pm25_avg:.1f} µg/m³\n"
                        f"- PM10: {result.pm10_avg:.1f} µg/m³\n"
                        f"- Temperature: {result.temp_avg:.1f}°C\n\n"
                        f"_{result.attribution}_"
                    )
                    update_conversation_context("analyze", hours, location_name, coords)
                    return ChatResponse(
                        response=response,
                        intent="analyze",
                        data={"location": location_name, "hours": hours, "safety": result.safety_level.value}
                    )
                
                else:  # forecast
                    # Weather forecast
                    days = max(1, (hours + 23) // 24)
                    days = min(days, 16)
                    
                    from tools.weather_client import WeatherClient
                    client = WeatherClient()
                    daily = await client.get_weekly_forecast(lat, lon, days=days)
                    
                    # Build forecast data for LLM
                    forecast_lines = []
                    for i in range(min(days, len(daily.dates))):
                        summary = daily.get_day_summary(i)
                        precip = summary.get('precipitation', 0) or 0
                        precip_str = f", {precip:.1f}mm" if precip > 0 else ""
                        forecast_lines.append(
                            f"- {summary['date']}: {summary['emoji']} {summary['weather']}, "
                            f"{summary['temp_min']:.0f}° to {summary['temp_max']:.0f}°C{precip_str}"
                        )
                    
                    # Get LLM summary
                    llm = get_llm_client()
                    llm_response = await llm.chat([
                        {"role": "system", "content": "You are a weather assistant. Summarize the forecast concisely. Mention best/worst days for outdoor activities."},
                        {"role": "user", "content": f"User asked: {query}\n\nLocation: {location_name}\n\n{days}-day forecast:\n" + "\n".join(forecast_lines)}
                    ], temperature=0.3, max_tokens=250)
                    
                    # Build visual table for short forecasts
                    if days <= 7:
                        from datetime import datetime as dt
                        table = []
                        for i in range(min(days, len(daily.dates))):
                            s = daily.get_day_summary(i)
                            try:
                                d = dt.strptime(s['date'], "%Y-%m-%d").strftime("%a %d")
                            except:
                                d = s['date']
                            table.append(f"| {d} | {s['emoji']} | {s['temp_min']:.0f}° - {s['temp_max']:.0f}° |")
                        response = f"📅 **{days}-Day Forecast for {location_name}**\n\n{llm_response}\n\n" + "\n".join(table) + "\n\n_Weather data by Open-Meteo.com_"
                    else:
                        response = f"📅 **{days}-Day Forecast for {location_name}**\n\n{llm_response}\n\n_Weather data by Open-Meteo.com_"
                    
                    update_conversation_context("forecast", hours, location_name, coords)
                    return ChatResponse(response=response, intent="forecast", data={"location": location_name, "days": days})
                    
            except Exception as e:
                logger.error(f"Request failed: {e}")
                return ChatResponse(response=f"Sorry, something went wrong: {str(e)}", intent=intent)
        
        # No location found - ask user
        update_conversation_context(intent, awaiting_location=True)
        if intent == "analyze":
            return ChatResponse(
                response="I'd be happy to check the air quality! Please tell me a location.\n\n**Examples:** \"Sofia\", \"Paris, France\", or coordinates",
                intent="analyze",
                data={"needs": "location"}
            )
        else:
            return ChatResponse(
                response="I'd be happy to check the forecast! Please tell me a location.\n\n**Example:** \"What's the weather this week in Paris?\"",
                intent="forecast",
                data={"needs": "location"}
            )
    
    # Handle help
    if intent == "help":
        response = (
            "👋 **Air & Insights Agent**\n\n"
            "I can help you with:\n\n"
            "🏃 **Air Quality** - \"Is it safe to run in Sofia?\"\n"
            "🌤️ **Weather Forecast** - \"What's the weather this week in Paris?\"\n"
            "🌟 **NASA APOD** - \"Show me today's astronomy picture\"\n\n"
            "Just ask naturally - I understand follow-ups too!"
        )
        return ChatResponse(response=response, intent="help")
    
    # Handle greetings and unknown intents with LLM
    llm = get_llm_client()
    try:
        llm_response = await llm.chat([
            {"role": "system", "content": (
                "You are Air & Insights Agent. You help with:\n"
                "1. Air quality - checking if safe to exercise outdoors\n"
                "2. Weather forecasts - up to 16 days\n"
                "3. NASA APOD - astronomy picture\n\n"
                "For greetings: respond warmly, briefly explain capabilities.\n"
                "For off-topic: politely redirect to what you CAN do.\n"
                "Be concise and friendly."
            )},
            {"role": "user", "content": query}
        ], temperature=0.5, max_tokens=150)
        return ChatResponse(response=llm_response, intent="greeting" if intent == "greeting" else "chat")
    except Exception as e:
        logger.error(f"LLM chat failed: {e}")
        return ChatResponse(
            response="Hello! 👋 I help with air quality, weather forecasts, and NASA's astronomy picture. What would you like to know?",
            intent="chat"
        )


# =============================================================================
# Helper Functions
# =============================================================================

async def geocode_location(location_name: str) -> Optional[dict]:
    """
    Use OpenStreetMap Nominatim API to geocode a location name.
    
    This is a free geocoding service that doesn't require an API key.
    
    Returns dict with:
    - coords: tuple (latitude, longitude) or None
    - location_name: str (display name from Nominatim)
    - location_type: str ("city", "country", "region", etc.)
    - country: str (country name if available)
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://nominatim.openstreetmap.org/search",
                params={
                    "q": location_name,
                    "format": "json",
                    "limit": 1,
                    "addressdetails": 1,  # Get address breakdown
                },
                headers={
                    "User-Agent": "AirInsightsAgent/1.0 (contact@example.com)",
                },
                timeout=10.0,
            )
            response.raise_for_status()
            results = response.json()
            
            if results:
                result = results[0]
                lat = float(result["lat"])
                lon = float(result["lon"])
                display_name = result.get("display_name", location_name)
                location_type = result.get("type", "unknown")
                osm_class = result.get("class", "")
                
                # Extract address details
                address = result.get("address", {})
                country = address.get("country", "")
                address_type = result.get("addresstype", "")
                
                # Determine if this is a country-level result
                # Use addresstype field (most reliable) or fallback to type+class check
                is_country = (
                    address_type == "country" or
                    (location_type in ["country", "administrative"] and 
                     osm_class == "boundary" and
                     not address.get("city") and 
                     not address.get("town") and
                     not address.get("village"))
                )
                
                # Extract just the city/place name (first part before comma)
                short_name = display_name.split(",")[0].strip()
                
                logger.info(f"Geocoded '{location_name}' -> {short_name} ({lat}, {lon}), type={location_type}, is_country={is_country}")
                return {
                    "coords": (lat, lon),
                    "location_name": short_name,
                    "location_type": "country" if is_country else location_type,
                    "is_country": is_country,
                    "country": country or short_name,
                    "display_name": display_name,
                }
        
        return None
        
    except Exception as e:
        logger.warning(f"Geocoding error for '{location_name}': {e}")
        return None


async def get_country_cities_from_llm(country_name: str) -> tuple[list, str]:
    """
    Use LLM to get major cities for a country.
    
    Returns:
        tuple: (list of city names, first city name for example)
    """
    try:
        llm = get_llm_client()
        
        prompt = f"""List exactly 4 major cities in {country_name} that would be good for checking air quality.
Return ONLY the city names separated by commas, nothing else.
Example format: Beijing, Shanghai, Guangzhou, Shenzhen"""
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant that provides concise geographic information. Return only what is asked, no explanations."},
            {"role": "user", "content": prompt},
        ]
        
        response = await llm.chat(messages, temperature=0.3, max_tokens=100)
        
        # Parse the response - split by comma and clean up
        cities = [city.strip() for city in response.strip().split(",") if city.strip()]
        
        if cities:
            logger.info(f"LLM suggested cities for {country_name}: {cities}")
            return cities[:4], cities[0]  # Return max 4 cities
        
        return [], ""
        
    except Exception as e:
        logger.warning(f"Failed to get cities from LLM for {country_name}: {e}")
        return [], ""


def extract_coordinates_from_pattern(query: str) -> Optional[tuple[float, float]]:
    """
    Extract latitude/longitude from explicit coordinate patterns in query text.
    
    Looks for patterns like:
    - 42.6977, 23.3219
    - lat 42.6977 lon 23.3219
    - (42.6977, 23.3219)
    """
    patterns = [
        r'(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)',  # 42.6977, 23.3219
        r'lat(?:itude)?\s*[=:]?\s*(-?\d+\.?\d*)\s*,?\s*lon(?:gitude)?\s*[=:]?\s*(-?\d+\.?\d*)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            try:
                lat = float(match.group(1))
                lon = float(match.group(2))
                # Validate ranges
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    return (lat, lon)
            except (ValueError, IndexError):
                continue
    
    return None


# =============================================================================
# Cache Stats Endpoint (for monitoring)
# =============================================================================

@router.get(
    "/cache/stats",
    tags=["System"],
    summary="Cache Statistics",
    description="Get cache hit/miss statistics for monitoring.",
)
async def get_cache_stats() -> dict:
    """Return cache statistics."""
    agent = get_agent()
    return agent.get_cache_stats()
