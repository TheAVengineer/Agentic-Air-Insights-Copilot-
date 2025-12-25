"""
FastAPI routes for the Air & Insights API.

This module is intentionally thin - it handles HTTP concerns only
and delegates business logic to proper service layers:

- agent/query_parser.py: LLM-based natural language understanding
- agent/orchestrator.py: Business logic coordination
- agent/memory.py: Caching and conversation context
- tools/geocoding.py: Location name to coordinates
- tools/weather_client.py: Weather/air quality data
- tools/nasa_client.py: NASA APOD data
"""

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException

from agent.orchestrator import AirInsightsAgent
from agent.query_parser import QueryParser
from agent.memory import ConversationContext
from tools.geocoding import GeocodingService
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

logger = logging.getLogger(__name__)

# =============================================================================
# Service Instances (Dependency Injection ready)
# =============================================================================

router = APIRouter()

# Services - could be injected via FastAPI Depends() for testing
_agent: Optional[AirInsightsAgent] = None
_query_parser: Optional[QueryParser] = None
_geocoding: Optional[GeocodingService] = None
_context: Optional[ConversationContext] = None


def get_agent() -> AirInsightsAgent:
    global _agent
    if _agent is None:
        _agent = AirInsightsAgent()
    return _agent


def get_query_parser() -> QueryParser:
    global _query_parser
    if _query_parser is None:
        _query_parser = QueryParser()
    return _query_parser


def get_geocoding() -> GeocodingService:
    global _geocoding
    if _geocoding is None:
        _geocoding = GeocodingService()
    return _geocoding


def get_context() -> ConversationContext:
    global _context
    if _context is None:
        _context = ConversationContext()
    return _context


def set_agent(agent: AirInsightsAgent) -> None:
    """For testing."""
    global _agent
    _agent = agent


# =============================================================================
# Health & Status Endpoints
# =============================================================================

@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Service health check."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.utcnow(),
    )


@router.get("/status/llm", tags=["System"])
async def llm_status():
    """
    Check LLM provider status.
    
    Returns which LLM providers are available:
    - GitHub Models (primary)
    - Ollama (fallback)
    """
    from llm.client import LLMClient
    client = LLMClient()
    
    # Check Ollama availability
    await client._check_ollama_available()
    
    return {
        "providers": client.get_provider_status(),
        "message": "Use Ollama as fallback when GitHub Models is rate limited. "
                   "Install: https://ollama.ai, then run: ollama pull llama3.2"
    }


@router.get("/status/rate-limits", tags=["System"])
async def rate_limits_status():
    """
    Check rate limit status for all LLM providers.
    
    Returns:
    - GitHub Models: Limited to ~150 requests/day (free tier)
    - Ollama: Unlimited (local)
    
    Useful for monitoring API usage and planning fallback strategies.
    """
    import httpx
    from llm.client import LLMClient
    
    client = LLMClient()
    status = {
        "github_models": {
            "provider": "GitHub Models (Azure)",
            "model": client.github_model,
            "rate_limit": "~150 requests/day (free tier)",
            "status": "unknown",
            "available": False,
            "reset_info": None,
        },
        "ollama": {
            "provider": "Ollama (local)",
            "model": client.ollama_model,
            "rate_limit": "unlimited",
            "status": "unknown",
            "available": False,
        },
        "active_provider": None,
        "recommendation": None,
    }
    
    # Check GitHub Models availability
    if client.github_api_key:
        try:
            # Make a minimal test request to check rate limits
            async with httpx.AsyncClient(timeout=5.0) as http_client:
                response = await http_client.post(
                    f"{client.GITHUB_MODELS_ENDPOINT}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {client.github_api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": client.github_model,
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 1,
                    }
                )
                
                if response.status_code == 200:
                    status["github_models"]["status"] = "available"
                    status["github_models"]["available"] = True
                    status["active_provider"] = "github_models"
                elif response.status_code == 429:
                    status["github_models"]["status"] = "rate_limited"
                    # Try to extract reset time from headers
                    retry_after = response.headers.get("Retry-After")
                    if retry_after:
                        try:
                            seconds = int(retry_after)
                            hours = seconds // 3600
                            minutes = (seconds % 3600) // 60
                            status["github_models"]["reset_info"] = f"Resets in ~{hours}h {minutes}m"
                        except ValueError:
                            status["github_models"]["reset_info"] = f"Retry after: {retry_after}"
                else:
                    status["github_models"]["status"] = f"error ({response.status_code})"
        except Exception as e:
            status["github_models"]["status"] = f"error: {str(e)[:50]}"
    else:
        status["github_models"]["status"] = "no_api_key"
    
    # Check Ollama availability
    try:
        async with httpx.AsyncClient(timeout=3.0) as http_client:
            response = await http_client.get(f"{client.ollama_endpoint}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "").split(":")[0] for m in models]
                
                if client.ollama_model in model_names:
                    status["ollama"]["status"] = "available"
                    status["ollama"]["available"] = True
                    status["ollama"]["installed_models"] = model_names
                    if not status["github_models"]["available"]:
                        status["active_provider"] = "ollama"
                else:
                    status["ollama"]["status"] = f"model_not_found (available: {model_names})"
    except Exception as e:
        status["ollama"]["status"] = f"not_running: {str(e)[:30]}"
    
    # Set recommendation
    if status["github_models"]["available"]:
        status["recommendation"] = "GitHub Models available - primary provider active"
    elif status["ollama"]["available"]:
        status["recommendation"] = "Using Ollama fallback - GitHub Models rate limited or unavailable"
    else:
        status["recommendation"] = "⚠️ No LLM available! Start Ollama: `ollama serve` then `ollama pull llama3.2`"
    
    return status


# =============================================================================
# Air Quality Analysis Endpoint
# =============================================================================

@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["Air Quality"],
    summary="Analyze air quality for outdoor exercise",
    description="""
**Use this tool when the user asks about:**
- Air quality, PM2.5, PM10, pollution levels
- Is it safe to run/exercise/jog outside?
- Should I go for a walk/bike ride?
- Outdoor activity safety recommendations

**What it does:**
Fetches real-time air quality data (PM2.5, PM10) and temperature for the specified 
location and time period, then provides actionable guidance about whether outdoor 
exercise is safe.

**Input:** Latitude, longitude, and forecast hours (1-384, up to 16 days)
**Output:** Air quality metrics with safety level and LLM-generated exercise guidance
""",
)
async def analyze_air_quality(request: AnalyzeRequest) -> AnalyzeResponse:
    """Analyze air quality for outdoor exercise safety."""
    logger.info(f"Analyze: lat={request.latitude}, lon={request.longitude}, hours={request.hours}")
    
    try:
        result = await get_agent().analyze(
            latitude=request.latitude,
            longitude=request.longitude,
            hours=request.hours,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"error": "VALIDATION_ERROR", "message": str(e)})
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail={"error": "ANALYSIS_FAILED", "message": str(e)})


# =============================================================================
# NASA APOD Endpoint
# =============================================================================

@router.get(
    "/apod/today", 
    response_model=APODResponse, 
    tags=["NASA APOD"],
    summary="Get NASA Astronomy Picture of the Day",
    description="""
**Use this tool when the user asks about:**
- NASA picture, astronomy picture, APOD
- Space photo, today's space image
- What's the astronomy picture today?
- Show me something cool from NASA

**What it does:**
Fetches NASA's Astronomy Picture of the Day including the image URL, title, 
explanation, and an AI-generated summary suitable for conversation.

**Input:** None required
**Output:** APOD image URL, title, explanation, and AI summary
""",
)
async def get_apod_today() -> APODResponse:
    """Get NASA Astronomy Picture of the Day."""
    logger.info("APOD request")
    
    try:
        return await get_agent().get_apod()
    except Exception as e:
        logger.error(f"APOD failed: {e}")
        raise HTTPException(status_code=500, detail={"error": "APOD_FAILED", "message": str(e)})


# =============================================================================
# Chat Endpoint - Natural Language Interface
# =============================================================================

@router.post(
    "/chat", 
    response_model=ChatResponse, 
    tags=["Chat"],
    summary="Natural language chat for weather, air quality, and NASA APOD",
    description="""
**Use this tool for any natural language conversation about:**
- Weather forecasts (current, today, tomorrow, next week, next 16 days)
- Historical weather (yesterday, last week, past 3 months - up to 92 days)
- Air quality and outdoor exercise safety
- NASA Astronomy Picture of the Day
- Location-aware queries (cities, countries, coordinates)

**What it does:**
Accepts natural language queries and intelligently routes them to the appropriate 
data source. Supports conversation context for follow-up questions like 
"what about tomorrow?" or "how about Paris?".

**Input:** Natural language message (e.g., "Is it safe to run in Sofia today?")
**Output:** Formatted response with data and AI-generated guidance

**Examples:**
- "What's the weather in London for the next 3 days?"
- "Is it safe to exercise outside in Tokyo?"
- "Show me the NASA astronomy picture"
- "How was the air quality yesterday in Berlin?"
""",
)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Natural language chat interface.
    
    Delegates to:
    - QueryParser for understanding the query
    - GeocodingService for location resolution
    - Agent for data fetching and analysis
    """
    query = request.message.strip()
    logger.info(f"Chat: {query[:100]}...")
    
    # Get services
    parser = get_query_parser()
    geocoding = get_geocoding()
    context = get_context()
    agent = get_agent()
    
    # Parse query with LLM
    parsed = await parser.parse(query, context.to_dict())
    logger.info(f"Parsed: intent={parsed.intent}, loc={parsed.location}, hours={parsed.hours}, past_days={parsed.past_days}, followup={parsed.is_followup}")
    
    # Smart recovery: If LLM returns "unknown" but query looks like a time-based follow-up
    # and we have weather context, treat it as a weather follow-up
    if parsed.intent == "unknown" and context.last_intent in ["analyze", "forecast"]:
        query_lower = query.lower()
        time_patterns = ["week", "day", "hour", "tomorrow", "yesterday", "month", "2 week", "3 day"]
        if any(p in query_lower for p in time_patterns):
            logger.info(f"Smart recovery: treating '{query}' as time-based follow-up")
            # Re-parse with forced follow-up context
            parsed.intent = context.last_intent
            parsed.is_followup = True
            # Extract time from query
            hours, past_days = parser._extract_time(query_lower)
            if hours != 6 or past_days != 0:  # Time was extracted
                parsed.hours = hours
                parsed.past_days = past_days
    
    # Handle by intent
    if parsed.intent == "apod":
        return await _handle_apod(agent, context)
    
    if parsed.intent in ["analyze", "forecast"]:
        return await _handle_weather_query(query, parsed, geocoding, agent, context)
    
    if parsed.intent == "help":
        return _handle_help()
    
    # For "unknown" intent - this means the query is genuinely off-topic
    # Don't try to force it into weather/air quality - let the LLM respond appropriately
    if parsed.intent == "unknown":
        return await _handle_off_topic(query)
    
    # Greeting - use LLM
    return await _handle_greeting(query)


# =============================================================================
# Intent Handlers
# =============================================================================

async def _handle_apod(agent: AirInsightsAgent, context: ConversationContext) -> ChatResponse:
    """Handle APOD requests."""
    try:
        apod = await agent.get_apod()
        context.update(intent="apod")
        return ChatResponse(
            response=f"🌟 **{apod.title}**\n\n{apod.summary}\n\n[View Image]({apod.url})\n\n_{apod.attribution}_",
            intent="apod",
            data={"title": apod.title, "url": apod.url}
        )
    except Exception as e:
        return ChatResponse(response=f"Sorry, couldn't fetch APOD: {e}", intent="apod")


async def _handle_weather_query(
    query: str,
    parsed,
    geocoding: GeocodingService,
    agent: AirInsightsAgent,
    context: ConversationContext
) -> ChatResponse:
    """Handle air quality analysis and weather forecast requests."""
    
    coords = None
    location_name = parsed.location
    
    # Check if coordinates were provided directly in the query
    if parsed.coordinates:
        coords = parsed.coordinates
        location_name = f"{coords[0]:.4f}, {coords[1]:.4f}"
        logger.info(f"Using direct coordinates: {coords}")
    
    # Geocode location if provided (and no direct coordinates)
    elif location_name:
        geo_result = await geocoding.geocode(location_name)
        if geo_result:
            if geo_result.is_country:
                # Country detected - ask for specific city
                cities, info = await geocoding.get_country_cities(geo_result.country or location_name)
                cities_str = ", ".join(cities[:5]) if cities else "major cities"
                context.update(intent=parsed.intent, awaiting_location=True)
                return ChatResponse(
                    response=f"🌍 **{geo_result.location_name}** is a country. Please specify a city.\n\n**Try:** {cities_str}",
                    intent=parsed.intent,
                    data={"needs": "city", "country": geo_result.location_name}
                )
            coords = geo_result.coords
            location_name = geo_result.location_name
    
    # Use context coordinates ONLY if this is a follow-up with a location in context
    # Don't silently use old coordinates for new queries without location
    if not coords and parsed.is_followup and context.last_coords:
        coords = context.last_coords
        location_name = context.last_location
        logger.info(f"Using context location for follow-up: {location_name}")
    
    # Still no location - ask for one
    if not coords:
        context.update(intent=parsed.intent, awaiting_location=True)
        return ChatResponse(
            response="Please specify a location.\n\n**Examples:** \"Sofia\", \"Paris, France\", \"42.69, 23.32\"",
            intent=parsed.intent,
            data={"needs": "location"}
        )
    
    # Execute the request
    lat, lon = coords
    hours = parsed.hours
    past_days = getattr(parsed, 'past_days', 0)  # Get past_days if available
    
    try:
        if parsed.intent == "analyze":
            result = await agent.analyze(lat, lon, hours, past_days=past_days)
            time_str = _format_time_period(hours, past_days)
            
            context.update(intent="analyze", location=location_name, coords=coords, hours=hours, past_days=past_days)
            return ChatResponse(
                response=(
                    f"📍 **Air Quality for {location_name}** ({time_str})\n\n"
                    f"{result.guidance_text}\n\n"
                    f"**Data:** PM2.5: {result.pm25_avg:.1f}, PM10: {result.pm10_avg:.1f}, Temp: {result.temp_avg:.1f}°C\n\n"
                    f"_{result.attribution}_"
                ),
                intent="analyze",
                data={"location": location_name, "safety": result.safety_level.value}
            )
        
        else:  # forecast
            return await _handle_forecast(query, lat, lon, hours, location_name, context, past_days)
            
    except Exception as e:
        logger.error(f"Weather query failed: {e}")
        return ChatResponse(response=f"Sorry, something went wrong: {e}", intent=parsed.intent)


async def _handle_forecast(
    query: str,
    lat: float,
    lon: float,
    hours: int,
    location_name: str,
    context: ConversationContext,
    past_days: int = 0
) -> ChatResponse:
    """Handle weather forecast requests (including historical data)."""
    from tools.weather_client import WeatherClient
    from datetime import datetime as dt
    
    days = max(1, (hours + 23) // 24)
    days = min(days, 16)
    
    client = WeatherClient()
    
    # For historical requests, get daily aggregated data
    if past_days > 0:
        daily = await client.get_historical_daily(lat, lon, past_days=past_days)
        
        time_label = "yesterday" if past_days == 1 else f"past {past_days} days"
        
        # Build detailed day-by-day list
        day_lines = []
        total_precip = 0
        temps_max = []
        temps_min = []
        
        for i in range(len(daily.dates)):
            s = daily.get_day_summary(i)
            try:
                date_str = dt.strptime(s['date'], "%Y-%m-%d").strftime("%a, %b %d")
            except:
                date_str = s['date']
            
            temp_max = s.get('temp_max')
            temp_min = s.get('temp_min')
            precip = s.get('precipitation', 0) or 0
            emoji = s.get('emoji', '❓')
            weather = s.get('weather', 'Unknown')
            
            # Skip days with no temperature data
            if temp_max is None or temp_min is None:
                continue
                
            temps_max.append(temp_max)
            temps_min.append(temp_min)
            total_precip += precip
            
            precip_str = f" 💧{precip:.1f}mm" if precip > 0 else ""
            day_lines.append(f"**{date_str}**: {emoji} {weather}, {temp_min:.0f}° to {temp_max:.0f}°C{precip_str}")
        
        # Summary stats
        avg_max = sum(temps_max) / len(temps_max) if temps_max else 0
        avg_min = sum(temps_min) / len(temps_min) if temps_min else 0
        overall_max = max(temps_max) if temps_max else 0
        overall_min = min(temps_min) if temps_min else 0
        
        days_list = "\n".join(day_lines)
        
        summary = f"📈 **Summary:** {overall_min:.0f}° to {overall_max:.0f}°C overall"
        if total_precip > 0:
            summary += f", total rain: {total_precip:.1f}mm"
        
        context.update(intent="forecast", location=location_name, coords=(lat, lon), hours=hours, past_days=past_days)
        return ChatResponse(
            response=(
                f"📊 **Historical Weather for {location_name}** ({time_label})\n\n"
                f"{days_list}\n\n"
                f"{summary}\n\n"
                f"_Weather by Open-Meteo.com_"
            ),
            intent="forecast",
            data={"location": location_name, "past_days": past_days, "historical": True}
        )
    
    # Future forecast (existing logic)
    daily = await client.get_weekly_forecast(lat, lon, days=days)
    
    # Build forecast summary
    forecast_lines = []
    for i in range(min(days, len(daily.dates))):
        s = daily.get_day_summary(i)
        forecast_lines.append(f"- {s['date']}: {s['emoji']} {s['weather']}, {s['temp_min']:.0f}° to {s['temp_max']:.0f}°C")
    
    # Try to get LLM summary, but fallback gracefully
    try:
        llm = LLMClient()
        llm_summary = await llm.chat([
            {"role": "system", "content": "Summarize weather concisely. Mention best days for outdoor activities."},
            {"role": "user", "content": f"User: {query}\n\n{days}-day forecast for {location_name}:\n" + "\n".join(forecast_lines)}
        ], temperature=0.3, max_tokens=200)
    except Exception as e:
        logger.warning(f"LLM summary failed: {e}, using basic summary")
        # Create a simple summary without LLM
        llm_summary = f"Here's your {days}-day forecast for {location_name}."
    
    # Build table for short forecasts
    table = ""
    if days <= 7:
        from datetime import datetime as dt
        table_lines = []
        for i in range(min(days, len(daily.dates))):
            s = daily.get_day_summary(i)
            try:
                d = dt.strptime(s['date'], "%Y-%m-%d").strftime("%a %d")
            except:
                d = s['date']
            table_lines.append(f"| {d} | {s['emoji']} | {s['temp_min']:.0f}° - {s['temp_max']:.0f}° |")
        table = "\n" + "\n".join(table_lines)
    
    context.update(intent="forecast", location=location_name, coords=(lat, lon), hours=hours)
    
    return ChatResponse(
        response=f"📅 **{days}-Day Forecast for {location_name}**\n\n{llm_summary}{table}\n\n_Weather by Open-Meteo.com_",
        intent="forecast",
        data={"location": location_name, "days": days}
    )


def _handle_help() -> ChatResponse:
    """Handle help requests."""
    return ChatResponse(
        response=(
            "👋 **Air & Insights Agent**\n\n"
            "🏃 **Air Quality** - \"Is it safe to run in Sofia?\"\n"
            "🌤️ **Weather Forecast** - \"Weather this week in Paris?\"\n"
            "🌟 **NASA APOD** - \"Today's astronomy picture\"\n\n"
            "I understand follow-ups too!"
        ),
        intent="help"
    )


async def _handle_off_topic(query: str) -> ChatResponse:
    """Handle off-topic queries with a polite redirect."""
    llm = LLMClient()
    try:
        response = await llm.chat([
            {"role": "system", "content": (
                "You are Air & Insights Agent - a specialized assistant for:\n"
                "1. Air quality analysis (PM2.5, PM10, pollution levels)\n"
                "2. Weather forecasts (temperature, rain, conditions)\n"
                "3. NASA Astronomy Picture of the Day\n\n"
                "The user asked something outside your capabilities. Politely:\n"
                "- Acknowledge their question briefly\n"
                "- Explain you can't help with that specific request\n"
                "- Redirect them to what you CAN help with\n"
                "Be friendly and concise (2-3 sentences max)."
            )},
            {"role": "user", "content": query}
        ], temperature=0.7, max_tokens=100)
        return ChatResponse(response=response, intent="unknown")
    except:
        return ChatResponse(
            response="I'm specialized in air quality, weather forecasts, and NASA's astronomy pictures. "
                     "I can't help with that, but feel free to ask me about the weather or air quality in your city! 🌤️",
            intent="unknown"
        )


async def _handle_greeting(query: str) -> ChatResponse:
    """Handle greetings and unknown intents with LLM."""
    llm = LLMClient()
    try:
        response = await llm.chat([
            {"role": "system", "content": (
                "You are Air & Insights Agent. Help with: air quality, weather forecasts, NASA APOD.\n"
                "For greetings: be warm, briefly explain capabilities.\n"
                "For off-topic: politely redirect. Be concise."
            )},
            {"role": "user", "content": query}
        ], temperature=0.5, max_tokens=150)
        return ChatResponse(response=response, intent="greeting")
    except:
        return ChatResponse(
            response="Hello! 👋 I help with air quality, weather, and NASA's astronomy picture.",
            intent="greeting"
        )


def _format_time_period(hours: int, past_days: int = 0) -> str:
    """Format hours/past_days into readable time period."""
    # Historical data
    if past_days > 0:
        if past_days == 1:
            return "yesterday"
        elif past_days <= 7:
            return f"past {past_days} days"
        elif past_days <= 30:
            weeks = past_days // 7
            return f"past {weeks} week{'s' if weeks > 1 else ''}"
        else:
            return f"past {past_days} days"
    
    # Future/current data
    if hours <= 6:
        return f"next {hours} hours"
    elif hours <= 12:
        return "today"
    elif hours <= 24:
        return "tomorrow"
    elif hours <= 48:
        return "next 2 days"
    elif hours <= 168:
        days = hours // 24
        return f"next {days} days"
    else:
        days = hours // 24
        return f"next {days} days"


# =============================================================================
# Cache Stats Endpoint
# =============================================================================

@router.get("/cache/stats", tags=["System"])
async def get_cache_stats() -> dict:
    """Get cache statistics."""
    return get_agent().get_cache_stats()
