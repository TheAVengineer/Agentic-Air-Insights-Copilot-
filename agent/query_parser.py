"""
Query Parser - LLM-First Natural Language Understanding.

Enterprise-grade design: The LLM handles ALL language understanding.
No hardcoded word mappings - the LLM understands context, nuance, and intent.

When LLM is unavailable, a minimal "safe mode" keeps the system functional.
"""

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

from llm.client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class ParsedQuery:
    """Structured result from query parsing."""
    intent: str  # "analyze" | "forecast" | "apod" | "help" | "greeting" | "unknown"
    location: Optional[str] = None
    hours: int = 6  # Positive = future, used with past_days for historical
    past_days: int = 0  # Days in the past (0-92), Open-Meteo supports up to 92 days
    is_followup: bool = False
    needs_location: bool = False
    coordinates: Optional[Tuple[float, float]] = None


class QueryParser:
    """
    LLM-First Query Parser.
    
    Design Philosophy:
    - LLM handles ALL natural language understanding
    - Single comprehensive prompt = single LLM call
    - Graceful degradation to safe mode when offline
    """
    
    SYSTEM_PROMPT = """You are a query parser for an Air & Weather Insights agent.

CAPABILITIES:
1. AIR QUALITY ANALYSIS - PM2.5, PM10, temperature for outdoor exercise
2. WEATHER FORECAST - Up to 16 days future OR 92 days historical
3. NASA APOD - Astronomy Picture of the Day

PARSE the user query into JSON.

COORDINATES:
- "45.69, 23.32" → coordinates: [45.69, 23.32]
- "lat 45.69 lon 23.32" → coordinates: [45.69, 23.32]
- Validate: lat -90 to 90, lon -180 to 180

LOCATION:
- Extract CITY names only, never countries
- "weather in Bulgaria" → location: "Sofia"

TIME - We support BOTH future AND past data:
FUTURE (hours from now) - DO NOT set past_days:
- "next X hours" → hours: X (NO past_days!)
- "next X days" → hours: X × 24 (NO past_days!)
- "next X weeks" / "coming X weeks" → hours: X × 168 (NO past_days!)
- "today" → hours: 12
- "tomorrow" → hours: 24
- "upcoming week" → hours: 168
- Day names (future): hours until that day
- Maximum future: 384 hours (16 days)
- IMPORTANT: For ANY future query ("next", "upcoming", "will be", "tomorrow"), DO NOT return past_days!

HOLIDAYS (calculate hours from current date):
- "christmas" / "xmas" → hours until December 25
- "new year" / "new years" → hours until January 1
- "easter" → hours until Easter

PAST / HISTORICAL DATA (set past_days > 0):
- "yesterday" → hours: 24, past_days: 1
- "last X days" / "past X days" → hours: X × 24, past_days: X (max 92)
- "last week" / "past week" → hours: 168, past_days: 7
- "last X weeks" → hours: X × 168, past_days: X × 7 (max 92)
- "last month" / "past month" → hours: 720, past_days: 30
- "last X months" / "past X months" → past_days: X × 30 (max 92)
- "X days ago" → hours: 24, past_days: X
- "how was the weather yesterday" → hours: 24, past_days: 1
- "what was the air quality last week" → hours: 168, past_days: 7
- Maximum past: 92 days (API limit)

HISTORICAL EXAMPLES (always set past_days for these):
- "weather yesterday in Sofia" → intent: "forecast", location: "Sofia", hours: 24, past_days: 1
- "air quality last week in Paris" → intent: "analyze", location: "Paris", hours: 168, past_days: 7
- "how was the weather last month" → intent: "forecast", hours: 720, past_days: 30
- "past 3 days weather in London" → intent: "forecast", location: "London", hours: 72, past_days: 3
- "was the air good yesterday" → intent: "analyze", hours: 24, past_days: 1

Word numbers: "two"=2, "three"=3, "couple"=2, "few"=3
IMPORTANT: Never return hours: 0, minimum is 1

INTENT - BE VERY CAREFUL:
- "greeting" → ONLY for pure greetings with NO weather/air content:
  - "hi", "hello", "hey", "how are you", "good morning", "what's up"
  - These are GREETINGS, not weather queries!
- "unknown" → For queries we CANNOT answer:
  - "what is today's date" → unknown (we don't have date info to share)
  - "what time is it" → unknown (we don't have time info)
  - "who are you" → unknown (meta question)
  - Math questions, jokes, personal requests → unknown
- "analyze" → air quality, PM2.5, PM10, pollution, safe to exercise, safe to run
- "forecast" → weather, temperature, rain, sunny, cold, hot
- "apod" → NASA, astronomy, space picture, "nasa pic", "the picture", "today's picture"
- "help" → help, what can you do

CRITICAL - GREETING vs WEATHER:
- "how are you" → greeting (NOT weather!)
- "how is the weather" → forecast
- "what is today's date" → unknown (NOT weather!)
- "what is today's weather" → forecast
- "hi, check Sofia air" → analyze (NOT greeting, has weather content)

CRITICAL - FOLLOW-UP & CONTEXT RULES:
1. ONLY mark is_followup: true if query is CLEARLY about weather/air quality/location/time
2. OFF-TOPIC queries (math, jokes, personal requests) → intent: "unknown", is_followup: false
3. "what about X" where X is a location/city → is_followup: true, use PREVIOUS intent (analyze/forecast)
4. "what about X" where X is off-topic → intent: "unknown", is_followup: false
5. For valid follow-ups, PRESERVE previous intent unless explicitly changed
6. For follow-ups with NO time mentioned → hours: null (will use previous)
7. LOCATION FOLLOW-UPS: "what about Paris?", "and London?", "how about NYC?" → 
   These are asking for the SAME analysis (air quality/weather) for a NEW location.
   Set is_followup: true, location: <new city>, intent: <previous intent>
8. TIME FOLLOW-UPS: When user asks for a different time period after a weather/air query,
   it's a follow-up! Examples: "for the next 2 weeks", "what about tomorrow", "and yesterday?"

EXAMPLES of "unknown" (NOT follow-ups):
- "how much is 1+1?" → unknown (math, not weather)
- "can you brush my teeth?" → unknown (personal request)
- "can you fix the weather?" → unknown (impossible request)
- "tell me a joke" → unknown (entertainment)
- "can you draw a board?" → unknown (visualization request - we can't draw)

EXAMPLES of LOCATION FOLLOW-UPS (previous intent was "analyze"):
- "what about Plovdiv?" → intent: "analyze", location: "Plovdiv", is_followup: true
- "and Paris?" → intent: "analyze", location: "Paris", is_followup: true
- "how about London?" → intent: "analyze", location: "London", is_followup: true

EXAMPLES of TIME FOLLOW-UPS (previous intent was "analyze" or "forecast"):
- "for the next 2 weeks" → intent: <previous>, hours: 336, is_followup: true
- "can you give me for the next 2 weeks" → intent: <previous>, hours: 336, is_followup: true  
- "what about tomorrow" → intent: <previous>, hours: 24, is_followup: true
- "and for the next 3 days" → intent: <previous>, hours: 72, is_followup: true

RESPOND WITH ONLY A JSON OBJECT - NO EXPLANATION, NO TEXT BEFORE OR AFTER:
{"intent": "...", "location": "...", "coordinates": null, "hours": 6, "past_days": 0, "is_followup": false, "needs_location": false}"""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()
    
    async def parse(self, query: str, context: Optional[dict] = None) -> ParsedQuery:
        """Parse query using LLM-first approach with safe mode fallback."""
        context = context or {}
        
        try:
            # Build context-aware prompt
            prompt = self._build_prompt(query, context)
            
            # Single LLM call - increased max_tokens for complete JSON
            response = await self.llm.chat([
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ], temperature=0.1, max_tokens=300)
            
            return self._parse_response(response, query, context)
            
        except Exception as e:
            logger.warning(f"LLM parsing failed: {e}, using fallback")
            return self._safe_mode_parse(query, context)
    
    def _build_prompt(self, query: str, context: dict) -> str:
        """Build context-aware prompt."""
        now = datetime.now()
        prompt = f"Date: {now.strftime('%Y-%m-%d')} ({now.strftime('%A')})\n"
        
        if context.get("last_location") or context.get("last_coords") or context.get("last_intent"):
            prompt += f"""
CONVERSATION CONTEXT (use for follow-ups):
- Previous intent: {context.get("last_intent") or "none"} ← PRESERVE unless user explicitly changes
- Previous location: {context.get("last_location") or "none"}
- Previous coordinates: {context.get("last_coords") or "none"}  
- Previous time period: {context.get("last_hours") or 6} hours ← PRESERVE if no new time mentioned
- Previous past_days: {context.get("last_past_days") or 0}
"""
        
        prompt += f'\nQuery: "{query}"'
        return prompt
    
    def _parse_response(self, response: str, query: str, context: dict) -> ParsedQuery:
        """Parse LLM JSON response."""
        # Clean markdown and extract JSON
        response = response.strip()
        if "```" in response:
            response = re.sub(r'```(?:json)?\n?', '', response)
            response = response.replace('```', '')
        
        # Try to extract JSON object from response (handles "Here is the JSON:" prefix)
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            response = json_match.group()
        
        try:
            result = json.loads(response.strip())
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON: {response[:100]}")
            return self._safe_mode_parse(query, context)
        
        # Extract coordinates from LLM response
        coordinates = None
        if result.get("coordinates"):
            coords = result["coordinates"]
            if isinstance(coords, list) and len(coords) == 2:
                try:
                    lat, lon = float(coords[0]), float(coords[1])
                    if -90 <= lat <= 90 and -180 <= lon <= 180:
                        coordinates = (lat, lon)
                except (ValueError, TypeError):
                    pass
        
        # IMPORTANT: If LLM didn't extract coordinates, try regex extraction from query
        # This ensures explicit coordinates like "42.6977, 23.3219" are always captured
        if not coordinates:
            coordinates = self._extract_coordinates(query)
            if coordinates:
                logger.info(f"Extracted coordinates from query text: {coordinates}")
        
        # Apply context for follow-ups
        location = result.get("location")
        intent = result.get("intent", "unknown")
        raw_hours = result.get("hours")  # Can be null/None for "use previous"
        raw_past_days = result.get("past_days")  # Can be null/None for "use previous"
        is_followup = result.get("is_followup", False)
        
        # Determine time period - null means "use previous context"
        # Only use context if BOTH hours and past_days are not specified
        time_specified = raw_hours is not None or raw_past_days is not None
        
        if time_specified:
            # User specified new time - use it
            hours = int(raw_hours) if raw_hours is not None else 6
            past_days = min(int(raw_past_days), 92) if raw_past_days is not None else 0
            
            # If hours is specified but past_days is not, default past_days to 0 (future)
            # This prevents context's past_days from bleeding into new queries
            if raw_hours is not None and raw_past_days is None:
                past_days = 0
        else:
            # No time specified - use context for follow-ups, default otherwise
            if is_followup and context:
                hours = context.get("last_hours", 6)
                past_days = context.get("last_past_days", 0)
            else:
                hours = 6
                past_days = 0
        
        # Ensure hours is at least 1 (never 0)
        if hours <= 0:
            hours = 1
        hours = min(hours, 384)
        past_days = min(past_days, 92)
        
        # If explicit coordinates were provided in the query, this is NOT a follow-up
        # (user is specifying a new location via coordinates)
        if coordinates:
            is_followup = False
        
        # IMPORTANT: If a location NAME is specified, clear any coordinates
        # The location will be geocoded by the routes handler
        # This prevents context coordinates from overriding a new city name
        if location and coordinates:
            # Check if coordinates were NOT extracted from query text
            query_coords = self._extract_coordinates(query)
            if not query_coords:
                # Coordinates came from LLM using context, but user specified a city name
                # Clear them so the city gets geocoded
                logger.info(f"Clearing context coordinates - user specified location: {location}")
                coordinates = None
        
        if is_followup:
            # Preserve location from context ONLY if no new location specified
            if not location and not coordinates:
                location = context.get("last_location")
                coordinates = context.get("last_coords")
            
            # Preserve intent from context unless explicitly changed
            if intent == "unknown":
                intent = context.get("last_intent", "analyze")
        
        needs_location = result.get("needs_location", False)
        if coordinates:
            needs_location = False
        
        parsed = ParsedQuery(
            intent=intent,
            location=location,
            hours=hours,
            past_days=past_days,
            is_followup=is_followup,
            needs_location=needs_location,
            coordinates=coordinates
        )
        
        logger.info(f"Parsed: intent={intent}, loc={location}, hours={hours}, past_days={past_days}, followup={is_followup}, coords={coordinates}")
        return parsed
    
    # ==========================================================================
    # FALLBACK PATTERNS - Enhanced regex for common queries when LLM fails
    # ==========================================================================
    
    # Pattern format: (regex, intent, location_group_index)
    # These provide robust fallback for common query formats
    FALLBACK_PATTERNS = [
        # Weather/Forecast queries
        (r'(?:weather|forecast)\s+(?:in|for|at)\s+([a-zA-Z\s]+?)(?:\s+for|\s*$)', 'forecast', 1),
        (r'(?:what|how)(?:\'?s| is| will be)\s+(?:the\s+)?weather\s+(?:in|for|at)\s+([a-zA-Z\s]+)', 'forecast', 1),
        (r'(?:is|will)\s+it\s+(?:rain|cold|hot|sunny|cloudy)\s+(?:in|at)\s+([a-zA-Z\s]+)', 'forecast', 1),
        (r'temperature\s+(?:in|for|at)\s+([a-zA-Z\s]+)', 'forecast', 1),
        
        # Air quality queries
        (r'air\s*quality\s+(?:in|for|at)\s+([a-zA-Z\s]+)', 'analyze', 1),
        (r'(?:pm2\.?5|pm10|pollution|aqi)\s+(?:in|for|at)\s+([a-zA-Z\s]+)', 'analyze', 1),
        (r'(?:is|can)\s+(?:it|i)\s+(?:safe|ok|good)\s+to\s+(?:run|exercise|jog|walk)\s+(?:in|at|outside)?\s*([a-zA-Z\s]*)', 'analyze', 1),
        (r'(?:should|can)\s+i\s+(?:go\s+)?(?:run|exercise|jog)\s+(?:in|at)?\s*([a-zA-Z\s]*)', 'analyze', 1),
        
        # NASA APOD queries
        (r'(?:nasa|astronomy)\s*(?:\'?s)?\s*(?:picture|photo|image|apod)', 'apod', None),
        (r'(?:picture|photo|image)\s+of\s+the\s+day', 'apod', None),
        (r'(?:show|get|fetch)\s+(?:me\s+)?(?:the\s+)?(?:nasa|astronomy|space)\s*(?:picture|photo|image)', 'apod', None),
        
        # Historical data queries
        (r'(?:weather|air\s*quality)\s+yesterday\s+(?:in|at|for)\s+([a-zA-Z\s]+)', None, 1),  # intent determined by keywords
        (r'(?:how|what)\s+was\s+(?:the\s+)?(?:weather|air\s*quality)\s+(?:in|at)\s+([a-zA-Z\s]+)\s+(?:yesterday|last)', None, 1),
    ]
    
    def _try_fallback_patterns(self, query: str, query_lower: str) -> Optional[Tuple[str, Optional[str]]]:
        """
        Try regex fallback patterns for common query formats.
        
        Returns: (intent, location) tuple or None if no pattern matches
        """
        for pattern, intent, loc_group in self.FALLBACK_PATTERNS:
            match = re.search(pattern, query_lower)
            if match:
                # Determine intent if not fixed
                if intent is None:
                    if 'air' in query_lower or 'quality' in query_lower or 'pm' in query_lower:
                        intent = 'analyze'
                    else:
                        intent = 'forecast'
                
                # Extract location if pattern has a location group
                location = None
                if loc_group is not None and match.lastindex >= loc_group:
                    loc = match.group(loc_group).strip()
                    if loc and len(loc) > 1:
                        # Capitalize properly
                        location = ' '.join(word.capitalize() for word in loc.split())
                
                logger.info(f"Fallback pattern matched: intent={intent}, location={location}")
                return (intent, location)
        
        return None
    
    # ==========================================================================
    # SAFE MODE - Minimal parsing when LLM unavailable
    # ==========================================================================
    
    def _safe_mode_parse(self, query: str, context: dict) -> ParsedQuery:
        """Safe mode: basic regex parsing when LLM offline."""
        query_lower = query.lower()
        
        # Try fallback patterns first for better accuracy
        fallback_result = self._try_fallback_patterns(query, query_lower)
        if fallback_result:
            fb_intent, fb_location = fallback_result
        else:
            fb_intent, fb_location = None, None
        
        # 1. Coordinates (always reliable)
        coordinates = self._extract_coordinates(query)
        
        # 2. Intent detection - use fallback pattern result if available
        intent = fb_intent if fb_intent else self._detect_intent(query_lower, context)
        
        # 3. Time extraction (hours and past_days)
        hours, past_days = self._extract_time(query_lower)
        time_was_specified = hours != 6 or past_days != 0  # Check if user specified time
        hours = max(1, hours)  # Ensure minimum of 1
        
        # 4. Check if query indicates FUTURE (should not use context past_days)
        is_future_query = any(p in query_lower for p in [
            "next", "upcoming", "coming", "tomorrow", "will be", "forecast", "this week"
        ])
        
        # 5. Location extraction - use fallback pattern result if available
        location = fb_location if fb_location else self._extract_location(query)
        
        # 6. Follow-up detection - also short queries are follow-ups
        is_short_query = len(query.split()) <= 4
        is_followup = any(p in query_lower for p in [
            "what about", "how about", "and for", "the next", "for next", "and "
        ]) or (is_short_query and context.get("last_intent"))
        
        # Apply context for follow-ups
        if is_followup:
            # Preserve location
            if not location and not coordinates:
                location = context.get("last_location")
                coordinates = context.get("last_coords")
            
            # Preserve intent unless explicitly changed
            if intent == "unknown":
                intent = context.get("last_intent") or "analyze"  # Default to analyze if no context
            
            # Preserve time period if not specified in this query
            # BUT if this is a future query, do NOT use context's past_days
            if not time_was_specified and context.get("last_hours"):
                hours = context.get("last_hours", 6)
                # Only use context past_days if this is NOT a future query
                if not is_future_query:
                    past_days = context.get("last_past_days", 0)
                else:
                    past_days = 0  # Future query, reset past_days
        
        # Default intent for coordinates
        if coordinates and intent == "unknown":
            intent = "analyze"
        
        needs_location = intent in ("analyze", "forecast") and not location and not coordinates
        
        logger.info(f"Safe mode parsed: intent={intent}, loc={location}, hours={hours}, past_days={past_days}")
        return ParsedQuery(
            intent=intent,
            location=location,
            hours=hours,
            past_days=past_days,
            is_followup=is_followup,
            needs_location=needs_location,
            coordinates=coordinates
        )
    
    def _detect_intent(self, q: str, context: Optional[dict] = None) -> str:
        """
        Detect intent from keywords (safe mode fallback only).
        
        Design: Keep simple - LLM handles nuance. This is just emergency fallback.
        Uses context to resolve ambiguous phrases like "the picture".
        """
        context = context or {}
        last_intent = context.get("last_intent", "")
        
        # GREETING - Must come first for proper detection
        # Match conversational patterns that are NOT about weather/air
        greeting_patterns = [
            r'^(hi|hello|hey)[\s!.?]*$',  # Simple greetings
            r'^how are you',              # How are you?
            r"^what'?s up",               # What's up?
            r'^good (morning|afternoon|evening)',  # Time-based greetings
        ]
        for pattern in greeting_patterns:
            if re.match(pattern, q):
                return "greeting"
        
        # GENERAL QUESTIONS - Not about weather (intent: unknown)
        general_question_patterns = [
            r"what('?s| is) (today'?s?|the) date",  # What is today's date
            r"what time is it",            # What time is it
            r"who are you",                # Who are you
            r"what are you",               # What are you
            r"tell me (a joke|about yourself)",  # Tell me a joke
            r"can you (draw|sing|dance)",  # Can you draw/sing
        ]
        for pattern in general_question_patterns:
            if re.search(pattern, q):
                return "unknown"
        
        # APOD - explicit keywords (high confidence)
        if any(k in q for k in ["apod", "astronomy picture", "picture of the day", "nasa pic", "nasa picture", "nasa photo"]):
            return "apod"
        
        # Context-aware: "the picture" / "show picture" only → APOD if last query was APOD
        if last_intent == "apod" and any(w in q for w in ["picture", "pic", "photo", "image"]):
            return "apod"
        
        if any(k in q for k in ["help", "what can you"]):
            return "help"
        
        # Normalize hyphens to spaces for matching
        q_normalized = q.replace("-", " ")
        
        if any(k in q_normalized for k in ["air quality", "pm2.5", "pm10", "pollution", "safe to run", "safe to exercise", "aqi"]):
            return "analyze"
        if any(k in q_normalized for k in ["weather", "forecast", "temperature", "rain", "cold", "hot"]):
            return "forecast"
        return "unknown"
    
    def _extract_time(self, q: str) -> Tuple[int, int]:
        """
        Extract time period - returns (hours, past_days).
        
        Supports both future and historical queries:
        - "next 3 days" → (72, 0)
        - "yesterday" → (24, 1)
        - "last week" → (168, 7)
        - "past 5 days" → (120, 5)
        """
        # Word to number
        w2n = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
               "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
               "couple": 2, "few": 3, "several": 5, "a": 1}
        
        def to_num(s):
            if not s: return 1
            return int(s) if s.isdigit() else w2n.get(s, 1)
        
        num = r'(\d+|one|two|three|four|five|six|seven|eight|nine|ten|couple|few|several|a)'
        
        # Check for PAST indicators first
        is_past = any(p in q for p in ["yesterday", "last", "past", "previous", "ago"])
        
        # Yesterday - special case
        if "yesterday" in q:
            return (24, 1)
        
        # "last/past X weeks"
        m = re.search(rf'(?:last|past|previous)\s*{num}?\s*weeks?', q)
        if m:
            n = to_num(m.group(1))
            days = min(n * 7, 92)
            return (days * 24, days)
        
        # "last/past X days"
        m = re.search(rf'(?:last|past|previous)\s*{num}?\s*days?', q)
        if m:
            days = min(to_num(m.group(1)), 92)
            return (days * 24, days)
        
        # "last/past X hours" - historical hourly data
        m = re.search(rf'(?:last|past|previous)\s*{num}?\s*hours?', q)
        if m:
            hours = to_num(m.group(1))
            # For past hours, we need at least 1 past_day to get historical data
            past_days = max(1, (hours + 23) // 24)
            return (hours, past_days)
        
        # "last/past X months"
        m = re.search(rf'(?:last|past|previous)\s*{num}?\s*months?', q)
        if m:
            months = to_num(m.group(1))
            days = min(months * 30, 92)  # Cap at 92 days
            return (days * 24, days)
        
        # "last month" (without number)
        if "last month" in q or "past month" in q:
            return (720, 30)
        
        # "X days/weeks ago"
        m = re.search(rf'{num}\s*days?\s*ago', q)
        if m:
            days = min(to_num(m.group(1)), 92)
            return (24, days)  # Show 1 day of data from X days ago
        
        m = re.search(rf'{num}\s*weeks?\s*ago', q)
        if m:
            days = min(to_num(m.group(1)) * 7, 92)
            return (24, days)
        
        # FUTURE time expressions
        # Weeks
        m = re.search(rf'{num}?\s*weeks?', q)
        if m and not is_past:
            return (min(to_num(m.group(1)) * 168, 384), 0)
        
        # Days
        m = re.search(rf'{num}\s*days?', q)
        if m and not is_past:
            return (min(to_num(m.group(1)) * 24, 384), 0)
        
        # Hours  
        m = re.search(rf'{num}\s*hours?', q)
        if m:
            return (min(to_num(m.group(1)), 384), 0)
        
        # Day of week (future)
        days_of_week = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
                        "friday": 4, "saturday": 5, "sunday": 6}
        for day, day_num in days_of_week.items():
            if day in q:
                today = datetime.now().weekday()
                until = (day_num - today) % 7 or 7
                return (until * 24, 0)
        
        # Special expressions
        if "fortnight" in q: return (336, 0)
        if "tomorrow" in q: return (24, 0)
        if "today" in q: return (12, 0)
        
        # Holiday dates - calculate hours until the holiday
        now = datetime.now()
        current_year = now.year
        
        # Christmas (December 25)
        if "christmas" in q:
            christmas = datetime(current_year, 12, 25)
            if christmas < now:
                christmas = datetime(current_year + 1, 12, 25)
            hours_until = int((christmas - now).total_seconds() / 3600)
            return (max(1, min(hours_until, 384)), 0)
        
        # New Year (January 1)
        if "new year" in q:
            new_year = datetime(current_year + 1, 1, 1)
            hours_until = int((new_year - now).total_seconds() / 3600)
            return (max(1, min(hours_until, 384)), 0)
        
        # Easter (approximate - first Sunday after first full moon after March 21)
        # Simplified: Use April 20 as average Easter date
        if "easter" in q:
            easter = datetime(current_year, 4, 20)
            if easter < now:
                easter = datetime(current_year + 1, 4, 20)
            hours_until = int((easter - now).total_seconds() / 3600)
            return (max(1, min(hours_until, 384)), 0)
        
        return (6, 0)  # Default: 6 hours future
    
    def _extract_location(self, query: str) -> Optional[str]:
        """Extract location."""
        # Skip day names as locations
        skip = {"monday", "tuesday", "wednesday", "thursday", "friday", 
                "saturday", "sunday", "the", "next", "today", "tomorrow"}
        
        patterns = [
            r'\bin\s+([A-Z][a-zA-Z]+)',
            r'(?:what|how)\s+about\s+([A-Z][a-zA-Z]+)',
            r'\bfor\s+([A-Z][a-zA-Z]+)',
        ]
        
        for pattern in patterns:
            m = re.search(pattern, query)
            if m and m.group(1).lower() not in skip:
                return m.group(1)
        return None
    
    def _extract_coordinates(self, query: str) -> Optional[Tuple[float, float]]:
        """Extract coordinates."""
        patterns = [
            r'lat(?:itude)?[:\s]*(-?\d+\.?\d*)[,\s]+lon(?:gitude)?[:\s]*(-?\d+\.?\d*)',
            r'(-?\d{1,3}\.\d+)[,\s]+(-?\d{1,3}\.\d+)',
        ]
        
        for pattern in patterns:
            m = re.search(pattern, query, re.IGNORECASE)
            if m:
                try:
                    lat, lon = float(m.group(1)), float(m.group(2))
                    if -90 <= lat <= 90 and -180 <= lon <= 180:
                        return (lat, lon)
                    if -90 <= lon <= 90 and -180 <= lat <= 180:
                        return (lon, lat)
                except ValueError:
                    continue
        return None
