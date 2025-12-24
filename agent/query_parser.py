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

PAST (past_days for historical data):
- "yesterday" → hours: 24, past_days: 1
- "last X days" / "past X days" → hours: X × 24, past_days: X (max 92)
- "last week" → hours: 168, past_days: 7
- "last X weeks" → hours: X × 168, past_days: X × 7 (max 92)
- "last month" → hours: 720, past_days: 30
- "last X months" / "past X months" → past_days: X × 30 (max 92)
- Maximum past: 92 days (API limit)

Word numbers: "two"=2, "three"=3, "couple"=2, "few"=3
IMPORTANT: Never return hours: 0, minimum is 1

INTENT:
- "analyze" → air quality, PM2.5, PM10, pollution, safe to exercise, safe to run
- "forecast" → weather, temperature, rain, sunny, cold, hot
- "apod" → NASA, astronomy, space picture, "nasa pic", "the picture", "today's picture", "give me the picture"
- "help" → help, what can you do
- "greeting" → hi, hello, how are you (ONLY if no other weather/air content)
- "unknown" → OFF-TOPIC queries that are NOT about weather, air quality, or APOD
  Examples of "unknown": math questions, personal requests, jokes, fixing things, drawing

CRITICAL - FOLLOW-UP & CONTEXT RULES:
1. ONLY mark is_followup: true if query is CLEARLY about weather/air quality/location
2. OFF-TOPIC queries (math, jokes, personal requests) → intent: "unknown", is_followup: false
3. "what about X" where X is a location/time → is_followup: true
4. "what about X" where X is off-topic → intent: "unknown", is_followup: false
5. For valid follow-ups, PRESERVE previous intent unless explicitly changed
6. For follow-ups with NO time mentioned → hours: null (will use previous)

EXAMPLES of "unknown" (NOT follow-ups):
- "how much is 1+1?" → unknown (math, not weather)
- "can you brush my teeth?" → unknown (personal request)
- "can you fix the weather?" → unknown (impossible request)
- "tell me a joke" → unknown (entertainment)
- "can you draw a board?" → unknown (visualization request - we can't draw)

OUTPUT (JSON only):
{
  "intent": "analyze|forecast|apod|help|greeting|unknown",
  "location": "city" or null,
  "coordinates": [lat, lon] or null,
  "hours": <integer or null if follow-up with no new time>,
  "past_days": <integer 0-92>,
  "is_followup": true|false,
  "needs_location": true|false
}"""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()
    
    async def parse(self, query: str, context: Optional[dict] = None) -> ParsedQuery:
        """Parse query using LLM-first approach with safe mode fallback."""
        context = context or {}
        
        try:
            # Build context-aware prompt
            prompt = self._build_prompt(query, context)
            
            # Single LLM call
            response = await self.llm.chat([
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ], temperature=0.1, max_tokens=150)
            
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
        # Clean markdown
        response = response.strip()
        if "```" in response:
            response = re.sub(r'```(?:json)?\n?', '', response)
            response = response.replace('```', '')
        
        try:
            result = json.loads(response.strip())
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON: {response[:100]}")
            return self._safe_mode_parse(query, context)
        
        # Extract coordinates
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
        
        if is_followup:
            # Preserve location from context
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
    # SAFE MODE - Minimal parsing when LLM unavailable
    # ==========================================================================
    
    def _safe_mode_parse(self, query: str, context: dict) -> ParsedQuery:
        """Safe mode: basic regex parsing when LLM offline."""
        query_lower = query.lower()
        
        # 1. Coordinates (always reliable)
        coordinates = self._extract_coordinates(query)
        
        # 2. Intent detection (pass context for smart resolution)
        intent = self._detect_intent(query_lower, context)
        
        # 3. Time extraction (hours and past_days)
        hours, past_days = self._extract_time(query_lower)
        time_was_specified = hours != 6 or past_days != 0  # Check if user specified time
        hours = max(1, hours)  # Ensure minimum of 1
        
        # 4. Check if query indicates FUTURE (should not use context past_days)
        is_future_query = any(p in query_lower for p in [
            "next", "upcoming", "coming", "tomorrow", "will be", "forecast", "this week"
        ])
        
        # 5. Location extraction
        location = self._extract_location(query)
        
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
                intent = context.get("last_intent", "analyze")
            
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
        
        # APOD - explicit keywords (high confidence)
        if any(k in q for k in ["apod", "astronomy picture", "picture of the day", "nasa pic", "nasa picture", "nasa photo"]):
            return "apod"
        
        # Context-aware: "the picture" / "show picture" only → APOD if last query was APOD
        if last_intent == "apod" and any(w in q for w in ["picture", "pic", "photo", "image"]):
            return "apod"
        
        if any(k in q for k in ["help", "what can you"]):
            return "help"
        if re.match(r'^(hi|hello|hey)[\s!.?]*$', q):
            return "greeting"
        if any(k in q for k in ["air quality", "pm2.5", "pm10", "pollution", "safe to run", "safe to exercise"]):
            return "analyze"
        if any(k in q for k in ["weather", "forecast", "temperature", "rain", "cold", "hot"]):
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
