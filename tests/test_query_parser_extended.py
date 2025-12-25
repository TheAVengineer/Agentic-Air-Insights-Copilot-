"""
Extended tests for agent/query_parser.py to increase coverage.
Tests _extract_time historical patterns, day-of-week, and edge cases.
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from agent.query_parser import QueryParser, ParsedQuery


class TestExtractTimeHistorical:
    """Test _extract_time with historical time expressions."""
    
    def setup_method(self):
        """Create parser instance."""
        self.parser = QueryParser()
    
    def test_yesterday(self):
        """Should extract 'yesterday' correctly."""
        hours, past_days = self.parser._extract_time("weather yesterday")
        assert past_days == 1
        assert hours == 24
    
    def test_last_week(self):
        """Should extract 'last week' correctly."""
        hours, past_days = self.parser._extract_time("what was the weather last week")
        assert past_days == 7
        assert hours == 168
    
    def test_last_2_weeks(self):
        """Should extract 'last 2 weeks' correctly."""
        hours, past_days = self.parser._extract_time("weather last 2 weeks")
        assert past_days == 14
        assert hours == 336
    
    def test_past_3_days(self):
        """Should extract 'past 3 days' correctly."""
        hours, past_days = self.parser._extract_time("air quality past 3 days")
        assert past_days == 3
        assert hours == 72
    
    def test_last_month(self):
        """Should extract 'last month' correctly."""
        hours, past_days = self.parser._extract_time("weather last month")
        assert past_days == 30
        assert hours == 720
    
    def test_5_days_ago(self):
        """Should extract 'X days ago' correctly."""
        hours, past_days = self.parser._extract_time("weather 5 days ago")
        assert past_days == 5


class TestExtractTimeFuture:
    """Test _extract_time with future time expressions."""
    
    def setup_method(self):
        """Create parser instance."""
        self.parser = QueryParser()
    
    def test_next_3_days(self):
        """Should extract 'next 3 days' correctly."""
        hours, past_days = self.parser._extract_time("weather next 3 days")
        assert hours == 72
        assert past_days == 0
    
    def test_next_week(self):
        """Should extract 'next week' correctly."""
        hours, past_days = self.parser._extract_time("forecast next week")
        assert hours == 168
        assert past_days == 0
    
    def test_5_hours(self):
        """Should extract 'X hours' correctly."""
        hours, past_days = self.parser._extract_time("next 5 hours")
        assert hours == 5
        assert past_days == 0
    
    def test_tomorrow(self):
        """Should extract 'tomorrow' correctly."""
        hours, past_days = self.parser._extract_time("weather tomorrow")
        assert hours == 24
        assert past_days == 0
    
    def test_today(self):
        """Should extract 'today' correctly."""
        hours, past_days = self.parser._extract_time("weather today")
        assert hours == 12
        assert past_days == 0
    
    def test_fortnight(self):
        """Should extract 'fortnight' correctly."""
        hours, past_days = self.parser._extract_time("weather fortnight")
        assert hours == 336
        assert past_days == 0
    
    def test_default(self):
        """Should return default for no time expression."""
        hours, past_days = self.parser._extract_time("air quality in sofia")
        assert hours == 6
        assert past_days == 0


class TestExtractTimeDayOfWeek:
    """Test _extract_time with day-of-week expressions."""
    
    def setup_method(self):
        """Create parser instance."""
        self.parser = QueryParser()
    
    def test_monday(self):
        """Should calculate hours until Monday."""
        hours, past_days = self.parser._extract_time("weather on monday")
        assert hours > 0
        assert hours <= 168  # Max 7 days
        assert past_days == 0
    
    def test_friday(self):
        """Should calculate hours until Friday."""
        hours, past_days = self.parser._extract_time("forecast for friday")
        assert hours > 0
        assert hours <= 168
        assert past_days == 0


class TestExtractTimeWordNumbers:
    """Test _extract_time with word numbers."""
    
    def setup_method(self):
        """Create parser instance."""
        self.parser = QueryParser()
    
    def test_two_days(self):
        """Should parse 'two days'."""
        hours, past_days = self.parser._extract_time("next two days")
        assert hours == 48
    
    def test_three_hours(self):
        """Should parse 'three hours'."""
        hours, past_days = self.parser._extract_time("next three hours")
        assert hours == 3
    
    def test_couple_days(self):
        """Should parse 'couple days'."""
        hours, past_days = self.parser._extract_time("next couple days")
        assert hours == 48


class TestExtractLocation:
    """Test _extract_location method."""
    
    def setup_method(self):
        """Create parser instance."""
        self.parser = QueryParser()
    
    def test_in_city(self):
        """Should extract 'in City' pattern."""
        location = self.parser._extract_location("weather in Sofia")
        assert location == "Sofia"
    
    def test_for_city(self):
        """Should extract 'for City' pattern."""
        location = self.parser._extract_location("weather for Berlin")
        assert location == "Berlin"
    
    def test_skip_day_names(self):
        """Should not extract day names as locations."""
        location = self.parser._extract_location("weather for Monday")
        assert location is None
    
    def test_no_location(self):
        """Should return None when no location found."""
        location = self.parser._extract_location("is it safe to run")
        assert location is None


class TestExtractCoordinates:
    """Test _extract_coordinates method."""
    
    def setup_method(self):
        """Create parser instance."""
        self.parser = QueryParser()
    
    def test_lat_lon_format(self):
        """Should extract lat/lon format."""
        coords = self.parser._extract_coordinates("weather at lat 42.69 lon 23.32")
        assert coords == (42.69, 23.32)
    
    def test_decimal_coordinates(self):
        """Should extract decimal coordinates."""
        coords = self.parser._extract_coordinates("weather at 42.6977, 23.3219")
        assert coords is not None
        assert abs(coords[0] - 42.6977) < 0.001
    
    def test_negative_coordinates(self):
        """Should extract negative coordinates."""
        coords = self.parser._extract_coordinates("weather at -33.87, 151.21")
        assert coords is not None
        assert coords[0] < 0


class TestDetectIntent:
    """Test _detect_intent method."""
    
    def setup_method(self):
        """Create parser instance."""
        self.parser = QueryParser()
    
    def test_apod_keywords(self):
        """Should detect APOD intent."""
        assert self.parser._detect_intent("show me the apod") == "apod"
        assert self.parser._detect_intent("astronomy picture of the day") == "apod"
        assert self.parser._detect_intent("nasa picture") == "apod"
    
    def test_help_intent(self):
        """Should detect help intent."""
        assert self.parser._detect_intent("help me") == "help"
        assert self.parser._detect_intent("what can you do") == "help"
    
    def test_greeting_intent(self):
        """Should detect greeting intent."""
        assert self.parser._detect_intent("hi") == "greeting"
        assert self.parser._detect_intent("hello") == "greeting"
    
    def test_analyze_intent(self):
        """Should detect analyze intent."""
        assert self.parser._detect_intent("air quality in sofia") == "analyze"
        assert self.parser._detect_intent("pm2.5 levels") == "analyze"
        assert self.parser._detect_intent("is it safe to run") == "analyze"
    
    def test_forecast_intent(self):
        """Should detect forecast intent."""
        assert self.parser._detect_intent("weather forecast") == "forecast"
        assert self.parser._detect_intent("will it rain tomorrow") == "forecast"
    
    def test_unknown_intent(self):
        """Should return unknown for unrecognized queries."""
        assert self.parser._detect_intent("random query") == "unknown"


class TestSafeModeParseExtended:
    """Extended tests for safe mode parsing."""
    
    def setup_method(self):
        """Create parser instance."""
        self.parser = QueryParser()
    
    def test_safe_mode_historical_query(self):
        """Should parse historical query in safe mode."""
        result = self.parser._safe_mode_parse("weather yesterday in Sofia", {})
        
        assert result.past_days == 1
        assert result.hours == 24
        assert result.location == "Sofia"
    
    def test_safe_mode_needs_location(self):
        """Should set needs_location when no location provided."""
        result = self.parser._safe_mode_parse("is the air quality good", {})
        
        assert result.needs_location is True


class TestBuildPrompt:
    """Test prompt building."""
    
    def setup_method(self):
        """Create parser instance."""
        self.parser = QueryParser()
    
    def test_build_prompt_includes_query(self):
        """Should include query in prompt."""
        prompt = self.parser._build_prompt("weather in Sofia", {})
        
        assert "weather in Sofia" in prompt
    
    def test_build_prompt_with_context(self):
        """Should include context in prompt."""
        context = {
            "last_intent": "analyze",
            "last_location": "Paris",
            "last_hours": 24
        }
        
        prompt = self.parser._build_prompt("what about tomorrow", context)
        
        assert "CONVERSATION CONTEXT" in prompt
        assert "analyze" in prompt


class TestFallbackPatterns:
    """Test regex fallback patterns for common queries."""
    
    def setup_method(self):
        """Create parser instance."""
        self.parser = QueryParser()
    
    def test_weather_in_city(self):
        """Should match 'weather in City' pattern."""
        result = self.parser._try_fallback_patterns("weather in London", "weather in london")
        assert result is not None
        intent, location = result
        assert intent == "forecast"
        assert location == "London"
    
    def test_forecast_for_city(self):
        """Should match 'forecast for City' pattern."""
        result = self.parser._try_fallback_patterns("forecast for Paris", "forecast for paris")
        assert result is not None
        intent, location = result
        assert intent == "forecast"
        assert location == "Paris"
    
    def test_air_quality_in_city(self):
        """Should match 'air quality in City' pattern."""
        result = self.parser._try_fallback_patterns("air quality in Sofia", "air quality in sofia")
        assert result is not None
        intent, location = result
        assert intent == "analyze"
        assert location == "Sofia"
    
    def test_safe_to_run_in_city(self):
        """Should match 'is it safe to run in City' pattern."""
        result = self.parser._try_fallback_patterns("is it safe to run in Berlin", "is it safe to run in berlin")
        assert result is not None
        intent, location = result
        assert intent == "analyze"
        assert location == "Berlin"
    
    def test_nasa_apod(self):
        """Should match NASA APOD patterns."""
        result = self.parser._try_fallback_patterns("show me the nasa picture", "show me the nasa picture")
        assert result is not None
        intent, _ = result
        assert intent == "apod"
    
    def test_picture_of_the_day(self):
        """Should match 'picture of the day' pattern."""
        result = self.parser._try_fallback_patterns("picture of the day", "picture of the day")
        assert result is not None
        intent, _ = result
        assert intent == "apod"
    
    def test_whats_the_weather(self):
        """Should match 'what's the weather in City' pattern."""
        result = self.parser._try_fallback_patterns("what's the weather in NYC", "what's the weather in nyc")
        assert result is not None
        intent, location = result
        assert intent == "forecast"
        assert location == "Nyc"  # Capitalized as expected
    
    def test_will_it_rain(self):
        """Should match 'will it rain in City' pattern."""
        result = self.parser._try_fallback_patterns("will it rain in Tokyo", "will it rain in tokyo")
        assert result is not None
        intent, location = result
        assert intent == "forecast"
        assert location == "Tokyo"
    
    def test_no_match(self):
        """Should return None for non-matching queries."""
        result = self.parser._try_fallback_patterns("random query", "random query")
        assert result is None
    
    def test_safe_mode_uses_fallback(self):
        """Safe mode should use fallback patterns."""
        result = self.parser._safe_mode_parse("weather in Madrid", {})
        assert result.intent == "forecast"
        assert result.location == "Madrid"
    
    def test_safe_mode_air_quality_fallback(self):
        """Safe mode should use fallback for air quality queries."""
        result = self.parser._safe_mode_parse("air quality in Tokyo", {})
        assert result.intent == "analyze"
        assert result.location == "Tokyo"

