"""
Tests for query parser module.

These tests verify:
- LLM-based natural language parsing
- Coordinate extraction
- Time period parsing (future/historical)
- Safe mode fallback when LLM unavailable
- Follow-up query handling
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from agent.query_parser import QueryParser, ParsedQuery


class TestParsedQuery:
    """Tests for ParsedQuery dataclass."""
    
    def test_parsed_query_defaults(self):
        """Default values should be set correctly."""
        query = ParsedQuery(intent="analyze")
        
        assert query.intent == "analyze"
        assert query.location is None
        assert query.hours == 6
        assert query.past_days == 0
        assert query.is_followup is False
        assert query.needs_location is False
        assert query.coordinates is None
    
    def test_parsed_query_with_coordinates(self):
        """Should store coordinates as tuple."""
        query = ParsedQuery(
            intent="analyze",
            coordinates=(42.6977, 23.3219)
        )
        
        assert query.coordinates == (42.6977, 23.3219)
    
    def test_parsed_query_with_all_fields(self):
        """Should store all provided fields."""
        query = ParsedQuery(
            intent="forecast",
            location="Sofia",
            hours=24,
            past_days=7,
            is_followup=True,
            needs_location=False,
            coordinates=(42.6977, 23.3219)
        )
        
        assert query.intent == "forecast"
        assert query.location == "Sofia"
        assert query.hours == 24
        assert query.past_days == 7
        assert query.is_followup is True
        assert query.coordinates == (42.6977, 23.3219)


class TestQueryParserSafeMode:
    """Tests for safe mode fallback parsing."""
    
    @pytest.fixture
    def parser(self):
        """Create parser with mocked LLM that always fails."""
        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(side_effect=Exception("LLM unavailable"))
        return QueryParser(llm_client=mock_llm)
    
    @pytest.mark.asyncio
    async def test_safe_mode_apod_intent(self, parser):
        """Should detect APOD intent in safe mode."""
        queries = [
            "show me the nasa picture",
            "astronomy picture of the day",
            "today's apod",
            "nasa apod",
        ]
        
        for query in queries:
            result = await parser.parse(query)
            assert result.intent == "apod", f"Failed for: {query}"
    
    @pytest.mark.asyncio
    async def test_safe_mode_help_intent(self, parser):
        """Should detect help intent in safe mode."""
        queries = [
            "help",
            "what can you do",
            "help me",
        ]
        
        for query in queries:
            result = await parser.parse(query)
            assert result.intent == "help", f"Failed for: {query}"
    
    @pytest.mark.asyncio
    async def test_safe_mode_greeting_intent(self, parser):
        """Should detect greeting intent in safe mode."""
        queries = [
            "hello",
            "hi",
            "hey",
        ]
        
        for query in queries:
            result = await parser.parse(query)
            assert result.intent == "greeting", f"Failed for: {query}"
    
    @pytest.mark.asyncio
    async def test_safe_mode_analyze_intent(self, parser):
        """Should detect analyze intent in safe mode."""
        queries = [
            "air quality in sofia",
            "pm2.5 levels",
            "is it safe to run",
            "pollution in london",
        ]
        
        for query in queries:
            result = await parser.parse(query)
            assert result.intent == "analyze", f"Failed for: {query}"
    
    @pytest.mark.asyncio
    async def test_safe_mode_forecast_intent(self, parser):
        """Should detect forecast intent in safe mode."""
        queries = [
            "weather forecast",
            "what's the temperature",
            "will it rain tomorrow",
        ]
        
        for query in queries:
            result = await parser.parse(query)
            assert result.intent == "forecast", f"Failed for: {query}"
    
    @pytest.mark.asyncio
    async def test_safe_mode_coordinate_extraction(self, parser):
        """Should extract coordinates in safe mode."""
        result = await parser.parse("weather at 42.6977, 23.3219")
        
        assert result.coordinates is not None
        assert abs(result.coordinates[0] - 42.6977) < 0.01
        assert abs(result.coordinates[1] - 23.3219) < 0.01
    
    @pytest.mark.asyncio
    async def test_safe_mode_coordinate_formats(self, parser):
        """Should handle various coordinate formats."""
        test_cases = [
            ("42.6977, 23.3219", (42.6977, 23.3219)),
            ("lat 42.6977 lon 23.3219", (42.6977, 23.3219)),
            ("-33.8688, 151.2093", (-33.8688, 151.2093)),  # Sydney
        ]
        
        for query, expected in test_cases:
            result = await parser.parse(f"weather at {query}")
            assert result.coordinates is not None, f"Failed for: {query}"
            assert abs(result.coordinates[0] - expected[0]) < 0.01
            assert abs(result.coordinates[1] - expected[1]) < 0.01
    
    @pytest.mark.asyncio
    async def test_safe_mode_hours_extraction(self, parser):
        """Should extract hours in safe mode."""
        test_cases = [
            ("next 12 hours", 12),
            ("for 24 hours", 24),
            ("next 3 days", 72),  # 3 * 24
        ]
        
        for query, expected_hours in test_cases:
            result = await parser.parse(f"weather {query}")
            assert result.hours == expected_hours, f"Failed for: {query}"
    
    @pytest.mark.asyncio
    async def test_safe_mode_location_extraction(self, parser):
        """Should extract location in safe mode."""
        # Note: Safe mode has limited location extraction - it requires "in" keyword
        test_cases = [
            ("weather in Sofia", "Sofia"),
            ("air quality in Paris", "Paris"),
            ("pollution in Tokyo", "Tokyo"),
        ]
        
        for query, expected_loc in test_cases:
            result = await parser.parse(query)
            assert result.location is not None, f"Failed for: {query}"
            assert expected_loc.lower() in result.location.lower()


class TestQueryParserLLMMode:
    """Tests for LLM-based parsing."""
    
    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM client."""
        mock = MagicMock()
        mock.chat = AsyncMock()
        return mock
    
    @pytest.fixture
    def parser(self, mock_llm):
        """Create parser with mock LLM."""
        return QueryParser(llm_client=mock_llm)
    
    @pytest.mark.asyncio
    async def test_llm_parsing_analyze(self, parser, mock_llm):
        """Should parse analyze intent via LLM."""
        mock_llm.chat.return_value = '''{"intent": "analyze", "location": "Sofia", "coordinates": null, "hours": 6, "past_days": 0, "is_followup": false, "needs_location": false}'''
        
        result = await parser.parse("Is it safe to run in Sofia?")
        
        assert result.intent == "analyze"
        assert result.location == "Sofia"
        assert result.hours == 6
    
    @pytest.mark.asyncio
    async def test_llm_parsing_apod(self, parser, mock_llm):
        """Should parse APOD intent via LLM."""
        mock_llm.chat.return_value = '''{"intent": "apod", "location": null, "coordinates": null, "hours": 6, "past_days": 0, "is_followup": false, "needs_location": false}'''
        
        result = await parser.parse("Show me today's NASA picture")
        
        assert result.intent == "apod"
    
    @pytest.mark.asyncio
    async def test_llm_parsing_with_coordinates(self, parser, mock_llm):
        """Should extract coordinates from LLM response."""
        mock_llm.chat.return_value = '''{"intent": "analyze", "location": null, "coordinates": [42.6977, 23.3219], "hours": 6, "past_days": 0, "is_followup": false, "needs_location": false}'''
        
        result = await parser.parse("Air quality at 42.6977, 23.3219")
        
        assert result.coordinates == (42.6977, 23.3219)
    
    @pytest.mark.asyncio
    async def test_llm_parsing_historical(self, parser, mock_llm):
        """Should parse historical data request."""
        mock_llm.chat.return_value = '''{"intent": "forecast", "location": "Berlin", "coordinates": null, "hours": 168, "past_days": 7, "is_followup": false, "needs_location": false}'''
        
        result = await parser.parse("Weather last week in Berlin")
        
        assert result.intent == "forecast"
        assert result.past_days == 7
        assert result.location == "Berlin"
    
    @pytest.mark.asyncio
    async def test_llm_parsing_followup(self, parser, mock_llm):
        """Should detect follow-up queries."""
        mock_llm.chat.return_value = '''{"intent": "analyze", "location": "Paris", "coordinates": null, "hours": null, "past_days": 0, "is_followup": true, "needs_location": false}'''
        
        context = {"last_location": "Sofia", "last_intent": "analyze"}
        result = await parser.parse("What about Paris?", context)
        
        assert result.is_followup is True
        assert result.location == "Paris"
    
    @pytest.mark.asyncio
    async def test_llm_fallback_on_error(self, parser, mock_llm):
        """Should fall back to safe mode on LLM error."""
        mock_llm.chat.side_effect = Exception("API Error")
        
        result = await parser.parse("show me nasa apod")
        
        # Should still work via safe mode
        assert result.intent == "apod"
    
    @pytest.mark.asyncio
    async def test_llm_parsing_unknown_intent(self, parser, mock_llm):
        """Should handle unknown/off-topic queries."""
        mock_llm.chat.return_value = '''{"intent": "unknown", "location": null, "coordinates": null, "hours": 6, "past_days": 0, "is_followup": false, "needs_location": false}'''
        
        result = await parser.parse("What is 2 + 2?")
        
        assert result.intent == "unknown"
    
    @pytest.mark.asyncio
    async def test_llm_parsing_malformed_json(self, parser, mock_llm):
        """Should handle malformed JSON from LLM."""
        mock_llm.chat.return_value = "This is not valid JSON"
        
        # Should fall back to safe mode
        result = await parser.parse("weather in London")
        
        assert result.intent in ["forecast", "analyze"]


class TestQueryParserContextBuilding:
    """Tests for context-aware prompt building."""
    
    @pytest.fixture
    def parser(self):
        """Create parser with mocked LLM."""
        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(return_value='{"intent": "analyze", "location": null, "coordinates": null, "hours": 6, "past_days": 0, "is_followup": true, "needs_location": false}')
        return QueryParser(llm_client=mock_llm)
    
    @pytest.mark.asyncio
    async def test_context_with_last_location(self, parser):
        """Should include last location in context."""
        context = {
            "last_location": "Sofia",
            "last_coords": (42.6977, 23.3219),
            "last_intent": "analyze"
        }
        
        result = await parser.parse("what about tomorrow?", context)
        
        # Parser should have passed context to LLM
        assert result.is_followup is True
    
    @pytest.mark.asyncio
    async def test_context_empty(self, parser):
        """Should handle empty context."""
        result = await parser.parse("weather in Paris", {})
        
        assert result is not None

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


    @pytest.fixture
    def parser(self):
        """Create parser with mocked LLM that always fails."""
        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(side_effect=Exception("LLM unavailable"))
        return QueryParser(llm_client=mock_llm)
    
    @pytest.mark.asyncio
    async def test_last_week(self, parser):
        """Should parse 'last week'."""
        result = await parser.parse("weather last week in London")
        
        assert result.past_days >= 7
    
    @pytest.mark.asyncio
    async def test_past_3_days(self, parser):
        """Should parse 'past 3 days'."""
        result = await parser.parse("air quality past 3 days in Berlin")
        
        assert result.past_days >= 3
    
    @pytest.mark.asyncio
    async def test_next_3_days(self, parser):
        """Should parse 'next 3 days'."""
        result = await parser.parse("forecast next 3 days in Vienna")
        
        assert result.past_days == 0
        assert result.hours >= 72
    
    @pytest.mark.asyncio
    async def test_tomorrow(self, parser):
        """Should parse 'tomorrow'."""
        result = await parser.parse("weather tomorrow in Paris")
        
        assert result.past_days == 0
    
    @pytest.mark.asyncio
    async def test_yesterday(self, parser):
        """Should parse 'yesterday'."""
        result = await parser.parse("weather yesterday in Madrid")
        
        assert result.past_days >= 1


class TestQueryParserSafeModeFollowups:
    """Test follow-up query handling in safe mode."""
    
    @pytest.fixture
    def parser(self):
        """Create parser with mocked LLM that always fails."""
        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(side_effect=Exception("LLM unavailable"))
        return QueryParser(llm_client=mock_llm)
    
    @pytest.mark.asyncio
    async def test_followup_preserves_location(self, parser):
        """Should preserve location from context for follow-up."""
        context = {
            "last_location": "Sofia",
            "last_coords": (42.69, 23.32),
            "last_intent": "analyze",
            "last_hours": 6
        }
        
        result = await parser.parse("what about now?", context)
        
        assert result.is_followup is True
        assert result.location == "Sofia"
    
    @pytest.mark.asyncio
    async def test_followup_short_query(self, parser):
        """Should detect short follow-up queries."""
        context = {
            "last_location": "Sofia",
            "last_intent": "analyze"
        }
        
        result = await parser.parse("and tomorrow?", context)
        
        assert result.is_followup is True


class TestQueryParserSafeModeIntents:
    """Test intent detection in safe mode."""
    
    @pytest.fixture
    def parser(self):
        """Create parser with mocked LLM that always fails."""
        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(side_effect=Exception("LLM unavailable"))
        return QueryParser(llm_client=mock_llm)
    
    @pytest.mark.asyncio
    async def test_analyze_intent(self, parser):
        """Should detect analyze intent."""
        result = await parser.parse("pm2.5 levels in London")
        
        assert result.intent == "analyze"
    
    @pytest.mark.asyncio
    async def test_forecast_intent(self, parser):
        """Should detect forecast intent."""
        result = await parser.parse("weather forecast for Paris")
        
        assert result.intent == "forecast"
    
    @pytest.mark.asyncio
    async def test_apod_intent(self, parser):
        """Should detect APOD intent."""
        result = await parser.parse("show the picture of the day")
        
        assert result.intent == "apod"
    
    @pytest.mark.asyncio
    async def test_greet_intent(self, parser):
        """Should detect greeting intent."""
        result = await parser.parse("hello")
        
        assert result.intent in ["greet", "greeting"]
    
    @pytest.mark.asyncio
    async def test_help_intent(self, parser):
        """Should detect help intent."""
        result = await parser.parse("help")
        
        assert result.intent == "help"


class TestQueryParserSafeModeCoordinates:
    """Test coordinate parsing in safe mode."""
    
    @pytest.fixture
    def parser(self):
        """Create parser with mocked LLM that always fails."""
        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(side_effect=Exception("LLM unavailable"))
        return QueryParser(llm_client=mock_llm)
    
    @pytest.mark.asyncio
    async def test_coordinates_with_comma(self, parser):
        """Should parse lat,lon format."""
        result = await parser.parse("weather at 42.69, 23.32")
        
        assert result.coordinates is not None
        assert result.coordinates[0] == pytest.approx(42.69, 0.01)
    
    @pytest.mark.asyncio
    async def test_negative_coordinates(self, parser):
        """Should parse negative coordinates."""
        result = await parser.parse("weather at -33.87, 151.21")
        
        assert result.coordinates is not None
        assert result.coordinates[0] < 0


class TestQueryParserLLMMode:
    """Test LLM-powered parsing."""
    
    @pytest.mark.asyncio
    async def test_llm_mode_parse(self):
        """Should use LLM for parsing when available."""
        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(return_value='{"intent": "analyze", "location": "London", "hours": 6, "past_days": 0, "is_followup": false, "needs_location": false}')
        
        parser = QueryParser(llm_client=mock_llm)
        result = await parser.parse("Is it safe to jog in London?")
        
        assert result.intent == "analyze"
        assert result.location == "London"
    
    @pytest.mark.asyncio
    async def test_llm_mode_fallback_to_safe(self):
        """Should fallback to safe mode on LLM error."""
        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(side_effect=Exception("LLM Error"))
        
        parser = QueryParser(llm_client=mock_llm)
        result = await parser.parse("weather in London")
        
        # Should still parse using fallback
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_llm_mode_invalid_json(self):
        """Should fallback when LLM returns invalid JSON."""
        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(return_value='not valid json')
        
        parser = QueryParser(llm_client=mock_llm)
        result = await parser.parse("weather in Tokyo")
        
        # Should still parse using fallback
        assert result is not None


class TestQueryParserNeedsLocation:
    """Test needs_location flag."""
    
    @pytest.fixture
    def parser(self):
        """Create parser with mocked LLM that always fails."""
        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(side_effect=Exception("LLM unavailable"))
        return QueryParser(llm_client=mock_llm)
    
    @pytest.mark.asyncio
    async def test_needs_location_true(self, parser):
        """Should set needs_location when location missing."""
        result = await parser.parse("is it safe to run?")
        
        assert result.needs_location is True
    
    @pytest.mark.asyncio
    async def test_needs_location_false_with_location(self, parser):
        """Should set needs_location false when location provided."""
        result = await parser.parse("is it safe to run in Paris?")
        
        assert result.needs_location is False


class TestParsedQueryDataclass:
    """Test ParsedQuery dataclass."""
    
    def test_parsed_query_initialization(self):
        """Should initialize correctly."""
        query = ParsedQuery(
            intent="analyze",
            location="Sofia",
            hours=6,
            past_days=0,
            is_followup=False,
            needs_location=False,
            coordinates=(42.69, 23.32)
        )
        
        assert query.intent == "analyze"
        assert query.location == "Sofia"
        assert query.hours == 6
        assert query.coordinates == (42.69, 23.32)
    
    def test_parsed_query_defaults(self):
        """Should have correct defaults."""
        query = ParsedQuery(intent="analyze")
        
        assert query.location is None
        assert query.coordinates is None
        assert query.is_followup is False
        assert query.hours == 6
        assert query.past_days == 0
