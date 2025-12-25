"""
Extended tests for api/routes.py to increase coverage.
Tests LLM status, forecast handlers, and helper functions.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from fastapi.testclient import TestClient
from fastapi import FastAPI
from api.routes import _handle_forecast, _handle_apod, _handle_help, _handle_off_topic, _handle_greeting
from tools.weather_client import DailyForecast

from api.routes import (
    router,
    get_agent,
    get_query_parser,
    get_geocoding,
    get_context,
    set_agent,
    _handle_apod,
    _handle_help,
    _handle_off_topic,
    _handle_greeting,
    _format_time_period,
)
from api.models import ChatResponse, APODResponse
from agent.memory import ConversationContext
from agent.query_parser import ParsedQuery


# Create test app
app = FastAPI()
app.include_router(router)
client = TestClient(app)


class TestHealthAndStatus:
    """Test health check and status endpoints."""
    
    def test_health_check(self):
        """Health endpoint should return healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"
        assert "timestamp" in data
    
    def test_rate_limits_endpoint(self):
        """Rate limits endpoint should return provider status."""
        with patch("httpx.AsyncClient") as mock_client:
            # Mock the HTTP client
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"models": [{"name": "llama3.2"}]}
            mock_response.headers = {}
            
            async_mock = AsyncMock()
            async_mock.get.return_value = mock_response
            async_mock.post.return_value = mock_response
            async_mock.__aenter__.return_value = async_mock
            async_mock.__aexit__.return_value = None
            mock_client.return_value = async_mock
            
            response = client.get("/status/rate-limits")
            assert response.status_code == 200
            data = response.json()
            
            # Check structure
            assert "github_models" in data
            assert "ollama" in data
            assert "active_provider" in data
            assert "recommendation" in data
            
            # Check GitHub Models structure
            assert "provider" in data["github_models"]
            assert "rate_limit" in data["github_models"]
            assert "status" in data["github_models"]
            
            # Check Ollama structure
            assert "provider" in data["ollama"]
            assert "rate_limit" in data["ollama"]
            assert data["ollama"]["rate_limit"] == "unlimited"


class TestFormatTimePeriod:
    """Test _format_time_period helper function."""
    
    def test_format_yesterday(self):
        """Should format 1 past day as 'yesterday'."""
        assert _format_time_period(24, past_days=1) == "yesterday"
    
    def test_format_past_days(self):
        """Should format past days correctly."""
        assert _format_time_period(72, past_days=3) == "past 3 days"
    
    def test_format_past_week(self):
        """Should format past week correctly."""
        result = _format_time_period(168, past_days=7)
        assert "7" in result or "week" in result.lower()
    
    def test_format_future_hours(self):
        """Should format future hours correctly."""
        assert _format_time_period(3, past_days=0) == "next 3 hours"
        assert _format_time_period(6, past_days=0) == "next 6 hours"
    
    def test_format_today(self):
        """Should format today correctly."""
        assert _format_time_period(12, past_days=0) == "today"
    
    def test_format_tomorrow(self):
        """Should format tomorrow correctly."""
        assert _format_time_period(24, past_days=0) == "tomorrow"
    
    def test_format_next_days(self):
        """Should format next days correctly."""
        assert _format_time_period(48, past_days=0) == "next 2 days"
        assert _format_time_period(72, past_days=0) == "next 3 days"


class TestHandleHelp:
    """Test help handler."""
    
    def test_handle_help_returns_chat_response(self):
        """Should return helpful guidance."""
        result = _handle_help()
        assert isinstance(result, ChatResponse)
        assert result.intent == "help"
        assert "Air Quality" in result.response


class TestHandleApod:
    """Test APOD handler."""
    
    @pytest.mark.asyncio
    async def test_handle_apod_success(self):
        """Should handle APOD requests successfully."""
        mock_agent = MagicMock()
        mock_agent.get_apod = AsyncMock(return_value=APODResponse(
            title="Test Image",
            url="https://example.com/image.jpg",
            explanation="Test explanation",
            summary="Test summary",
            date="2025-12-24"
        ))
        context = ConversationContext()
        
        result = await _handle_apod(mock_agent, context)
        
        assert isinstance(result, ChatResponse)
        assert result.intent == "apod"
        assert "Test Image" in result.response
    
    @pytest.mark.asyncio
    async def test_handle_apod_failure(self):
        """Should handle APOD errors gracefully."""
        mock_agent = MagicMock()
        mock_agent.get_apod = AsyncMock(side_effect=Exception("API error"))
        context = ConversationContext()
        
        result = await _handle_apod(mock_agent, context)
        
        assert isinstance(result, ChatResponse)
        assert "couldn't fetch APOD" in result.response


class TestHandleOffTopic:
    """Test off-topic handler."""
    
    @pytest.mark.asyncio
    async def test_handle_off_topic_llm_failure(self):
        """Should fallback when LLM fails."""
        with patch("api.routes.LLMClient") as MockLLM:
            mock_client = MagicMock()
            mock_client.chat = AsyncMock(side_effect=Exception("LLM error"))
            MockLLM.return_value = mock_client
            
            result = await _handle_off_topic("Random question")
            
            assert isinstance(result, ChatResponse)
            assert "specialized" in result.response.lower()


class TestHandleGreeting:
    """Test greeting handler."""
    
    @pytest.mark.asyncio
    async def test_handle_greeting_llm_failure(self):
        """Should fallback when LLM fails."""
        with patch("api.routes.LLMClient") as MockLLM:
            mock_client = MagicMock()
            mock_client.chat = AsyncMock(side_effect=Exception("LLM error"))
            MockLLM.return_value = mock_client
            
            result = await _handle_greeting("Hi there")
            
            assert isinstance(result, ChatResponse)
            assert "Hello" in result.response


class TestChatEndpoint:
    """Test the main chat endpoint."""
    
    def test_chat_help_intent(self):
        """Should handle help intent."""
        with patch("api.routes.get_query_parser") as mock_parser_fn:
            mock_parser = MagicMock()
            mock_parser.parse = AsyncMock(return_value=ParsedQuery(
                intent="help",
                location=None,
                hours=6,
                past_days=0,
                is_followup=False,
                needs_location=False,
                coordinates=None
            ))
            mock_parser_fn.return_value = mock_parser
            
            response = client.post("/chat", json={"message": "help"})
            assert response.status_code == 200
            data = response.json()
            assert data["intent"] == "help"


class TestServiceGetters:
    """Test service getter functions."""
    
    def test_get_agent_singleton(self):
        """Should return same agent instance."""
        agent1 = get_agent()
        agent2 = get_agent()
        assert agent1 is agent2
    
    def test_get_query_parser_singleton(self):
        """Should return same parser instance."""
        parser1 = get_query_parser()
        parser2 = get_query_parser()
        assert parser1 is parser2
    
    def test_get_geocoding_singleton(self):
        """Should return same geocoding instance."""
        geo1 = get_geocoding()
        geo2 = get_geocoding()
        assert geo1 is geo2
    
    def test_get_context_singleton(self):
        """Should return same context instance."""
        ctx1 = get_context()
        ctx2 = get_context()
        assert ctx1 is ctx2
    
    def test_set_agent(self):
        """Should allow setting agent for testing."""
        mock_agent = MagicMock()
        set_agent(mock_agent)
        assert get_agent() is mock_agent

class TestHandleForecastHistorical:
    """Test _handle_forecast with historical data."""
    
    @pytest.mark.asyncio
    async def test_handle_forecast_yesterday(self):
        """Should handle yesterday forecast."""
        mock_daily = DailyForecast(
            dates=["2025-12-23"],
            temp_max=[10.0],
            temp_min=[2.0],
            precipitation_sum=[0.0],
            precipitation_probability=[0],
            weather_code=[0],
            sunrise=["07:00"],
            sunset=["16:30"]
        )
        
        context = ConversationContext()
        
        with patch("tools.weather_client.WeatherClient") as MockWeather:
            mock_client = MagicMock()
            mock_client.get_historical_daily = AsyncMock(return_value=mock_daily)
            MockWeather.return_value = mock_client
            
            result = await _handle_forecast(
                "Weather yesterday",
                42.69, 23.32, 24,
                "Sofia", context, past_days=1
            )
            
            assert isinstance(result, ChatResponse)
            assert result.intent == "forecast"
            assert "Historical" in result.response
    
    @pytest.mark.asyncio
    async def test_handle_forecast_past_week(self):
        """Should handle past week forecast."""
        mock_daily = DailyForecast(
            dates=["2025-12-17", "2025-12-18", "2025-12-19", "2025-12-20", 
                   "2025-12-21", "2025-12-22", "2025-12-23"],
            temp_max=[8.0, 9.0, 10.0, 11.0, 10.0, 9.0, 8.0],
            temp_min=[0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0],
            precipitation_sum=[0.0, 2.0, 0.0, 0.0, 5.0, 0.0, 0.0],
            precipitation_probability=[0] * 7,
            weather_code=[0, 61, 0, 0, 61, 0, 0],
            sunrise=["07:00"] * 7,
            sunset=["16:30"] * 7
        )
        
        context = ConversationContext()
        
        with patch("tools.weather_client.WeatherClient") as MockWeather:
            mock_client = MagicMock()
            mock_client.get_historical_daily = AsyncMock(return_value=mock_daily)
            MockWeather.return_value = mock_client
            
            result = await _handle_forecast(
                "Weather last week",
                42.69, 23.32, 168,
                "Sofia", context, past_days=7
            )
            
            assert isinstance(result, ChatResponse)
            assert "Historical" in result.response
            # Check precipitation summary is included
            assert "rain" in result.response.lower() or "💧" in result.response


class TestHandleForecastFuture:
    """Test _handle_forecast with future data."""
    
    @pytest.mark.asyncio
    async def test_handle_forecast_next_3_days(self):
        """Should handle 3 day forecast."""
        mock_daily = DailyForecast(
            dates=["2025-12-24", "2025-12-25", "2025-12-26"],
            temp_max=[10.0, 12.0, 8.0],
            temp_min=[2.0, 4.0, 1.0],
            precipitation_sum=[0.0, 5.0, 0.0],
            precipitation_probability=[10, 80, 20],
            weather_code=[0, 61, 3],
            sunrise=["07:00", "07:01", "07:02"],
            sunset=["16:30", "16:31", "16:32"]
        )
        
        context = ConversationContext()
        
        with patch("tools.weather_client.WeatherClient") as MockWeather:
            mock_client = MagicMock()
            mock_client.get_weekly_forecast = AsyncMock(return_value=mock_daily)
            MockWeather.return_value = mock_client
            
            with patch("llm.client.LLMClient") as MockLLM:
                llm_client = MagicMock()
                llm_client.chat = AsyncMock(return_value="Nice weather ahead!")
                MockLLM.return_value = llm_client
                
                result = await _handle_forecast(
                    "Weather next 3 days",
                    42.69, 23.32, 72,
                    "Sofia", context, past_days=0
                )
                
                assert isinstance(result, ChatResponse)
                assert result.intent == "forecast"
                assert "Sofia" in result.response
    
    @pytest.mark.asyncio
    async def test_handle_forecast_llm_failure_fallback(self):
        """Should fallback gracefully when LLM fails."""
        mock_daily = DailyForecast(
            dates=["2025-12-24"],
            temp_max=[10.0],
            temp_min=[2.0],
            precipitation_sum=[0.0],
            precipitation_probability=[10],
            weather_code=[0],
            sunrise=["07:00"],
            sunset=["16:30"]
        )
        
        context = ConversationContext()
        
        with patch("tools.weather_client.WeatherClient") as MockWeather:
            mock_client = MagicMock()
            mock_client.get_weekly_forecast = AsyncMock(return_value=mock_daily)
            MockWeather.return_value = mock_client
            
            with patch("llm.client.LLMClient") as MockLLM:
                llm_client = MagicMock()
                llm_client.chat = AsyncMock(side_effect=Exception("LLM error"))
                MockLLM.return_value = llm_client
                
                result = await _handle_forecast(
                    "Weather tomorrow",
                    42.69, 23.32, 24,
                    "Sofia", context, past_days=0
                )
                
                assert isinstance(result, ChatResponse)
                assert "forecast" in result.response.lower()


class TestHandleHelpers:
    """Test helper endpoint handlers."""
    
    @pytest.mark.asyncio
    async def test_handle_apod(self):
        """Should handle APOD request."""
        mock_agent = MagicMock()
        mock_agent.get_apod = AsyncMock(return_value=APODResponse(
            title="Test Image",
            url="https://example.com/image.jpg",
            explanation="Test explanation",
            summary="Test summary",
            date="2025-12-24"
        ))
        
        context = ConversationContext()
        
        result = await _handle_apod(mock_agent, context)
        
        assert result.intent == "apod"
        assert "Test Image" in result.response
    
    @pytest.mark.asyncio
    async def test_handle_apod_error(self):
        """Should handle APOD error."""
        mock_agent = MagicMock()
        mock_agent.get_apod = AsyncMock(side_effect=Exception("APOD error"))
        
        context = ConversationContext()
        
        result = await _handle_apod(mock_agent, context)
        
        assert result.intent == "apod"
        assert "problem" in result.response.lower() or "error" in result.response.lower()
    
    @pytest.mark.asyncio
    async def test_handle_help(self):
        """Should return help text."""
        result = _handle_help()  # This is not async
        
        assert result.intent == "help"
        assert "air quality" in result.response.lower()
    
    @pytest.mark.asyncio
    async def test_handle_off_topic(self):
        """Should handle off-topic queries."""
        with patch("api.routes.LLMClient") as MockLLM:
            llm_client = MagicMock()
            llm_client.chat = AsyncMock(return_value="I can help with weather!")
            MockLLM.return_value = llm_client
            
            result = await _handle_off_topic("What is the meaning of life?")
            
            assert result.intent == "unknown"
    
    @pytest.mark.asyncio
    async def test_handle_off_topic_llm_error(self):
        """Should handle off-topic with LLM error."""
        with patch("api.routes.LLMClient") as MockLLM:
            llm_client = MagicMock()
            llm_client.chat = AsyncMock(side_effect=Exception("LLM error"))
            MockLLM.return_value = llm_client
            
            result = await _handle_off_topic("What is the meaning of life?")
            
            assert result.intent == "unknown"
            assert "specialized" in result.response.lower()
    
    @pytest.mark.asyncio
    async def test_handle_greeting(self):
        """Should handle greetings."""
        with patch("api.routes.LLMClient") as MockLLM:
            llm_client = MagicMock()
            llm_client.chat = AsyncMock(return_value="Hello! I can help with weather!")
            MockLLM.return_value = llm_client
            
            result = await _handle_greeting("Hello!")
            
            assert result.intent == "greeting"
    
    @pytest.mark.asyncio
    async def test_handle_greeting_llm_error(self):
        """Should handle greeting with LLM error."""
        with patch("api.routes.LLMClient") as MockLLM:
            llm_client = MagicMock()
            llm_client.chat = AsyncMock(side_effect=Exception("LLM error"))
            MockLLM.return_value = llm_client
            
            result = await _handle_greeting("Hello!")
            
            assert result.intent == "greeting"
            assert "Hello" in result.response


class TestFormatTimePeriod:
    """Test _format_time_period function."""
    
    def test_format_time_period_import(self):
        """Should import the function."""
        from api.routes import _format_time_period
        
        assert callable(_format_time_period)
    
    def test_format_yesterday(self):
        """Should format yesterday."""
        from api.routes import _format_time_period
        
        result = _format_time_period(24, past_days=1)
        assert result == "yesterday"
    
    def test_format_past_days(self):
        """Should format past days."""
        from api.routes import _format_time_period
        
        result = _format_time_period(72, past_days=3)
        assert "3 days" in result
    
    def test_format_past_weeks(self):
        """Should format past weeks."""
        from api.routes import _format_time_period
        
        result = _format_time_period(168, past_days=14)
        assert "2 week" in result
    
    def test_format_next_hours(self):
        """Should format next hours."""
        from api.routes import _format_time_period
        
        result = _format_time_period(6, past_days=0)
        assert "6 hours" in result
    
    def test_format_today(self):
        """Should format today."""
        from api.routes import _format_time_period
        
        result = _format_time_period(12, past_days=0)
        assert "today" in result
    
    def test_format_tomorrow(self):
        """Should format tomorrow."""
        from api.routes import _format_time_period
        
        result = _format_time_period(24, past_days=0)
        assert "tomorrow" in result
    
    def test_format_next_2_days(self):
        """Should format next 2 days."""
        from api.routes import _format_time_period
        
        result = _format_time_period(48, past_days=0)
        assert "2 days" in result
