"""
Tests for /chat endpoint with various intents.

These tests verify:
- Natural language chat interface
- Intent routing (analyze, forecast, apod, help, greeting)
- Location handling and geocoding
- Follow-up conversations
- Error handling
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from fastapi.testclient import TestClient
from api.models import AnalyzeResponse, APODResponse, SafetyLevel, DataQuality


@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    mock = MagicMock()
    
    # Mock analyze response
    mock.analyze = AsyncMock(return_value=AnalyzeResponse(
        pm25_avg=15.5,
        pm10_avg=28.3,
        temp_avg=18.2,
        guidance_text="✅ Great conditions for outdoor exercise!",
        safety_level=SafetyLevel.SAFE,
        data_quality=DataQuality.HIGH,
        forecast_hours=6,
        attribution="Weather data by Open-Meteo.com",
        cached=False,
        timestamp=datetime.utcnow(),
    ))
    
    # Mock APOD response
    mock.get_apod = AsyncMock(return_value=APODResponse(
        title="Test APOD",
        url="https://example.com/image.jpg",
        explanation="Test explanation",
        summary="Test summary in 2 lines.",
        date="2024-12-24",
        media_type="image",
        attribution="Image from NASA Astronomy Picture of the Day",
    ))
    
    return mock


@pytest.fixture
def mock_query_parser():
    """Create a mock query parser."""
    from agent.query_parser import ParsedQuery
    
    mock = MagicMock()
    mock.parse = AsyncMock(return_value=ParsedQuery(
        intent="analyze",
        location="Sofia",
        hours=6,
        past_days=0,
        is_followup=False,
        needs_location=False,
        coordinates=None
    ))
    return mock


@pytest.fixture
def mock_geocoding():
    """Create a mock geocoding service."""
    from tools.geocoding import GeocodingResult
    
    mock = MagicMock()
    mock.geocode = AsyncMock(return_value=GeocodingResult(
        coords=(42.6977, 23.3219),
        location_name="Sofia",
        is_country=False,
        country="Bulgaria"
    ))
    return mock


@pytest.fixture
def client(mock_agent, mock_query_parser, mock_geocoding):
    """Create test client with mocked dependencies."""
    from api.routes import set_agent, router
    from main import app
    
    set_agent(mock_agent)
    
    # Patch the service getters
    with patch('api.routes.get_query_parser', return_value=mock_query_parser):
        with patch('api.routes.get_geocoding', return_value=mock_geocoding):
            with TestClient(app) as test_client:
                yield test_client


class TestChatAnalyzeIntent:
    """Tests for chat endpoint with analyze intent."""
    
    def test_chat_analyze_with_location(self, client, mock_agent, mock_query_parser, mock_geocoding):
        """Should analyze air quality for a location."""
        from agent.query_parser import ParsedQuery
        
        mock_query_parser.parse.return_value = ParsedQuery(
            intent="analyze",
            location="Sofia",
            hours=6,
            coordinates=None
        )
        
        with patch('api.routes.get_query_parser', return_value=mock_query_parser):
            with patch('api.routes.get_geocoding', return_value=mock_geocoding):
                response = client.post(
                    "/chat",
                    json={"message": "Is it safe to run in Sofia?"}
                )
        
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "intent" in data
    
    def test_chat_analyze_with_coordinates(self, client, mock_agent, mock_query_parser):
        """Should analyze air quality with direct coordinates."""
        from agent.query_parser import ParsedQuery
        
        mock_query_parser.parse.return_value = ParsedQuery(
            intent="analyze",
            location=None,
            hours=6,
            coordinates=(42.6977, 23.3219)
        )
        
        with patch('api.routes.get_query_parser', return_value=mock_query_parser):
            response = client.post(
                "/chat",
                json={"message": "Air quality at 42.6977, 23.3219"}
            )
        
        assert response.status_code == 200


class TestChatAPODIntent:
    """Tests for chat endpoint with APOD intent."""
    
    def test_chat_apod_request(self, client, mock_agent, mock_query_parser):
        """Should return APOD data."""
        from agent.query_parser import ParsedQuery
        
        mock_query_parser.parse.return_value = ParsedQuery(
            intent="apod",
            location=None,
            hours=6
        )
        
        with patch('api.routes.get_query_parser', return_value=mock_query_parser):
            response = client.post(
                "/chat",
                json={"message": "Show me NASA picture"}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["intent"] == "apod"


class TestChatHelpIntent:
    """Tests for chat endpoint with help intent."""
    
    def test_chat_help_request(self, client, mock_query_parser):
        """Should return help information."""
        from agent.query_parser import ParsedQuery
        
        mock_query_parser.parse.return_value = ParsedQuery(
            intent="help",
            location=None,
            hours=6
        )
        
        with patch('api.routes.get_query_parser', return_value=mock_query_parser):
            response = client.post(
                "/chat",
                json={"message": "help"}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["intent"] == "help"
        assert "Air Quality" in data["response"] or "air quality" in data["response"].lower()


class TestChatErrorHandling:
    """Tests for chat endpoint error handling."""
    
    def test_chat_empty_message(self, client):
        """Should handle empty message."""
        response = client.post(
            "/chat",
            json={"message": ""}
        )
        
        # Empty message should still be processed
        assert response.status_code in [200, 422]
    
    def test_chat_missing_message(self, client):
        """Should reject request without message field."""
        response = client.post(
            "/chat",
            json={}
        )
        
        assert response.status_code == 422  # Validation error


class TestChatCountryHandling:
    """Tests for country detection and city prompting."""
    
    def test_chat_country_detected(self, client, mock_query_parser, mock_geocoding):
        """Should prompt for city when country is detected."""
        from agent.query_parser import ParsedQuery
        from tools.geocoding import GeocodingResult
        
        mock_query_parser.parse.return_value = ParsedQuery(
            intent="analyze",
            location="Bulgaria",
            hours=6,
            coordinates=None
        )
        
        mock_geocoding.geocode.return_value = GeocodingResult(
            coords=(42.7339, 25.4858),
            location_name="Bulgaria",
            is_country=True,
            country="Bulgaria"
        )
        mock_geocoding.get_country_cities = AsyncMock(return_value=(
            ["Sofia", "Plovdiv", "Varna"],
            "Bulgaria is in Southeast Europe."
        ))
        
        with patch('api.routes.get_query_parser', return_value=mock_query_parser):
            with patch('api.routes.get_geocoding', return_value=mock_geocoding):
                response = client.post(
                    "/chat",
                    json={"message": "Weather in Bulgaria"}
                )
        
        assert response.status_code == 200
        data = response.json()
        # Should ask for specific city
        assert "country" in data["response"].lower() or "city" in data["response"].lower()


class TestChatModels:
    """Tests for chat request/response models."""
    
    def test_chat_request_valid(self, client, mock_query_parser):
        """Should accept valid chat request."""
        from agent.query_parser import ParsedQuery
        
        mock_query_parser.parse.return_value = ParsedQuery(
            intent="greeting",
            hours=6
        )
        
        with patch('api.routes.get_query_parser', return_value=mock_query_parser):
            with patch('api.routes._handle_greeting', new_callable=AsyncMock) as mock_greeting:
                from api.models import ChatResponse
                mock_greeting.return_value = ChatResponse(
                    response="Hello! I can help with weather and air quality.",
                    intent="greeting"
                )
                
                response = client.post(
                    "/chat",
                    json={"message": "Hello"}
                )
        
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
    
    def test_chat_response_structure(self, client, mock_agent, mock_query_parser, mock_geocoding):
        """Should return proper response structure."""
        from agent.query_parser import ParsedQuery
        
        mock_query_parser.parse.return_value = ParsedQuery(
            intent="analyze",
            location="Sofia",
            hours=6,
            coordinates=None
        )
        
        with patch('api.routes.get_query_parser', return_value=mock_query_parser):
            with patch('api.routes.get_geocoding', return_value=mock_geocoding):
                response = client.post(
                    "/chat",
                    json={"message": "Air quality in Sofia"}
                )
        
        assert response.status_code == 200
        data = response.json()
        
        # Required fields
        assert "response" in data
        assert "intent" in data
        
        # Response should be a string
        assert isinstance(data["response"], str)


class TestTimeFormatting:
    """Tests for time period formatting."""
    
    def test_format_time_period_hours(self):
        """Should format hours correctly."""
        from api.routes import _format_time_period
        
        assert _format_time_period(6) == "next 6 hours"
        assert _format_time_period(12) == "today"
        assert _format_time_period(24) == "tomorrow"
    
    def test_format_time_period_days(self):
        """Should format days correctly."""
        from api.routes import _format_time_period
        
        assert "2 days" in _format_time_period(48)
        assert "days" in _format_time_period(72)
    
    def test_format_time_period_past(self):
        """Should format past periods correctly."""
        from api.routes import _format_time_period
        
        assert _format_time_period(24, past_days=1) == "yesterday"
        assert "past" in _format_time_period(168, past_days=7)


class TestChatIntegration:
    """Integration tests for chat flow."""
    
    @pytest.mark.asyncio
    async def test_full_analyze_flow(self, mock_agent):
        """Test complete analyze flow."""
        from api.routes import _handle_weather_query
        from agent.query_parser import ParsedQuery
        from agent.memory import ConversationContext
        from tools.geocoding import GeocodingService, GeocodingResult
        
        parsed = ParsedQuery(
            intent="analyze",
            location="Sofia",
            hours=6,
            coordinates=None
        )
        
        mock_geocoding = MagicMock()
        mock_geocoding.geocode = AsyncMock(return_value=GeocodingResult(
            coords=(42.6977, 23.3219),
            location_name="Sofia",
            is_country=False,
            country="Bulgaria"
        ))
        
        context = ConversationContext()
        
        result = await _handle_weather_query(
            query="Is it safe to run in Sofia?",
            parsed=parsed,
            geocoding=mock_geocoding,
            agent=mock_agent,
            context=context
        )
        
        assert result is not None
        assert result.intent == "analyze"
        assert "Sofia" in result.response
