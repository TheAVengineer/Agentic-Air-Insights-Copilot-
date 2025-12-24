"""
API integration tests.

These tests verify:
- Endpoint responses
- Request validation
- Error handling
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime

from fastapi.testclient import TestClient


# We need to mock the agent before importing the app
@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    from api.models import AnalyzeResponse, APODResponse, SafetyLevel, DataQuality
    
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
        date="2024-12-20",
        media_type="image",
        attribution="Image from NASA Astronomy Picture of the Day",
    ))
    
    # Mock cache stats
    mock.get_cache_stats = MagicMock(return_value={
        "hits": 10,
        "misses": 5,
        "size": 3,
    })
    
    return mock


@pytest.fixture
def client(mock_agent):
    """Create test client with mocked agent."""
    from api.routes import set_agent
    from main import app
    
    set_agent(mock_agent)
    
    with TestClient(app) as client:
        yield client


class TestHealthEndpoint:
    """Tests for /health endpoint."""
    
    def test_health_check(self, client):
        """Health check should return 200."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "timestamp" in data


class TestAnalyzeEndpoint:
    """Tests for /analyze endpoint."""
    
    def test_analyze_valid_request(self, client):
        """Valid request should return analysis."""
        response = client.post(
            "/analyze",
            json={
                "latitude": 42.6977,
                "longitude": 23.3219,
                "hours": 6,
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "pm25_avg" in data
        assert "pm10_avg" in data
        assert "temp_avg" in data
        assert "guidance_text" in data
        assert "attribution" in data
    
    def test_analyze_default_hours(self, client):
        """Hours should default to 6."""
        response = client.post(
            "/analyze",
            json={
                "latitude": 42.6977,
                "longitude": 23.3219,
            }
        )
        
        assert response.status_code == 200
    
    def test_analyze_invalid_latitude(self, client):
        """Invalid latitude should return 422."""
        response = client.post(
            "/analyze",
            json={
                "latitude": 100.0,  # Invalid
                "longitude": 23.3219,
                "hours": 6,
            }
        )
        
        assert response.status_code == 422
    
    def test_analyze_invalid_longitude(self, client):
        """Invalid longitude should return 422."""
        response = client.post(
            "/analyze",
            json={
                "latitude": 42.6977,
                "longitude": 200.0,  # Invalid
                "hours": 6,
            }
        )
        
        assert response.status_code == 422
    
    def test_analyze_invalid_hours(self, client):
        """Invalid hours should return 422."""
        response = client.post(
            "/analyze",
            json={
                "latitude": 42.6977,
                "longitude": 23.3219,
                "hours": 500,  # Invalid (max 384 for 16-day forecast)
            }
        )
        
        assert response.status_code == 422
    
    def test_analyze_missing_required_field(self, client):
        """Missing latitude should return 422."""
        response = client.post(
            "/analyze",
            json={
                "longitude": 23.3219,
                "hours": 6,
            }
        )
        
        assert response.status_code == 422


class TestAPODEndpoint:
    """Tests for /apod/today endpoint."""
    
    def test_get_apod(self, client):
        """Should return APOD data."""
        response = client.get("/apod/today")
        
        assert response.status_code == 200
        data = response.json()
        assert "title" in data
        assert "url" in data
        assert "explanation" in data
        assert "summary" in data
        assert "attribution" in data


class TestChatEndpoint:
    """Tests for /chat endpoint."""
    
    def test_chat_air_quality(self, client):
        """Should detect air quality intent."""
        response = client.post(
            "/chat",
            json={
                "message": "What's the air quality at 42.6977, 23.3219 for 6 hours?"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert data["intent"] == "analyze"
    
    def test_chat_apod(self, client):
        """Should detect APOD intent."""
        response = client.post(
            "/chat",
            json={
                "message": "Show me today's astronomy picture"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["intent"] == "apod"
    
    def test_chat_help(self, client):
        """Should detect help intent."""
        response = client.post(
            "/chat",
            json={
                "message": "What can you help me with?"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["intent"] == "help"
    
    def test_chat_unknown(self, client):
        """Should handle unknown queries."""
        response = client.post(
            "/chat",
            json={
                "message": "What's the meaning of life?"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["intent"] == "unknown"


class TestCacheStatsEndpoint:
    """Tests for /cache/stats endpoint."""
    
    def test_get_cache_stats(self, client):
        """Should return cache statistics."""
        response = client.get("/cache/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert "hits" in data
        assert "misses" in data
        assert "size" in data


class TestOpenAPISchema:
    """Tests for OpenAPI schema export."""
    
    def test_openapi_available(self, client):
        """OpenAPI schema should be available."""
        response = client.get("/openapi.json")
        
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "paths" in data
        assert "/analyze" in data["paths"]
        assert "/apod/today" in data["paths"]
    
    def test_openapi_export_endpoint(self, client):
        """Export endpoint should return schema."""
        response = client.get("/openapi-export.json")
        
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data


class TestCoordinateExtraction:
    """Tests for coordinate extraction from chat (via QueryParser)."""
    
    def test_extract_coordinates(self):
        """Should extract coordinates from query using QueryParser."""
        from agent.query_parser import QueryParser
        
        parser = QueryParser()
        
        # Standard format
        coords = parser._extract_coordinates("Check 42.6977, 23.3219")
        assert coords == (42.6977, 23.3219)
        
        # With negative longitude
        coords = parser._extract_coordinates("Location -33.86, 151.21")
        assert coords == (-33.86, 151.21)
        
        # No coordinates
        coords = parser._extract_coordinates("Show me today's APOD")
        assert coords is None
    
    def test_extract_hours(self):
        """Should extract hours from query using QueryParser."""
        from agent.query_parser import QueryParser
        
        parser = QueryParser()
        
        # _extract_time returns (hours, past_days) tuple
        assert parser._extract_time("for the next 6 hours")[0] == 6
        assert parser._extract_time("next 3 days")[0] == 72  # 3*24
        assert parser._extract_time("no hours mentioned")[0] == 6  # Default


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
