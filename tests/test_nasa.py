"""
Tests for NASA APOD client.

These tests verify:
- APOD data fetching
- API key handling (DEMO_KEY fallback)
- Retry logic
- Error handling
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import date
import httpx
import os

from tools.nasa_client import NASAClient, APODData


class TestAPODData:
    """Tests for APODData dataclass."""
    
    def test_apod_data_required_fields(self):
        """Should create APOD data with required fields."""
        data = APODData(
            title="Test Image",
            url="https://example.com/image.jpg",
            explanation="Test explanation",
            date="2024-12-24"
        )
        
        assert data.title == "Test Image"
        assert data.url == "https://example.com/image.jpg"
        assert data.explanation == "Test explanation"
        assert data.date == "2024-12-24"
    
    def test_apod_data_optional_fields(self):
        """Should handle optional fields."""
        data = APODData(
            title="Test",
            url="https://example.com/image.jpg",
            explanation="Test",
            date="2024-12-24",
            media_type="image",
            hdurl="https://example.com/hd.jpg",
            copyright="NASA"
        )
        
        assert data.media_type == "image"
        assert data.hdurl == "https://example.com/hd.jpg"
        assert data.copyright == "NASA"
    
    def test_apod_data_defaults(self):
        """Should set default values."""
        data = APODData(
            title="Test",
            url="https://example.com/image.jpg",
            explanation="Test",
            date="2024-12-24"
        )
        
        assert data.media_type == "image"
        assert data.hdurl is None
        assert data.copyright is None
    
    def test_apod_data_to_dict(self):
        """Should convert to dictionary."""
        data = APODData(
            title="Test Image",
            url="https://example.com/image.jpg",
            explanation="Test explanation",
            date="2024-12-24",
            copyright="NASA"
        )
        
        result = data.to_dict()
        
        assert result["title"] == "Test Image"
        assert result["url"] == "https://example.com/image.jpg"
        assert result["copyright"] == "NASA"


class TestNASAClient:
    """Tests for NASAClient class."""
    
    @pytest.fixture
    def client(self):
        """Create NASA client with DEMO_KEY."""
        return NASAClient(api_key="DEMO_KEY")
    
    @pytest.fixture
    def client_with_key(self):
        """Create NASA client with custom API key."""
        return NASAClient(api_key="test_api_key_12345")
    
    def test_init_with_demo_key(self, client):
        """Should initialize with DEMO_KEY."""
        assert client.api_key == "DEMO_KEY"
    
    def test_init_with_custom_key(self, client_with_key):
        """Should use provided API key."""
        assert client_with_key.api_key == "test_api_key_12345"
    
    def test_init_from_environment(self):
        """Should read API key from environment."""
        with patch.dict(os.environ, {"NASA_API_KEY": "env_key_123"}):
            client = NASAClient()
            assert client.api_key == "env_key_123"
    
    def test_init_fallback_to_demo(self):
        """Should fall back to DEMO_KEY if no key provided."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove NASA_API_KEY if it exists
            os.environ.pop("NASA_API_KEY", None)
            client = NASAClient()
            assert client.api_key == "DEMO_KEY"
    
    @pytest.mark.asyncio
    async def test_get_apod_success(self, client):
        """Should fetch APOD data successfully."""
        mock_response = {
            "title": "Orion Nebula",
            "url": "https://apod.nasa.gov/image.jpg",
            "explanation": "The Orion Nebula is a diffuse nebula...",
            "date": "2024-12-24",
            "media_type": "image",
            "hdurl": "https://apod.nasa.gov/hd.jpg"
        }
        
        with patch.object(client, '_request_with_retry', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            result = await client.get_apod()
            
            assert result.title == "Orion Nebula"
            assert result.media_type == "image"
            assert result.hdurl == "https://apod.nasa.gov/hd.jpg"
    
    @pytest.mark.asyncio
    async def test_get_apod_with_date(self, client):
        """Should fetch APOD for specific date."""
        mock_response = {
            "title": "Past Image",
            "url": "https://apod.nasa.gov/past.jpg",
            "explanation": "Past explanation",
            "date": "2024-01-01",
            "media_type": "image"
        }
        
        with patch.object(client, '_request_with_retry', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            result = await client.get_apod(date(2024, 1, 1))
            
            assert result.date == "2024-01-01"
    
    @pytest.mark.asyncio
    async def test_get_today(self, client):
        """Should fetch today's APOD."""
        mock_response = {
            "title": "Today's Image",
            "url": "https://apod.nasa.gov/today.jpg",
            "explanation": "Today's explanation",
            "date": "2024-12-24",
            "media_type": "image"
        }
        
        with patch.object(client, '_request_with_retry', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            result = await client.get_today()
            
            assert result.title == "Today's Image"
    
    @pytest.mark.asyncio
    async def test_get_apod_video_type(self, client):
        """Should handle video media type."""
        mock_response = {
            "title": "Space Video",
            "url": "https://youtube.com/watch?v=xyz",
            "explanation": "Video explanation",
            "date": "2024-12-24",
            "media_type": "video"
        }
        
        with patch.object(client, '_request_with_retry', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            result = await client.get_apod()
            
            assert result.media_type == "video"
    
    @pytest.mark.asyncio
    async def test_get_apod_with_copyright(self, client):
        """Should handle copyright field."""
        mock_response = {
            "title": "Copyrighted Image",
            "url": "https://example.com/image.jpg",
            "explanation": "Explanation",
            "date": "2024-12-24",
            "media_type": "image",
            "copyright": "John Doe"
        }
        
        with patch.object(client, '_request_with_retry', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            result = await client.get_apod()
            
            assert result.copyright == "John Doe"
    
    @pytest.mark.asyncio
    async def test_get_apod_api_error(self, client):
        """Should raise error on API failure."""
        with patch.object(client, '_request_with_retry', new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = httpx.RequestError("API Error")
            
            with pytest.raises(Exception):
                await client.get_apod()


class TestNASAClientRetry:
    """Tests for NASA client retry logic."""
    
    @pytest.fixture
    def client(self):
        """Create NASA client."""
        return NASAClient(api_key="DEMO_KEY")
    
    @pytest.mark.asyncio
    async def test_retry_on_server_error(self, client):
        """Should retry on 5xx errors."""
        call_count = 0
        
        async def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                response = MagicMock()
                response.status_code = 500
                response.text = "Server Error"
                raise httpx.HTTPStatusError("Server Error", request=MagicMock(), response=response)
            
            return MagicMock(
                status_code=200,
                json=lambda: {
                    "title": "Test",
                    "url": "https://example.com/image.jpg",
                    "explanation": "Test",
                    "date": "2024-12-24"
                },
                raise_for_status=lambda: None
            )
        
        with patch('httpx.AsyncClient.get', new_callable=AsyncMock, side_effect=mock_get):
            with patch('asyncio.sleep', new_callable=AsyncMock):
                result = await client._request_with_retry(
                    "https://api.nasa.gov/test",
                    {"api_key": "DEMO_KEY"},
                    "NASA_TEST"
                )
                
                assert result is not None
                assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_no_retry_on_client_error(self, client):
        """Should not retry on 4xx errors (except 429)."""
        response = MagicMock()
        response.status_code = 404
        response.text = "Not Found"
        
        with patch('httpx.AsyncClient.get', new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = httpx.HTTPStatusError(
                "Not Found",
                request=MagicMock(),
                response=response
            )
            
            with pytest.raises(httpx.HTTPStatusError):
                await client._request_with_retry(
                    "https://api.nasa.gov/test",
                    {"api_key": "DEMO_KEY"},
                    "NASA_TEST"
                )
            
            # Should only be called once (no retry)
            assert mock_get.call_count == 1
    
    @pytest.mark.asyncio
    async def test_retry_on_rate_limit(self, client):
        """Should retry on 429 (rate limit)."""
        call_count = 0
        
        async def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                response = MagicMock()
                response.status_code = 429
                response.text = "Rate limited"
                raise httpx.HTTPStatusError("Rate limited", request=MagicMock(), response=response)
            
            return MagicMock(
                status_code=200,
                json=lambda: {"title": "Test", "url": "test", "explanation": "test", "date": "2024-12-24"},
                raise_for_status=lambda: None
            )
        
        with patch('httpx.AsyncClient.get', new_callable=AsyncMock, side_effect=mock_get):
            with patch('asyncio.sleep', new_callable=AsyncMock):
                result = await client._request_with_retry(
                    "https://api.nasa.gov/test",
                    {},
                    "NASA_TEST"
                )
                
                assert call_count == 2  # Retried once


class TestNASAClientEdgeCases:
    """Edge case tests for NASA client."""
    
    @pytest.fixture
    def client(self):
        """Create NASA client."""
        return NASAClient(api_key="DEMO_KEY")
    
    @pytest.mark.asyncio
    async def test_missing_optional_fields(self, client):
        """Should handle missing optional fields."""
        mock_response = {
            "title": "Minimal",
            "url": "https://example.com/image.jpg",
            "explanation": "Test",
            "date": "2024-12-24"
            # No hdurl, copyright, media_type
        }
        
        with patch.object(client, '_request_with_retry', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            result = await client.get_apod()
            
            assert result.title == "Minimal"
            assert result.media_type == "image"  # Default
            assert result.hdurl is None
            assert result.copyright is None
    
    @pytest.mark.asyncio
    async def test_empty_title(self, client):
        """Should handle empty title."""
        mock_response = {
            "title": "",
            "url": "https://example.com/image.jpg",
            "explanation": "Test",
            "date": "2024-12-24"
        }
        
        with patch.object(client, '_request_with_retry', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            result = await client.get_apod()
            
            # Should use "Unknown" or empty string
            assert result.title is not None
