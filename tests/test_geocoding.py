"""
Tests for geocoding service.

These tests verify:
- Location name to coordinates conversion
- Country vs city detection
- LLM-powered city suggestions
- Error handling for invalid locations
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx

from tools.geocoding import GeocodingService, GeocodingResult


class TestGeocodingResult:
    """Tests for GeocodingResult dataclass."""
    
    def test_geocoding_result_city(self):
        """Should create result for a city."""
        result = GeocodingResult(
            coords=(42.6977, 23.3219),
            location_name="Sofia",
            is_country=False,
            country="Bulgaria"
        )
        
        assert result.coords == (42.6977, 23.3219)
        assert result.location_name == "Sofia"
        assert result.is_country is False
        assert result.country == "Bulgaria"
    
    def test_geocoding_result_country(self):
        """Should create result for a country."""
        result = GeocodingResult(
            coords=(42.7339, 25.4858),
            location_name="Bulgaria",
            is_country=True,
            country="Bulgaria"
        )
        
        assert result.is_country is True
        assert result.location_name == "Bulgaria"
    
    def test_geocoding_result_no_coords(self):
        """Should handle None coordinates."""
        result = GeocodingResult(
            coords=None,
            location_name="Unknown",
            is_country=False,
            country=None
        )
        
        assert result.coords is None


class TestGeocodingService:
    """Tests for GeocodingService class."""
    
    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM client."""
        mock = MagicMock()
        mock.chat = AsyncMock(return_value='{"cities": ["Sofia", "Plovdiv", "Varna"], "info": "Bulgaria is in Southeast Europe."}')
        return mock
    
    @pytest.fixture
    def service(self, mock_llm):
        """Create geocoding service with mock LLM."""
        return GeocodingService(llm_client=mock_llm)
    
    @pytest.mark.asyncio
    async def test_geocode_city_success(self, service):
        """Should geocode a city successfully."""
        mock_response = {
            "lat": "42.6977",
            "lon": "23.3219",
            "display_name": "Sofia, Sofia City, Bulgaria",
            "addresstype": "city",
            "address": {
                "city": "Sofia",
                "country": "Bulgaria"
            }
        }
        
        with patch.object(httpx.AsyncClient, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = MagicMock(
                status_code=200,
                json=lambda: [mock_response],
                raise_for_status=lambda: None
            )
            
            result = await service.geocode("Sofia")
            
            assert result is not None
            assert result.coords == (42.6977, 23.3219)
            assert result.location_name == "Sofia"
            assert result.is_country is False
            assert result.country == "Bulgaria"
    
    @pytest.mark.asyncio
    async def test_geocode_country_detection(self, service):
        """Should detect when location is a country."""
        mock_response = {
            "lat": "42.7339",
            "lon": "25.4858",
            "display_name": "Bulgaria",
            "addresstype": "country",
            "type": "country",
            "address": {
                "country": "Bulgaria"
            }
        }
        
        with patch.object(httpx.AsyncClient, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = MagicMock(
                status_code=200,
                json=lambda: [mock_response],
                raise_for_status=lambda: None
            )
            
            result = await service.geocode("Bulgaria")
            
            assert result is not None
            assert result.is_country is True
    
    @pytest.mark.asyncio
    async def test_geocode_not_found(self, service):
        """Should return None for unknown location."""
        with patch.object(httpx.AsyncClient, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = MagicMock(
                status_code=200,
                json=lambda: [],  # Empty result
                raise_for_status=lambda: None
            )
            
            result = await service.geocode("NonexistentPlace12345")
            
            assert result is None
    
    @pytest.mark.asyncio
    async def test_geocode_api_error(self, service):
        """Should handle API errors gracefully."""
        with patch.object(httpx.AsyncClient, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = httpx.RequestError("Connection failed")
            
            result = await service.geocode("Sofia")
            
            assert result is None
    
    @pytest.mark.asyncio
    async def test_geocode_town_detection(self, service):
        """Should handle town address type."""
        mock_response = {
            "lat": "42.1354",
            "lon": "24.7453",
            "display_name": "Plovdiv, Plovdiv Province, Bulgaria",
            "addresstype": "town",
            "address": {
                "town": "Plovdiv",
                "country": "Bulgaria"
            }
        }
        
        with patch.object(httpx.AsyncClient, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = MagicMock(
                status_code=200,
                json=lambda: [mock_response],
                raise_for_status=lambda: None
            )
            
            result = await service.geocode("Plovdiv")
            
            assert result is not None
            assert result.is_country is False
            assert result.location_name == "Plovdiv"
    
    @pytest.mark.asyncio
    async def test_geocode_village_detection(self, service):
        """Should handle village address type."""
        mock_response = {
            "lat": "42.0",
            "lon": "24.0",
            "display_name": "Small Village, Bulgaria",
            "addresstype": "village",
            "address": {
                "village": "Small Village",
                "country": "Bulgaria"
            }
        }
        
        with patch.object(httpx.AsyncClient, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = MagicMock(
                status_code=200,
                json=lambda: [mock_response],
                raise_for_status=lambda: None
            )
            
            result = await service.geocode("Small Village")
            
            assert result is not None
            assert result.is_country is False


class TestGeocodingCountryCities:
    """Tests for get_country_cities method."""
    
    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM client."""
        mock = MagicMock()
        mock.chat = AsyncMock()
        return mock
    
    @pytest.fixture
    def service(self, mock_llm):
        """Create geocoding service with mock LLM."""
        return GeocodingService(llm_client=mock_llm)
    
    @pytest.mark.asyncio
    async def test_get_country_cities_success(self, service, mock_llm):
        """Should return cities for a country."""
        mock_llm.chat.return_value = '{"cities": ["Sofia", "Plovdiv", "Varna", "Burgas", "Ruse"], "info": "Bulgaria is in Southeast Europe."}'
        
        cities, info = await service.get_country_cities("Bulgaria")
        
        assert len(cities) == 5
        assert "Sofia" in cities
        assert "Plovdiv" in cities
        assert "Bulgaria" in info
    
    @pytest.mark.asyncio
    async def test_get_country_cities_with_code_block(self, service, mock_llm):
        """Should handle JSON wrapped in code block."""
        mock_llm.chat.return_value = '```json\n{"cities": ["Tokyo", "Osaka", "Kyoto"], "info": "Japan is an island nation."}\n```'
        
        cities, info = await service.get_country_cities("Japan")
        
        assert len(cities) == 3
        assert "Tokyo" in cities
    
    @pytest.mark.asyncio
    async def test_get_country_cities_llm_error(self, service, mock_llm):
        """Should handle LLM errors gracefully."""
        mock_llm.chat.side_effect = Exception("LLM Error")
        
        cities, info = await service.get_country_cities("France")
        
        # Should return empty/default values
        assert isinstance(cities, list)
        assert isinstance(info, str)
    
    @pytest.mark.asyncio
    async def test_get_country_cities_malformed_json(self, service, mock_llm):
        """Should handle malformed JSON from LLM."""
        mock_llm.chat.return_value = "This is not valid JSON at all"
        
        cities, info = await service.get_country_cities("Germany")
        
        # Should return empty/default values
        assert isinstance(cities, list)
    
    @pytest.mark.asyncio
    async def test_get_country_cities_limits_to_five(self, service, mock_llm):
        """Should limit cities to 5."""
        mock_llm.chat.return_value = '{"cities": ["A", "B", "C", "D", "E", "F", "G", "H"], "info": "Test"}'
        
        cities, info = await service.get_country_cities("TestCountry")
        
        assert len(cities) <= 5


class TestGeocodingValidation:
    """Tests for geocoding input validation."""
    
    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM client."""
        mock = MagicMock()
        mock.chat = AsyncMock(return_value='{"cities": ["Sofia"], "info": "Bulgaria"}')
        return mock
    
    @pytest.fixture
    def service(self, mock_llm):
        """Create geocoding service with mock LLM."""
        return GeocodingService(llm_client=mock_llm)
    
    @pytest.mark.asyncio
    async def test_rejects_single_character(self, service):
        """Should reject single character inputs like '0' or 'm'."""
        result = await service.geocode("0")
        assert result is None
        
        result = await service.geocode("m")
        assert result is None
        
        result = await service.geocode("x")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_rejects_empty_string(self, service):
        """Should reject empty string."""
        result = await service.geocode("")
        assert result is None
        
        result = await service.geocode("   ")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_rejects_numeric_only(self, service):
        """Should reject numeric-only inputs."""
        result = await service.geocode("123")
        assert result is None
        
        result = await service.geocode("12345")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_rejects_symbol_only(self, service):
        """Should reject symbol-only inputs."""
        result = await service.geocode("..")
        assert result is None
        
        result = await service.geocode("!@#")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_accepts_short_valid_cities(self, service):
        """Should accept valid short city names like Rome, Nice, Oslo."""
        # Note: This tests validation passes for 2+ char alpha strings
        with patch.object(httpx.AsyncClient, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = MagicMock(
                status_code=200,
                json=lambda: [{
                    "lat": "41.9028",
                    "lon": "12.4964",
                    "display_name": "Rome, Italy",
                    "addresstype": "city",
                    "address": {"city": "Rome", "country": "Italy"}
                }],
                raise_for_status=lambda: None
            )
            
            # Short but valid city names should work
            result = await service.geocode("Rome")
            assert result is not None
    
    @pytest.mark.asyncio
    async def test_accepts_valid_city_names(self, service):
        """Should accept valid city names."""
        with patch.object(httpx.AsyncClient, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = MagicMock(
                status_code=200,
                json=lambda: [{
                    "lat": "42.6977",
                    "lon": "23.3219",
                    "display_name": "Sofia, Bulgaria",
                    "addresstype": "city",
                    "address": {"city": "Sofia", "country": "Bulgaria"}
                }],
                raise_for_status=lambda: None
            )
            
            result = await service.geocode("Sofia")
            assert result is not None
            assert result.location_name == "Sofia"
