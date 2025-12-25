"""
Tests for agent orchestrator.

These tests verify:
- Full agentic flow (Plan→Fetch→Validate→Cache→Reason→Respond)
- Safety level determination
- Fallback guidance generation
- APOD retrieval and caching
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from agent.orchestrator import AirInsightsAgent
from agent.memory import AgentMemory
from api.models import SafetyLevel, DataQuality
from tools.weather_client import AirQualityData, WeatherData
from tools.nasa_client import APODData


class TestAirInsightsAgentInit:
    """Tests for agent initialization."""
    
    def test_init_default(self):
        """Should initialize with default components."""
        agent = AirInsightsAgent()
        
        assert agent.llm is not None
        assert agent.weather is not None
        assert agent.nasa is not None
        assert agent.memory is not None
        assert agent.planner is not None
    
    def test_init_custom_components(self):
        """Should accept custom components."""
        mock_llm = MagicMock()
        mock_weather = MagicMock()
        mock_nasa = MagicMock()
        mock_memory = MagicMock()
        
        agent = AirInsightsAgent(
            llm_client=mock_llm,
            weather_client=mock_weather,
            nasa_client=mock_nasa,
            memory=mock_memory
        )
        
        assert agent.llm == mock_llm
        assert agent.weather == mock_weather
        assert agent.nasa == mock_nasa
        assert agent.memory == mock_memory
    
    def test_init_loads_thresholds(self):
        """Should load thresholds from policy."""
        agent = AirInsightsAgent()
        
        assert agent.thresholds is not None
        assert "pm25" in agent.thresholds
        assert "pm10" in agent.thresholds


class TestSafetyLevelDetermination:
    """Tests for safety level calculation."""
    
    @pytest.fixture
    def agent(self):
        """Create agent instance."""
        return AirInsightsAgent()
    
    def test_safe_conditions(self, agent):
        """Should return SAFE for good conditions."""
        level = agent._determine_safety_level(
            pm25_avg=15.0,  # Good
            pm10_avg=30.0,  # Good
            temp_avg=20.0   # Optimal
        )
        
        assert level == SafetyLevel.SAFE
    
    def test_moderate_pm25(self, agent):
        """Should return MODERATE for moderate PM2.5."""
        level = agent._determine_safety_level(
            pm25_avg=55.0,  # Moderate (above 50)
            pm10_avg=30.0,
            temp_avg=20.0
        )
        
        assert level == SafetyLevel.MODERATE
    
    def test_unhealthy_sensitive_pm25(self, agent):
        """Should return UNHEALTHY_SENSITIVE for high PM2.5."""
        level = agent._determine_safety_level(
            pm25_avg=80.0,  # Unhealthy for sensitive
            pm10_avg=30.0,
            temp_avg=20.0
        )
        
        assert level == SafetyLevel.UNHEALTHY_SENSITIVE
    
    def test_unhealthy_pm25(self, agent):
        """Should return UNHEALTHY for very high PM2.5."""
        level = agent._determine_safety_level(
            pm25_avg=110.0,  # Unhealthy
            pm10_avg=30.0,
            temp_avg=20.0
        )
        
        assert level == SafetyLevel.UNHEALTHY
    
    def test_very_unhealthy_pm25(self, agent):
        """Should return VERY_UNHEALTHY for dangerous PM2.5."""
        level = agent._determine_safety_level(
            pm25_avg=160.0,  # Very unhealthy
            pm10_avg=30.0,
            temp_avg=20.0
        )
        
        assert level == SafetyLevel.VERY_UNHEALTHY
    
    def test_hazardous_pm25(self, agent):
        """Should return HAZARDOUS for extreme PM2.5."""
        level = agent._determine_safety_level(
            pm25_avg=250.0,  # Hazardous
            pm10_avg=30.0,
            temp_avg=20.0
        )
        
        assert level == SafetyLevel.HAZARDOUS
    
    def test_unhealthy_cold_temperature(self, agent):
        """Should return UNHEALTHY for freezing temps."""
        level = agent._determine_safety_level(
            pm25_avg=15.0,
            pm10_avg=30.0,
            temp_avg=-5.0  # Below 0°C
        )
        
        assert level == SafetyLevel.UNHEALTHY
    
    def test_unhealthy_hot_temperature(self, agent):
        """Should return UNHEALTHY for extreme heat."""
        level = agent._determine_safety_level(
            pm25_avg=15.0,
            pm10_avg=30.0,
            temp_avg=38.0  # Above 35°C
        )
        
        assert level == SafetyLevel.UNHEALTHY
    
    def test_worst_condition_wins(self, agent):
        """Should use worst condition for safety level."""
        # PM2.5 good but PM10 very high - PM10 is checked after PM2.5
        # Note: The implementation checks PM2.5 first, then PM10, then temp
        # So we need PM2.5 to be safe and PM10 to be hazardous
        level = agent._determine_safety_level(
            pm25_avg=20.0,   # Safe (below 25)
            pm10_avg=450.0,  # Hazardous (above 400)
            temp_avg=20.0
        )
        
        assert level == SafetyLevel.HAZARDOUS


class TestAgentAnalyze:
    """Tests for analyze method."""
    
    @pytest.fixture
    def mock_weather(self):
        """Create mock weather client."""
        mock = MagicMock()
        mock.get_combined_data = AsyncMock(return_value=(
            AirQualityData(
                pm25=[15.0, 18.0, 20.0],
                pm10=[25.0, 28.0, 30.0],
                timestamps=["2024-12-24T00:00", "2024-12-24T01:00", "2024-12-24T02:00"]
            ),
            WeatherData(
                temperature=[18.0, 20.0, 22.0],
                timestamps=["2024-12-24T00:00", "2024-12-24T01:00", "2024-12-24T02:00"]
            )
        ))
        return mock
    
    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM client."""
        mock = MagicMock()
        mock.generate_guidance = AsyncMock(return_value="✅ Great conditions for outdoor exercise!")
        return mock
    
    @pytest.fixture
    def mock_memory(self):
        """Create mock memory."""
        mock = MagicMock()
        mock.get = MagicMock(return_value=None)  # Cache miss
        mock.set = MagicMock()
        return mock
    
    @pytest.fixture
    def agent(self, mock_weather, mock_llm, mock_memory):
        """Create agent with mocks."""
        return AirInsightsAgent(
            weather_client=mock_weather,
            llm_client=mock_llm,
            memory=mock_memory
        )
    
    @pytest.mark.asyncio
    async def test_analyze_success(self, agent, mock_weather, mock_llm):
        """Should analyze air quality successfully."""
        result = await agent.analyze(42.6977, 23.3219, hours=6)
        
        assert result.pm25_avg == pytest.approx(17.67, rel=0.1)
        assert result.pm10_avg == pytest.approx(27.67, rel=0.1)
        assert result.temp_avg == pytest.approx(20.0, rel=0.1)
        assert result.guidance_text == "✅ Great conditions for outdoor exercise!"
        assert result.safety_level == SafetyLevel.SAFE
        assert result.cached is False
    
    @pytest.mark.asyncio
    async def test_analyze_cache_hit(self, mock_weather, mock_llm):
        """Should return cached result on cache hit."""
        mock_memory = MagicMock()
        mock_memory.get = MagicMock(return_value={
            "pm25_avg": 15.0,
            "pm10_avg": 25.0,
            "temp_avg": 20.0,
            "guidance_text": "Cached guidance",
            "safety_level": SafetyLevel.SAFE,
            "data_quality": DataQuality.HIGH,
            "forecast_hours": 6,
            "attribution": "Weather data by Open-Meteo.com",
            "cached": False,
            "timestamp": datetime.utcnow()
        })
        
        agent = AirInsightsAgent(
            weather_client=mock_weather,
            llm_client=mock_llm,
            memory=mock_memory
        )
        
        result = await agent.analyze(42.6977, 23.3219, hours=6)
        
        assert result.cached is True
        assert result.guidance_text == "Cached guidance"
        # Weather API should not be called
        mock_weather.get_combined_data.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_analyze_invalid_latitude(self, agent):
        """Should raise error for invalid latitude."""
        with pytest.raises(ValueError, match="Latitude"):
            await agent.analyze(100.0, 23.3219, hours=6)
    
    @pytest.mark.asyncio
    async def test_analyze_invalid_longitude(self, agent):
        """Should raise error for invalid longitude."""
        with pytest.raises(ValueError, match="Longitude"):
            await agent.analyze(42.6977, 200.0, hours=6)
    
    @pytest.mark.asyncio
    async def test_analyze_llm_fallback(self, mock_weather, mock_memory):
        """Should use fallback guidance when LLM fails."""
        mock_llm = MagicMock()
        mock_llm.generate_guidance = AsyncMock(side_effect=Exception("LLM Error"))
        
        agent = AirInsightsAgent(
            weather_client=mock_weather,
            llm_client=mock_llm,
            memory=mock_memory
        )
        
        result = await agent.analyze(42.6977, 23.3219, hours=6)
        
        # Should have fallback guidance
        assert result.guidance_text is not None
        assert "PM2.5" in result.guidance_text or "µg/m³" in result.guidance_text
    
    @pytest.mark.asyncio
    async def test_analyze_stores_in_cache(self, agent, mock_memory):
        """Should store result in cache."""
        await agent.analyze(42.6977, 23.3219, hours=6)
        
        mock_memory.set.assert_called_once()


class TestAgentFallbackGuidance:
    """Tests for fallback guidance generation."""
    
    @pytest.fixture
    def agent(self):
        """Create agent instance."""
        return AirInsightsAgent()
    
    def test_fallback_safe(self, agent):
        """Should generate safe guidance."""
        guidance = agent._generate_fallback_guidance(
            pm25_avg=15.0,
            pm10_avg=25.0,
            temp_avg=20.0,
            safety_level=SafetyLevel.SAFE
        )
        
        assert "15.0" in guidance or "15" in guidance
        assert "µg/m³" in guidance
    
    def test_fallback_hazardous(self, agent):
        """Should generate hazardous guidance."""
        guidance = agent._generate_fallback_guidance(
            pm25_avg=250.0,
            pm10_avg=450.0,
            temp_avg=20.0,
            safety_level=SafetyLevel.HAZARDOUS
        )
        
        assert guidance is not None


class TestAgentAPOD:
    """Tests for APOD retrieval."""
    
    @pytest.fixture
    def mock_nasa(self):
        """Create mock NASA client."""
        mock = MagicMock()
        mock.get_today = AsyncMock(return_value=APODData(
            title="Test Image",
            url="https://apod.nasa.gov/test.jpg",
            explanation="Test explanation about space.",
            date="2024-12-24",
            media_type="image",
            hdurl="https://apod.nasa.gov/hd.jpg"
        ))
        return mock
    
    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM client."""
        mock = MagicMock()
        mock.generate_guidance = AsyncMock(return_value="A beautiful image of space showing stars.")
        return mock
    
    @pytest.fixture
    def mock_memory(self):
        """Create mock memory."""
        mock = MagicMock()
        mock.get = MagicMock(return_value=None)
        mock.set = MagicMock()
        return mock
    
    @pytest.fixture
    def agent(self, mock_nasa, mock_llm, mock_memory):
        """Create agent with mocks."""
        return AirInsightsAgent(
            nasa_client=mock_nasa,
            llm_client=mock_llm,
            memory=mock_memory
        )
    
    @pytest.mark.asyncio
    async def test_get_apod_success(self, agent, mock_nasa):
        """Should fetch APOD successfully."""
        result = await agent.get_apod()
        
        assert result.title == "Test Image"
        assert result.url == "https://apod.nasa.gov/test.jpg"
        assert result.summary == "A beautiful image of space showing stars."
    
    @pytest.mark.asyncio
    async def test_get_apod_cache_hit(self, mock_nasa, mock_llm):
        """Should return cached APOD."""
        mock_memory = MagicMock()
        mock_memory.get = MagicMock(return_value={
            "title": "Cached Image",
            "url": "https://cached.jpg",
            "explanation": "Cached explanation",
            "summary": "Cached summary",
            "date": "2024-12-24",
            "media_type": "image",
            "hdurl": None,
            "attribution": "NASA"
        })
        
        agent = AirInsightsAgent(
            nasa_client=mock_nasa,
            llm_client=mock_llm,
            memory=mock_memory
        )
        
        result = await agent.get_apod()
        
        assert result.title == "Cached Image"
        mock_nasa.get_today.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_get_apod_llm_fallback(self, mock_nasa, mock_memory):
        """Should use explanation excerpt when LLM fails."""
        mock_llm = MagicMock()
        mock_llm.generate_guidance = AsyncMock(side_effect=Exception("LLM Error"))
        
        # Update NASA mock to return longer explanation
        mock_nasa.get_today = AsyncMock(return_value=APODData(
            title="Test",
            url="https://test.jpg",
            explanation="First sentence about space. Second sentence about stars. Third sentence.",
            date="2024-12-24"
        ))
        
        agent = AirInsightsAgent(
            nasa_client=mock_nasa,
            llm_client=mock_llm,
            memory=mock_memory
        )
        
        result = await agent.get_apod()
        
        # Should use first two sentences as fallback
        assert "First sentence" in result.summary
        assert "Second sentence" in result.summary


class TestAgentCacheStats:
    """Tests for cache statistics."""
    
    def test_get_cache_stats(self):
        """Should return cache stats from memory."""
        mock_memory = MagicMock()
        mock_memory.stats = {"hits": 10, "misses": 5, "size": 3}
        
        agent = AirInsightsAgent(memory=mock_memory)
        
        stats = agent.get_cache_stats()
        
        assert stats["hits"] == 10
        assert stats["misses"] == 5
        assert stats["size"] == 3


class TestPM10OnlyThresholds:
    """Tests for PM10 threshold branches when PM2.5 is safe."""
    
    @pytest.fixture
    def agent(self):
        """Create agent instance."""
        return AirInsightsAgent()
    
    def test_pm10_very_unhealthy(self, agent):
        """Should return VERY_UNHEALTHY for high PM10 when PM2.5 is safe."""
        level = agent._determine_safety_level(
            pm25_avg=20.0,   # Safe (below 50)
            pm10_avg=350.0,  # Very unhealthy (300-400)
            temp_avg=20.0
        )
        assert level == SafetyLevel.VERY_UNHEALTHY
    
    def test_pm10_unhealthy(self, agent):
        """Should return UNHEALTHY for elevated PM10 when PM2.5 is safe."""
        level = agent._determine_safety_level(
            pm25_avg=20.0,   # Safe (below 50)
            pm10_avg=250.0,  # Unhealthy (200-300)
            temp_avg=20.0
        )
        assert level == SafetyLevel.UNHEALTHY
    
    def test_pm10_unhealthy_sensitive(self, agent):
        """Should return UNHEALTHY_SENSITIVE for moderate PM10 when PM2.5 is safe."""
        level = agent._determine_safety_level(
            pm25_avg=20.0,   # Safe (below 50)
            pm10_avg=175.0,  # Unhealthy sensitive (150-200)
            temp_avg=20.0
        )
        assert level == SafetyLevel.UNHEALTHY_SENSITIVE
    
    def test_pm10_moderate(self, agent):
        """Should return MODERATE for slightly elevated PM10 when PM2.5 is safe."""
        level = agent._determine_safety_level(
            pm25_avg=20.0,   # Safe (below 50)
            pm10_avg=125.0,  # Moderate (100-150)
            temp_avg=20.0
        )
        assert level == SafetyLevel.MODERATE


class TestAPIFailures:
    """Tests for API failure handling."""
    
    @pytest.fixture
    def mock_memory(self):
        """Create mock memory with cache miss."""
        mock = MagicMock()
        mock.get = MagicMock(return_value=None)
        return mock
    
    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM client."""
        mock = MagicMock()
        mock.generate_guidance = AsyncMock(return_value="Test guidance")
        return mock
    
    @pytest.mark.asyncio
    async def test_analyze_weather_api_failure(self, mock_memory, mock_llm):
        """Should propagate exception when weather API fails."""
        mock_weather = MagicMock()
        mock_weather.get_combined_data = AsyncMock(
            side_effect=Exception("Weather API unavailable")
        )
        
        agent = AirInsightsAgent(
            weather_client=mock_weather,
            llm_client=mock_llm,
            memory=mock_memory
        )
        
        with pytest.raises(Exception, match="Weather API unavailable"):
            await agent.analyze(42.6977, 23.3219, hours=6)
    
    @pytest.mark.asyncio
    async def test_get_apod_nasa_api_failure(self, mock_memory, mock_llm):
        """Should propagate exception when NASA API fails."""
        mock_nasa = MagicMock()
        mock_nasa.get_today = AsyncMock(
            side_effect=Exception("NASA API unavailable")
        )
        
        agent = AirInsightsAgent(
            nasa_client=mock_nasa,
            llm_client=mock_llm,
            memory=mock_memory
        )
        
        with pytest.raises(Exception, match="NASA API unavailable"):
            await agent.get_apod()
