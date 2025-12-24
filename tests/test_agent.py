"""
Tests for agent orchestration and memory.

These tests verify:
- Cache operations (get, set, TTL expiration)
- Agent planning
- Safety level determination
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

from agent.memory import AgentMemory, CacheEntry
from agent.planner import AgentPlanner, Intent, ToolType, ExecutionPlan


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""
    
    def test_cache_entry_not_expired(self):
        """Fresh entry should not be expired."""
        entry = CacheEntry(
            value="test",
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(hours=1),
        )
        
        assert not entry.is_expired
    
    def test_cache_entry_expired(self):
        """Old entry should be expired."""
        entry = CacheEntry(
            value="test",
            created_at=datetime.utcnow() - timedelta(hours=2),
            expires_at=datetime.utcnow() - timedelta(hours=1),
        )
        
        assert entry.is_expired
    
    def test_cache_entry_touch(self):
        """Touch should increment hits."""
        entry = CacheEntry(
            value="test",
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(hours=1),
        )
        
        assert entry.hits == 0
        entry.touch()
        assert entry.hits == 1
        entry.touch()
        assert entry.hits == 2


class TestAgentMemory:
    """Tests for AgentMemory cache system."""
    
    def test_memory_init(self):
        """Memory should initialize with default TTL."""
        memory = AgentMemory()
        
        assert memory.ttl_seconds == 600  # Default from policy
        assert memory.size == 0
    
    def test_memory_init_custom_ttl(self):
        """Memory should accept custom TTL."""
        memory = AgentMemory(ttl_seconds=300)
        
        assert memory.ttl_seconds == 300
    
    def test_memory_set_and_get(self):
        """Should store and retrieve values."""
        memory = AgentMemory()
        
        memory.set("test", {"data": "value"}, key1="a", key2="b")
        result = memory.get("test", key1="a", key2="b")
        
        assert result == {"data": "value"}
    
    def test_memory_get_miss(self):
        """Should return None for missing keys."""
        memory = AgentMemory()
        
        result = memory.get("nonexistent", key="value")
        
        assert result is None
    
    def test_memory_get_expired(self):
        """Should return None for expired entries."""
        import time
        memory = AgentMemory(ttl_seconds=1)  # 1 second TTL
        
        memory.set("test", "value", key="a")
        # Wait for entry to expire
        time.sleep(1.1)
        result = memory.get("test", key="a")
        
        assert result is None
    
    def test_memory_invalidate(self):
        """Should remove specific entry."""
        memory = AgentMemory()
        
        memory.set("test", "value", key="a")
        assert memory.get("test", key="a") == "value"
        
        removed = memory.invalidate("test", key="a")
        assert removed
        assert memory.get("test", key="a") is None
    
    def test_memory_invalidate_nonexistent(self):
        """Should return False for missing entry."""
        memory = AgentMemory()
        
        removed = memory.invalidate("nonexistent", key="a")
        
        assert not removed
    
    def test_memory_clear(self):
        """Should remove all entries."""
        memory = AgentMemory()
        
        memory.set("a", 1, k="1")
        memory.set("b", 2, k="2")
        memory.set("c", 3, k="3")
        
        count = memory.clear()
        
        assert count == 3
        assert memory.size == 0
    
    def test_memory_stats(self):
        """Should track hit/miss statistics."""
        memory = AgentMemory()
        
        memory.set("test", "value", key="a")
        memory.get("test", key="a")  # Hit
        memory.get("test", key="a")  # Hit
        memory.get("nonexistent", key="b")  # Miss
        
        stats = memory.stats
        
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["size"] == 1
    
    def test_memory_key_generation(self):
        """Keys should be consistent for same params."""
        memory = AgentMemory()
        
        # Same params in different order should generate same key
        key1 = memory._generate_key("test", a=1, b=2)
        key2 = memory._generate_key("test", b=2, a=1)
        
        assert key1 == key2
    
    def test_memory_custom_ttl_per_entry(self):
        """Should allow custom TTL per entry."""
        memory = AgentMemory(ttl_seconds=600)
        
        # Set with custom shorter TTL
        memory.set("short", "value", ttl_seconds=1, key="a")
        
        # Entry should exist immediately
        assert memory.get("short", key="a") == "value"


class TestAgentPlanner:
    """Tests for AgentPlanner."""
    
    def test_planner_init(self):
        """Planner should initialize."""
        planner = AgentPlanner()
        assert planner is not None
    
    def test_plan_air_quality_analysis(self):
        """Should create valid air quality plan."""
        planner = AgentPlanner()
        
        plan = planner.plan_air_quality_analysis(
            latitude=42.6977,
            longitude=23.3219,
            hours=6,
        )
        
        assert plan.intent == Intent.ANALYZE_AIR_QUALITY
        assert len(plan.steps) == 5
        
        # Check step order
        assert plan.steps[0].tool == ToolType.CACHE_LOOKUP
        assert plan.steps[1].tool == ToolType.AIR_QUALITY_API
        assert plan.steps[2].tool == ToolType.WEATHER_API
        assert plan.steps[3].tool == ToolType.LLM_REASONING
        assert plan.steps[4].tool == ToolType.CACHE_STORE
    
    def test_plan_apod_request(self):
        """Should create valid APOD plan."""
        planner = AgentPlanner()
        
        plan = planner.plan_apod_request()
        
        assert plan.intent == Intent.GET_APOD
        assert len(plan.steps) == 4
        
        # Check step order
        assert plan.steps[0].tool == ToolType.CACHE_LOOKUP
        assert plan.steps[1].tool == ToolType.NASA_APOD_API
        assert plan.steps[2].tool == ToolType.LLM_REASONING
        assert plan.steps[3].tool == ToolType.CACHE_STORE
    
    def test_plan_step_dependencies(self):
        """Steps should have correct dependencies."""
        planner = AgentPlanner()
        plan = planner.plan_air_quality_analysis(42.0, 23.0, 6)
        
        # Cache lookup has no dependencies
        assert plan.steps[0].depends_on == []
        
        # API calls depend on cache lookup
        assert 1 in plan.steps[1].depends_on  # Air quality depends on cache
        assert 1 in plan.steps[2].depends_on  # Weather depends on cache
        
        # LLM depends on both API calls
        assert 2 in plan.steps[3].depends_on
        assert 3 in plan.steps[3].depends_on


class TestExecutionPlan:
    """Tests for ExecutionPlan."""
    
    def test_plan_is_complete_false(self):
        """Plan with pending steps is not complete."""
        planner = AgentPlanner()
        plan = planner.plan_air_quality_analysis(42.0, 23.0, 6)
        
        assert not plan.is_complete
    
    def test_plan_is_complete_true(self):
        """Plan with all completed steps is complete."""
        planner = AgentPlanner()
        plan = planner.plan_air_quality_analysis(42.0, 23.0, 6)
        
        for step in plan.steps:
            step.status = "completed"
        
        assert plan.is_complete
    
    def test_plan_has_failures(self):
        """Plan should detect failed steps."""
        planner = AgentPlanner()
        plan = planner.plan_air_quality_analysis(42.0, 23.0, 6)
        
        plan.steps[1].status = "failed"
        
        assert plan.has_failures
    
    def test_plan_get_next_steps(self):
        """Should return steps ready to execute."""
        planner = AgentPlanner()
        plan = planner.plan_air_quality_analysis(42.0, 23.0, 6)
        
        # Initially only cache lookup is ready
        ready = plan.get_next_steps()
        assert len(ready) == 1
        assert ready[0].step_id == 1
        
        # Mark cache lookup complete
        plan.steps[0].status = "completed"
        
        # Now both API calls are ready (parallel)
        ready = plan.get_next_steps()
        assert len(ready) == 2
    
    def test_plan_update_step_status(self):
        """Should update step status correctly."""
        planner = AgentPlanner()
        plan = planner.plan_air_quality_analysis(42.0, 23.0, 6)
        
        planner.update_step_status(
            plan, step_id=1, status="completed", result={"cached": False}
        )
        
        assert plan.steps[0].status == "completed"
        assert plan.steps[0].result == {"cached": False}


class TestWeatherDataModels:
    """Tests for weather data models."""
    
    def test_air_quality_data_averages(self):
        """Should calculate correct averages."""
        from tools.weather_client import AirQualityData
        
        data = AirQualityData(
            pm25=[10.0, 20.0, 30.0],
            pm10=[50.0, 60.0, 70.0],
            timestamps=["t1", "t2", "t3"],
        )
        
        assert data.pm25_avg == 20.0
        assert data.pm10_avg == 60.0
    
    def test_air_quality_data_with_nulls(self):
        """Should handle null values in averages."""
        from tools.weather_client import AirQualityData
        
        data = AirQualityData(
            pm25=[10.0, None, 30.0],
            pm10=[50.0, None, None],
            timestamps=["t1", "t2", "t3"],
        )
        
        assert data.pm25_avg == 20.0  # (10 + 30) / 2
        assert data.pm10_avg == 50.0  # Only one valid value
    
    def test_air_quality_data_quality(self):
        """Should calculate data quality correctly."""
        from tools.weather_client import AirQualityData
        
        data = AirQualityData(
            pm25=[10.0, None, 30.0],  # 2/3 valid
            pm10=[50.0, 60.0, 70.0],  # 3/3 valid
            timestamps=["t1", "t2", "t3"],
        )
        
        # 5/6 valid = 0.833...
        assert 0.8 < data.data_quality < 0.9
    
    def test_weather_data_averages(self):
        """Should calculate temperature averages."""
        from tools.weather_client import WeatherData
        
        data = WeatherData(
            temperature=[15.0, 20.0, 25.0],
            timestamps=["t1", "t2", "t3"],
        )
        
        assert data.temp_avg == 20.0
        assert data.temp_min == 15.0
        assert data.temp_max == 25.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
