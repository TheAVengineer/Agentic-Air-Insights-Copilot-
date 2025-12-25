"""
Extended tests for agent/memory.py to increase coverage.
Tests ConversationContext and additional cache functionality.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch

from agent.memory import (
    AgentMemory,
    CacheEntry,
    ConversationContext,
)


class TestCacheEntry:
    """Test CacheEntry dataclass."""
    
    def test_cache_entry_creation(self):
        """Should create cache entry with correct fields."""
        now = datetime.utcnow()
        expires = now + timedelta(seconds=300)
        
        entry = CacheEntry(
            value={"test": "data"},
            created_at=now,
            expires_at=expires
        )
        
        assert entry.value == {"test": "data"}
        assert entry.created_at == now
        assert entry.expires_at == expires
        assert entry.hits == 0
    
    def test_cache_entry_is_expired(self):
        """Should detect expired entries."""
        now = datetime.utcnow()
        past = now - timedelta(seconds=10)
        
        entry = CacheEntry(
            value="test",
            created_at=past,
            expires_at=past + timedelta(seconds=5)  # Expired 5 seconds ago
        )
        
        assert entry.is_expired is True
    
    def test_cache_entry_not_expired(self):
        """Should detect non-expired entries."""
        now = datetime.utcnow()
        future = now + timedelta(seconds=300)
        
        entry = CacheEntry(
            value="test",
            created_at=now,
            expires_at=future
        )
        
        assert entry.is_expired is False
    
    def test_cache_entry_touch(self):
        """Should increment hits on touch."""
        now = datetime.utcnow()
        entry = CacheEntry(
            value="test",
            created_at=now,
            expires_at=now + timedelta(seconds=300)
        )
        
        assert entry.hits == 0
        entry.touch()
        assert entry.hits == 1
        entry.touch()
        assert entry.hits == 2


class TestAgentMemoryExtended:
    """Extended tests for AgentMemory."""
    
    def test_cache_key_generation_long_key(self):
        """Should hash very long keys."""
        cache = AgentMemory(ttl_seconds=300)
        
        # Create a key that would be > 100 chars
        key = cache._generate_key(
            "test",
            very_long_param="x" * 100,
            another_long_param="y" * 100
        )
        
        # Should be hashed to shorter form
        assert len(key) < 100
        assert key.startswith("test_")
    
    def test_cache_get_expired_entry(self):
        """Should return None and clean up expired entries."""
        cache = AgentMemory(ttl_seconds=1)
        
        # Set a value
        cache.set("test", value="data", lat=42.0)
        
        # Manually expire it
        key = cache._generate_key("test", lat=42.0)
        cache._cache[key].expires_at = datetime.utcnow() - timedelta(seconds=10)
        
        # Get should return None and clean up
        result = cache.get("test", lat=42.0)
        assert result is None
        assert key not in cache._cache
        assert cache._stats["expired"] >= 1
    
    def test_cache_invalidate_existing(self):
        """Should invalidate existing entry."""
        cache = AgentMemory(ttl_seconds=300)
        
        cache.set("test", value="data", key=1)
        assert cache.get("test", key=1) == "data"
        
        result = cache.invalidate("test", key=1)
        assert result is True
        assert cache.get("test", key=1) is None
        assert cache._stats["evictions"] >= 1
    
    def test_cache_invalidate_nonexistent(self):
        """Should return False for non-existent entry."""
        cache = AgentMemory(ttl_seconds=300)
        
        result = cache.invalidate("test", key=999)
        assert result is False
    
    def test_cache_clear(self):
        """Should clear all entries."""
        cache = AgentMemory(ttl_seconds=300)
        
        cache.set("test", value="data1", key=1)
        cache.set("test", value="data2", key=2)
        cache.set("test", value="data3", key=3)
        
        assert cache.size == 3
        
        cleared = cache.clear()
        assert cleared == 3
        assert cache.size == 0
    
    def test_cache_cleanup_expired(self):
        """Should clean up all expired entries."""
        cache = AgentMemory(ttl_seconds=300)
        
        # Add entries
        cache.set("test", value="data1", key=1)
        cache.set("test", value="data2", key=2)
        cache.set("test", value="data3", key=3)
        
        # Expire some entries
        key1 = cache._generate_key("test", key=1)
        key2 = cache._generate_key("test", key=2)
        cache._cache[key1].expires_at = datetime.utcnow() - timedelta(seconds=10)
        cache._cache[key2].expires_at = datetime.utcnow() - timedelta(seconds=10)
        
        # Cleanup
        cleaned = cache.cleanup_expired()
        assert cleaned == 2
        assert cache.size == 1
    
    def test_cache_cleanup_no_expired(self):
        """Should return 0 when no expired entries."""
        cache = AgentMemory(ttl_seconds=300)
        
        cache.set("test", value="data", key=1)
        
        cleaned = cache.cleanup_expired()
        assert cleaned == 0
    
    def test_cache_stats_property(self):
        """Should return cache statistics."""
        cache = AgentMemory(ttl_seconds=300)
        
        # Some operations
        cache.set("test", value="data", key=1)
        cache.get("test", key=1)  # hit
        cache.get("test", key=2)  # miss
        
        stats = cache.stats
        assert "hits" in stats
        assert "misses" in stats
        assert "size" in stats
        assert "ttl_seconds" in stats
        assert stats["ttl_seconds"] == 300
    
    def test_cache_get_cache_info(self):
        """Should return detailed cache information."""
        cache = AgentMemory(ttl_seconds=300)
        
        cache.set("test", value="data1", key=1)
        cache.set("test", value="data2", key=2)
        
        # Access one entry
        cache.get("test", key=1)
        
        info = cache.get_cache_info()
        
        assert "stats" in info
        assert "entries" in info
        assert len(info["entries"]) == 2
        
        # Check entry structure
        entry = info["entries"][0]
        assert "key" in entry
        assert "hits" in entry
        assert "created_at" in entry
        assert "expires_in_seconds" in entry
        assert "is_expired" in entry
    
    def test_cache_custom_ttl(self):
        """Should support custom TTL per entry."""
        cache = AgentMemory(ttl_seconds=300)
        
        # Set with custom TTL
        cache.set("test", value="data", ttl_seconds=10, key=1)
        
        # Should still be valid
        assert cache.get("test", key=1) == "data"


class TestConversationContext:
    """Test ConversationContext class."""
    
    def test_context_default_values(self):
        """Should have correct default values."""
        ctx = ConversationContext()
        
        assert ctx.last_intent is None
        assert ctx.last_location is None
        assert ctx.last_coords is None
        assert ctx.last_hours == 6
        assert ctx.last_past_days == 0
        assert ctx.awaiting_location is False
    
    def test_context_update_intent(self):
        """Should update intent."""
        ctx = ConversationContext()
        ctx.update(intent="analyze")
        
        assert ctx.last_intent == "analyze"
    
    def test_context_update_location(self):
        """Should update location."""
        ctx = ConversationContext()
        ctx.update(location="Sofia")
        
        assert ctx.last_location == "Sofia"
    
    def test_context_update_coords(self):
        """Should update coordinates."""
        ctx = ConversationContext()
        ctx.update(coords=(42.69, 23.32))
        
        assert ctx.last_coords == (42.69, 23.32)
    
    def test_context_update_hours(self):
        """Should update hours."""
        ctx = ConversationContext()
        ctx.update(hours=24)
        
        assert ctx.last_hours == 24
    
    def test_context_update_past_days(self):
        """Should update past_days."""
        ctx = ConversationContext()
        ctx.update(past_days=7)
        
        assert ctx.last_past_days == 7
    
    def test_context_update_past_days_zero(self):
        """Should update past_days to zero."""
        ctx = ConversationContext()
        ctx.update(past_days=7)
        ctx.update(past_days=0)
        
        assert ctx.last_past_days == 0
    
    def test_context_update_awaiting_location(self):
        """Should update awaiting_location."""
        ctx = ConversationContext()
        ctx.update(awaiting_location=True)
        
        assert ctx.awaiting_location is True
    
    def test_context_update_multiple(self):
        """Should update multiple fields at once."""
        ctx = ConversationContext()
        ctx.update(
            intent="forecast",
            location="Paris",
            coords=(48.85, 2.35),
            hours=168,
            past_days=0,
            awaiting_location=False
        )
        
        assert ctx.last_intent == "forecast"
        assert ctx.last_location == "Paris"
        assert ctx.last_coords == (48.85, 2.35)
        assert ctx.last_hours == 168
        assert ctx.last_past_days == 0
        assert ctx.awaiting_location is False
    
    def test_context_to_dict(self):
        """Should convert to dictionary."""
        ctx = ConversationContext()
        ctx.update(
            intent="analyze",
            location="Sofia",
            coords=(42.69, 23.32),
            hours=12,
            past_days=1
        )
        
        d = ctx.to_dict()
        
        assert d["last_intent"] == "analyze"
        assert d["last_location"] == "Sofia"
        assert d["last_coords"] == (42.69, 23.32)
        assert d["last_hours"] == 12
        assert d["last_past_days"] == 1
        assert d["awaiting_location"] is False
    
    def test_context_clear(self):
        """Should reset all values."""
        ctx = ConversationContext()
        ctx.update(
            intent="analyze",
            location="Sofia",
            coords=(42.69, 23.32),
            hours=12,
            past_days=1,
            awaiting_location=True
        )
        
        ctx.clear()
        
        assert ctx.last_intent is None
        assert ctx.last_location is None
        assert ctx.last_coords is None
        assert ctx.last_hours == 6
        assert ctx.last_past_days == 0
        assert ctx.awaiting_location is False
    
    def test_context_partial_update_preserves_values(self):
        """Should preserve values not being updated."""
        ctx = ConversationContext()
        ctx.update(
            intent="analyze",
            location="Sofia",
            coords=(42.69, 23.32)
        )
        
        # Update only hours
        ctx.update(hours=24)
        
        # Other values should be preserved
        assert ctx.last_intent == "analyze"
        assert ctx.last_location == "Sofia"
        assert ctx.last_coords == (42.69, 23.32)
        assert ctx.last_hours == 24
