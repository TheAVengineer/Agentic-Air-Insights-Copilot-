"""
Agent short-term memory with TTL-based caching.

The memory system stores recent API responses to:
- Reduce external API calls
- Improve response time for repeated queries
- Respect API rate limits

Cache key format: "{latitude}_{longitude}_{hours}"
Default TTL: 10 minutes (configurable in policies)
"""

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional, Generic, TypeVar

from policies import SAFETY_RULES

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class CacheEntry(Generic[T]):
    """A single cache entry with value and expiration."""
    
    value: T
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime = field(default_factory=datetime.utcnow)
    hits: int = 0
    
    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        return datetime.utcnow() > self.expires_at
    
    def touch(self) -> None:
        """Record a cache hit."""
        self.hits += 1


class AgentMemory:
    """
    Short-term memory for the agent with TTL-based caching.
    
    Features:
    - Thread-safe in-memory cache
    - Automatic expiration based on TTL
    - Cache key generation from parameters
    - Statistics tracking (hits, misses, expirations)
    
    Usage:
        memory = AgentMemory()
        
        # Check cache first
        cached = memory.get("weather", lat=42.69, lon=23.32, hours=6)
        if cached:
            return cached
        
        # Fetch and store
        data = await fetch_weather(...)
        memory.set("weather", data, lat=42.69, lon=23.32, hours=6)
    """
    
    def __init__(self, ttl_seconds: Optional[int] = None):
        """
        Initialize agent memory.
        
        Args:
            ttl_seconds: Cache TTL in seconds. Defaults to policy value (600s).
        """
        agent_config = SAFETY_RULES.get("agent", {})
        self.ttl_seconds = ttl_seconds or agent_config.get("cache_ttl_seconds", 600)
        
        self._cache: dict[str, CacheEntry] = {}
        self._stats = {
            "hits": 0,
            "misses": 0,
            "expired": 0,
            "evictions": 0,
        }
        
        logger.info(f"AgentMemory initialized with TTL={self.ttl_seconds}s")
    
    def _generate_key(self, prefix: str, **kwargs) -> str:
        """
        Generate a cache key from prefix and keyword arguments.
        
        Args:
            prefix: Cache type prefix (e.g., "weather", "air_quality")
            **kwargs: Parameters to include in key (lat, lon, hours, etc.)
            
        Returns:
            Unique cache key string
        """
        # Sort kwargs for consistent key generation
        sorted_items = sorted(kwargs.items())
        key_parts = [prefix] + [f"{k}={v}" for k, v in sorted_items]
        key_string = "_".join(str(p) for p in key_parts)
        
        # Use hash for very long keys
        if len(key_string) > 100:
            key_hash = hashlib.md5(key_string.encode()).hexdigest()[:16]
            return f"{prefix}_{key_hash}"
        
        return key_string
    
    def get(self, prefix: str, **kwargs) -> Optional[Any]:
        """
        Retrieve a value from cache if it exists and hasn't expired.
        
        Args:
            prefix: Cache type prefix
            **kwargs: Parameters to identify the cached item
            
        Returns:
            Cached value or None if not found/expired
        """
        key = self._generate_key(prefix, **kwargs)
        entry = self._cache.get(key)
        
        if entry is None:
            self._stats["misses"] += 1
            logger.debug(f"Cache MISS: {key}")
            return None
        
        if entry.is_expired:
            self._stats["expired"] += 1
            logger.debug(f"Cache EXPIRED: {key}")
            del self._cache[key]
            return None
        
        entry.touch()
        self._stats["hits"] += 1
        logger.debug(f"Cache HIT: {key} (hits={entry.hits})")
        return entry.value
    
    def set(
        self,
        prefix: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Store a value in cache with TTL.
        
        Args:
            prefix: Cache type prefix
            value: Value to cache
            ttl_seconds: Optional custom TTL (overrides default)
            **kwargs: Parameters to identify the cached item
            
        Returns:
            The generated cache key
        """
        key = self._generate_key(prefix, **kwargs)
        ttl = ttl_seconds or self.ttl_seconds
        now = datetime.utcnow()
        
        self._cache[key] = CacheEntry(
            value=value,
            created_at=now,
            expires_at=now + timedelta(seconds=ttl),
        )
        
        logger.debug(f"Cache SET: {key} (TTL={ttl}s)")
        return key
    
    def invalidate(self, prefix: str, **kwargs) -> bool:
        """
        Manually invalidate a cache entry.
        
        Args:
            prefix: Cache type prefix
            **kwargs: Parameters to identify the cached item
            
        Returns:
            True if entry was removed, False if not found
        """
        key = self._generate_key(prefix, **kwargs)
        if key in self._cache:
            del self._cache[key]
            self._stats["evictions"] += 1
            logger.debug(f"Cache INVALIDATE: {key}")
            return True
        return False
    
    def clear(self) -> int:
        """
        Clear all cache entries.
        
        Returns:
            Number of entries cleared
        """
        count = len(self._cache)
        self._cache.clear()
        logger.info(f"Cache CLEARED: {count} entries")
        return count
    
    def cleanup_expired(self) -> int:
        """
        Remove all expired entries from cache.
        
        Returns:
            Number of entries removed
        """
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.is_expired
        ]
        
        for key in expired_keys:
            del self._cache[key]
        
        if expired_keys:
            self._stats["expired"] += len(expired_keys)
            logger.info(f"Cache CLEANUP: {len(expired_keys)} expired entries")
        
        return len(expired_keys)
    
    @property
    def stats(self) -> dict:
        """Get cache statistics."""
        return {
            **self._stats,
            "size": len(self._cache),
            "ttl_seconds": self.ttl_seconds,
        }
    
    @property
    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)
    
    def get_cache_info(self) -> dict:
        """Get detailed cache information for debugging."""
        now = datetime.utcnow()
        entries = []
        
        for key, entry in self._cache.items():
            remaining = (entry.expires_at - now).total_seconds()
            entries.append({
                "key": key,
                "hits": entry.hits,
                "created_at": entry.created_at.isoformat(),
                "expires_in_seconds": max(0, remaining),
                "is_expired": entry.is_expired,
            })
        
        return {
            "stats": self.stats,
            "entries": entries,
        }


# =============================================================================
# Conversation Context Memory
# =============================================================================

@dataclass
class ConversationContext:
    """Stores context from the conversation for follow-up handling."""
    last_intent: Optional[str] = None  # "analyze", "forecast", "apod"
    last_location: Optional[str] = None  # City/place name
    last_coords: Optional[tuple] = None  # (lat, lon)
    last_hours: int = 6  # Time period (future)
    last_past_days: int = 0  # Historical time period
    awaiting_location: bool = False  # True when we asked for location
    
    def update(
        self,
        intent: str = None,
        location: str = None,
        coords: tuple = None,
        hours: int = None,
        past_days: int = None,
        awaiting_location: bool = False
    ) -> None:
        """Update context with new values."""
        if intent:
            self.last_intent = intent
        if location:
            self.last_location = location
        if coords:
            self.last_coords = coords
        if hours:
            self.last_hours = hours
        if past_days is not None:
            self.last_past_days = past_days
        self.awaiting_location = awaiting_location
    
    def to_dict(self) -> dict:
        """Convert to dictionary for query parser."""
        return {
            "last_intent": self.last_intent,
            "last_location": self.last_location,
            "last_coords": self.last_coords,
            "last_hours": self.last_hours,
            "last_past_days": self.last_past_days,
            "awaiting_location": self.awaiting_location,
        }
    
    def clear(self) -> None:
        """Reset conversation context."""
        self.last_intent = None
        self.last_location = None
        self.last_coords = None
        self.last_hours = 6
        self.last_past_days = 0
        self.awaiting_location = False
