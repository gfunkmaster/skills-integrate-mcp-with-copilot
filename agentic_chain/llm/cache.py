"""
Response Cache - Cache LLM responses to reduce API costs.

This module provides caching functionality for LLM responses,
enabling cost optimization by avoiding duplicate API calls.
"""

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .base import LLMMessage, LLMResponse, LLMUsage


logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A cached LLM response entry."""
    
    response: LLMResponse
    created_at: float
    hits: int = 0
    
    def is_expired(self, ttl: int) -> bool:
        """Check if the cache entry has expired."""
        return time.time() - self.created_at > ttl


class ResponseCache:
    """
    In-memory LRU cache for LLM responses.
    
    Features:
    - LRU eviction when max size is reached
    - TTL-based expiration
    - Thread-safe operations
    - Cache statistics
    
    Example:
        >>> cache = ResponseCache(max_size=1000, ttl=3600)
        >>> cache.set(messages, params, response)
        >>> cached = cache.get(messages, params)
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        ttl: int = 3600,
        enabled: bool = True,
    ):
        """
        Initialize the response cache.
        
        Args:
            max_size: Maximum number of cached responses.
            ttl: Time-to-live in seconds for cached entries.
            enabled: Whether caching is enabled.
        """
        self.max_size = max_size
        self.ttl = ttl
        self.enabled = enabled
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
    
    def _make_key(
        self,
        messages: list[LLMMessage],
        params: dict,
    ) -> str:
        """
        Generate a cache key from messages and parameters.
        
        Args:
            messages: List of LLM messages.
            params: Request parameters (model, temperature, etc.).
            
        Returns:
            Hash string as cache key.
        """
        # Create a deterministic representation
        key_data = {
            "messages": [
                {"role": m.role.value, "content": m.content}
                for m in messages
            ],
            "params": {
                k: v for k, v in sorted(params.items())
                if k not in ["api_key", "timeout"]  # Exclude sensitive/variable params
            },
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def get(
        self,
        messages: list[LLMMessage],
        params: dict,
    ) -> Optional[LLMResponse]:
        """
        Get a cached response.
        
        Args:
            messages: List of LLM messages.
            params: Request parameters.
            
        Returns:
            Cached LLMResponse if found and valid, None otherwise.
        """
        if not self.enabled:
            return None
        
        key = self._make_key(messages, params)
        
        with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                self._misses += 1
                return None
            
            # Check expiration
            if entry.is_expired(self.ttl):
                del self._cache[key]
                self._misses += 1
                return None
            
            # Update hit count and move to end (LRU)
            entry.hits += 1
            self._cache.move_to_end(key)
            self._hits += 1
            
            logger.debug(f"Cache hit for key {key[:16]}...")
            return entry.response
    
    def set(
        self,
        messages: list[LLMMessage],
        params: dict,
        response: LLMResponse,
    ) -> None:
        """
        Cache a response.
        
        Args:
            messages: List of LLM messages.
            params: Request parameters.
            response: LLM response to cache.
        """
        if not self.enabled:
            return
        
        key = self._make_key(messages, params)
        
        with self._lock:
            # Remove oldest if at capacity
            while len(self._cache) >= self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                self._evictions += 1
            
            # Add new entry
            self._cache[key] = CacheEntry(
                response=response,
                created_at=time.time(),
            )
            
            logger.debug(f"Cached response for key {key[:16]}...")
    
    def invalidate(
        self,
        messages: Optional[list[LLMMessage]] = None,
        params: Optional[dict] = None,
    ) -> int:
        """
        Invalidate cached entries.
        
        Args:
            messages: Optional messages to match for invalidation.
            params: Optional params to match for invalidation.
            
        Returns:
            Number of entries invalidated.
        """
        if messages and params:
            # Invalidate specific entry
            key = self._make_key(messages, params)
            with self._lock:
                if key in self._cache:
                    del self._cache[key]
                    return 1
            return 0
        
        # Clear all entries
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count
    
    def prune_expired(self) -> int:
        """
        Remove all expired entries.
        
        Returns:
            Number of entries removed.
        """
        removed = 0
        current_time = time.time()
        
        with self._lock:
            # Find expired keys
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired(self.ttl)
            ]
            
            # Remove expired entries
            for key in expired_keys:
                del self._cache[key]
                removed += 1
        
        if removed > 0:
            logger.debug(f"Pruned {removed} expired cache entries")
        
        return removed
    
    def get_stats(self) -> dict:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics.
        """
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0
            
            return {
                "enabled": self.enabled,
                "size": len(self._cache),
                "max_size": self.max_size,
                "ttl": self.ttl,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(hit_rate, 2),
                "evictions": self._evictions,
            }
    
    def reset_stats(self) -> None:
        """Reset cache statistics."""
        with self._lock:
            self._hits = 0
            self._misses = 0
            self._evictions = 0


class DiskCache(ResponseCache):
    """
    Disk-based cache for LLM responses.
    
    Persists cached responses to disk for persistence across
    program restarts.
    
    Features:
    - Persistent storage
    - Automatic loading on init
    - Periodic saving
    
    Example:
        >>> cache = DiskCache(cache_dir="/tmp/llm_cache")
        >>> cache.set(messages, params, response)
    """
    
    def __init__(
        self,
        cache_dir: str,
        max_size: int = 1000,
        ttl: int = 3600,
        enabled: bool = True,
    ):
        """
        Initialize the disk cache.
        
        Args:
            cache_dir: Directory to store cache files.
            max_size: Maximum number of cached responses.
            ttl: Time-to-live in seconds.
            enabled: Whether caching is enabled.
        """
        super().__init__(max_size=max_size, ttl=ttl, enabled=enabled)
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._index_file = self.cache_dir / "cache_index.json"
        
        # Load existing cache
        self._load_index()
    
    def _load_index(self) -> None:
        """Load cache index from disk."""
        if not self._index_file.exists():
            logger.debug(f"No cache index file found at {self._index_file}")
            return
        
        try:
            with open(self._index_file, 'r') as f:
                index = json.load(f)
            
            current_time = time.time()
            for key, entry_data in index.items():
                # Skip expired entries
                if current_time - entry_data["created_at"] > self.ttl:
                    continue
                
                # Load response from file
                response_file = self.cache_dir / f"{key}.json"
                if response_file.exists():
                    with open(response_file, 'r') as f:
                        response_data = json.load(f)
                    
                    response = LLMResponse(
                        content=response_data["content"],
                        model=response_data["model"],
                        usage=LLMUsage(**response_data["usage"]),
                        finish_reason=response_data.get("finish_reason"),
                    )
                    
                    self._cache[key] = CacheEntry(
                        response=response,
                        created_at=entry_data["created_at"],
                        hits=entry_data.get("hits", 0),
                    )
            
            logger.info(f"Loaded {len(self._cache)} cached responses from disk")
        except Exception as e:
            logger.warning(f"Error loading cache index: {e}")
    
    def _save_index(self) -> None:
        """Save cache index to disk."""
        try:
            index = {}
            for key, entry in self._cache.items():
                index[key] = {
                    "created_at": entry.created_at,
                    "hits": entry.hits,
                }
            
            with open(self._index_file, 'w') as f:
                json.dump(index, f)
        except Exception as e:
            logger.warning(f"Error saving cache index: {e}")
    
    def _save_response(self, key: str, response: LLMResponse) -> None:
        """Save a response to disk."""
        try:
            response_file = self.cache_dir / f"{key}.json"
            response_data = {
                "content": response.content,
                "model": response.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                    "estimated_cost": response.usage.estimated_cost,
                },
                "finish_reason": response.finish_reason,
            }
            
            with open(response_file, 'w') as f:
                json.dump(response_data, f)
        except Exception as e:
            logger.warning(f"Error saving cached response: {e}")
    
    def set(
        self,
        messages: list[LLMMessage],
        params: dict,
        response: LLMResponse,
    ) -> None:
        """Cache a response and persist to disk."""
        if not self.enabled:
            return
        
        key = self._make_key(messages, params)
        
        with self._lock:
            # Call parent method to add to memory cache
            super().set(messages, params, response)
            
            # Save to disk
            self._save_response(key, response)
            self._save_index()
    
    def invalidate(
        self,
        messages: Optional[list[LLMMessage]] = None,
        params: Optional[dict] = None,
    ) -> int:
        """Invalidate cached entries and update disk."""
        count = super().invalidate(messages, params)
        
        if count > 0:
            with self._lock:
                self._save_index()
        
        return count
    
    def clear_disk(self) -> None:
        """Clear all disk cache files."""
        try:
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
            logger.info("Cleared disk cache")
        except Exception as e:
            logger.warning(f"Error clearing disk cache: {e}")


# Global cache instance
_global_cache: Optional[ResponseCache] = None


def get_cache(
    enabled: bool = True,
    max_size: int = 1000,
    ttl: int = 3600,
    storage: str = "memory",
    disk_path: Optional[str] = None,
) -> ResponseCache:
    """
    Get or create the global cache instance.
    
    Args:
        enabled: Whether caching is enabled.
        max_size: Maximum number of cached responses.
        ttl: Time-to-live in seconds.
        storage: "memory" or "disk"
        disk_path: Path for disk cache (required if storage="disk")
        
    Returns:
        ResponseCache instance.
    """
    global _global_cache
    
    if _global_cache is None:
        if storage == "disk" and disk_path:
            _global_cache = DiskCache(
                cache_dir=disk_path,
                max_size=max_size,
                ttl=ttl,
                enabled=enabled,
            )
        else:
            _global_cache = ResponseCache(
                max_size=max_size,
                ttl=ttl,
                enabled=enabled,
            )
    
    return _global_cache


def clear_cache() -> None:
    """Clear the global cache."""
    global _global_cache
    if _global_cache:
        _global_cache.invalidate()
