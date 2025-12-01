"""Tests for LLM response cache."""

import pytest
import time
import tempfile
import shutil
from pathlib import Path

from agentic_chain.llm.cache import (
    ResponseCache,
    DiskCache,
    CacheEntry,
    get_cache,
    clear_cache,
)
from agentic_chain.llm.base import (
    LLMMessage,
    LLMResponse,
    LLMUsage,
    MessageRole,
)


class TestCacheEntry:
    """Test cases for CacheEntry."""
    
    def test_create_entry(self):
        """Test creating a cache entry."""
        response = LLMResponse(
            content="Test",
            model="gpt-4",
            usage=LLMUsage(),
        )
        entry = CacheEntry(response=response, created_at=time.time())
        
        assert entry.response == response
        assert entry.hits == 0
    
    def test_entry_not_expired(self):
        """Test entry is not expired."""
        response = LLMResponse(
            content="Test",
            model="gpt-4",
            usage=LLMUsage(),
        )
        entry = CacheEntry(response=response, created_at=time.time())
        
        assert entry.is_expired(ttl=3600) is False
    
    def test_entry_expired(self):
        """Test entry is expired."""
        response = LLMResponse(
            content="Test",
            model="gpt-4",
            usage=LLMUsage(),
        )
        entry = CacheEntry(response=response, created_at=time.time() - 7200)
        
        assert entry.is_expired(ttl=3600) is True


class TestResponseCache:
    """Test cases for ResponseCache."""
    
    def test_cache_disabled(self):
        """Test cache when disabled."""
        cache = ResponseCache(enabled=False)
        messages = [LLMMessage(role=MessageRole.USER, content="Test")]
        params = {"model": "gpt-4"}
        
        response = LLMResponse(
            content="Response",
            model="gpt-4",
            usage=LLMUsage(),
        )
        
        cache.set(messages, params, response)
        result = cache.get(messages, params)
        
        assert result is None
    
    def test_cache_set_and_get(self):
        """Test setting and getting cached response."""
        cache = ResponseCache()
        messages = [LLMMessage(role=MessageRole.USER, content="Test")]
        params = {"model": "gpt-4", "temperature": 0.7}
        
        response = LLMResponse(
            content="Cached response",
            model="gpt-4",
            usage=LLMUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        
        cache.set(messages, params, response)
        result = cache.get(messages, params)
        
        assert result is not None
        assert result.content == "Cached response"
    
    def test_cache_miss(self):
        """Test cache miss."""
        cache = ResponseCache()
        messages = [LLMMessage(role=MessageRole.USER, content="Test")]
        params = {"model": "gpt-4"}
        
        result = cache.get(messages, params)
        
        assert result is None
    
    def test_cache_different_params(self):
        """Test that different params result in different cache keys."""
        cache = ResponseCache()
        messages = [LLMMessage(role=MessageRole.USER, content="Test")]
        
        response1 = LLMResponse(content="Response 1", model="gpt-4", usage=LLMUsage())
        response2 = LLMResponse(content="Response 2", model="gpt-4", usage=LLMUsage())
        
        cache.set(messages, {"model": "gpt-4", "temperature": 0.5}, response1)
        cache.set(messages, {"model": "gpt-4", "temperature": 0.9}, response2)
        
        result1 = cache.get(messages, {"model": "gpt-4", "temperature": 0.5})
        result2 = cache.get(messages, {"model": "gpt-4", "temperature": 0.9})
        
        assert result1.content == "Response 1"
        assert result2.content == "Response 2"
    
    def test_cache_ttl_expiration(self):
        """Test that expired entries are not returned."""
        cache = ResponseCache(ttl=1)  # 1 second TTL
        messages = [LLMMessage(role=MessageRole.USER, content="Test")]
        params = {"model": "gpt-4"}
        
        response = LLMResponse(content="Short-lived", model="gpt-4", usage=LLMUsage())
        cache.set(messages, params, response)
        
        # Should be available immediately
        assert cache.get(messages, params) is not None
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Should be expired
        assert cache.get(messages, params) is None
    
    def test_cache_max_size_eviction(self):
        """Test LRU eviction when max size is reached."""
        cache = ResponseCache(max_size=3)
        
        # Add 3 entries
        for i in range(3):
            messages = [LLMMessage(role=MessageRole.USER, content=f"Message {i}")]
            params = {"model": "gpt-4"}
            response = LLMResponse(content=f"Response {i}", model="gpt-4", usage=LLMUsage())
            cache.set(messages, params, response)
        
        # Add a 4th entry (should evict the first)
        messages = [LLMMessage(role=MessageRole.USER, content="Message 3")]
        response = LLMResponse(content="Response 3", model="gpt-4", usage=LLMUsage())
        cache.set(messages, {"model": "gpt-4"}, response)
        
        # First entry should be evicted
        first_messages = [LLMMessage(role=MessageRole.USER, content="Message 0")]
        assert cache.get(first_messages, {"model": "gpt-4"}) is None
        
        # Last entry should still be there
        assert cache.get(messages, {"model": "gpt-4"}) is not None
    
    def test_cache_invalidate_specific(self):
        """Test invalidating a specific entry."""
        cache = ResponseCache()
        messages = [LLMMessage(role=MessageRole.USER, content="Test")]
        params = {"model": "gpt-4"}
        
        response = LLMResponse(content="Response", model="gpt-4", usage=LLMUsage())
        cache.set(messages, params, response)
        
        count = cache.invalidate(messages, params)
        
        assert count == 1
        assert cache.get(messages, params) is None
    
    def test_cache_invalidate_all(self):
        """Test invalidating all entries."""
        cache = ResponseCache()
        
        # Add multiple entries
        for i in range(5):
            messages = [LLMMessage(role=MessageRole.USER, content=f"Message {i}")]
            response = LLMResponse(content=f"Response {i}", model="gpt-4", usage=LLMUsage())
            cache.set(messages, {"model": "gpt-4"}, response)
        
        count = cache.invalidate()
        
        assert count == 5
    
    def test_cache_prune_expired(self):
        """Test pruning expired entries."""
        cache = ResponseCache(ttl=1)
        
        # Add entries
        for i in range(3):
            messages = [LLMMessage(role=MessageRole.USER, content=f"Message {i}")]
            response = LLMResponse(content=f"Response {i}", model="gpt-4", usage=LLMUsage())
            cache.set(messages, {"model": "gpt-4"}, response)
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Prune
        removed = cache.prune_expired()
        
        assert removed == 3
    
    def test_cache_stats(self):
        """Test cache statistics."""
        cache = ResponseCache()
        messages = [LLMMessage(role=MessageRole.USER, content="Test")]
        params = {"model": "gpt-4"}
        
        response = LLMResponse(content="Response", model="gpt-4", usage=LLMUsage())
        cache.set(messages, params, response)
        
        # Hit
        cache.get(messages, params)
        # Miss
        cache.get([LLMMessage(role=MessageRole.USER, content="Other")], params)
        
        stats = cache.get_stats()
        
        assert stats["size"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 50.0
    
    def test_cache_reset_stats(self):
        """Test resetting cache statistics."""
        cache = ResponseCache()
        messages = [LLMMessage(role=MessageRole.USER, content="Test")]
        params = {"model": "gpt-4"}
        
        response = LLMResponse(content="Response", model="gpt-4", usage=LLMUsage())
        cache.set(messages, params, response)
        cache.get(messages, params)
        
        cache.reset_stats()
        stats = cache.get_stats()
        
        assert stats["hits"] == 0
        assert stats["misses"] == 0


class TestDiskCache:
    """Test cases for DiskCache."""
    
    @pytest.fixture
    def cache_dir(self):
        """Create a temporary directory for cache."""
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        shutil.rmtree(tmpdir)
    
    def test_disk_cache_set_and_get(self, cache_dir):
        """Test setting and getting from disk cache."""
        cache = DiskCache(cache_dir=cache_dir)
        messages = [LLMMessage(role=MessageRole.USER, content="Test")]
        params = {"model": "gpt-4"}
        
        response = LLMResponse(
            content="Disk cached",
            model="gpt-4",
            usage=LLMUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        
        cache.set(messages, params, response)
        result = cache.get(messages, params)
        
        assert result is not None
        assert result.content == "Disk cached"
    
    def test_disk_cache_persistence(self, cache_dir):
        """Test that disk cache persists across instances."""
        messages = [LLMMessage(role=MessageRole.USER, content="Persistent test")]
        params = {"model": "gpt-4"}
        
        response = LLMResponse(
            content="Persistent response",
            model="gpt-4",
            usage=LLMUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        
        # Create first cache and add entry
        cache1 = DiskCache(cache_dir=cache_dir)
        cache1.set(messages, params, response)
        
        # Create second cache (should load from disk)
        cache2 = DiskCache(cache_dir=cache_dir)
        result = cache2.get(messages, params)
        
        assert result is not None
        assert result.content == "Persistent response"
    
    def test_disk_cache_clear(self, cache_dir):
        """Test clearing disk cache."""
        cache = DiskCache(cache_dir=cache_dir)
        messages = [LLMMessage(role=MessageRole.USER, content="Test")]
        params = {"model": "gpt-4"}
        
        response = LLMResponse(content="Response", model="gpt-4", usage=LLMUsage())
        cache.set(messages, params, response)
        
        cache.clear_disk()
        
        # Files should be removed
        cache_files = list(Path(cache_dir).glob("*.json"))
        # Only index file might remain or be empty
        assert len([f for f in cache_files if f.name != "cache_index.json"]) == 0


class TestGlobalCache:
    """Test cases for global cache functions."""
    
    def test_get_cache_creates_instance(self):
        """Test that get_cache creates a cache instance."""
        # Clear any existing global cache
        import agentic_chain.llm.cache as cache_module
        cache_module._global_cache = None
        
        cache = get_cache()
        
        assert cache is not None
        assert isinstance(cache, ResponseCache)
    
    def test_get_cache_returns_same_instance(self):
        """Test that get_cache returns the same instance."""
        # Clear any existing global cache
        import agentic_chain.llm.cache as cache_module
        cache_module._global_cache = None
        
        cache1 = get_cache()
        cache2 = get_cache()
        
        assert cache1 is cache2
    
    def test_clear_cache(self):
        """Test clearing the global cache."""
        # Clear any existing global cache
        import agentic_chain.llm.cache as cache_module
        cache_module._global_cache = None
        
        cache = get_cache()
        messages = [LLMMessage(role=MessageRole.USER, content="Test")]
        response = LLMResponse(content="Response", model="gpt-4", usage=LLMUsage())
        cache.set(messages, {"model": "gpt-4"}, response)
        
        clear_cache()
        
        stats = cache.get_stats()
        assert stats["size"] == 0
