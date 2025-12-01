"""
Rate limiting middleware using token bucket algorithm.
"""

import asyncio
import hashlib
import logging
import os
import time
from typing import Callable, Dict, Optional

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class TokenBucket:
    """Token bucket rate limiter."""
    
    def __init__(self, rate: float, capacity: int):
        """
        Initialize token bucket.
        
        Args:
            rate: Tokens added per second
            capacity: Maximum tokens in bucket
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()
    
    async def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens from the bucket.
        
        Returns True if tokens were consumed, False if not enough tokens.
        """
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_update
            self.last_update = now
            
            # Add tokens based on elapsed time
            self.tokens = min(
                self.capacity,
                self.tokens + elapsed * self.rate
            )
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            return False
    
    @property
    def wait_time(self) -> float:
        """Calculate time to wait for one token."""
        if self.tokens >= 1:
            return 0
        return (1 - self.tokens) / self.rate


class RateLimiter:
    """Rate limiter with per-client buckets."""
    
    def __init__(
        self,
        rate: float = 10.0,
        capacity: int = 100,
        enabled: bool = True,
    ):
        """
        Initialize rate limiter.
        
        Args:
            rate: Requests per second allowed
            capacity: Burst capacity
            enabled: Whether rate limiting is enabled
        """
        self.rate = rate
        self.capacity = capacity
        self.enabled = enabled
        self._buckets: Dict[str, TokenBucket] = {}
        self._lock = asyncio.Lock()
        self._cleanup_interval = 300  # 5 minutes
        self._last_cleanup = time.monotonic()
    
    def _get_or_create_bucket(self, client_id: str) -> TokenBucket:
        """Get or create a bucket for a client (not thread-safe, use with lock)."""
        if client_id not in self._buckets:
            self._buckets[client_id] = TokenBucket(self.rate, self.capacity)
        return self._buckets[client_id]
    
    def get_client_id(self, request: Request) -> str:
        """Get client identifier from request."""
        # Use API key if available, otherwise use IP
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"key:{hashlib.sha256(api_key.encode()).hexdigest()[:16]}"
        
        # Use forwarded IP if behind proxy
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return f"ip:{forwarded.split(',')[0].strip()}"
        
        # Use client IP
        client = request.client
        if client:
            return f"ip:{client.host}"
        
        return "unknown"
    
    async def check_rate_limit(self, request: Request) -> bool:
        """
        Check if request should be rate limited.
        
        Returns True if allowed, raises HTTPException if limited.
        """
        if not self.enabled:
            return True
        
        # Cleanup old buckets periodically
        await self._maybe_cleanup()
        
        client_id = self.get_client_id(request)
        
        async with self._lock:
            bucket = self._get_or_create_bucket(client_id)
        
        if await bucket.consume():
            return True
        
        wait_time = bucket.wait_time
        logger.warning(f"Rate limit exceeded for {client_id}")
        
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Try again in {wait_time:.1f} seconds.",
            headers={"Retry-After": str(int(wait_time) + 1)},
        )
    
    async def _maybe_cleanup(self):
        """Clean up old buckets."""
        now = time.monotonic()
        if now - self._last_cleanup < self._cleanup_interval:
            return
        
        self._last_cleanup = now
        
        async with self._lock:
            # Remove buckets that are at full capacity (inactive)
            to_remove = [
                client_id for client_id, bucket in self._buckets.items()
                if bucket.tokens >= bucket.capacity
            ]
            
            for client_id in to_remove:
                del self._buckets[client_id]
        
        if to_remove:
            logger.debug(f"Cleaned up {len(to_remove)} rate limit buckets")


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for rate limiting."""
    
    def __init__(
        self,
        app,
        rate: Optional[float] = None,
        capacity: Optional[int] = None,
    ):
        super().__init__(app)
        
        # Get config from environment or use defaults
        rate = rate or float(os.environ.get("AGENTIC_CHAIN_RATE_LIMIT", "10"))
        capacity = capacity or int(os.environ.get("AGENTIC_CHAIN_RATE_CAPACITY", "100"))
        enabled = os.environ.get("AGENTIC_CHAIN_RATE_LIMIT_ENABLED", "true").lower() == "true"
        
        self.limiter = RateLimiter(rate=rate, capacity=capacity, enabled=enabled)
    
    async def dispatch(self, request: Request, call_next: Callable):
        """Process request with rate limiting."""
        # Skip rate limiting for health check
        if request.url.path == "/api/v1/health":
            return await call_next(request)
        
        await self.limiter.check_rate_limit(request)
        return await call_next(request)


# Global rate limiter instance for use in dependencies
rate_limiter = RateLimiter()
