"""
Request/response logging middleware.
"""

import json
import logging
import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("agentic_chain.api.requests")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging HTTP requests and responses."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process and log request/response."""
        # Generate request ID
        request_id = str(uuid.uuid4())[:8]
        
        # Start timing
        start_time = time.perf_counter()
        
        # Log request
        client_ip = self._get_client_ip(request)
        logger.info(
            f"[{request_id}] {request.method} {request.url.path} "
            f"from {client_ip}"
        )
        
        # Add request ID to state for use in handlers
        request.state.request_id = request_id
        
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration = time.perf_counter() - start_time
            
            # Log response
            logger.info(
                f"[{request_id}] {response.status_code} "
                f"({duration*1000:.2f}ms)"
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            duration = time.perf_counter() - start_time
            logger.error(
                f"[{request_id}] ERROR: {type(e).__name__}: {e} "
                f"({duration*1000:.2f}ms)"
            )
            raise
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address from request."""
        # Check for forwarded header (when behind proxy)
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        
        # Use client IP
        if request.client:
            return request.client.host
        
        return "unknown"
