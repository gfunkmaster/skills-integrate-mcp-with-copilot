"""
API key authentication system.
"""

import hashlib
import hmac
import logging
import os
import secrets
from typing import Optional

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

logger = logging.getLogger(__name__)

# API key header name
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_api_keys() -> set:
    """
    Get valid API keys from environment.
    
    API keys can be set via:
    - AGENTIC_CHAIN_API_KEY: Single API key
    - AGENTIC_CHAIN_API_KEYS: Comma-separated list of API keys
    """
    keys = set()
    
    # Single key
    single_key = os.environ.get("AGENTIC_CHAIN_API_KEY")
    if single_key:
        keys.add(single_key.strip())
    
    # Multiple keys
    multi_keys = os.environ.get("AGENTIC_CHAIN_API_KEYS", "")
    for key in multi_keys.split(","):
        key = key.strip()
        if key:
            keys.add(key)
    
    return keys


def is_auth_enabled() -> bool:
    """Check if authentication is enabled."""
    return bool(get_api_keys())


def verify_api_key(api_key: Optional[str]) -> bool:
    """
    Verify if the provided API key is valid.
    
    Uses constant-time comparison to prevent timing attacks.
    """
    valid_keys = get_api_keys()
    
    if not valid_keys:
        # Auth disabled if no keys configured
        return True
    
    if not api_key:
        return False
    
    # Use constant-time comparison
    for valid_key in valid_keys:
        if secrets.compare_digest(api_key, valid_key):
            return True
    
    return False


async def require_api_key(
    api_key: Optional[str] = Security(API_KEY_HEADER),
) -> Optional[str]:
    """
    FastAPI dependency to require API key authentication.
    
    Raises HTTPException 401 if authentication fails.
    Returns the API key if valid, or None if auth is disabled.
    """
    if not is_auth_enabled():
        return None
    
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Provide X-API-Key header.",
        )
    
    if not verify_api_key(api_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key.",
        )
    
    return api_key


def generate_api_key() -> str:
    """Generate a secure API key."""
    return secrets.token_urlsafe(32)
