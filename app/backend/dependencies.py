"""
Dependency Injection

Provides API key verification, rate limiting, and other dependencies.
"""

import time
from collections import defaultdict
from typing import Optional

from fastapi import Header, HTTPException, Request

from app.backend.config import default_config


# Rate limiting storage
_rate_limit_store = defaultdict(list)


async def verify_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
) -> str:
    """
    Verify API key from request header.
    
    Args:
        x_api_key: API key from X-API-Key header
        
    Returns:
        Verified API key
        
    Raises:
        HTTPException: 401 if API key is invalid
    """
    # Skip verification in debug mode without API key
    if not default_config.api_key:
        return "debug-mode"
    
    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="API key required. Provide X-API-Key header.",
        )
    
    if x_api_key != default_config.api_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key.",
        )
    
    return x_api_key


async def check_rate_limit(request: Request) -> None:
    """
    Check rate limit for request.
    
    Args:
        request: FastAPI request object
        
    Raises:
        HTTPException: 429 if rate limit exceeded
    """
    if not default_config.rate_limit_enabled:
        return
    
    # Get client IP
    client_ip = request.client.host if request.client else "unknown"
    
    # Get current time
    now = time.time()
    window_start = now - default_config.rate_limit_window_seconds
    
    # Clean old entries
    _rate_limit_store[client_ip] = [
        t for t in _rate_limit_store[client_ip] if t > window_start
    ]
    
    # Check limit
    if len(_rate_limit_store[client_ip]) >= default_config.rate_limit_requests:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Try again later.",
        )
    
    # Record request
    _rate_limit_store[client_ip].append(now)


def get_model(request: Request):
    """
    Get loaded model from application state.
    
    Args:
        request: FastAPI request object
        
    Returns:
        Loaded model
        
    Raises:
        HTTPException: 503 if model not loaded
    """
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service unavailable.",
        )
    return model


def get_model_info(request: Request):
    """
    Get model info from application state.
    
    Args:
        request: FastAPI request object
        
    Returns:
        Model metadata
        
    Raises:
        HTTPException: 503 if model info not available
    """
    model_info = getattr(request.app.state, "model_info", None)
    if model_info is None:
        raise HTTPException(
            status_code=503,
            detail="Model information not available.",
        )
    return model_info
