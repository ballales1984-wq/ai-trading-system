"""
Cache Management API
====================
API endpoints for cache management (restart and clear cache).

SECURITY NOTE: Clear operations require admin authentication.
"""

import logging
import os
import secrets
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel

from app.core.cache import get_cache_manager

# Try to import in-memory cache from src package
# This handles cases where PYTHONPATH may not include project root
try:
    from src.utils_cache import get_cache
    IN_MEMORY_CACHE_AVAILABLE = True
except ModuleNotFoundError as e:
    # Fall back only when src package is unavailable. Re-raise other import errors.
    if e.name not in {"src", "src.utils_cache"}:
        raise
    IN_MEMORY_CACHE_AVAILABLE = False

    def get_cache():
        """Fallback when in-memory cache is not available."""
        raise RuntimeError("In-memory cache not available - src.utils_cache not in PYTHONPATH")

logger = logging.getLogger(__name__)

router = APIRouter()
CACHE_NAMESPACE_PREFIX = os.getenv("REDIS_CACHE_PREFIX", "cache:")


def _validate_cache_pattern(pattern: str) -> str:
    """Allow deletes only inside the cache namespace."""
    if not pattern:
        return f"{CACHE_NAMESPACE_PREFIX}*"

    if not pattern.startswith(CACHE_NAMESPACE_PREFIX):
        raise HTTPException(
            status_code=400,
            detail=f"Pattern must start with '{CACHE_NAMESPACE_PREFIX}'"
        )

    return pattern


# =============================================================================
# RESPONSE MODELS
# =============================================================================

class CacheClearResponse(BaseModel):
    """Response for cache clear operation."""
    success: bool
    message: str
    cleared_count: int = 0


class CacheStatsResponse(BaseModel):
    """Response for cache statistics."""
    in_memory: Dict[str, Any]
    redis: Dict[str, Any]


# =============================================================================
# AUTHENTICATION
# =============================================================================

async def verify_admin_access(x_admin_key: str = Header(None, alias="X-Admin-Key")) -> bool:
    """
    Verify admin access for destructive operations.
    
    Requires X-Admin-Key header matching ADMIN_SECRET_KEY environment variable.
    In production, this should be replaced with proper JWT authentication.
    """
    admin_key = os.getenv("ADMIN_SECRET_KEY", "")
    
    # If no admin key is configured, deny access (fail secure)
    if not admin_key:
        logger.error("ADMIN_SECRET_KEY not configured - denying admin access")
        raise HTTPException(
            status_code=503,
            detail="Admin access not configured. Set ADMIN_SECRET_KEY environment variable."
        )
    
    if not x_admin_key or not secrets.compare_digest(x_admin_key, admin_key):
        logger.warning("Invalid admin key attempt")
        raise HTTPException(
            status_code=403,
            detail="Invalid or missing admin credentials"
        )
    
    return True


# =============================================================================
# IN-MEMORY CACHE OPERATIONS
# =============================================================================

@router.delete("/in-memory", response_model=CacheClearResponse)
async def clear_in_memory_cache(
    _: bool = Depends(verify_admin_access)
):
    """
    Clear in-memory cache.
    
    Clears all cached data from the in-memory cache (src/utils_cache.py).
    
    **Requires admin authentication via X-Admin-Key header.**
    """
    if not IN_MEMORY_CACHE_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="In-memory cache not available"
        )
    
    try:
        cache = get_cache()
        stats_before = cache.get_stats()
        cache.clear()
        
        logger.info(f"In-memory cache cleared. Was holding {stats_before.get('size', 0)} entries")
        
        return CacheClearResponse(
            success=True,
            message="In-memory cache cleared successfully",
            cleared_count=stats_before.get('size', 0)
        )
    except Exception as e:
        logger.error(f"Error clearing in-memory cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/in-memory/stats")
async def get_in_memory_cache_stats():
    """
    Get in-memory cache statistics.
    
    Returns cache hits, misses, hit rate and current size.
    """
    if not IN_MEMORY_CACHE_AVAILABLE:
        return {
            "in_memory": {
                "available": False,
                "error": "In-memory cache not available"
            }
        }
    
    try:
        cache = get_cache()
        stats = cache.get_stats()
        stats["available"] = True
        return {
            "in_memory": stats
        }
    except Exception as e:
        logger.error(f"Error getting in-memory cache stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# REDIS CACHE OPERATIONS
# =============================================================================

@router.delete("/redis", response_model=CacheClearResponse)
async def clear_redis_cache(
    pattern: str = "",
    _: bool = Depends(verify_admin_access)
):
    """
    Clear Redis cache.
    
    Clears keys from Redis matching the pattern in cache namespace.
    Default behavior clears only keys under REDIS_CACHE_PREFIX.
    
    **Requires admin authentication via X-Admin-Key header.**
    """
    try:
        cache_manager = await get_cache_manager()
        
        if not cache_manager.is_connected:
            raise HTTPException(status_code=503, detail="Redis not connected")
        
        safe_pattern = _validate_cache_pattern(pattern)

        # Use SCAN + UNLINK to avoid blocking Redis with KEYS.
        deleted = 0
        cursor = 0
        while True:
            cursor, keys = await cache_manager.scan(
                cursor=cursor,
                match=safe_pattern,
                count=500
            )
            if keys:
                deleted += await cache_manager.unlink(*keys)
            if cursor == 0:
                break
        
        logger.info(f"Redis cache cleared. Deleted {deleted} keys matching pattern '{safe_pattern}'")
        
        return CacheClearResponse(
            success=True,
            message=f"Redis cache cleared successfully (pattern: {safe_pattern})",
            cleared_count=deleted
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing Redis cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/redis/stats")
async def get_redis_cache_stats():
    """
    Get Redis cache statistics.
    
    Returns connection status, latency, memory usage and key count.
    """
    try:
        cache_manager = await get_cache_manager()
        
        if not cache_manager.is_connected:
            return {
                "redis": {
                    "connected": False,
                    "error": "Redis not connected"
                }
            }
        
        health = await cache_manager.health_check()
        
        return {
            "redis": health
        }
    except Exception as e:
        logger.error(f"Error getting Redis cache stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# COMBINED OPERATIONS
# =============================================================================

@router.delete("/", response_model=CacheClearResponse)
async def clear_all_cache(
    _: bool = Depends(verify_admin_access)
):
    """
    Clear ALL caches (both in-memory and Redis).
    
    This restarts the cache by clearing both in-memory and Redis caches.
    
    **Requires admin authentication via X-Admin-Key header.**
    """
    try:
        total_cleared = 0
        
        # Clear in-memory cache
        if IN_MEMORY_CACHE_AVAILABLE:
            try:
                in_memory_cache = get_cache()
                in_memory_stats = in_memory_cache.get_stats()
                in_memory_cache.clear()
                total_cleared += in_memory_stats.get('size', 0)
            except Exception as e:
                logger.warning(f"Could not clear in-memory cache: {e}")
        
        # Clear Redis cache
        try:
            cache_manager = await get_cache_manager()
            if cache_manager.is_connected:
                cursor = 0
                safe_pattern = f"{CACHE_NAMESPACE_PREFIX}*"
                while True:
                    cursor, keys = await cache_manager.scan(
                        cursor=cursor,
                        match=safe_pattern,
                        count=500
                    )
                    if keys:
                        total_cleared += await cache_manager.unlink(*keys)
                    if cursor == 0:
                        break
        except Exception as e:
            logger.warning(f"Could not clear Redis cache: {e}")
        
        logger.info(f"All caches cleared. Total entries: {total_cleared}")
        
        return CacheClearResponse(
            success=True,
            message="All caches cleared successfully (in-memory + Redis)",
            cleared_count=total_cleared
        )
    except Exception as e:
        logger.error(f"Error clearing all caches: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=CacheStatsResponse)
async def get_all_cache_stats():
    """
    Get statistics from ALL caches.
    
    Returns both in-memory and Redis cache statistics.
    """
    try:
        # In-memory stats
        in_memory_stats = {"available": False}
        if IN_MEMORY_CACHE_AVAILABLE:
            try:
                in_memory_cache = get_cache()
                in_memory_stats = in_memory_cache.get_stats()
                in_memory_stats["available"] = True
            except Exception as e:
                logger.warning(f"Could not get in-memory stats: {e}")
                in_memory_stats["error"] = str(e)
        
        # Redis stats
        redis_stats = {"connected": False}
        try:
            cache_manager = await get_cache_manager()
            if cache_manager.is_connected:
                redis_stats = await cache_manager.health_check()
        except Exception as e:
            logger.warning(f"Could not get Redis stats: {e}")
        
        return CacheStatsResponse(
            in_memory=in_memory_stats,
            redis=redis_stats
        )
    except Exception as e:
        logger.error(f"Error getting all cache stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
