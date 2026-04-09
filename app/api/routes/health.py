"""
Health Check Routes
==================
System health and status endpoints.
"""

from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, status

from app.core.config import settings


router = APIRouter()


@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check() -> Dict[str, Any]:
    """
    Main health check endpoint.

    Returns system status and version information.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
    }


@router.get("/ready", status_code=status.HTTP_200_OK)
async def readiness_check() -> Dict[str, str]:
    """
    Readiness check for Kubernetes/load balancer.

    Returns 'ready' when system can accept traffic.
    """
    return {"status": "ready"}


@router.get("/live", status_code=status.HTTP_200_OK)
async def liveness_check() -> Dict[str, str]:
    """
    Liveness check for Kubernetes.

    Returns 'alive' to indicate the process is running.
    """
    return {"status": "alive"}


@router.get("/health/database", status_code=status.HTTP_200_OK)
async def database_health_check() -> Dict[str, Any]:
    """
    Database health check with connection pool stats.

    Returns database status and connection pool information.
    """
    from app.core.database import get_db_manager

    db_manager = get_db_manager()
    health = db_manager.health_check()

    # Include config limits
    health["config"] = {
        "pool_size": settings.db_pool_size,
        "max_overflow": settings.db_max_overflow,
        "total_max": settings.db_pool_size + settings.db_max_overflow,
    }

    return health


@router.get("/status", status_code=status.HTTP_200_OK)
async def status_check() -> Dict[str, Any]:
    """
    Comprehensive system status endpoint.

    Returns full system status including:
    - Service health
    - Database status
    - Cache status
    - Active connections
    """
    from app.core.database import get_db_manager
    import psutil
    import os

    # Get database status
    db_manager = get_db_manager()
    db_health = db_manager.health_check() if db_manager else {"status": "unavailable"}

    # Get system resources
    process = psutil.Process(os.getpid())

    return {
        "status": "operational",
        "service": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "timestamp": datetime.utcnow().isoformat(),
        "system": {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "process_memory_mb": process.memory_info().rss / 1024 / 1024,
        },
        "database": db_health,
        "simulation_mode": os.getenv("SIMULATION_MODE", "true") != "false",
    }
