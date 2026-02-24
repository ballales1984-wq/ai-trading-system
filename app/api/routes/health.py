"""
Health Check Routes
==================
System health and status endpoints.
"""

from datetime import datetime
from typing import Dict, Any
from pydantic import BaseModel
import logging

from fastapi import APIRouter, status

from app.core.config import settings


router = APIRouter()
logger = logging.getLogger(__name__)


class ClientEvent(BaseModel):
    level: str
    event: str
    details: Dict[str, Any] | None = None


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


@router.post("/health/client-events", status_code=status.HTTP_202_ACCEPTED)
async def ingest_client_event(payload: ClientEvent) -> Dict[str, str]:
    """
    Lightweight endpoint to ingest frontend runtime events.
    """
    logger.info(
        "client_event level=%s event=%s details=%s",
        payload.level,
        payload.event,
        payload.details or {},
    )
    return {"status": "accepted"}
