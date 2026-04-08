"""
Main FastAPI Application
=======================
Hedge Fund Trading System API.

Security Features:
- JWT Authentication
- Rate Limiting (60 req/min, 1000/hr, 10000/day)
- Security Headers (HSTS, CSP, X-Frame-Options, etc.)
- Audit Logging
- Request Monitoring & Metrics
"""

from datetime import datetime
from typing import Any

# pyre-ignore[21]: Missing module attribute
from fastapi import FastAPI

# pyre-ignore[21]: Missing module attribute
from fastapi import Request, Response

# pyre-ignore[21]: Missing module attribute
from fastapi.middleware.cors import CORSMiddleware

# pyre-ignore[21]: Missing module attribute
from fastapi.staticfiles import StaticFiles

# pyre-ignore[21]: Missing module attribute
from fastapi.responses import JSONResponse

# pyre-ignore[21]: Missing module attribute
from app.core.config import settings

# pyre-ignore[21]: Missing module attribute
from app.core.logging import setup_logging, get_logger

# pyre-ignore[21]: Missing module attribute
from app.core.rate_limiter import default_rate_limiter, RateLimitExceeded

# pyre-ignore[21]: Missing module attribute
from app.core.security_middleware import (
    setup_security_middleware,
    create_monitoring_routes,
    SecurityHeadersConfig,
    get_monitoring_middleware,
)

# Import route routers
from app.api.routes import (
    news,
    market,
    portfolio,
    orders,
    health,
    strategy,
    waitlist,
    cache,
    auth,
    risk,
    ws,
    agents,
)

from app.scheduler import init_scheduler, get_scheduler

# Import audit classes
from app.compliance.audit import AuditLogger, AuditEvent, AuditEventType

# Setup logging — must be defined BEFORE any usage (e.g. in the Prometheus try/except)
setup_logging()
logger = get_logger(__name__)

try:
    from app.metrics import get_metrics_app, instrument_requests
except ImportError:
    logger.warning("Prometheus metrics disabled - install prometheus_client")
    get_metrics_app = None
    instrument_requests = None

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Hedge Fund Trading System API",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

@app.on_event("startup")
async def startup_event():
    """Startup lifecycle event."""
    # Initialize and start scheduler
    try:
        scheduler = init_scheduler()
        await scheduler.start()
        logger.info("Global Task Scheduler started")
    except Exception as e:
        logger.error(f"Failed to start Task Scheduler: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown lifecycle event."""
    # Stop scheduler
    try:
        scheduler = get_scheduler()
        await scheduler.stop()
        logger.info("Global Task Scheduler stopped")
    except Exception as e:
        logger.error(f"Error stopping Task Scheduler: {e}")

# CORS middleware - Allow frontend to communicate with backend
# Uses the cors_origins list from settings, with fallback to common dev origins
cors_origins = getattr(
    settings,
    "cors_origins",
    [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8000",
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info(f"CORS enabled for origins: {cors_origins}")


# Security Response class for headers
class SecurityResponse(JSONResponse):
    def __init__(self, content: Any, status_code: int = 200, headers: dict | None = None, **kwargs):
        # pyre-ignore[6]: Expected positional arguments
        super().__init__(content, status_code, headers or {}, **kwargs)
        security_headers = {
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains; preload",
            "X-Frame-Options": "DENY",
            "X-Content-Type-Options": "nosniff",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
            "Content-Security-Policy": "default-src 'self'; script-src-elem 'self' 'unsafe-inline' 'unsafe-eval' https://pagead2.googlesyndication.com https://www.googletagmanager.com https://*.google-analytics.com; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self' data:; connect-src 'self' ws: wss: wss://*.onrender.com https://*.google-analytics.com https://*.analytics.google.com; frame-ancestors 'none';",
            "Cross-Origin-Embedder-Policy": "require-corp",
            "Cross-Origin-Opener-Policy": "same-origin",
            "Cross-Origin-Resource-Policy": "same-origin",
        }
        self.headers.update(security_headers)


# Rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_id = request.client.host
    try:
        default_rate_limiter.check_rate_limit(client_id)
    except RateLimitExceeded as e:
        return SecurityResponse(
            content={"error": "Rate limit exceeded", "retry_after": e.retry_after},
            status_code=429,
            headers={"Retry-After": str(e.retry_after)},
        )
    response = await call_next(request)
    # Add security headers to response
    security_headers = {
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains; preload",
        "X-Frame-Options": "DENY",
        "X-Content-Type-Options": "nosniff",
        "X-XSS-Protection": "1; mode=block",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
        "Content-Security-Policy": "default-src 'self'; script-src-elem 'self' 'unsafe-inline' 'unsafe-eval' https://pagead2.googlesyndication.com https://www.googletagmanager.com https://*.google-analytics.com; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self' data:; connect-src 'self' ws: wss: wss://*.onrender.com https://*.google-analytics.com https://*.analytics.google.com; frame-ancestors 'none';",
        "Cross-Origin-Embedder-Policy": "require-corp",
        "Cross-Origin-Opener-Policy": "same-origin",
        "Cross-Origin-Resource-Policy": "same-origin",
    }
    for key, value in security_headers.items():
        response.headers[key] = value
    return response


# Global audit logger
audit_logger = AuditLogger()

# Include routers
app.include_router(news, prefix=f"{settings.api_prefix}/news", tags=["News"])

app.include_router(market, prefix=f"{settings.api_prefix}/market", tags=["Market"])

app.include_router(portfolio, prefix=f"{settings.api_prefix}/portfolio", tags=["Portfolio"])

app.include_router(orders, prefix=f"{settings.api_prefix}/orders", tags=["Orders"])

app.include_router(health, prefix=settings.api_prefix, tags=["Health"])

app.include_router(strategy, prefix=f"{settings.api_prefix}/strategy", tags=["Strategy"])

app.include_router(waitlist, prefix=f"{settings.api_prefix}/waitlist", tags=["Waitlist"])

app.include_router(cache, prefix=f"{settings.api_prefix}/cache", tags=["Cache"])

app.include_router(auth, prefix=f"{settings.api_prefix}/auth", tags=["Auth"])

app.include_router(risk, prefix=f"{settings.api_prefix}/risk", tags=["Risk"])

app.include_router(ws, prefix="/ws", tags=["WebSocket"])

app.include_router(
    agents, prefix=f"{settings.api_prefix}/agents/autonomous", tags=["Autonomous Agent"]
)


# Health check endpoint (no audit logging to avoid log spam)
@app.get("/health")
async def health_check():
    """Health check endpoint - no rate limiting or audit for fast monitoring."""
    return {
        "status": "healthy",
        "version": settings.app_version,
        "environment": settings.environment,
    }


# Rate limit stats endpoint
@app.get("/api/v1/rate-limit/stats")
async def rate_limit_stats(request: Request):
    """Get rate limit statistics for current client."""
    client_id = request.client.host
    try:
        stats = default_rate_limiter.get_stats(client_id)
        audit_logger.log_event(
            AuditEvent(
                ip_address=client_id, action="Rate limit stats query", details={"stats": stats}
            )
        )
        return SecurityResponse(stats)
    except Exception as e:
        return SecurityResponse({"error": str(e)}, status_code=500)


# Comprehensive Monitoring Endpoints
@app.get("/api/monitoring/metrics")
async def get_metrics():
    """
    Get application monitoring metrics.

    Returns:
        - requests: Request counts by endpoint, status, user
        - performance: Response time percentiles (p50, p95, p99)
        - errors: Error count and rate
        - uptime: Application uptime
    """
    monitoring = get_monitoring_middleware()
    if monitoring is None:
        return JSONResponse(status_code=503, content={"error": "Monitoring not enabled"})
    return monitoring.get_metrics()


@app.get("/api/monitoring/health")
async def detailed_health():
    """
    Get detailed health status with metrics.

    Returns:
        - status: healthy | degraded | unhealthy
        - monitoring: enabled status
        - metrics: current performance metrics
    """
    monitoring = get_monitoring_middleware()

    if monitoring is None:
        return {
            "status": "healthy",
            "monitoring": "disabled",
            "timestamp": datetime.now().isoformat(),
        }

    metrics = monitoring.get_metrics()

    # Determine health status
    error_rate = metrics.get("errors", {}).get("rate", 0)
    avg_response = metrics.get("performance", {}).get("avg_response_time_ms", 0)

    if error_rate > 10 or avg_response > 1000:
        status = "unhealthy"
    elif error_rate > 5 or avg_response > 500:
        status = "degraded"
    else:
        status = "healthy"

    return {
        "status": status,
        "monitoring": "enabled",
        "metrics": metrics,
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/api/monitoring/reset")
async def reset_metrics():
    """Reset monitoring metrics."""
    monitoring = get_monitoring_middleware()
    if monitoring is None:
        return JSONResponse(status_code=503, content={"error": "Monitoring not enabled"})
    monitoring.reset_metrics()
    return {"message": "Metrics reset successfully"}


@app.get("/api/security/headers")
async def security_headers_info():
    """
    Get security headers configuration.

    Returns current security headers and rate limiting settings.
    """
    return {
        "security_level": "standard",
        "headers": {
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains; preload",
            "X-Frame-Options": "DENY",
            "X-Content-Type-Options": "nosniff",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
            "Content-Security-Policy": "default-src 'self'; script-src-elem 'self' 'unsafe-inline' 'unsafe-eval' https://pagead2.googlesyndication.com https://www.googletagmanager.com https://*.google-analytics.com; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self' data:; connect-src 'self' ws: wss: wss://*.onrender.com https://*.google-analytics.com https://*.analytics.google.com; frame-ancestors 'none';",
            "Cross-Origin-Embedder-Policy": "require-corp",
            "Cross-Origin-Opener-Policy": "same-origin",
            "Cross-Origin-Resource-Policy": "same-origin",
        },
        "rate_limiting": {
            "enabled": True,
            "requests_per_minute": 60,
            "requests_per_hour": 1000,
            "requests_per_day": 10000,
            "burst_size": 10,
        },
    }


# Audit log query endpoint
@app.get("/api/audit/events")
async def get_audit_events(
    request: Request, event_type: str | None = None, user_id: str | None = None, limit: int = 100
):
    """
    Query audit events.

    Args:
        event_type: Filter by event type
        user_id: Filter by user ID
        limit: Maximum number of events to return
    """
    # In production, this would require admin authentication
    events = audit_logger.events[-limit:]

    if event_type:
        events = [e for e in events if e.event_type.value == event_type]
    if user_id:
        events = [e for e in events if e.user_id == user_id]

    return {"events": [e.to_dict() for e in events], "total": len(events)}


@app.get("/api/audit/stats")
async def get_audit_stats():
    """Get audit statistics."""
    return audit_logger.get_stats()


# Serve frontend static files (for Render deployment)
try:
    app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="frontend")
    logger.info("Frontend static files mounted at / from frontend/dist/")
except Exception as e:
    logger.warning(f"Could not mount frontend static files: {e}")

if get_metrics_app:
    app.mount("/metrics", get_metrics_app())
    logger.info("Prometheus metrics available at /metrics")
else:
    logger.warning("Metrics endpoint disabled")

if __name__ == "__main__":
    # pyre-ignore[21]: Missing module attribute
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=settings.debug)
