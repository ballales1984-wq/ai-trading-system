"""
Security Middleware Module
==========================
HTTP security headers, rate limiting integration, and security monitoring.

Author: AI Trading System
"""

import time
import logging
from typing import Callable, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict

from fastapi import FastAPI, Request, Response, HTTPException, Depends
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.core.rate_limiter import RateLimiter, RateLimitConfig, RateLimitExceeded
from app.core.security import jwt_manager, User, UserRole
from app.compliance.audit import AuditLogger, AuditEvent, AuditEventType, RiskLevel

logger = logging.getLogger(__name__)


class SecurityHeadersConfig(Enum):
    """Security header levels."""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"


@dataclass
class SecurityPolicyConfig:
    """Content Security Policy configuration."""
    default_src: tuple = ("'self'",)
    script_src: tuple = ("'self'", "'unsafe-inline'")
    style_src: tuple = ("'self'", "'unsafe-inline'")
    img_src: tuple = ("'self'", "data:", "https:")
    font_src: tuple = ("'self'",)
    connect_src: tuple = ("'self'", "wss:", "https:")
    frame_ancestors: tuple = ("'none'",)
    base_uri: tuple = ("'self'",)
    form_action: tuple = ("'self'",)
    upgrade_insecure_requests: bool = True


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Security Headers Middleware
    ===========================
    Adds comprehensive HTTP security headers to all responses.
    """
    
    def __init__(
        self,
        app: ASGIApp,
        policy_level: SecurityHeadersConfig = SecurityHeadersConfig.STANDARD,
        csp_config: Optional[SecurityPolicyConfig] = None,
        allowed_hosts: Optional[list] = None,
        https_only: bool = False,
    ):
        super().__init__(app)
        self.policy_level = policy_level
        self.csp_config = csp_config or SecurityPolicyConfig()
        self.allowed_hosts = allowed_hosts or ["*"]
        self.https_only = https_only
        
        # Track blocked IPs
        self._blocked_ips: Dict[str, datetime] = {}
        self._ip_request_counts: Dict[str, list] = defaultdict(list)
    
    def _build_csp(self) -> str:
        """Build Content Security Policy header."""
        csp_parts = []
        
        if self.csp_config.default_src:
            csp_parts.append(f"default-src {' '.join(self.csp_config.default_src)}")
        if self.csp_config.script_src:
            csp_parts.append(f"script-src {' '.join(self.csp_config.script_src)}")
        if self.csp_config.style_src:
            csp_parts.append(f"style-src {' '.join(self.csp_config.style_src)}")
        if self.csp_config.img_src:
            csp_parts.append(f"img-src {' '.join(self.csp_config.img_src)}")
        if self.csp_config.font_src:
            csp_parts.append(f"font-src {' '.join(self.csp_config.font_src)}")
        if self.csp_config.connect_src:
            csp_parts.append(f"connect-src {' '.join(self.csp_config.connect_src)}")
        if self.csp_config.frame_ancestors:
            csp_parts.append(f"frame-ancestors {' '.join(self.csp_config.frame_ancestors)}")
        if self.csp_config.base_uri:
            csp_parts.append(f"base-uri {' '.join(self.csp_config.base_uri)}")
        if self.csp_config.form_action:
            csp_parts.append(f"form-action {' '.join(self.csp_config.form_action)}")
        if self.csp_config.upgrade_insecure_requests:
            csp_parts.append("upgrade-insecure-requests")
        
        return "; ".join(csp_parts)
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"
    
    def _is_ip_blocked(self, ip: str) -> bool:
        """Check if IP is temporarily blocked."""
        if ip in self._blocked_ips:
            if datetime.now() < self._blocked_ips[ip]:
                return True
            else:
                del self._blocked_ips[ip]
        return False
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        client_ip = self._get_client_ip(request)
        
        # Check if IP is blocked
        if self._is_ip_blocked(client_ip):
            return JSONResponse(
                status_code=403,
                content={"error": "Access denied", "reason": "ip_blocked"}
            )
        
        # Track request for abuse detection
        self._ip_request_counts[client_ip].append(time.time())
        
        # Clean old entries (older than 1 minute)
        current_time = time.time()
        self._ip_request_counts[client_ip] = [
            t for t in self._ip_request_counts[client_ip]
            if current_time - t < 60
        ]
        
        # Block if too many requests (100+ per minute)
        if len(self._ip_request_counts[client_ip]) > 100:
            self._blocked_ips[client_ip] = datetime.now() + timedelta(minutes=5)
            logger.warning(f"Blocked IP {client_ip} due to excessive requests")
            return JSONResponse(
                status_code=429,
                content={"error": "Too many requests", "retry_after": 300}
            )
        
        # Process request
        response = await call_next(request)
        
        # Add security headers based on policy level
        if self.policy_level in (SecurityHeadersConfig.STANDARD, SecurityHeadersConfig.STRICT):
            # HSTS (only for HTTPS)
            if self.https_only or request.url.scheme == "https":
                response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        # X-Frame-Options
        if self.policy_level != SecurityHeadersConfig.BASIC:
            response.headers["X-Frame-Options"] = "DENY"
        
        # X-Content-Type-Options
        response.headers["X-Content-Type-Options"] = "nosniff"
        
        # X-XSS-Protection (legacy but still useful)
        if self.policy_level == SecurityHeadersConfig.STRICT:
            response.headers["X-XSS-Protection"] = "1; mode=block"
        
        # Referrer-Policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Permissions-Policy
        if self.policy_level != SecurityHeadersConfig.BASIC:
            response.headers["Permissions-Policy"] = (
                "geolocation=(), "
                "midi=(), "
                "notifications=(), "
                "push=(), "
                "sync-xhr=(), "
                "microphone=(), "
                "camera=(), "
                "magnetometer=(), "
                "gyroscope=(), "
                "fullscreen=(self), "
                "payment=()"
            )
        
        # Content Security Policy
        if self.policy_level != SecurityHeadersConfig.BASIC:
            response.headers["Content-Security-Policy"] = self._build_csp()
        
        # Cache-Control for sensitive pages
        if request.url.path.startswith("/api/auth"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, proxy-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        
        # Remove server identification
        response.headers["Server"] = "AI-Trading-System"
        response.headers["X-Powered-By"] = "AI Trading System"
        
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate Limiting Middleware
    ========================
    Integrates rate limiting into FastAPI.
    """
    
    def __init__(
        self,
        app: ASGIApp,
        rate_limiter: Optional[RateLimiter] = None,
        exempt_paths: Optional[list] = None,
        use_redis: bool = False,
    ):
        super().__init__(app)
        self.rate_limiter = rate_limiter or RateLimiter(
            RateLimitConfig(
                requests_per_minute=60,
                requests_per_hour=1000,
                requests_per_day=10000,
                burst_size=10,
            )
        )
        self.exempt_paths = exempt_paths or [
            "/health",
            "/docs",
            "/redoc",
            "/openapi.json",
        ]
        self.use_redis = use_redis
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier."""
        # Try to get API key first
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"api:{api_key}"
        
        # Then try JWT token
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]
            payload = jwt_manager.verify_token(token)
            if payload:
                return f"user:{payload.username}"
        
        # Fall back to IP
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return f"ip:{forwarded.split(',')[0].strip()}"
        
        return f"ip:{request.client.host if request.client else 'unknown'}"
    
    def _is_exempt(self, path: str) -> bool:
        """Check if path is exempt from rate limiting."""
        for exempt_path in self.exempt_paths:
            if path.startswith(exempt_path):
                return True
        return False
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for exempt paths
        if self._is_exempt(request.url.path):
            return await call_next(request)
        
        client_id = self._get_client_id(request)
        
        try:
            self.rate_limiter.check_rate_limit(client_id, request.url.path)
        except RateLimitExceeded as e:
            logger.warning(f"Rate limit exceeded for {client_id}: {e}")
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "message": "Too many requests. Please slow down.",
                    "retry_after": e.retry_after,
                },
                headers={
                    "Retry-After": str(e.retry_after),
                    "X-RateLimit-Limit": "60",
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(time.time()) + e.retry_after),
                }
            )
        
        # Add rate limit headers to successful responses
        stats = self.rate_limiter.get_stats(client_id)
        
        response = await call_next(request)
        
        # Add rate limit headers
        remaining = max(0, 60 - stats.get("count", 0))
        response.headers["X-RateLimit-Limit"] = "60"
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(time.time()) + 60)
        
        return response


@dataclass
class MonitoringMetrics:
    """Application monitoring metrics."""
    requests_total: int = 0
    requests_by_endpoint: Dict[str, int] = field(default_factory=dict)
    requests_by_status: Dict[int, int] = field(default_factory=dict)
    requests_by_user: Dict[str, int] = field(default_factory=dict)
    response_times: list = field(default_factory=list)
    errors: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    
    # System metrics
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    active_connections: int = 0


class MonitoringMiddleware(BaseHTTPMiddleware):
    """
    Monitoring Middleware
    =====================
    Tracks request metrics and performance.
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.metrics = MonitoringMetrics()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        # Track request
        self.metrics.requests_total += 1
        
        # Get endpoint
        endpoint = request.url.path
        self.metrics.requests_by_endpoint[endpoint] = \
            self.metrics.requests_by_endpoint.get(endpoint, 0) + 1
        
        # Get user if authenticated
        auth_header = request.headers.get("Authorization")
        user_id = "anonymous"
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]
            payload = jwt_manager.verify_token(token)
            if payload:
                user_id = payload.username
                self.metrics.requests_by_user[user_id] = \
                    self.metrics.requests_by_user.get(user_id, 0) + 1
        
        try:
            response = await call_next(request)
            
            # Track status code
            status = response.status_code
            self.metrics.requests_by_status[status] = \
                self.metrics.requests_by_status.get(status, 0) + 1
            
            return response
        except Exception as e:
            self.metrics.errors += 1
            raise
        finally:
            # Track response time
            duration = time.time() - start_time
            self.metrics.response_times.append(duration)
            
            # Keep only last 1000 response times
            if len(self.metrics.response_times) > 1000:
                self.metrics.response_times = self.metrics.response_times[-1000:]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        response_times = self.metrics.response_times
        
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        max_response_time = max(response_times) if response_times else 0
        min_response_time = min(response_times) if response_times else 0
        
        # Calculate percentiles
        sorted_times = sorted(response_times)
        p50 = sorted_times[len(sorted_times) // 2] if sorted_times else 0
        p95 = sorted_times[int(len(sorted_times) * 0.95)] if sorted_times else 0
        p99 = sorted_times[int(len(sorted_times) * 0.99)] if sorted_times else 0
        
        uptime = (datetime.now() - self.metrics.start_time).total_seconds()
        
        return {
            "requests": {
                "total": self.metrics.requests_total,
                "by_endpoint": dict(self.metrics.requests_by_endpoint),
                "by_status": dict(self.metrics.requests_by_status),
                "by_user": dict(self.metrics.requests_by_user),
            },
            "performance": {
                "avg_response_time_ms": round(avg_response_time * 1000, 2),
                "min_response_time_ms": round(min_response_time * 1000, 2),
                "max_response_time_ms": round(max_response_time * 1000, 2),
                "p50_ms": round(p50 * 1000, 2),
                "p95_ms": round(p95 * 1000, 2),
                "p99_ms": round(p99 * 1000, 2),
            },
            "errors": {
                "total": self.metrics.errors,
                "rate": round(self.metrics.errors / self.metrics.requests_total * 100, 2) 
                    if self.metrics.requests_total > 0 else 0,
            },
            "uptime": {
                "seconds": int(uptime),
                "formatted": str(timedelta(seconds=int(uptime))),
            },
        }
    
    def reset_metrics(self):
        """Reset all metrics."""
        self.metrics = MonitoringMetrics()


# Global monitoring instance
monitoring_middleware: Optional[MonitoringMiddleware] = None


def get_monitoring_middleware() -> Optional[MonitoringMiddleware]:
    """Get the global monitoring middleware instance."""
    return monitoring_middleware


def setup_security_middleware(
    app: FastAPI,
    security_level: SecurityHeadersConfig = SecurityHeadersConfig.STANDARD,
    enable_rate_limiting: bool = True,
    enable_monitoring: bool = True,
    allowed_hosts: Optional[list] = None,
) -> None:
    """
    Setup all security middleware.
    
    Args:
        app: FastAPI application
        security_level: Security headers level
        enable_rate_limiting: Enable rate limiting
        enable_monitoring: Enable monitoring middleware
        allowed_hosts: List of allowed hosts
    """
    global monitoring_middleware
    
    # Add monitoring middleware first (outermost)
    if enable_monitoring:
        monitoring = MonitoringMiddleware(app)
        monitoring_middleware = monitoring
        logger.info("Monitoring middleware enabled")
    
    # Add rate limiting middleware
    if enable_rate_limiting:
        app.add_middleware(
            RateLimitMiddleware,
            exempt_paths=["/health", "/docs", "/redoc", "/openapi.json"]
        )
        logger.info("Rate limiting middleware enabled")
    
    # Add security headers middleware (innermost)
    app.add_middleware(
        SecurityHeadersMiddleware,
        policy_level=security_level,
        allowed_hosts=allowed_hosts,
    )
    logger.info(f"Security headers middleware enabled with level: {security_level.value}")


# API Routes for monitoring and security

def create_monitoring_routes(app: FastAPI):
    """Create monitoring and security API routes."""
    
    @app.get("/api/monitoring/metrics")
    async def get_metrics():
        """Get application monitoring metrics."""
        if monitoring_middleware is None:
            return {"error": "Monitoring not enabled"}
        return monitoring_middleware.get_metrics()
    
    @app.get("/api/monitoring/health")
    async def detailed_health():
        """Get detailed health status."""
        if monitoring_middleware is None:
            return {
                "status": "healthy",
                "monitoring": "disabled"
            }
        
        metrics = monitoring_middleware.get_metrics()
        
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
        if monitoring_middleware is None:
            return {"error": "Monitoring not enabled"}
        monitoring_middleware.reset_metrics()
        return {"message": "Metrics reset successfully"}
    
    @app.get("/api/security/headers")
    async def security_headers_info():
        """Get security headers configuration."""
        return {
            "security_level": "standard",
            "headers": {
                "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
                "X-Frame-Options": "DENY",
                "X-Content-Type-Options": "nosniff",
                "X-XSS-Protection": "1; mode=block",
                "Referrer-Policy": "strict-origin-when-cross-origin",
                "Permissions-Policy": "geolocation=(),midi=(),notifications=(),push=()",
            },
            "rate_limiting": {
                "enabled": True,
                "requests_per_minute": 60,
                "requests_per_hour": 1000,
            }
        }


# Audit logging integration

def log_security_event(
    event_type: AuditEventType,
    user_id: Optional[str] = None,
    username: Optional[str] = None,
    ip_address: Optional[str] = None,
    details: Optional[Dict] = None,
):
    """Log a security event to audit trail."""
    # This would integrate with the existing AuditLogger
    # For now, just log to the standard logger
    logger.info(
        f"Security Event: {event_type.value}",
        extra={
            "event_type": event_type.value,
            "user_id": user_id,
            "username": username,
            "ip_address": ip_address,
            "details": details or {},
        }
    )


__all__ = [
    "SecurityHeadersConfig",
    "SecurityPolicyConfig",
    "SecurityHeadersMiddleware",
    "RateLimitMiddleware",
    "MonitoringMiddleware",
    "MonitoringMetrics",
    "setup_security_middleware",
    "create_monitoring_routes",
    "log_security_event",
    "get_monitoring_middleware",
]
