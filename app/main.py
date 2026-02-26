"""
Main FastAPI Application
=======================
Hedge Fund Trading System API.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from app.core.config import settings
from app.core.logging import setup_logging, get_logger
from app.api.routes import health, orders, portfolio, strategy, risk, market, waitlist
from app.api.routes import cache, news


from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path


# Setup logging
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application lifespan manager."""
    # Startup
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Environment: {settings.environment}")
    
    # Initialize services
    await _initialize_services()
    
    yield
    
    # Shutdown
    logger.info("Shutting down application")
    await _shutdown_services()


async def _initialize_services():
    """Initialize application services."""
    # Initialize database connection
    # Initialize broker connections
    # Start background tasks
    logger.info("Services initialized")


async def _shutdown_services():
    """Cleanup on shutdown."""
    logger.info("Services stopped")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="""
    ## Hedge Fund Trading System
    
    Professional-grade trading platform with:
    - Multi-asset support (crypto, forex, stocks, futures)
    - Multi-strategy execution
    - Institutional risk management (VaR, CVaR, Monte Carlo)
    - Real-time portfolio tracking
    - Paper trading and live trading modes
    """,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests."""
    logger.info(f"Request: {request.method} {request.url.path}")
    
    response = await call_next(request)
    
    logger.info(f"Response: {response.status_code}")
    return response


# Validation error handler
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors with detailed messages."""
    errors = []
    for error in exc.errors():
        errors.append({
            "field": ".".join(str(loc) for loc in error.get("loc", [])),
            "message": error.get("msg"),
            "type": error.get("type"),
        })
    
    logger.warning(f"Validation error on {request.url.path}: {errors}")
    
    return JSONResponse(
        status_code=422,
        content={
            "detail": errors,
            "path": str(request.url.path),
            "method": request.method,
        }
    )


# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle uncaught exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "path": str(request.url)
        }
    )


# Include routers
app.include_router(
    health.router,
    prefix=settings.api_prefix,
    tags=["Health"]
)

app.include_router(
    orders.router,
    prefix=f"{settings.api_prefix}/orders",
    tags=["Orders"]
)

app.include_router(
    portfolio.router,
    prefix=f"{settings.api_prefix}/portfolio",
    tags=["Portfolio"]
)

app.include_router(
    strategy.router,
    prefix=f"{settings.api_prefix}/strategy",
    tags=["Strategy"]
)

app.include_router(
    risk.router,
    prefix=f"{settings.api_prefix}/risk",
    tags=["Risk"]
)

app.include_router(
    market.router,
    prefix=f"{settings.api_prefix}/market",
    tags=["Market"]
)

app.include_router(
    waitlist.router,
    prefix=f"{settings.api_prefix}",
    tags=["Waitlist"]
)

app.include_router(
    cache.router,
    prefix=f"{settings.api_prefix}/cache",
    tags=["Cache"]
)

app.include_router(
    news.router,
    prefix=f"{settings.api_prefix}/news",
    tags=["News"]
)


# Serve landing page
LANDING_DIR = Path(__file__).parent.parent / "landing"
FRONTEND_DIR = Path(__file__).parent.parent / "frontend" / "dist"

if LANDING_DIR.exists():
    app.mount("/landing", StaticFiles(directory=str(LANDING_DIR), html=True), name="landing")

# Serve frontend in production
if FRONTEND_DIR.exists():
    app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR / "assets")), name="assets")


# Root endpoint - serve landing page or frontend
@app.get("/")
async def root():
    """Root endpoint - serve landing page if available."""
    landing_file = LANDING_DIR / "index.html"
    if landing_file.exists():
        return FileResponse(str(landing_file))
    
    # Fallback to frontend
    frontend_file = FRONTEND_DIR / "index.html"
    if frontend_file.exists():
        return FileResponse(str(frontend_file))
    
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs"
    }


# Serve frontend for SPA routes
@app.get("/dashboard")
@app.get("/portfolio")
@app.get("/market")
@app.get("/orders")
async def serve_spa():
    """Serve frontend SPA for client-side routes."""
    frontend_file = FRONTEND_DIR / "index.html"
    if frontend_file.exists():
        return FileResponse(str(frontend_file))
    
    # Try to serve from /frontend path
    frontend_file = Path("/app/frontend/dist/index.html")
    if frontend_file.exists():
        return FileResponse(str(frontend_file))
    
    return {"error": "Frontend not built"}


# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": settings.app_version,
        "environment": settings.environment
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )
