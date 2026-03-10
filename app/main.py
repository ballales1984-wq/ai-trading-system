"""
Main FastAPI Application
=======================
Hedge Fund Trading System API.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.core.config import settings
from app.core.logging import setup_logging, get_logger
from app.api.routes import news, market, portfolio, orders, health, strategy, waitlist, cache, auth, risk

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Hedge Fund Trading System API",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# CORS middleware - Allow frontend to communicate with backend
# Uses the cors_origins list from settings, with fallback to common dev origins
cors_origins = getattr(settings, 'cors_origins', [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:8000",
])

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info(f"CORS enabled for origins: {cors_origins}")

# Include routers
app.include_router(
    news,
    prefix=f"{settings.api_prefix}/news",
    tags=["News"]
)

app.include_router(
    market,
    prefix=f"{settings.api_prefix}/market",
    tags=["Market"]
)

app.include_router(
    portfolio,
    prefix=f"{settings.api_prefix}/portfolio",
    tags=["Portfolio"]
)

app.include_router(
    orders,
    prefix=f"{settings.api_prefix}/orders",
    tags=["Orders"]
)

app.include_router(
    health,
    prefix=settings.api_prefix,
    tags=["Health"]
)

app.include_router(
    strategy,
    prefix=f"{settings.api_prefix}/strategy",
    tags=["Strategy"]
)

app.include_router(
    waitlist,
    prefix=f"{settings.api_prefix}/waitlist",
    tags=["Waitlist"]
)

app.include_router(
    cache,
    prefix=f"{settings.api_prefix}/cache",
    tags=["Cache"]
)

app.include_router(
    auth,
    prefix=f"{settings.api_prefix}/auth",
    tags=["Auth"]
)

app.include_router(
    risk,
    prefix=f"{settings.api_prefix}/risk",
    tags=["Risk"]
)

# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": settings.app_version,
        "environment": settings.environment
    }

# Serve frontend static files (for Render deployment)
try:
    app.mount("/", StaticFiles(directory="dist", html=True), name="frontend")
    logger.info("Frontend static files mounted at / from dist/")
except Exception as e:
    logger.warning(f"Could not mount frontend static files: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )
