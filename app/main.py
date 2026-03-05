"""
Main FastAPI Application
=======================
Hedge Fund Trading System API.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator
import threading
import time

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from app.core.config import settings
from app.core.logging import setup_logging, get_logger
from app.api.routes import health, orders, portfolio, strategy, market, waitlist
from app.api.routes import cache, news, auth
from app.api.routes.payments import router as payments
from app.api.routes.risk import router as risk


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
    # Start background tasks for auto trading
    
    # Start auto trader in background thread
    try:
        import threading
        import time
        import asyncio
        import httpx
        
        async def execute_trade_via_api(symbol: str, side: str, quantity: float, order_type: str = "MARKET"):
            """Execute a trade via the internal API."""
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        "http://127.0.0.1:8000/api/v1/orders",
                        json={
                            "symbol": symbol,
                            "side": side,
                            "order_type": order_type,
                            "quantity": quantity,
                            "price": None if order_type == "MARKET" else None
                        }
                    )
                    return response.json() if response.status_code == 200 else None
            except Exception as e:
                logger.warning(f"Trade execution error: {e}")
                return None
        
        async def run_auto_trader_async():
            """Run auto trader in background - simplified version using API."""
            import random
            from datetime import datetime
            
            logger.info("Starting simplified auto trader (API-based)...")
            
            # Assets to trade
            assets = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOTUSDT"]
            
            # Initial delay to let server fully start
            await asyncio.sleep(5)
            
            cycle_count = 0
            
            logger.info("=" * 60)
            logger.info("AUTO TRADER STARTED (API-based mode)")
            logger.info(f"Mode: DRY RUN (paper trading)")
            logger.info(f"Assets: {assets}")
            logger.info(f"Loop interval: 60s")
            logger.info("=" * 60)
            
            while True:
                try:
                    cycle_count += 1
                    logger.info(f"\n=== Trading Cycle {cycle_count} ===")
                    
                    # Get current prices
                    async with httpx.AsyncClient(timeout=10.0) as client:
                        try:
                            response = await client.get("http://127.0.0.1:8000/api/v1/market/prices")
                            prices = response.json() if response.status_code == 200 else {}
                        except:
                            prices = {}
                    
                    # Generate random trading decisions
                    num_trades = random.randint(1, 3)
                    trades_executed = 0
                    
                    for _ in range(num_trades):
                        symbol = random.choice(assets)
                        side = random.choice(["BUY", "SELL"])
                        
                        # Get approximate quantity based on price
                        price = prices.get(symbol, {}).get("price", 1000)
                        quantity = round(random.uniform(0.01, 0.5), 4)
                        
                        # Execute trade
                        result = await execute_trade_via_api(symbol, side, quantity)
                        if result:
                            trades_executed += 1
                            logger.info(f"Executed: {side} {quantity} {symbol}")
                        
                        await asyncio.sleep(1)  # Small delay between trades
                    
                    logger.info(f"Cycle {cycle_count} completed: {trades_executed} trades")
                    
                    # Wait for next cycle (60 seconds)
                    await asyncio.sleep(60)
                    
                except Exception as e:
                    logger.exception(f"Error in trading cycle: {e}")
                    await asyncio.sleep(60)
        
        # Start auto trader in daemon thread using new event loop
        def start_background_trader():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(run_auto_trader_async())
            except Exception as e:
                logger.error(f"Auto trader error: {e}")
        
        trader_thread = threading.Thread(target=start_background_trader, daemon=True)
        trader_thread.start()
        logger.info("Auto trader thread started (API-based)")
    except Exception as e:
        logger.warning(f"Could not start auto trader: {e}")
    
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
    allow_origin_regex=r"https://.*\.vercel\.app|https://.*\.ngrok-free\.app",
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
    health,
    prefix=settings.api_prefix,
    tags=["Health"]
)

app.include_router(
    orders,
    prefix=f"{settings.api_prefix}/orders",
    tags=["Orders"]
)

app.include_router(
    portfolio,
    prefix=f"{settings.api_prefix}/portfolio",
    tags=["Portfolio"]
)

app.include_router(
    strategy,
    prefix=f"{settings.api_prefix}/strategy",
    tags=["Strategy"]
)

app.include_router(
    market,
    prefix=f"{settings.api_prefix}/market",
    tags=["Market"]
)

app.include_router(
    waitlist,
    prefix=f"{settings.api_prefix}",
    tags=["Waitlist"]
)

app.include_router(
    cache,
    prefix=f"{settings.api_prefix}/cache",
    tags=["Cache"]
)

app.include_router(
    news,
    prefix=f"{settings.api_prefix}/news",
    tags=["News"]
)

app.include_router(
    auth,
    prefix=f"{settings.api_prefix}/auth",
    tags=["Authentication"]
)

# Include payments router
app.include_router(
    payments,
    prefix=f"{settings.api_prefix}/payments",
    tags=["Payments"]
)

# Include risk router
app.include_router(
    risk,
    prefix=f"{settings.api_prefix}/risk",
    tags=["Risk"]
)


@app.on_event("startup")
async def startup_event():
    """Log startup information."""
    logger.info("Application starting up...")
    
    # Log all registered routes
    for route in app.routes:
        if hasattr(route, 'path'):
            logger.info(f"Registered route: {route.path}")


# Serve frontend static files
FRONTEND_DIR = Path(__file__).parent.parent / "frontend" / "dist"
logger.info(f"FRONTEND_DIR: {FRONTEND_DIR}, exists: {FRONTEND_DIR.exists()}")

if FRONTEND_DIR.exists():
    # Mount static assets first (lower priority than explicit routes)
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR / "assets")), name="assets")


# Serve frontend for SPA routes
@app.get("/")
@app.get("/dashboard")
@app.get("/portfolio")
@app.get("/market")
@app.get("/orders")
@app.get("/login")
@app.get("/payment")
async def serve_spa(request: Request):
    """Serve frontend SPA for client-side routes."""
    logger.info(f"serve_spa called: {request.url}")
    
    # First check the relative path
    frontend_file = FRONTEND_DIR / "index.html"
    if frontend_file.exists():
        logger.info(f"Serving from relative path: {frontend_file}")
        return FileResponse(str(frontend_file))
    
    # Try to serve from absolute path
    frontend_file = Path("c:/ai-trading-system/frontend/dist/index.html")
    if frontend_file.exists():
        logger.info(f"Serving from absolute path: {frontend_file}")
        return FileResponse(str(frontend_file))
    
    logger.warning("Frontend not built - index.html not found")
    return {"error": "Frontend not built"}


# Favicon handler - return 204 No Content to avoid 404
@app.route("/favicon.ico", methods=["GET", "HEAD"])
async def favicon(request: Request):
    """Serve favicon or return empty response."""
    return Response(status_code=204)


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
