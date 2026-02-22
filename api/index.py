"""
Vercel Serverless Entry Point
=============================
Simplified entry point for Vercel serverless functions.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from mangum import Mangum

# Create a minimal FastAPI app for Vercel
app = FastAPI(
    title="AI Trading System API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "AI Trading System API", "status": "running"}


@app.get("/api/v1/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "ai-trading-system"}


@app.get("/api/v1/portfolio/summary")
async def portfolio_summary():
    """Portfolio summary endpoint."""
    return {
        "total_value": 1000000.00,
        "cash": 500000.00,
        "invested": 500000.00,
        "pnl": 15000.00,
        "pnl_percent": 1.5
    }


@app.get("/api/v1/market/prices")
async def market_prices():
    """Market prices endpoint."""
    return {
        "prices": [
            {"symbol": "BTC/USDT", "price": 45000.00, "change_24h": 2.5},
            {"symbol": "ETH/USDT", "price": 3000.00, "change_24h": 1.8},
            {"symbol": "SOL/USDT", "price": 100.00, "change_24h": -0.5},
        ]
    }


@app.get("/api/v1/orders")
async def orders():
    """Orders endpoint."""
    return {"orders": []}


@app.get("/api/v1/strategy/performance")
async def strategy_performance():
    """Strategy performance endpoint."""
    return {
        "strategies": [
            {"name": "Momentum", "return": 12.5, "trades": 45},
            {"name": "Mean Reversion", "return": 8.3, "trades": 32},
        ]
    }


@app.get("/api/v1/risk/metrics")
async def risk_metrics():
    """Risk metrics endpoint."""
    return {
        "var_95": 0.02,
        "cvar_95": 0.03,
        "sharpe_ratio": 1.5,
        "max_drawdown": 0.08
    }


# In-memory waitlist storage
_waitlist = []


@app.post("/api/v1/waitlist")
async def join_waitlist(request: Request):
    """Add email to waitlist."""
    try:
        data = await request.json()
        email = data.get("email")
        source = data.get("source", "landing_page")
        
        if not email:
            return JSONResponse(
                status_code=400,
                content={"success": False, "message": "Email is required"}
            )
        
        # Check if email already exists
        for entry in _waitlist:
            if entry["email"] == email:
                return {
                    "success": True,
                    "message": "You're already on the waitlist!",
                    "position": entry["position"]
                }
        
        # Add new entry
        position = len(_waitlist) + 1
        _waitlist.append({
            "email": email,
            "source": source,
            "position": position
        })
        
        return {
            "success": True,
            "message": "Successfully joined the waitlist!",
            "position": position
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": str(e)}
        )


@app.get("/api/v1/waitlist/count")
async def waitlist_count():
    """Get waitlist count."""
    return {"count": len(_waitlist)}


# Vercel requires the handler to be named 'handler'
# Mangum wraps FastAPI as an ASGI handler for serverless
handler = Mangum(app)
