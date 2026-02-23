"""
Vercel Python entrypoint for API routes.

Important:
- Export the ASGI app as `app` (no Mangum handler) for Vercel Python runtime.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


app = FastAPI(
    title="AI Trading System API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Position(BaseModel):
    position_id: str
    symbol: str
    side: str
    quantity: float
    entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float
    leverage: float
    margin_used: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    opened_at: str
    updated_at: str


class OrderCreate(BaseModel):
    symbol: str
    side: str
    order_type: str = "MARKET"
    quantity: float = Field(..., gt=0)
    price: Optional[float] = None
    stop_price: Optional[float] = None
    strategy_id: Optional[str] = None
    broker: str = "demo"


_positions: List[Position] = [
    Position(
        position_id=str(uuid4()),
        symbol="BTCUSDT",
        side="LONG",
        quantity=1.5,
        entry_price=42000.0,
        current_price=43500.0,
        market_value=65250.0,
        unrealized_pnl=2250.0,
        realized_pnl=0.0,
        leverage=1.0,
        margin_used=31500.0,
        opened_at="2026-02-15T10:00:00",
        updated_at=datetime.utcnow().isoformat(),
    ),
    Position(
        position_id=str(uuid4()),
        symbol="ETHUSDT",
        side="LONG",
        quantity=15.0,
        entry_price=2200.0,
        current_price=2350.0,
        market_value=35250.0,
        unrealized_pnl=2250.0,
        realized_pnl=0.0,
        leverage=1.0,
        margin_used=16500.0,
        opened_at="2026-02-16T14:30:00",
        updated_at=datetime.utcnow().isoformat(),
    ),
]

_orders: Dict[str, Dict[str, Any]] = {}
_waitlist: List[Dict[str, Any]] = []


@app.get("/")
async def root() -> Dict[str, str]:
    return {"message": "AI Trading System API", "status": "running"}


@app.get("/health")
async def health_root() -> Dict[str, str]:
    return {"status": "healthy", "service": "ai-trading-system"}


@app.get("/api/v1/health")
async def health_api() -> Dict[str, str]:
    return {"status": "healthy", "service": "ai-trading-system"}


@app.get("/api/v1/portfolio/summary")
async def portfolio_summary() -> Dict[str, Any]:
    cash = 500000.0
    market_value = sum(p.market_value for p in _positions)
    unrealized = sum(p.unrealized_pnl for p in _positions)
    realized = sum(p.realized_pnl for p in _positions)
    total_value = cash + market_value
    total_pnl = unrealized + realized
    return {
        "total_value": total_value,
        "cash_balance": cash,
        "market_value": market_value,
        "total_pnl": total_pnl,
        "unrealized_pnl": unrealized,
        "realized_pnl": realized,
        "daily_pnl": 250.0,
        "daily_return_pct": 0.025,
        "total_return_pct": 0.5,
        "leverage": 1.0,
        "buying_power": cash,
        "num_positions": len(_positions),
    }


@app.get("/api/v1/portfolio/positions")
async def portfolio_positions(symbol: Optional[str] = Query(None)) -> List[Dict[str, Any]]:
    positions = _positions
    if symbol:
        positions = [p for p in positions if p.symbol == symbol]
    return [p.model_dump() for p in positions]


@app.get("/api/v1/portfolio/performance")
async def portfolio_performance() -> Dict[str, Any]:
    return {
        "total_return": 50000.0,
        "total_return_pct": 5.0,
        "sharpe_ratio": 1.85,
        "sortino_ratio": 2.34,
        "max_drawdown": -8500.0,
        "max_drawdown_pct": -8.5,
        "calmar_ratio": 1.2,
        "win_rate": 0.62,
        "profit_factor": 1.75,
        "avg_win": 2500.0,
        "avg_loss": -1200.0,
        "num_trades": 45,
        "num_winning_trades": 28,
        "num_losing_trades": 17,
    }


@app.get("/api/v1/portfolio/allocation")
async def portfolio_allocation() -> Dict[str, Any]:
    return {
        "by_asset_class": {"crypto": 100.0},
        "by_sector": {"crypto": 100.0},
        "by_symbol": {"BTCUSDT": 65.0, "ETHUSDT": 35.0},
    }


@app.get("/api/v1/portfolio/history")
async def portfolio_history(days: int = Query(30, ge=1, le=365)) -> Dict[str, Any]:
    base = 1_000_000.0
    history = []
    for i in range(days):
        date = (datetime.utcnow() - timedelta(days=(days - i))).strftime("%Y-%m-%d")
        value = base + (i * 850.0)
        history.append({"date": date, "value": value, "daily_return": 0.08})
    return {"history": history}


@app.get("/api/v1/market/prices")
async def market_prices() -> Dict[str, Any]:
    now = datetime.utcnow().isoformat()
    return {
        "timestamp": now,
        "markets": [
            {
                "symbol": "BTCUSDT",
                "price": 45000.0,
                "change_24h": 1125.0,
                "change_pct_24h": 2.5,
                "high_24h": 45800.0,
                "low_24h": 43800.0,
                "volume_24h": 1000000.0,
                "timestamp": now,
            },
            {
                "symbol": "ETHUSDT",
                "price": 3000.0,
                "change_24h": 54.0,
                "change_pct_24h": 1.8,
                "high_24h": 3050.0,
                "low_24h": 2920.0,
                "volume_24h": 800000.0,
                "timestamp": now,
            },
        ],
    }


@app.get("/api/v1/market/price/{symbol}")
async def market_price(symbol: str) -> Dict[str, Any]:
    now = datetime.utcnow().isoformat()
    price = 45000.0 if symbol.upper().startswith("BTC") else 3000.0
    return {
        "symbol": symbol.upper(),
        "price": price,
        "change_24h": price * 0.01,
        "change_pct_24h": 1.0,
        "high_24h": price * 1.02,
        "low_24h": price * 0.98,
        "volume_24h": 500000.0,
        "timestamp": now,
    }


@app.get("/api/v1/market/candles/{symbol}")
async def market_candles(
    symbol: str, interval: str = Query("1h"), limit: int = Query(100, ge=1, le=1000)
) -> List[Dict[str, Any]]:
    _ = interval
    base = 45000.0 if symbol.upper().startswith("BTC") else 3000.0
    candles = []
    for i in range(limit):
        ts = (datetime.utcnow() - timedelta(hours=(limit - i))).isoformat()
        candles.append(
            {
                "timestamp": ts,
                "open": base,
                "high": base * 1.01,
                "low": base * 0.99,
                "close": base * 1.002,
                "volume": 1000.0 + i,
            }
        )
    return candles


@app.get("/api/v1/market/orderbook/{symbol}")
async def market_orderbook(symbol: str) -> Dict[str, Any]:
    base = 45000.0 if symbol.upper().startswith("BTC") else 3000.0
    return {
        "symbol": symbol.upper(),
        "bids": [[base - i, 1.0 + i * 0.1] for i in range(1, 11)],
        "asks": [[base + i, 1.0 + i * 0.1] for i in range(1, 11)],
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/api/v1/orders")
async def list_orders() -> List[Dict[str, Any]]:
    return list(_orders.values())


@app.get("/api/v1/orders/{order_id}")
async def get_order(order_id: str) -> Dict[str, Any]:
    order = _orders.get(order_id)
    if not order:
        raise HTTPException(status_code=404, detail=f"Order {order_id} not found")
    return order


@app.post("/api/v1/orders", status_code=status.HTTP_201_CREATED)
async def create_order(order: OrderCreate) -> Dict[str, Any]:
    now = datetime.utcnow().isoformat()
    order_id = str(uuid4())
    row = {
        "order_id": order_id,
        "symbol": order.symbol,
        "side": order.side,
        "order_type": order.order_type,
        "quantity": order.quantity,
        "price": order.price,
        "stop_price": order.stop_price,
        "status": "PENDING",
        "filled_quantity": 0.0,
        "average_price": None,
        "commission": 0.0,
        "created_at": now,
        "updated_at": now,
        "strategy_id": order.strategy_id,
        "broker": order.broker,
        "error_message": None,
    }
    _orders[order_id] = row
    return row


@app.delete("/api/v1/orders/{order_id}", status_code=status.HTTP_204_NO_CONTENT)
async def cancel_order(order_id: str) -> None:
    if order_id not in _orders:
        raise HTTPException(status_code=404, detail=f"Order {order_id} not found")
    _orders[order_id]["status"] = "CANCELLED"
    _orders[order_id]["updated_at"] = datetime.utcnow().isoformat()


@app.post("/api/v1/orders/{order_id}/execute")
async def execute_order(order_id: str) -> Dict[str, Any]:
    if order_id not in _orders:
        raise HTTPException(status_code=404, detail=f"Order {order_id} not found")
    _orders[order_id]["status"] = "FILLED"
    _orders[order_id]["filled_quantity"] = _orders[order_id]["quantity"]
    _orders[order_id]["average_price"] = _orders[order_id]["price"] or 0.0
    _orders[order_id]["updated_at"] = datetime.utcnow().isoformat()
    return _orders[order_id]


@app.post("/api/v1/waitlist")
async def join_waitlist(request: Request) -> Dict[str, Any]:
    try:
        data = await request.json()
        email = data.get("email")
        source = data.get("source", "landing_page")
        if not email:
            return JSONResponse(
                status_code=400,
                content={"success": False, "message": "Email is required"},
            )

        for entry in _waitlist:
            if entry["email"] == email:
                return {
                    "success": True,
                    "message": "You're already on the waitlist!",
                    "position": entry["position"],
                }

        position = len(_waitlist) + 1
        _waitlist.append({"email": email, "source": source, "position": position})
        return {
            "success": True,
            "message": "Successfully joined the waitlist!",
            "position": position,
        }
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": str(exc)},
        )


@app.get("/api/v1/waitlist/count")
async def waitlist_count() -> Dict[str, int]:
    return {"count": len(_waitlist)}
