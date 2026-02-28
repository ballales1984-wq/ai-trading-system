"""
Portfolio Routes
================
REST API for portfolio management and positions.
"""

import os
import random
from datetime import datetime
from typing import List, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, status, Query
from pydantic import BaseModel, Field

from app.core.data_adapter import get_data_adapter
from app.api.mock_data import (
    DEMO_MODE,
    get_portfolio_summary as mock_portfolio_summary,
    get_positions as mock_positions,
    get_performance_metrics as mock_performance,
    get_portfolio_history as mock_history,
    get_allocation as mock_allocation,
)


router = APIRouter()

# ============================================================================
# CONFIGURABLE PORTFOLIO BALANCE
# ============================================================================

# Get initial balance from environment variable or use default
PAPER_INITIAL_BALANCE = float(os.getenv("PAPER_INITIAL_BALANCE", "500000"))

# In-memory portfolio store (can be updated via API)
portfolio_data = {
    "cash_balance": PAPER_INITIAL_BALANCE,
    "positions": [],
    "initialized": False,
}


def _initialize_default_positions():
    """Initialize default positions based on initial balance."""
    if not portfolio_data["initialized"]:
        # Create default positions based on balance
        cash = portfolio_data["cash_balance"]
        
        # BTC position (30% of portfolio)
        btc_value = cash * 0.30
        btc_price = 66461.78  # Current BTC price
        btc_qty = btc_value / btc_price
        
        # ETH position (20% of portfolio)
        eth_value = cash * 0.20
        eth_price = 3333.52  # Current ETH price
        eth_qty = eth_value / eth_price
        
        # SOL position (10% of portfolio)
        sol_value = cash * 0.10
        sol_price = 152.12
        sol_qty = sol_value / sol_price
        
        portfolio_data["positions"] = [
            {
                "position_id": str(uuid4()),
                "symbol": "BTCUSDT",
                "side": "LONG",
                "quantity": round(btc_qty, 4),
                "entry_price": btc_price * 0.95,  # Buy at 5% discount
                "current_price": btc_price,
                "market_value": btc_value,
                "unrealized_pnl": btc_value * 0.05,
                "realized_pnl": 0.0,
                "leverage": 1.0,
                "margin_used": btc_value,
                "opened_at": "2026-02-15T10:00:00",
                "updated_at": "2026-02-28T18:00:00",
            },
            {
                "position_id": str(uuid4()),
                "symbol": "ETHUSDT",
                "side": "LONG",
                "quantity": round(eth_qty, 4),
                "entry_price": eth_price * 0.95,
                "current_price": eth_price,
                "market_value": eth_value,
                "unrealized_pnl": eth_value * 0.05,
                "realized_pnl": 0.0,
                "leverage": 1.0,
                "margin_used": eth_value,
                "opened_at": "2026-02-16T14:30:00",
                "updated_at": "2026-02-28T18:00:00",
            },
            {
                "position_id": str(uuid4()),
                "symbol": "SOLUSDT",
                "side": "LONG",
                "quantity": round(sol_qty, 4),
                "entry_price": sol_price * 0.95,
                "current_price": sol_price,
                "market_value": sol_value,
                "unrealized_pnl": sol_value * 0.05,
                "realized_pnl": 0.0,
                "leverage": 1.0,
                "margin_used": sol_value,
                "opened_at": "2026-02-17T09:00:00",
                "updated_at": "2026-02-28T18:00:00",
            },
        ]
        portfolio_data["initialized"] = True


# Initialize default positions
_initialize_default_positions()

class Position(BaseModel):
    """Portfolio position model."""
    position_id: str = Field(default_factory=lambda: str(uuid4()))
    symbol: str
    side: str = Field(..., description="LONG or SHORT")
    quantity: float
    entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float = 0.0
    leverage: float = 1.0
    margin_used: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    opened_at: datetime
    updated_at: datetime


class PortfolioSummary(BaseModel):
    """Portfolio summary model."""
    total_value: float
    cash_balance: float
    market_value: float
    total_pnl: float
    unrealized_pnl: float
    realized_pnl: float
    daily_pnl: float
    daily_return_pct: float
    total_return_pct: float
    leverage: float
    buying_power: float
    num_positions: int


class PerformanceMetrics(BaseModel):
    """Performance metrics model."""
    total_return: float
    total_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    num_trades: int
    num_winning_trades: int
    num_losing_trades: int


class HistoryEntry(BaseModel):
    """Portfolio history entry."""
    date: str
    value: float
    daily_return: float


class PortfolioHistory(BaseModel):
    """Portfolio history response."""
    history: List[HistoryEntry]


# ============================================================================
# IN-MEMORY PORTFOLIO STORE
# ============================================================================

# Note: portfolio_data is already initialized above with _initialize_default_positions()
# Do NOT redefine it here or it will reset the positions


# ============================================================================
# ROUTES
# ============================================================================

# Endpoint to update paper trading balance
@router.post("/balance", status_code=200)
async def update_balance(new_balance: float = Query(..., ge=1000, le=100000000, description="New initial balance for paper trading")):
    """
    Update the paper trading initial balance.
    This allows changing the portfolio value without restarting the server.
    """
    global portfolio_data
    
    # Update the balance
    portfolio_data["cash_balance"] = new_balance
    portfolio_data["initialized"] = False  # Reinitialize positions
    
    # Reinitialize with new balance
    _initialize_default_positions()
    
    return {
        "success": True,
        "message": f"Balance updated to ${new_balance:,.2f}",
        "new_balance": new_balance,
        "positions_count": len(portfolio_data["positions"])
    }


@router.get("/balance")
async def get_balance():
    """Get current paper trading balance."""
    return {
        "current_balance": portfolio_data["cash_balance"],
        "positions_count": len(portfolio_data["positions"])
    }


@router.get("/summary", response_model=PortfolioSummary)
async def get_portfolio_summary() -> PortfolioSummary:
    """
    Get portfolio summary.
    
    Returns total portfolio value, cash, positions, and P&L.
    """
    # When DEMO_MODE is true but we have custom portfolio_data (via /balance endpoint)
    # use the custom portfolio_data instead of mock data
    if DEMO_MODE and len(portfolio_data.get("positions", [])) > 0:
        # Use our configurable portfolio
        positions = portfolio_data["positions"]
        cash = portfolio_data["cash_balance"]
        market_value = sum(p.get("market_value", 0) for p in positions)
        unrealized_pnl = sum(p.get("unrealized_pnl", 0) for p in positions)
        
        total_value = cash + market_value
        total_pnl = unrealized_pnl
        daily_pnl = total_value * 0.02  # 2% daily assumption
        daily_return_pct = 2.0
        total_return_pct = (total_pnl / total_value) * 100 if total_value > 0 else 0
        
        return PortfolioSummary(
            total_value=total_value,
            cash_balance=cash,
            market_value=market_value,
            total_pnl=total_pnl,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=0.0,
            daily_pnl=daily_pnl,
            daily_return_pct=daily_return_pct,
            total_return_pct=total_return_pct,
            leverage=1.0,
            buying_power=total_value,
            num_positions=len(positions)
        )
    
    # Use mock data if demo mode is enabled
    if DEMO_MODE:
        data = mock_portfolio_summary()
        return PortfolioSummary(
            total_value=data["total_value"],
            cash_balance=data["cash"],
            market_value=data["invested"],
            total_pnl=data["unrealized_pnl"],
            unrealized_pnl=data["unrealized_pnl"],
            realized_pnl=0.0,
            daily_pnl=data["daily_pnl"],
            daily_return_pct=data["daily_return_pct"],
            total_return_pct=data["total_return_pct"],
            leverage=1.0,
            buying_power=data["total_value"],
            num_positions=data["num_positions"]
        )
    
    # Try to get real data first
    adapter = get_data_adapter()
    real_data = adapter.get_portfolio_summary()
    
    # Use real data if available, otherwise fallback to mock
    if real_data.get('total_value', 0) > 0 or real_data.get('num_positions', 0) > 0:
        positions = adapter.get_positions()
        cash = real_data.get('cash_balance', 0)
    else:
        positions = portfolio_data["positions"]
        cash = portfolio_data["cash_balance"]
    
    # Use real data if available
    if real_data.get('total_value', 0) > 0:
        total_value = real_data.get('total_value', cash + sum(p.get('market_value', 0) for p in positions))
        total_pnl = real_data.get('total_pnl', 0)
        unrealized_pnl = real_data.get('unrealized_pnl', sum(p.get('unrealized_pnl', 0) for p in positions))
        realized_pnl = real_data.get('realized_pnl', sum(p.get('realized_pnl', 0) for p in positions))
        daily_pnl = real_data.get('daily_pnl', 0)
        daily_return_pct = real_data.get('daily_return_pct', 0)
        total_return_pct = real_data.get('total_return_pct', 0)
        market_value = real_data.get('market_value', sum(p.get('market_value', 0) for p in positions))
        margin_used = sum(p.get('margin_used', 0) for p in positions)
    else:
        # Fallback to calculated values
        market_value = sum(p.get("market_value", 0) for p in positions)
        unrealized_pnl = sum(p.get("unrealized_pnl", 0) for p in positions)
        realized_pnl = sum(p.get("realized_pnl", 0) for p in positions)
        margin_used = sum(p.get("margin_used", 0) for p in positions)
        
        total_value = cash + market_value
        total_pnl = unrealized_pnl + realized_pnl
        
        # Assume starting capital of 1M
        starting_capital = 1000000.0
        daily_pnl = total_pnl * 0.1  # Simulated daily P&L
        daily_return_pct = (daily_pnl / starting_capital) * 100
        total_return_pct = ((total_value - starting_capital) / starting_capital) * 100
    
    return PortfolioSummary(
        total_value=total_value,
        cash_balance=cash,
        market_value=market_value,
        total_pnl=total_pnl,
        unrealized_pnl=unrealized_pnl,
        realized_pnl=realized_pnl,
        daily_pnl=daily_pnl,
        daily_return_pct=daily_return_pct,
        total_return_pct=total_return_pct,
        leverage=1.0,
        buying_power=cash + margin_used,
        num_positions=len(positions)
    )


@router.get("/positions", response_model=List[Position])
async def list_positions(
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    side: Optional[str] = Query(None, description="Filter by side: LONG, SHORT"),
) -> List[Position]:
    """
    List all open positions.
    """
    # When DEMO_MODE is true but we have custom portfolio_data, use it
    if DEMO_MODE and len(portfolio_data.get("positions", [])) > 0:
        positions = portfolio_data["positions"]
        if symbol:
            positions = [p for p in positions if p["symbol"] == symbol]
        if side:
            positions = [p for p in positions if p["side"] == side]
        return [Position(
            position_id=p["position_id"],
            symbol=p["symbol"],
            side=p["side"],
            quantity=p["quantity"],
            entry_price=p["entry_price"],
            current_price=p["current_price"],
            market_value=p["market_value"],
            unrealized_pnl=p.get("unrealized_pnl", 0),
            realized_pnl=0.0,
            leverage=1.0,
            margin_used=p.get("margin_used", p["market_value"]),
            opened_at=datetime.fromisoformat(p["opened_at"]),
            updated_at=datetime.fromisoformat(p["updated_at"]),
        ) for p in positions]
    
    # Use mock data if demo mode is enabled
    if DEMO_MODE:
        positions = mock_positions()
        if symbol:
            positions = [p for p in positions if p["symbol"] == symbol]
        if side:
            positions = [p for p in positions if p["side"] == side]
        return [Position(
            position_id=str(uuid4()),
            symbol=p["symbol"],
            side=p["side"],
            quantity=p["quantity"],
            entry_price=p["entry_price"],
            current_price=p["current_price"],
            market_value=p["market_value"],
            unrealized_pnl=p["unrealized_pnl"],
            realized_pnl=0.0,
            leverage=1.0,
            margin_used=p["market_value"],
            opened_at=datetime.fromisoformat(p["opened_at"]),
            updated_at=datetime.utcnow(),
        ) for p in positions]
    
    # Try to get real positions first
    adapter = get_data_adapter()
    real_positions = adapter.get_positions()
    
    if real_positions:
        positions = real_positions
    else:
        positions = portfolio_data["positions"]
    
    if symbol:
        positions = [p for p in positions if p["symbol"] == symbol]
    if side:
        positions = [p for p in positions if p["side"] == side]
    
    return [Position(**p) for p in positions]


@router.get("/positions/{position_id}", response_model=Position)
async def get_position(position_id: str) -> Position:
    """
    Get position by ID.
    """
    for p in portfolio_data["positions"]:
        if p["position_id"] == position_id:
            return Position(**p)
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Position {position_id} not found"
    )


@router.get("/performance", response_model=PerformanceMetrics)
async def get_performance_metrics() -> PerformanceMetrics:
    """
    Get portfolio performance metrics.
    
    Returns Sharpe ratio, max drawdown, win rate, etc.
    """
    # Use mock data if demo mode is enabled
    if DEMO_MODE:
        data = mock_performance()
        return PerformanceMetrics(
            total_return=data["total_return_pct"] * 1000,  # Approximate
            total_return_pct=data["total_return_pct"],
            sharpe_ratio=data["sharpe_ratio"],
            sortino_ratio=data["sortino_ratio"],
            max_drawdown=data["max_drawdown_pct"] * 1000,  # Approximate
            max_drawdown_pct=data["max_drawdown_pct"],
            calmar_ratio=data["calmar_ratio"],
            win_rate=data["win_rate"],
            profit_factor=data["profit_factor"],
            avg_win=data["avg_win"],
            avg_loss=data["avg_loss"],
            num_trades=data["total_trades"],
            num_winning_trades=data["winning_trades"],
            num_losing_trades=data["losing_trades"],
        )
    
    # Simulated performance data
    return PerformanceMetrics(
        total_return=50000.0,
        total_return_pct=5.0,
        sharpe_ratio=1.85,
        sortino_ratio=2.34,
        max_drawdown=-8500.0,
        max_drawdown_pct=-8.5,
        calmar_ratio=1.2,
        win_rate=0.62,
        profit_factor=1.75,
        avg_win=2500.0,
        avg_loss=-1200.0,
        num_trades=45,
        num_winning_trades=28,
        num_losing_trades=17,
    )


@router.get("/allocation")
async def get_allocation() -> dict:
    """
    Get portfolio allocation by asset class and sector.
    """
    # Use mock data if demo mode is enabled
    if DEMO_MODE:
        data = mock_allocation()
        return {
            "by_asset_class": {
                "crypto": 100.0,
            },
            "by_sector": {
                "crypto": 100.0,
            },
            "by_symbol": {item["symbol"]: item["percentage"] for item in data["by_asset"]},
            "total": data["total"],
            "cash": data["cash"],
        }
    
    return {
        "by_asset_class": {
            "crypto": 85.0,
            "forex": 10.0,
            "stocks": 5.0,
        },
        "by_sector": {
            "crypto": 85.0,
            "technology": 5.0,
            "financial": 10.0,
        },
        "by_symbol": {
            "BTCUSDT": 65.0,
            "ETHUSDT": 20.0,
            "EURUSD": 10.0,
            "AAPL": 5.0,
        }
    }


@router.get("/history", response_model=PortfolioHistory)
async def get_portfolio_history(
    days: int = Query(default=30, ge=1, le=365, description="Number of days to retrieve history"),
) -> PortfolioHistory:
    """
    Get portfolio value history.
    
    Returns historical portfolio values for the specified number of days.
    Default is 30 days if not specified.
    """
    # Use mock data if demo mode is enabled
    if DEMO_MODE:
        data = mock_history(days)
        history = [HistoryEntry(
            date=h["date"],
            value=h["value"],
            daily_return=h["daily_return"],
        ) for h in data["history"]]
        return PortfolioHistory(history=history)
    
    # Try to get real history first
    adapter = get_data_adapter()
    real_history = adapter.get_portfolio_history(days=days)
    
    if real_history:
        history = [HistoryEntry(**h) for h in real_history]
    else:
        # Fallback to simulated data
        import random
        
        # Ensure days is within valid range
        days = max(1, min(365, days))
        
        history = []
        base_value = 1000000.0
        
        for i in range(days):
            date = f"2026-01-{i+1:02d}" if i < 48 else f"2026-02-{i-47:02d}"
            daily_return = random.uniform(-0.02, 0.025)
            base_value *= (1 + daily_return)
            
            history.append(HistoryEntry(
                date=date,
                value=base_value,
                daily_return=daily_return * 100,
            ))
    
    return PortfolioHistory(history=history)
