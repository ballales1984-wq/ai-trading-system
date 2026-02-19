"""
Portfolio Routes
================
REST API for portfolio management and positions.
"""

from datetime import datetime
from typing import List, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, status, Query
from pydantic import BaseModel, Field

from app.core.data_adapter import get_data_adapter


router = APIRouter()


# ============================================================================
# DATA MODELS
# ============================================================================

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

# Sample portfolio data
portfolio_data = {
    "cash_balance": 500000.0,
    "positions": [
        {
            "position_id": str(uuid4()),
            "symbol": "BTCUSDT",
            "side": "LONG",
            "quantity": 1.5,
            "entry_price": 42000.0,
            "current_price": 43500.0,
            "market_value": 65250.0,
            "unrealized_pnl": 2250.0,
            "realized_pnl": 0.0,
            "leverage": 1.0,
            "margin_used": 31500.0,
            "opened_at": "2026-02-15T10:00:00",
            "updated_at": "2026-02-18T22:00:00",
        },
        {
            "position_id": str(uuid4()),
            "symbol": "ETHUSDT",
            "side": "LONG",
            "quantity": 15.0,
            "entry_price": 2200.0,
            "current_price": 2350.0,
            "market_value": 35250.0,
            "unrealized_pnl": 2250.0,
            "realized_pnl": 0.0,
            "leverage": 1.0,
            "margin_used": 16500.0,
            "opened_at": "2026-02-16T14:30:00",
            "updated_at": "2026-02-18T22:00:00",
        },
    ]
}


# ============================================================================
# ROUTES
# ============================================================================

@router.get("/summary", response_model=PortfolioSummary)
async def get_portfolio_summary() -> PortfolioSummary:
    """
    Get portfolio summary.
    
    Returns total portfolio value, cash, positions, and P&L.
    """
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
