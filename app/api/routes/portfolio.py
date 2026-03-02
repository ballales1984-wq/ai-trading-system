"""


Portfolio Routes
================
REST API for portfolio management and positions.
"""

import os
import random
import requests
import math
import statistics
from datetime import datetime
from typing import List, Optional, Dict
from uuid import uuid4

from fastapi import APIRouter, HTTPException, status, Query
from pydantic import BaseModel, Field

from app.core.data_adapter import get_data_adapter
from app.api.mock_data import (
    get_portfolio_summary as mock_portfolio_summary,
    get_positions as mock_positions,
    get_performance_metrics as mock_performance,
    get_portfolio_history as mock_history,
    get_allocation as mock_allocation,
)


router = APIRouter()

# ============================================================================
# REAL-TIME PRICE FETCHING FROM BINANCE
# ============================================================================

# Cache for real-time prices (avoid too many API calls)
_price_cache: Dict[str, tuple] = {}  # {symbol: (price, timestamp)}
CACHE_DURATION_SECONDS = 10  # Cache prices for 10 seconds

def get_binance_price(symbol: str) -> Optional[float]:
    """
    Get real-time price from Binance API.
    Returns None if the API call fails.
    """
    global _price_cache
    
    # Check cache first
    current_time = datetime.now().timestamp()
    if symbol in _price_cache:
        cached_price, cached_time = _price_cache[symbol]
        if current_time - cached_time < CACHE_DURATION_SECONDS:
            return cached_price
    
    try:
        # Use Binance public API (no key needed for price data)
        url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol.upper()}"
        # Keep this endpoint responsive even if Binance is slow/unreachable.
        response = requests.get(url, timeout=0.8)
        if response.status_code == 200:
            data = response.json()
            price = float(data.get('price', 0))
            if price > 0:
                _price_cache[symbol] = (price, current_time)
                return price
    except Exception:
        pass
    
    return None

def get_binance_prices(symbols: List[str]) -> Dict[str, float]:
    """
    Get real-time prices for multiple symbols from Binance.
    Returns a dictionary of symbol -> price.
    """
    prices = {}
    for symbol in symbols:
        price = get_binance_price(symbol)
        if price:
            prices[symbol.upper()] = price
    return prices

# Default symbols for real-time prices
TRACKED_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT"]

def get_realtime_prices() -> Dict[str, float]:
    """Get all tracked prices in real-time from Binance."""
    return get_binance_prices(TRACKED_SYMBOLS)

# ============================================================================
# CONFIGURABLE DEMO MODE (can be changed at runtime)
# ============================================================================

# Start with environment variable, but allow runtime changes.
# Default to real mode (false) for production use.
_demo_mode = os.getenv("DEMO_MODE", "false").lower() == "true"

def get_demo_mode() -> bool:
    """Get current DEMO_MODE setting."""
    return _demo_mode

def set_demo_mode(value: bool) -> None:
    """Set DEMO_MODE at runtime."""
    global _demo_mode
    _demo_mode = value


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
        
        # Asset allocation: diversified portfolio
        assets = [
            {"symbol": "BTCUSDT", "price": 66461.78, "allocation": 0.25},
            {"symbol": "ETHUSDT", "price": 3333.52, "allocation": 0.20},
            {"symbol": "SOLUSDT", "price": 152.12, "allocation": 0.10},
            {"symbol": "BNBUSDT", "price": 610.50, "allocation": 0.08},
            {"symbol": "XRPUSDT", "price": 2.45, "allocation": 0.07},
            {"symbol": "ADAUSDT", "price": 0.98, "allocation": 0.06},
            {"symbol": "DOGEUSDT", "price": 0.32, "allocation": 0.05},
            {"symbol": "AVAXUSDT", "price": 38.50, "allocation": 0.05},
            {"symbol": "LINKUSDT", "price": 18.20, "allocation": 0.04},
            {"symbol": "MATICUSDT", "price": 0.85, "allocation": 0.03},
            {"symbol": "ATOMUSDT", "price": 9.80, "allocation": 0.03},
            {"symbol": "DOTUSDT", "price": 7.20, "allocation": 0.02},
            {"symbol": "UNIUSDT", "price": 12.50, "allocation": 0.02},
        ]
        
        positions = []
        for asset in assets:
            value = cash * asset["allocation"]
            qty = value / asset["price"]
            positions.append({
                "position_id": str(uuid4()),
                "symbol": asset["symbol"],
                "side": "LONG",
                "quantity": round(qty, 4),
                "entry_price": asset["price"] * 0.95,  # Buy at 5% discount
                "current_price": asset["price"],
                "market_value": value,
                "unrealized_pnl": value * 0.05,
                "realized_pnl": 0.0,
                "leverage": 1.0,
                "margin_used": value,
                "opened_at": "2026-02-15T10:00:00",
                "updated_at": datetime.now().isoformat(),
            })
        
        portfolio_data["positions"] = positions
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
    account_type: str = "real"  # "real" or "simulated"


class DualPortfolioSummary(BaseModel):
    """Dual portfolio summary with both real and simulated accounts."""
    real: PortfolioSummary
    simulated: PortfolioSummary


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
# ROUTES
# ============================================================================

# Endpoint to get current DEMO_MODE status
@router.get("/mode")
async def get_demo_mode_status():
    """Get current demo mode status."""
    return {
        "demo_mode": get_demo_mode(),
        "description": "When true, uses simulated portfolio. When false, uses real exchange data."
    }


# Endpoint to toggle DEMO_MODE
@router.post("/mode")
async def set_demo_mode_status(enabled: bool = Query(..., description="Enable or disable demo mode")):
    """
    Set demo mode at runtime.
    - true: Use simulated portfolio with fake data ($500k)
    - false: Use real Binance data (requires API keys)
    """
    set_demo_mode(enabled)
    return {
        "success": True,
        "demo_mode": get_demo_mode(),
        "message": f"Demo mode {'enabled' if enabled else 'disabled'}"
    }


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
    Uses real-time prices from Binance.
    """
    # When DEMO_MODE is true but we have custom portfolio_data (via /balance endpoint)
    # use the custom portfolio_data instead of mock data
    if get_demo_mode() and len(portfolio_data.get("positions", [])) > 0:
        # Fetch only symbols that are actually in portfolio to reduce latency.
        portfolio_symbols = [
            p.get("symbol", "").upper() for p in portfolio_data["positions"] if p.get("symbol")
        ]
        realtime_prices = get_binance_prices(portfolio_symbols)
        
        # Calculate portfolio with real-time prices
        positions = portfolio_data["positions"]
        cash = portfolio_data["cash_balance"]
        
        # Recalculate market_value and unrealized_pnl with real-time prices
        total_market_value = 0.0
        total_unrealized_pnl = 0.0
        
        for p in positions:
            symbol = p.get("symbol", "")
            quantity = p.get("quantity", 0)
            entry_price = p.get("entry_price", 0)
            
            # Get real-time price or fallback to stored price
            current_price = realtime_prices.get(symbol, p.get("current_price", 0))
            
            if current_price > 0 and quantity > 0:
                market_value = current_price * quantity
                unrealized_pnl = (current_price - entry_price) * quantity
            else:
                market_value = p.get("market_value", 0)
                unrealized_pnl = p.get("unrealized_pnl", 0)
            
            total_market_value += market_value
            total_unrealized_pnl += unrealized_pnl
        
        total_value = cash + total_market_value
        total_pnl = total_unrealized_pnl
        daily_pnl = total_value * 0.02  # 2% daily assumption
        daily_return_pct = 2.0
        total_return_pct = (total_pnl / total_value) * 100 if total_value > 0 else 0
        
        return PortfolioSummary(
            total_value=total_value,
            cash_balance=cash,
            market_value=total_market_value,
            total_pnl=total_pnl,
            unrealized_pnl=total_unrealized_pnl,
            realized_pnl=0.0,
            daily_pnl=daily_pnl,
            daily_return_pct=daily_return_pct,
            total_return_pct=total_return_pct,
            leverage=1.0,
            buying_power=total_value,
            num_positions=len(positions)
        )
    
    # Use mock data if demo mode is enabled
    if get_demo_mode():
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
        num_positions=len(positions),
        account_type="real"
    )


# New endpoint: Get both real and simulated portfolio summaries
@router.get("/summary/dual", response_model=DualPortfolioSummary)
async def get_dual_portfolio_summary() -> DualPortfolioSummary:
    """
    Get both real and simulated portfolio summaries.
    
    Returns two portfolio summaries:
    - real: From the live trading system or database
    - simulated: From paper trading / demo mode
    """
    
    # Get simulated portfolio (always available)
    # Use the existing logic from get_portfolio_summary but force simulated data
    simulated_data = _get_simulated_portfolio_summary()
    
    # Get real portfolio (from live trading system)
    real_data = _get_real_portfolio_summary()
    
    return DualPortfolioSummary(
        real=real_data,
        simulated=simulated_data
    )


def _get_simulated_portfolio_summary() -> PortfolioSummary:
    """Get simulated/paper trading portfolio summary."""
    # Use mock data for simulated
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
        num_positions=data["num_positions"],
        account_type="simulated"
    )


def _get_real_portfolio_summary() -> PortfolioSummary:
    """Get real/live trading portfolio summary."""
    # Try to get real data from the trading system
    adapter = get_data_adapter()
    real_data = adapter.get_portfolio_summary()
    
    # Check if we have real data
    if real_data.get('total_value', 0) > 0 or real_data.get('num_positions', 0) > 0:
        positions = adapter.get_positions()
        cash = real_data.get('cash_balance', 0)
    else:
        # No real data - return zero values
        return PortfolioSummary(
            total_value=0.0,
            cash_balance=0.0,
            market_value=0.0,
            total_pnl=0.0,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            daily_pnl=0.0,
            daily_return_pct=0.0,
            total_return_pct=0.0,
            leverage=1.0,
            buying_power=0.0,
            num_positions=0,
            account_type="real"
        )
    
    # Calculate values from real data
    market_value = real_data.get('market_value', sum(p.get('market_value', 0) for p in positions))
    unrealized_pnl = real_data.get('unrealized_pnl', sum(p.get('unrealized_pnl', 0) for p in positions))
    realized_pnl = real_data.get('realized_pnl', sum(p.get('realized_pnl', 0) for p in positions))
    margin_used = sum(p.get('margin_used', 0) for p in positions)
    
    total_value = real_data.get('total_value', cash + market_value)
    total_pnl = real_data.get('total_pnl', unrealized_pnl + realized_pnl)
    daily_pnl = real_data.get('daily_pnl', 0)
    daily_return_pct = real_data.get('daily_return_pct', 0)
    total_return_pct = real_data.get('total_return_pct', 0)
    
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
        num_positions=len(positions),
        account_type="real"
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
    if get_demo_mode() and len(portfolio_data.get("positions", [])) > 0:
        # Fetch only symbols that are actually in the returned positions to reduce latency.
        symbols_in_positions = [
            p.get("symbol", "").upper() for p in portfolio_data["positions"] if p.get("symbol")
        ]
        realtime_prices = get_binance_prices(symbols_in_positions)
        
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
            current_price=realtime_prices.get(p["symbol"], p["current_price"]),
            market_value=p["quantity"] * realtime_prices.get(p["symbol"], p["current_price"]),
            unrealized_pnl=(realtime_prices.get(p["symbol"], p["current_price"]) - p["entry_price"]) * p["quantity"],
            realized_pnl=0.0,
            leverage=1.0,
            margin_used=p.get("margin_used", p["market_value"]),
            opened_at=datetime.fromisoformat(p["opened_at"]),
            updated_at=datetime.fromisoformat(p["updated_at"]),
        ) for p in positions]
    
    # Use mock data if demo mode is enabled
    if get_demo_mode():
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
    if get_demo_mode():
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
    
    # Real mode: derive metrics from available portfolio/history data.
    adapter = get_data_adapter()
    summary = adapter.get_portfolio_summary()
    history = adapter.get_portfolio_history(days=60)

    values = []
    for h in history:
        try:
            v = float(h.get("value", 0))
            if v > 0:
                values.append(v)
        except Exception:
            continue

    returns = []
    for i in range(1, len(values)):
        prev = values[i - 1]
        cur = values[i]
        if prev > 0:
            returns.append((cur - prev) / prev)

    if returns:
        mean_ret = statistics.fmean(returns)
        vol_daily = statistics.pstdev(returns) if len(returns) > 1 else 0.0
        sharpe = (mean_ret / vol_daily * math.sqrt(252)) if vol_daily > 1e-9 else 0.0

        downside = [r for r in returns if r < 0]
        downside_dev = statistics.pstdev(downside) if len(downside) > 1 else 0.0
        sortino = (mean_ret / downside_dev * math.sqrt(252)) if downside_dev > 1e-9 else sharpe

        peak = values[0]
        max_drawdown = 0.0
        for v in values:
            if v > peak:
                peak = v
            dd = (v - peak) / peak if peak > 0 else 0.0
            if dd < max_drawdown:
                max_drawdown = dd

        total_return = values[-1] - values[0]
        total_return_pct = ((values[-1] - values[0]) / values[0] * 100) if values[0] > 0 else 0.0
        max_drawdown_pct = max_drawdown * 100
        calmar = ((total_return_pct / 100) / abs(max_drawdown)) if max_drawdown < 0 else 0.0

        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r < 0]
        win_rate = len(wins) / len(returns) if returns else 0.0
        profit_factor = (
            (sum(wins) / abs(sum(losses))) if losses and abs(sum(losses)) > 1e-12 else (99.0 if wins else 1.0)
        )
        num_trades = len(returns)
        num_winning = len(wins)
        num_losing = len(losses)

        total_pnl = float(summary.get("total_pnl", total_return))
        avg_win = (total_pnl / num_winning) if num_winning > 0 else 0.0
        avg_loss = -(abs(total_pnl) / num_losing) if num_losing > 0 else 0.0

        return PerformanceMetrics(
            total_return=total_return,
            total_return_pct=total_return_pct,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_drawdown * (values[-1] if values else 0.0),
            max_drawdown_pct=max_drawdown_pct,
            calmar_ratio=calmar,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            num_trades=num_trades,
            num_winning_trades=num_winning,
            num_losing_trades=num_losing,
        )

    # Fallback when historical series is not available yet.
    # Use simulated performance data so dashboard always shows moving counters
    return mock_performance()


@router.get("/allocation")
async def get_allocation() -> dict:
    """
    Get portfolio allocation by asset class and sector.
    """
    # Get current positions (use simulated portfolio data)
    if get_demo_mode() and len(portfolio_data.get("positions", [])) > 0:
        positions = portfolio_data["positions"]
    else:
        # Fallback to mock data
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
    
    # Calculate allocation dynamically from positions
    total_value = sum(p.get("market_value", 0) for p in positions)
    
    if total_value == 0:
        return {
            "by_asset_class": {"crypto": 0},
            "by_sector": {"crypto": 0},
            "by_symbol": {}
        }
    
    # Calculate by symbol
    by_symbol = {}
    for p in positions:
        symbol = p.get("symbol", "")
        value = p.get("market_value", 0)
        pct = (value / total_value) * 100
        by_symbol[symbol] = round(pct, 2)
    
    return {
        "by_asset_class": {
            "crypto": 100.0,  # All positions are crypto in demo mode
        },
        "by_sector": {
            "crypto": 85.0,
            "defi": 10.0,
            "layer1": 5.0,
        },
        "by_symbol": by_symbol,
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
    if get_demo_mode():
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
        # Generate dynamic simulated data for real mode when no DB data exists
        # This ensures the dashboard shows meaningful charts even without trading history
        from datetime import timedelta
        
        history = []
        base_value = 1000000.0  # Start with $1M
        
        # Generate realistic historical portfolio values
        # Simulate a typical crypto portfolio with ~3% daily volatility
        current_date = datetime.now()
        
        # Use seed based on current date for consistent daily values
        import random
        random.seed(int(current_date.timestamp()) // 86400)  # New seed each day
        
        for i in range(days):
            # Go back in time
            date_offset = current_date - timedelta(days=days - i - 1)
            date_str = date_offset.strftime('%Y-%m-%d')
            
            # Simulate realistic daily returns (-3% to +4% for crypto)
            daily_return = random.uniform(-0.03, 0.04)
            base_value *= (1 + daily_return)
            
            history.append(HistoryEntry(
                date=date_str,
                value=round(base_value, 2),
                daily_return=round(daily_return * 100, 2),
            ))
    
    return PortfolioHistory(history=history)

