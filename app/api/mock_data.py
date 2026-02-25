"""
Mock Data for Demo Mode
=======================
Provides realistic demo data for public demonstrations.
Safe to use in production - no real API keys or sensitive data.
"""

import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import math
import os

# Demo mode configuration - set to True for demo mode by default
# Can be overridden by environment variable DEMO_MODE
DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() == "true"


# Base prices for demo assets
BASE_PRICES = {
    "BTC/USDT": 67500.00,
    "ETH/USDT": 3450.00,
    "SOL/USDT": 145.00,
    "BNB/USDT": 580.00,
    "XRP/USDT": 0.52,
    "ADA/USDT": 0.45,
    "DOGE/USDT": 0.12,
    "AVAX/USDT": 35.50,
}

# Initial portfolio state
INITIAL_PORTFOLIO_VALUE = 100000.00


def _random_change(base: float, volatility: float = 0.02) -> float:
    """Generate a random price change within volatility range."""
    change = random.uniform(-volatility, volatility)
    return base * (1 + change)


def _generate_price_history(base_price: float, days: int = 30, trend: float = 0.001) -> List[Dict]:
    """Generate realistic price history with slight upward trend."""
    history = []
    price = base_price * 0.85  # Start 15% lower
    
    for i in range(days):
        date = datetime.utcnow() - timedelta(days=days - i - 1)
        # Add some randomness with slight trend
        daily_change = random.gauss(trend, 0.02)
        price = price * (1 + daily_change)
        
        history.append({
            "date": date.isoformat(),
            "price": round(price, 2),
            "volume": random.randint(1000000, 10000000),
        })
    
    return history


def get_portfolio_summary() -> Dict[str, Any]:
    """Get demo portfolio summary."""
    # Simulate some daily variation
    daily_return = random.uniform(-0.02, 0.03)
    total_value = INITIAL_PORTFOLIO_VALUE * (1 + random.uniform(0.15, 0.35))
    daily_pnl = total_value * daily_return
    
    return {
        "total_value": round(total_value, 2),
        "cash": round(total_value * 0.25, 2),
        "invested": round(total_value * 0.75, 2),
        "daily_pnl": round(daily_pnl, 2),
        "daily_return_pct": round(daily_return * 100, 2),
        "unrealized_pnl": round(total_value - INITIAL_PORTFOLIO_VALUE, 2),
        "total_return_pct": round((total_value / INITIAL_PORTFOLIO_VALUE - 1) * 100, 2),
        "num_positions": 5,
        "last_updated": datetime.utcnow().isoformat(),
    }


def get_positions() -> List[Dict[str, Any]]:
    """Get demo open positions."""
    positions = [
        {
            "symbol": "BTC/USDT",
            "side": "LONG",
            "entry_price": 62000.00,
            "current_price": 67500.00,
            "quantity": 0.5,
            "market_value": 33750.00,
            "unrealized_pnl": 2750.00,
            "unrealized_pnl_pct": 8.87,
            "opened_at": (datetime.utcnow() - timedelta(days=15)).isoformat(),
        },
        {
            "symbol": "ETH/USDT",
            "side": "LONG",
            "entry_price": 3100.00,
            "current_price": 3450.00,
            "quantity": 5.0,
            "market_value": 17250.00,
            "unrealized_pnl": 1750.00,
            "unrealized_pnl_pct": 11.29,
            "opened_at": (datetime.utcnow() - timedelta(days=10)).isoformat(),
        },
        {
            "symbol": "SOL/USDT",
            "side": "LONG",
            "entry_price": 130.00,
            "current_price": 145.00,
            "quantity": 50.0,
            "market_value": 7250.00,
            "unrealized_pnl": 750.00,
            "unrealized_pnl_pct": 11.54,
            "opened_at": (datetime.utcnow() - timedelta(days=7)).isoformat(),
        },
        {
            "symbol": "BNB/USDT",
            "side": "LONG",
            "entry_price": 550.00,
            "current_price": 580.00,
            "quantity": 10.0,
            "market_value": 5800.00,
            "unrealized_pnl": 300.00,
            "unrealized_pnl_pct": 5.45,
            "opened_at": (datetime.utcnow() - timedelta(days=5)).isoformat(),
        },
        {
            "symbol": "AVAX/USDT",
            "side": "LONG",
            "entry_price": 32.00,
            "current_price": 35.50,
            "quantity": 100.0,
            "market_value": 3550.00,
            "unrealized_pnl": 350.00,
            "unrealized_pnl_pct": 10.94,
            "opened_at": (datetime.utcnow() - timedelta(days=3)).isoformat(),
        },
    ]
    return positions


def get_performance_metrics() -> Dict[str, Any]:
    """Get demo performance metrics."""
    return {
        "total_return_pct": 25.5,
        "annualized_return_pct": 42.3,
        "sharpe_ratio": 1.85,
        "sortino_ratio": 2.45,
        "calmar_ratio": 3.12,
        "max_drawdown_pct": -8.5,
        "win_rate": 0.68,
        "profit_factor": 2.1,
        "avg_trade_duration_hours": 48,
        "total_trades": 156,
        "winning_trades": 106,
        "losing_trades": 50,
        "avg_win": 450.00,
        "avg_loss": -215.00,
        "best_trade": 2350.00,
        "worst_trade": -890.00,
        "last_updated": datetime.utcnow().isoformat(),
    }


def get_portfolio_history(days: int = 30) -> Dict[str, Any]:
    """Get demo portfolio equity history."""
    history = []
    value = INITIAL_PORTFOLIO_VALUE
    
    for i in range(days):
        date = datetime.utcnow() - timedelta(days=days - i - 1)
        # Simulate daily returns with slight upward bias
        daily_return = random.gauss(0.003, 0.015)
        value = value * (1 + daily_return)
        
        history.append({
            "date": date.isoformat(),
            "value": round(value, 2),
            "daily_return": round(daily_return * 100, 4),
            "cash": round(value * 0.25, 2),
            "invested": round(value * 0.75, 2),
        })
    
    return {
        "history": history,
        "initial_value": INITIAL_PORTFOLIO_VALUE,
        "final_value": round(value, 2),
        "total_return_pct": round((value / INITIAL_PORTFOLIO_VALUE - 1) * 100, 2),
    }


def get_allocation() -> Dict[str, Any]:
    """Get demo portfolio allocation."""
    total = 67500.00  # Approximate total from positions
    
    return {
        "by_asset": [
            {"symbol": "BTC/USDT", "value": 33750.00, "percentage": 50.0},
            {"symbol": "ETH/USDT", "value": 17250.00, "percentage": 25.6},
            {"symbol": "SOL/USDT", "value": 7250.00, "percentage": 10.7},
            {"symbol": "BNB/USDT", "value": 5800.00, "percentage": 8.6},
            {"symbol": "AVAX/USDT", "value": 3550.00, "percentage": 5.3},
        ],
        "by_side": {
            "long": {"value": 67500.00, "percentage": 100.0},
            "short": {"value": 0, "percentage": 0.0},
        },
        "cash": 25000.00,
        "total": 92500.00,
    }


def get_market_prices() -> Dict[str, Any]:
    """Get demo market prices with simulated 24h changes."""
    markets = []
    
    for symbol, base_price in BASE_PRICES.items():
        # Simulate 24h change
        change_pct = random.uniform(-5, 5)
        current_price = base_price * (1 + change_pct / 100)
        
        markets.append({
            "symbol": symbol,
            "price": round(current_price, 2),
            "change_pct_24h": round(change_pct, 2),
            "high_24h": round(current_price * 1.03, 2),
            "low_24h": round(current_price * 0.97, 2),
            "volume_24h": random.randint(10000000, 500000000),
            "market_cap": round(current_price * random.randint(10000000, 100000000), 0),
            "last_updated": datetime.utcnow().isoformat(),
        })
    
    return {
        "markets": markets,
        "total_market_cap": sum(m["market_cap"] for m in markets),
        "last_updated": datetime.utcnow().isoformat(),
    }


def get_price_data(symbol: str) -> Dict[str, Any]:
    """Get demo price data for a specific symbol."""
    symbol = symbol.upper()
    
    # Normalize symbol format
    if "/" not in symbol:
        symbol = symbol.replace("USDT", "/USDT")
    
    if symbol not in BASE_PRICES:
        return {"error": f"Symbol {symbol} not found"}
    
    base_price = BASE_PRICES[symbol]
    change_pct = random.uniform(-5, 5)
    current_price = base_price * (1 + change_pct / 100)
    
    return {
        "symbol": symbol,
        "price": round(current_price, 2),
        "change_24h": round(current_price - base_price, 2),
        "change_pct_24h": round(change_pct, 2),
        "high_24h": round(current_price * 1.03, 2),
        "low_24h": round(current_price * 0.97, 2),
        "volume_24h": random.randint(10000000, 500000000),
        "bid": round(current_price * 0.9999, 2),
        "ask": round(current_price * 1.0001, 2),
        "last_updated": datetime.utcnow().isoformat(),
    }


def get_candle_data(symbol: str, interval: str = "1h", limit: int = 100) -> List[Dict[str, Any]]:
    """Get demo OHLCV candle data."""
    symbol = symbol.upper()
    if "/" not in symbol:
        symbol = symbol.replace("USDT", "/USDT")
    
    base_price = BASE_PRICES.get(symbol, 100.0)
    candles = []
    current_time = datetime.utcnow()
    
    # Interval in minutes
    interval_minutes = {
        "1m": 1, "5m": 5, "15m": 15, "30m": 30,
        "1h": 60, "4h": 240, "1d": 1440,
    }.get(interval, 60)
    
    price = base_price * 0.95
    
    for i in range(limit):
        timestamp = current_time - timedelta(minutes=interval_minutes * (limit - i))
        
        # Generate realistic OHLCV
        volatility = 0.005
        open_price = price
        high_price = open_price * (1 + random.uniform(0, volatility * 2))
        low_price = open_price * (1 - random.uniform(0, volatility * 2))
        close_price = open_price * (1 + random.uniform(-volatility, volatility))
        
        candles.append({
            "timestamp": timestamp.isoformat(),
            "open": round(open_price, 2),
            "high": round(max(open_price, high_price, close_price), 2),
            "low": round(min(open_price, low_price, close_price), 2),
            "close": round(close_price, 2),
            "volume": random.randint(10000, 1000000),
        })
        
        price = close_price
    
    return candles


def get_orders(status: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get demo orders with P&L data for trade history."""
    # Handle FastAPI Query objects and convert to string if needed
    if status is not None and hasattr(status, 'upper'):
        status_str = status.upper()
    elif status is not None:
        status_str = str(status).upper()
    else:
        status_str = None
    
    # Current prices for P&L calculation
    current_prices = {
        "BTC/USDT": 67500.00,
        "ETH/USDT": 3450.00,
        "SOL/USDT": 145.00,
        "BNB/USDT": 580.00,
        "AVAX/USDT": 35.50,
        "XRP/USDT": 0.52,
        "DOGE/USDT": 0.12,
    }
    
    all_orders = [

        {
            "id": "ORD-001",
            "symbol": "BTC/USDT",
            "side": "BUY",
            "type": "LIMIT",
            "quantity": 0.5,
            "price": 62000.00,
            "filled_quantity": 0.5,
            "status": "FILLED",
            "created_at": (datetime.utcnow() - timedelta(days=15)).isoformat(),
            "filled_at": (datetime.utcnow() - timedelta(days=15)).isoformat(),
            "pnl": 2750.00,  # (67500 - 62000) * 0.5
            "pnl_pct": 8.87,
        },
        {
            "id": "ORD-002",
            "symbol": "ETH/USDT",
            "side": "BUY",
            "type": "LIMIT",
            "quantity": 5.0,
            "price": 3100.00,
            "filled_quantity": 5.0,
            "status": "FILLED",
            "created_at": (datetime.utcnow() - timedelta(days=10)).isoformat(),
            "filled_at": (datetime.utcnow() - timedelta(days=10)).isoformat(),
            "pnl": 1750.00,  # (3450 - 3100) * 5
            "pnl_pct": 11.29,
        },
        {
            "id": "ORD-003",
            "symbol": "SOL/USDT",
            "side": "BUY",
            "type": "MARKET",
            "quantity": 50.0,
            "price": 130.00,
            "filled_quantity": 50.0,
            "status": "FILLED",
            "created_at": (datetime.utcnow() - timedelta(days=7)).isoformat(),
            "filled_at": (datetime.utcnow() - timedelta(days=7)).isoformat(),
            "pnl": 750.00,  # (145 - 130) * 50
            "pnl_pct": 11.54,
        },
        {
            "id": "ORD-004",
            "symbol": "BNB/USDT",
            "side": "BUY",
            "type": "LIMIT",
            "quantity": 10.0,
            "price": 550.00,
            "filled_quantity": 10.0,
            "status": "FILLED",
            "created_at": (datetime.utcnow() - timedelta(days=5)).isoformat(),
            "filled_at": (datetime.utcnow() - timedelta(days=5)).isoformat(),
            "pnl": 300.00,  # (580 - 550) * 10
            "pnl_pct": 5.45,
        },
        {
            "id": "ORD-005",
            "symbol": "AVAX/USDT",
            "side": "BUY",
            "type": "LIMIT",
            "quantity": 100.0,
            "price": 32.00,
            "filled_quantity": 100.0,
            "status": "FILLED",
            "created_at": (datetime.utcnow() - timedelta(days=3)).isoformat(),
            "filled_at": (datetime.utcnow() - timedelta(days=3)).isoformat(),
            "pnl": 350.00,  # (35.50 - 32) * 100
            "pnl_pct": 10.94,
        },
        {
            "id": "ORD-006",
            "symbol": "XRP/USDT",
            "side": "BUY",
            "type": "LIMIT",
            "quantity": 1000.0,
            "price": 0.48,
            "filled_quantity": 0,
            "status": "PENDING",
            "created_at": (datetime.utcnow() - timedelta(hours=2)).isoformat(),
            "filled_at": None,
            "pnl": 0.00,
            "pnl_pct": 0.00,
        },
        {
            "id": "ORD-007",
            "symbol": "DOGE/USDT",
            "side": "SELL",
            "type": "LIMIT",
            "quantity": 5000.0,
            "price": 0.15,
            "filled_quantity": 0,
            "status": "CANCELLED",
            "created_at": (datetime.utcnow() - timedelta(days=1)).isoformat(),
            "filled_at": None,
            "pnl": 0.00,
            "pnl_pct": 0.00,
        },
    ]
    
    if status_str:
        return [o for o in all_orders if o["status"] == status_str]
    
    return all_orders




def get_risk_metrics() -> Dict[str, Any]:
    """Get demo risk metrics."""
    return {
        "var_95": 2500.00,  # Value at Risk (95% confidence)
        "var_99": 4200.00,  # Value at Risk (99% confidence)
        "cvar_95": 3100.00,  # Conditional VaR
        "portfolio_beta": 1.15,
        "volatility_annualized": 0.35,
        "correlation_matrix": {
            "BTC/ETH": 0.85,
            "BTC/SOL": 0.72,
            "ETH/SOL": 0.78,
        },
        "concentration_risk": {
            "max_position_pct": 50.0,
            "top_3_positions_pct": 86.3,
        },
        "leverage": 1.0,
        "margin_used_pct": 0.0,
        "liquidation_price": None,
        "last_updated": datetime.utcnow().isoformat(),
    }


def get_strategy_signals() -> List[Dict[str, Any]]:
    """Get demo trading signals."""
    return [
        {
            "symbol": "BTC/USDT",
            "signal": "BUY",
            "confidence": 0.78,
            "strategy": "momentum",
            "entry_price": 67000.00,
            "target_price": 72000.00,
            "stop_loss": 64000.00,
            "risk_reward": 1.67,
            "generated_at": datetime.utcnow().isoformat(),
        },
        {
            "symbol": "ETH/USDT",
            "signal": "HOLD",
            "confidence": 0.65,
            "strategy": "mean_reversion",
            "entry_price": None,
            "target_price": None,
            "stop_loss": None,
            "risk_reward": None,
            "generated_at": datetime.utcnow().isoformat(),
        },
        {
            "symbol": "SOL/USDT",
            "signal": "BUY",
            "confidence": 0.82,
            "strategy": "breakout",
            "entry_price": 144.00,
            "target_price": 165.00,
            "stop_loss": 135.00,
            "risk_reward": 2.33,
            "generated_at": datetime.utcnow().isoformat(),
        },
    ]


def get_market_sentiment() -> Dict[str, Any]:
    """Get demo market sentiment data (Fear & Greed Index)."""
    # Generate a random fear & greed index between 20 and 80
    fear_greed_index = random.randint(20, 80)
    
    # Determine sentiment label based on index
    if fear_greed_index <= 20:
        sentiment_label = "Extreme Fear"
        sentiment_emoji = "😱"
    elif fear_greed_index <= 40:
        sentiment_label = "Fear"
        sentiment_emoji = "😰"
    elif fear_greed_index <= 60:
        sentiment_label = "Neutral"
        sentiment_emoji = "😐"
    elif fear_greed_index <= 80:
        sentiment_label = "Greed"
        sentiment_emoji = "🤑"
    else:
        sentiment_label = "Extreme Greed"
        sentiment_emoji = "🚀"
    
    return {
        "fear_greed_index": fear_greed_index,
        "sentiment_label": sentiment_label,
        "sentiment_emoji": sentiment_emoji,
        "btc_dominance": round(random.uniform(52.0, 58.0), 2),
        "market_momentum": round(random.uniform(-5.0, 15.0), 2),
        "last_updated": datetime.utcnow().isoformat(),
    }


# Export all functions

__all__ = [
    "DEMO_MODE",
    "get_portfolio_summary",
    "get_positions",
    "get_performance_metrics",
    "get_portfolio_history",
    "get_allocation",
    "get_market_prices",
    "get_price_data",
    "get_candle_data",
    "get_orders",
    "get_risk_metrics",
    "get_strategy_signals",
    "get_market_sentiment",
]
