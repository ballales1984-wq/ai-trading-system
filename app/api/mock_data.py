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
import os

# Demo mode configuration - set to True for demo mode by default
# Can be overridden by environment variable DEMO_MODE
DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() == "true"


# Base prices for demo assets - 30+ assets including crypto, forex, and commodities
BASE_PRICES = {
    # Major Cryptocurrencies (10)
    "BTC/USDT": 67500.00,
    "ETH/USDT": 3450.00,
    "SOL/USDT": 145.00,
    "BNB/USDT": 580.00,
    "XRP/USDT": 0.52,
    "ADA/USDT": 0.45,
    "DOGE/USDT": 0.12,
    "AVAX/USDT": 35.50,
    "DOT/USDT": 7.20,
    "MATIC/USDT": 0.58,
    "LINK/USDT": 14.80,
    "UNI/USDT": 6.90,
    "LTC/USDT": 72.00,
    "BCH/USDT": 340.00,
    "XLM/USDT": 0.11,
    "VET/USDT": 0.025,
    "FIL/USDT": 5.40,
    "TRX/USDT": 0.11,
    "ETC/USDT": 18.50,
    "XMR/USDT": 165.00,
    # Forex Pairs (8)
    "EUR/USD": 1.0850,
    "GBP/USD": 1.2650,
    "USD/JPY": 148.50,
    "USD/CHF": 0.8850,
    "AUD/USD": 0.6650,
    "USD/CAD": 1.3450,
    "NZD/USD": 0.6250,
    "EUR/GBP": 0.8580,
    # Commodities (4)
    "XAU/USD": 2025.00,
    "XAG/USD": 22.80,
    "USOIL": 72.50,
    "BRENT": 77.20,
    # Indices (4)
    "US30": 37500.00,
    "US500": 4780.00,
    "USTEC": 16800.00,
    "DE30": 16600.00,
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
        # Additional crypto orders
        {
            "id": "ORD-008",
            "symbol": "LTC/USDT",
            "side": "BUY",
            "type": "MARKET",
            "quantity": 10.0,
            "price": 68.00,
            "filled_quantity": 10.0,
            "status": "FILLED",
            "created_at": (datetime.utcnow() - timedelta(days=12)).isoformat(),
            "filled_at": (datetime.utcnow() - timedelta(days=12)).isoformat(),
            "pnl": 40.00,  # (72 - 68) * 10
            "pnl_pct": 5.88,
        },
        {
            "id": "ORD-009",
            "symbol": "LINK/USDT",
            "side": "BUY",
            "type": "LIMIT",
            "quantity": 50.0,
            "price": 13.50,
            "filled_quantity": 50.0,
            "status": "FILLED",
            "created_at": (datetime.utcnow() - timedelta(days=8)).isoformat(),
            "filled_at": (datetime.utcnow() - timedelta(days=8)).isoformat(),
            "pnl": 65.00,  # (14.80 - 13.50) * 50
            "pnl_pct": 9.63,
        },
        {
            "id": "ORD-010",
            "symbol": "ADA/USDT",
            "side": "BUY",
            "type": "LIMIT",
            "quantity": 1000.0,
            "price": 0.42,
            "filled_quantity": 1000.0,
            "status": "FILLED",
            "created_at": (datetime.utcnow() - timedelta(days=6)).isoformat(),
            "filled_at": (datetime.utcnow() - timedelta(days=6)).isoformat(),
            "pnl": 30.00,  # (0.45 - 0.42) * 1000
            "pnl_pct": 7.14,
        },
        {
            "id": "ORD-011",
            "symbol": "DOT/USDT",
            "side": "BUY",
            "type": "MARKET",
            "quantity": 100.0,
            "price": 6.80,
            "filled_quantity": 100.0,
            "status": "FILLED",
            "created_at": (datetime.utcnow() - timedelta(days=4)).isoformat(),
            "filled_at": (datetime.utcnow() - timedelta(days=4)).isoformat(),
            "pnl": 40.00,  # (7.20 - 6.80) * 100
            "pnl_pct": 5.88,
        },
        {
            "id": "ORD-012",
            "symbol": "MATIC/USDT",
            "side": "BUY",
            "type": "LIMIT",
            "quantity": 500.0,
            "price": 0.55,
            "filled_quantity": 500.0,
            "status": "FILLED",
            "created_at": (datetime.utcnow() - timedelta(days=2)).isoformat(),
            "filled_at": (datetime.utcnow() - timedelta(days=2)).isoformat(),
            "pnl": 15.00,  # (0.58 - 0.55) * 500
            "pnl_pct": 5.45,
        },
        # Forex orders
        {
            "id": "ORD-013",
            "symbol": "EUR/USD",
            "side": "BUY",
            "type": "LIMIT",
            "quantity": 10000.0,
            "price": 1.0800,
            "filled_quantity": 10000.0,
            "status": "FILLED",
            "created_at": (datetime.utcnow() - timedelta(days=11)).isoformat(),
            "filled_at": (datetime.utcnow() - timedelta(days=11)).isoformat(),
            "pnl": 50.00,  # (1.0850 - 1.0800) * 10000
            "pnl_pct": 0.46,
        },
        {
            "id": "ORD-014",
            "symbol": "GBP/USD",
            "side": "BUY",
            "type": "LIMIT",
            "quantity": 8000.0,
            "price": 1.2500,
            "filled_quantity": 8000.0,
            "status": "FILLED",
            "created_at": (datetime.utcnow() - timedelta(days=9)).isoformat(),
            "filled_at": (datetime.utcnow() - timedelta(days=9)).isoformat(),
            "pnl": 120.00,  # (1.2650 - 1.2500) * 8000
            "pnl_pct": 1.20,
        },
        {
            "id": "ORD-015",
            "symbol": "USD/JPY",
            "side": "SELL",
            "type": "LIMIT",
            "quantity": 1000.0,
            "price": 150.00,
            "filled_quantity": 0,
            "status": "PENDING",
            "created_at": (datetime.utcnow() - timedelta(hours=1)).isoformat(),
            "filled_at": None,
            "pnl": 0.00,
            "pnl_pct": 0.00,
        },
        # Commodities orders
        {
            "id": "ORD-016",
            "symbol": "XAU/USD",
            "side": "BUY",
            "type": "LIMIT",
            "quantity": 10.0,
            "price": 1980.00,
            "filled_quantity": 10.0,
            "status": "FILLED",
            "created_at": (datetime.utcnow() - timedelta(days=14)).isoformat(),
            "filled_at": (datetime.utcnow() - timedelta(days=14)).isoformat(),
            "pnl": 450.00,  # (2025 - 1980) * 10
            "pnl_pct": 2.27,
        },
        {
            "id": "ORD-017",
            "symbol": "XAG/USD",
            "side": "BUY",
            "type": "MARKET",
            "quantity": 100.0,
            "price": 22.00,
            "filled_quantity": 100.0,
            "status": "FILLED",
            "created_at": (datetime.utcnow() - timedelta(days=3)).isoformat(),
            "filled_at": (datetime.utcnow() - timedelta(days=3)).isoformat(),
            "pnl": 80.00,  # (22.80 - 22.00) * 100
            "pnl_pct": 3.64,
        },
        {
            "id": "ORD-018",
            "symbol": "USOIL",
            "side": "BUY",
            "type": "LIMIT",
            "quantity": 50.0,
            "price": 70.00,
            "filled_quantity": 50.0,
            "status": "FILLED",
            "created_at": (datetime.utcnow() - timedelta(days=7)).isoformat(),
            "filled_at": (datetime.utcnow() - timedelta(days=7)).isoformat(),
            "pnl": 125.00,  # (72.50 - 70.00) * 50
            "pnl_pct": 3.57,
        },
        # Indices orders
        {
            "id": "ORD-019",
            "symbol": "US500",
            "side": "BUY",
            "type": "LIMIT",
            "quantity": 5.0,
            "price": 4650.00,
            "filled_quantity": 5.0,
            "status": "FILLED",
            "created_at": (datetime.utcnow() - timedelta(days=13)).isoformat(),
            "filled_at": (datetime.utcnow() - timedelta(days=13)).isoformat(),
            "pnl": 650.00,  # (4780 - 4650) * 5
            "pnl_pct": 2.80,
        },
        {
            "id": "ORD-020",
            "symbol": "US30",
            "side": "BUY",
            "type": "MARKET",
            "quantity": 1.0,
            "price": 37000.00,
            "filled_quantity": 1.0,
            "status": "FILLED",
            "created_at": (datetime.utcnow() - timedelta(days=5)).isoformat(),
            "filled_at": (datetime.utcnow() - timedelta(days=5)).isoformat(),
            "pnl": 500.00,  # (37500 - 37000) * 1
            "pnl_pct": 1.35,
        },
    ]
    
    if status_str:
        return [o for o in all_orders if o["status"] == status_str]
    
    return all_orders




def get_risk_metrics() -> Dict[str, Any]:
    """Get demo risk metrics."""
    # Generate correlation matrix for all assets
    assets = list(BASE_PRICES.keys())
    n = len(assets)
    
    # Create realistic correlation matrix
    matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append(1.0)
            else:
                # Crypto assets are highly correlated with each other
                asset_i = assets[i]
                asset_j = assets[j]
                
                # Same category correlations
                if "/USDT" in asset_i and "/USDT" in asset_j:
                    corr = random.uniform(0.65, 0.95)  # Crypto-crypto
                elif ("/USD" in asset_i and "/USD" in asset_j and "XAU" not in asset_i and "XAG" not in asset_i and "USOIL" not in asset_i and "BRENT" not in asset_i):
                    corr = random.uniform(0.40, 0.85)  # Forex-forex
                elif ("US30" in asset_i or "US500" in asset_i or "USTEC" in asset_i or "DE30" in asset_i) and \
                     ("US30" in asset_j or "US500" in asset_j or "USTEC" in asset_j or "DE30" in asset_j):
                    corr = random.uniform(0.70, 0.95)  # Indices-indices
                elif ("XAU" in asset_i or "XAG" in asset_i) and ("XAU" in asset_j or "XAG" in asset_j):
                    corr = random.uniform(0.75, 0.90)  # Gold-Silver
                elif ("USOIL" in asset_i or "BRENT" in asset_i) and ("USOIL" in asset_j or "BRENT" in asset_j):
                    corr = random.uniform(0.85, 0.98)  # Oil-Oil
                else:
                    # Cross-category correlations (lower)
                    corr = random.uniform(-0.15, 0.35)
                
                row.append(round(corr, 2))
        matrix.append(row)
    
    return {
        "var_95": 2500.00,  # Value at Risk (95% confidence)
        "var_99": 4200.00,  # Value at Risk (99% confidence)
        "cvar_95": 3100.00,  # Conditional VaR
        "portfolio_beta": 1.15,
        "volatility_annualized": 0.35,
        "assets": assets,
        "correlation_matrix": matrix,
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
    
    # Determine sentiment label and trading indicator based on index
    if fear_greed_index <= 20:
        sentiment_label = "Extreme Fear"
        trading_indicator = "STRONG_SELL"
    elif fear_greed_index <= 40:
        sentiment_label = "Fear"
        trading_indicator = "SELL"
    elif fear_greed_index <= 60:
        sentiment_label = "Neutral"
        trading_indicator = "HOLD"
    elif fear_greed_index <= 80:
        sentiment_label = "Greed"
        trading_indicator = "BUY"
    else:
        sentiment_label = "Extreme Greed"
        trading_indicator = "STRONG_BUY"
    
    return {
        "fear_greed_index": fear_greed_index,
        "sentiment_label": sentiment_label,
        "trading_indicator": trading_indicator,
        "btc_dominance": round(random.uniform(52.0, 58.0), 2),
        "market_momentum": round(random.uniform(-5.0, 15.0), 2),
        "last_updated": datetime.utcnow().isoformat(),
    }



def get_news(symbol: Optional[str] = None, limit: int = 10, refresh: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get mock crypto news feed.
    
    When refresh is provided, generates dynamic data with varied timestamps and slightly
    randomized sentiment scores to simulate fresh news.
    """
    # Use refresh parameter to generate dynamic data
    import time
    seed = int(time.time() // 60) if refresh else 0  # Change every minute
    
    # Dynamic news variations based on time seed
    news_variations = [
        {
            "id": "news-001",
            "title": "Bitcoin Surges Past $67K as Institutional Adoption Accelerates",
            "source": "CoinDesk",
            "url": "https://coindesk.com/bitcoin-surge",
            "summary": "Major financial institutions continue to add Bitcoin to their portfolios, driving prices to new highs.",
            "sentiment": "positive",
            "sentiment_score": 0.85,
            "symbols": ["BTC/USDT"],
            "published_at": (datetime.utcnow() - timedelta(hours=2)).isoformat(),
            "category": "market",
        },
        {
            "id": "news-002",
            "title": "Ethereum Layer 2 Solutions See Record Transaction Volume",
            "source": "Cointelegraph",
            "url": "https://cointelegraph.com/eth-l2",
            "summary": "Arbitrum and Optimism process record number of transactions as gas fees remain low.",
            "sentiment": "positive",
            "sentiment_score": 0.78,
            "symbols": ["ETH/USDT"],
            "published_at": (datetime.utcnow() - timedelta(hours=4)).isoformat(),
            "category": "technology",
        },
        {
            "id": "news-003",
            "title": "Solana Network Experiences Brief Outage, Quickly Recovered",
            "source": "CryptoPanic",
            "url": "https://cryptopanic.com/solana-outage",
            "summary": "Network downtime lasted approximately 20 minutes before validators restored consensus.",
            "sentiment": "negative",
            "sentiment_score": -0.45,
            "symbols": ["SOL/USDT"],
            "published_at": (datetime.utcnow() - timedelta(hours=6)).isoformat(),
            "category": "network",
        },
        {
            "id": "news-004",
            "title": "SEC Approves New Spot Bitcoin ETF Applications",
            "source": "Bloomberg Crypto",
            "url": "https://bloomberg.com/sec-etf",
            "summary": "Regulatory approval paves way for increased institutional investment in cryptocurrency markets.",
            "sentiment": "positive",
            "sentiment_score": 0.92,
            "symbols": ["BTC/USDT", "ETH/USDT"],
            "published_at": (datetime.utcnow() - timedelta(hours=8)).isoformat(),
            "category": "regulation",
        },
        {
            "id": "news-005",
            "title": "DeFi Protocol Launches Revolutionary Yield Farming Strategy",
            "source": "DeFi Pulse",
            "url": "https://defipulse.com/yield-farming",
            "summary": "New automated strategy promises 15-20% APY with reduced impermanent loss risk.",
            "sentiment": "positive",
            "sentiment_score": 0.65,
            "symbols": ["ETH/USDT", "SOL/USDT"],
            "published_at": (datetime.utcnow() - timedelta(hours=12)).isoformat(),
            "category": "defi",
        },
        {
            "id": "news-006",
            "title": "Major Exchange Announces Trading Fee Reduction",
            "source": "CryptoNews",
            "url": "https://cryptonews.com/fee-reduction",
            "summary": "Competition among exchanges heats up as trading fees drop by 25% across spot markets.",
            "sentiment": "positive",
            "sentiment_score": 0.55,
            "symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"],
            "published_at": (datetime.utcnow() - timedelta(hours=16)).isoformat(),
            "category": "exchange",
        },
        {
            "id": "news-007",
            "title": "Crypto Market Volatility Expected Ahead of Fed Decision",
            "source": "Reuters",
            "url": "https://reuters.com/crypto-fed",
            "summary": "Analysts predict increased volatility as Federal Reserve prepares interest rate announcement.",
            "sentiment": "neutral",
            "sentiment_score": 0.0,
            "symbols": ["BTC/USDT", "ETH/USDT"],
            "published_at": (datetime.utcnow() - timedelta(hours=20)).isoformat(),
            "category": "macro",
        },
        {
            "id": "news-008",
            "title": "New Blockchain Bridge Connects Ethereum and Solana",
            "source": "TechCrunch",
            "url": "https://techcrunch.com/blockchain-bridge",
            "summary": "Cross-chain interoperability solution enables seamless asset transfers between major networks.",
            "sentiment": "positive",
            "sentiment_score": 0.72,
            "symbols": ["ETH/USDT", "SOL/USDT"],
            "published_at": (datetime.utcnow() - timedelta(hours=24)).isoformat(),
            "category": "technology",
        },
        # Additional dynamic news that appear when refresh is used
        {
            "id": "news-009",
            "title": "Bitcoin ETF Inflows Reach Record $500 Million in Single Day",
            "source": "CoinDesk",
            "url": "https://coindesk.com/etf-inflows",
            "summary": "Spot Bitcoin ETFs see massive institutional demand as market sentiment turns bullish.",
            "sentiment": "positive",
            "sentiment_score": 0.88,
            "symbols": ["BTC/USDT"],
            "published_at": (datetime.utcnow() - timedelta(minutes=30)).isoformat(),
            "category": "market",
        },
        {
            "id": "news-010",
            "title": "Ethereum Staking Yields Increase as Network Activity Rises",
            "source": "Cointelegraph",
            "url": "https://cointelegraph.com/eth-staking",
            "summary": "Staking rewards reach 5% APY as Ethereum network transaction volume increases significantly.",
            "sentiment": "positive",
            "sentiment_score": 0.75,
            "symbols": ["ETH/USDT"],
            "published_at": (datetime.utcnow() - timedelta(minutes=45)).isoformat(),
            "category": "defi",
        },
        {
            "id": "news-011",
            "title": "Regulatory Concerns Rise as Stablecoin Market Cap Hits $150B",
            "source": "Bloomberg",
            "url": "https://bloomberg.com/stablecoins",
            "summary": "Global regulators increasing scrutiny on stablecoins following rapid market growth.",
            "sentiment": "negative",
            "sentiment_score": -0.35,
            "symbols": ["USDT/USDT", "USDC/USDT"],
            "published_at": (datetime.utcnow() - timedelta(hours=1)).isoformat(),
            "category": "regulation",
        },
        {
            "id": "news-012",
            "title": "AI Crypto Tokens Surge as NVIDIA Announces New Chipset",
            "source": "CryptoNews",
            "url": "https://cryptonews.com/ai-tokens",
            "summary": "AI-related cryptocurrency tokens rally 20%+ on news of AI hardware advancement.",
            "sentiment": "positive",
            "sentiment_score": 0.82,
            "symbols": ["FET/USDT", "AGIX/USDT", "RNDR/USDT"],
            "published_at": (datetime.utcnow() - timedelta(hours=1, minutes=30)).isoformat(),
            "category": "technology",
        },
    ]
    
    # When refresh is used, shuffle and vary the news based on time seed
    if refresh and seed > 0:
        import random
        random.seed(seed)
        random.shuffle(news_variations)
        # Vary timestamps slightly for freshness perception
        for i, news in enumerate(news_variations):
            # Adjust time to appear more recent when refreshing
            hours_offset = i * 0.5  # Spread articles by 30 min intervals
            news["published_at"] = (datetime.utcnow() - timedelta(minutes=int(hours_offset * 60))).isoformat()
            # Add slight variation to sentiment score
            news["sentiment_score"] = round(news["sentiment_score"] + random.uniform(-0.1, 0.1), 2)
    
    all_news = news_variations
    
    # Filter by symbol if provided
    if symbol:
        symbol_normalized = symbol.replace("USDT", "/USDT") if "/" not in symbol else symbol
        filtered = [n for n in all_news if symbol_normalized in n.get("symbols", [])]
    else:
        filtered = all_news
    
    # Sort by published date (newest first) and apply limit
    filtered.sort(key=lambda x: x["published_at"], reverse=True)
    return filtered[:limit]


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
    "get_news",
]

# Backward compatibility aliases
MOCK_PRICES = BASE_PRICES


def get_mock_ticker(symbol: str):
    """Get mock ticker data."""
    return get_price_data(symbol)


def get_mock_orderbook(symbol: str, depth: int = 10):
    """Get mock orderbook data."""
    import random
    ticker = get_price_data(symbol)
    if "error" in ticker:
        return ticker
    
    price = ticker.get("price", 100.0)
    
    bids = []
    asks = []
    for i in range(depth):
        bid_price = price * (1 - 0.001 * (i + 1))
        ask_price = price * (1 + 0.001 * (i + 1))
        bids.append([round(bid_price, 2), round(random.uniform(100, 10000), 2)])
        asks.append([round(ask_price, 2), round(random.uniform(100, 10000), 2)])
    
    return {
        "symbol": symbol,
        "bids": bids,
        "asks": asks,
    }


def get_mock_ohlcv(symbol: str, interval: str = "1h", limit: int = 100):
    """Get mock OHLCV data."""
    return get_candle_data(symbol, interval, limit)
