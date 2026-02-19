"""
Market Data Routes
==================
REST API for market data and price feeds.
"""

from datetime import datetime, timedelta
from typing import List, Optional
import random

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field


router = APIRouter()


# ============================================================================
# DATA MODELS
# ============================================================================

class PriceData(BaseModel):
    """Price data for a symbol."""
    symbol: str
    price: float
    change_24h: float
    change_pct_24h: float
    high_24h: float
    low_24h: float
    volume_24h: float
    timestamp: datetime


class CandleData(BaseModel):
    """OHLCV candle data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class OrderBook(BaseModel):
    """Order book data."""
    symbol: str
    bids: List[List[float]]
    asks: List[List[float]]
    timestamp: datetime


class MarketOverview(BaseModel):
    """Overview of multiple markets."""
    timestamp: datetime
    markets: List[PriceData]


# ============================================================================
# ROUTES
# ============================================================================

@router.get("/price/{symbol}", response_model=PriceData)
async def get_price(symbol: str) -> PriceData:
    """
    Get current price for a symbol.
    """
    # Simulated price data
    base_prices = {
        "BTCUSDT": 43500.0,
        "ETHUSDT": 2350.0,
        "SOLUSDT": 95.0,
        "BNBUSDT": 310.0,
        "EURUSD": 1.0850,
    }
    
    base_price = base_prices.get(symbol.upper(), 100.0)
    change = random.uniform(-5, 5)
    
    return PriceData(
        symbol=symbol.upper(),
        price=base_price,
        change_24h=base_price * change / 100,
        change_pct_24h=change,
        high_24h=base_price * 1.03,
        low_24h=base_price * 0.97,
        volume_24h=random.uniform(1000000, 10000000),
        timestamp=datetime.utcnow(),
    )


@router.get("/prices", response_model=MarketOverview)
async def get_all_prices() -> MarketOverview:
    """
    Get prices for all tracked symbols.
    """
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "EURUSD"]
    markets = []
    
    for symbol in symbols:
        price_data = await get_price(symbol)
        markets.append(price_data)
    
    return MarketOverview(
        timestamp=datetime.utcnow(),
        markets=markets,
    )


@router.get("/candles/{symbol}", response_model=List[CandleData])
async def get_candles(
    symbol: str,
    interval: str = Query("1h", description="Time interval: 1m, 5m, 15m, 1h, 4h, 1d"),
    limit: int = Query(100, ge=1, le=1000),
) -> List[CandleData]:
    """
    Get OHLCV candle data for a symbol.
    """
    # Generate sample candles
    base_price = 43500.0 if "BTC" in symbol.upper() else 2350.0
    candles = []
    
    for i in range(limit):
        timestamp = datetime.utcnow() - timedelta(hours=limit - i)
        open_price = base_price * (1 + random.uniform(-0.02, 0.02))
        close_price = open_price * (1 + random.uniform(-0.01, 0.01))
        high_price = max(open_price, close_price) * (1 + random.uniform(0, 0.005))
        low_price = min(open_price, close_price) * (1 - random.uniform(0, 0.005))
        
        candles.append(CandleData(
            timestamp=timestamp,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=random.uniform(100, 1000),
        ))
    
    return candles


@router.get("/orderbook/{symbol}", response_model=OrderBook)
async def get_orderbook(symbol: str) -> OrderBook:
    """
    Get current order book for a symbol.
    """
    base_price = 43500.0 if "BTC" in symbol.upper() else 2350.0
    
    # Generate order book levels
    bids = []
    asks = []
    
    for i in range(20):
        bids.append([base_price - (i + 1) * 0.5, random.uniform(1, 10)])
        asks.append([base_price + (i + 1) * 0.5, random.uniform(1, 10)])
    
    return OrderBook(
        symbol=symbol.upper(),
        bids=bids,
        asks=asks,
        timestamp=datetime.utcnow(),
    )


@router.get("/ticker/24h/{symbol}", response_model=PriceData)
async def get_24h_ticker(symbol: str) -> PriceData:
    """
    Get 24-hour ticker data.
    """
    return await get_price(symbol)


@router.get("/trades/{symbol}")
async def get_recent_trades(
    symbol: str,
    limit: int = Query(50, ge=1, le=500),
) -> List[dict]:
    """
    Get recent trades for a symbol.
    """
    base_price = 43500.0 if "BTC" in symbol.upper() else 2350.0
    
    trades = []
    for i in range(limit):
        price = base_price * (1 + random.uniform(-0.001, 0.001))
        quantity = random.uniform(0.001, 1.0)
        
        trades.append({
            "id": str(i),
            "price": price,
            "quantity": quantity,
            "time": (datetime.utcnow() - timedelta(seconds=i * 10)).isoformat(),
            "is_buyer_maker": random.choice([True, False]),
        })
    
    return trades


@router.get("/futures/funding/{symbol}")
async def get_funding_rate(symbol: str) -> dict:
    """
    Get futures funding rate for a symbol.
    """
    return {
        "symbol": symbol.upper(),
        "funding_rate": 0.0001,
        "next_funding_time": (datetime.utcnow() + timedelta(hours=8)).isoformat(),
        "predicted_rate": 0.0001,
    }


@router.get("/index/{symbol}")
async def get_index_price(symbol: str) -> dict:
    """
    Get index price (for derivatives).
    """
    base_price = 43500.0 if "BTC" in symbol.upper() else 2350.0
    
    return {
        "symbol": symbol.upper(),
        "index_price": base_price,
        "mark_price": base_price * 1.0001,
        "last_update": datetime.utcnow().isoformat(),
    }
