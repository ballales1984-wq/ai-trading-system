"""
Market Data Routes
==================
REST API for market data and price feeds.
"""

from datetime import datetime, timedelta
from typing import List, Optional
import random
import os
import logging

from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


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


# ============================================================================
# COINMARKETCAP ENDPOINTS
# ============================================================================

def _get_cmc_client():
    """Lazy-load CoinMarketCap client."""
    try:
        from src.external.coinmarketcap_client import CoinMarketCapClient
        api_key = os.getenv("COINMARKETCAP_API_KEY", "")
        if not api_key:
            return None
        return CoinMarketCapClient(api_key=api_key)
    except ImportError:
        logger.warning("CoinMarketCap client not available")
        return None


@router.get("/cmc/global")
async def get_cmc_global_metrics() -> dict:
    """
    Get global crypto market metrics from CoinMarketCap.
    Returns BTC dominance, total market cap, active cryptos, etc.
    """
    client = _get_cmc_client()
    if not client:
        raise HTTPException(status_code=503, detail="CoinMarketCap API not configured")
    
    try:
        metrics = await client.get_global_metrics()
        return {
            "source": "coinmarketcap",
            "timestamp": datetime.utcnow().isoformat(),
            "data": metrics,
        }
    except Exception as e:
        logger.error(f"CMC global metrics error: {e}")
        raise HTTPException(status_code=502, detail=str(e))


@router.get("/cmc/quote/{symbol}")
async def get_cmc_quote(symbol: str) -> dict:
    """
    Get real-time quote for a cryptocurrency from CoinMarketCap.
    """
    client = _get_cmc_client()
    if not client:
        raise HTTPException(status_code=503, detail="CoinMarketCap API not configured")
    
    try:
        quote = await client.get_quote(symbol.upper())
        return {
            "source": "coinmarketcap",
            "symbol": symbol.upper(),
            "timestamp": datetime.utcnow().isoformat(),
            "data": quote,
        }
    except Exception as e:
        logger.error(f"CMC quote error for {symbol}: {e}")
        raise HTTPException(status_code=502, detail=str(e))


@router.get("/cmc/sentiment")
async def get_cmc_market_sentiment() -> dict:
    """
    Get market sentiment proxy from CoinMarketCap data.
    Uses BTC dominance, volume ratios, and market momentum.
    """
    client = _get_cmc_client()
    if not client:
        raise HTTPException(status_code=503, detail="CoinMarketCap API not configured")
    
    try:
        sentiment = await client.get_market_sentiment_proxy()
        return {
            "source": "coinmarketcap",
            "timestamp": datetime.utcnow().isoformat(),
            "data": sentiment,
        }
    except Exception as e:
        logger.error(f"CMC sentiment error: {e}")
        raise HTTPException(status_code=502, detail=str(e))


@router.get("/cmc/top")
async def get_cmc_top_cryptos(
    limit: int = Query(20, ge=1, le=100),
) -> dict:
    """
    Get top cryptocurrencies by market cap from CoinMarketCap.
    """
    client = _get_cmc_client()
    if not client:
        raise HTTPException(status_code=503, detail="CoinMarketCap API not configured")
    
    try:
        listings = await client.get_listings(limit=limit)
        return {
            "source": "coinmarketcap",
            "timestamp": datetime.utcnow().isoformat(),
            "count": len(listings) if isinstance(listings, list) else 0,
            "data": listings,
        }
    except Exception as e:
        logger.error(f"CMC top cryptos error: {e}")
        raise HTTPException(status_code=502, detail=str(e))
