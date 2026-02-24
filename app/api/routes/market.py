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
import asyncio

from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel, Field

from app.api.mock_data import (
    DEMO_MODE,
    get_market_prices as mock_market_prices,
    get_price_data as mock_price_data,
    get_candle_data as mock_candle_data,
)

logger = logging.getLogger(__name__)


router = APIRouter()

try:
    from src.external.sentiment_apis import create_sentiment_clients_from_env
except Exception:
    create_sentiment_clients_from_env = None


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


class NewsItem(BaseModel):
    """News/sentiment item for dashboard feed."""
    timestamp: datetime
    title: str
    source: str
    url: str
    sentiment_score: float = 0.0


class NewsResponse(BaseModel):
    """News feed response."""
    query: str
    count: int
    items: List[NewsItem]


# ============================================================================
# ROUTES
# ============================================================================

@router.get("/price/{symbol}", response_model=PriceData)
async def get_price(symbol: str) -> PriceData:
    """
    Get current price for a symbol.
    """
    # Use mock data if demo mode is enabled
    if DEMO_MODE:
        data = mock_price_data(symbol)
        return PriceData(
            symbol=data["symbol"],
            price=data["price"],
            change_24h=data["change_24h"],
            change_pct_24h=data["change_pct_24h"],
            high_24h=data["high_24h"],
            low_24h=data["low_24h"],
            volume_24h=data["volume_24h"],
            timestamp=datetime.utcnow(),
        )
    
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
    # Use mock data if demo mode is enabled
    if DEMO_MODE:
        data = mock_market_prices()
        markets = []
        for m in data["markets"]:
            markets.append(PriceData(
                symbol=m["symbol"],
                price=m["price"],
                change_24h=m["price"] * m["change_pct_24h"] / 100,
                change_pct_24h=m["change_pct_24h"],
                high_24h=m["high_24h"],
                low_24h=m["low_24h"],
                volume_24h=m["volume_24h"],
                timestamp=datetime.utcnow(),
            ))
        return MarketOverview(
            timestamp=datetime.utcnow(),
            markets=markets,
        )
    
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
    # Use mock data if demo mode is enabled
    if DEMO_MODE:
        data = mock_candle_data(symbol, interval, limit)
        return [CandleData(
            timestamp=datetime.fromisoformat(c["timestamp"]),
            open=c["open"],
            high=c["high"],
            low=c["low"],
            close=c["close"],
            volume=c["volume"],
        ) for c in data]
    
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


@router.get("/news", response_model=NewsResponse)
async def get_market_news(
    query: str = Query("bitcoin", min_length=2, max_length=100),
    limit: int = Query(10, ge=1, le=50),
) -> NewsResponse:
    """
    Get latest market news/sentiment feed.
    Uses configured sentiment providers from environment.
    """
    if not create_sentiment_clients_from_env:
        raise HTTPException(status_code=503, detail="Sentiment providers not available")

    clients = create_sentiment_clients_from_env()
    if not clients:
        raise HTTPException(status_code=503, detail="No sentiment provider configured")

    items: List[NewsItem] = []
    # Keep latency bounded for dashboard usage
    max_per_client = min(20, max(5, limit))

    for client in clients:
        try:
            records = await asyncio.wait_for(
                client.fetch(query=query, limit=max_per_client),
                timeout=4.0,
            )
        except asyncio.TimeoutError:
            logger.warning("news_provider_timeout provider=%s", getattr(client, "name", "unknown"))
            continue
        except Exception as exc:
            logger.warning("news_provider_error provider=%s error=%s", getattr(client, "name", "unknown"), exc)
            continue

        for rec in records:
            payload = rec.payload or {}
            title = str(payload.get("title") or payload.get("text") or "").strip()
            if not title:
                continue

            source = str(payload.get("source") or payload.get("domain") or rec.source_api or "unknown")
            url = str(payload.get("url") or "")
            sentiment_score = float(payload.get("sentiment_score") or 0.0)

            items.append(
                NewsItem(
                    timestamp=rec.timestamp,
                    title=title,
                    source=source,
                    url=url,
                    sentiment_score=sentiment_score,
                )
            )

    # Deduplicate by title+source and order by newest
    unique = {}
    for item in items:
        key = f"{item.source.lower()}::{item.title.lower()}"
        if key not in unique:
            unique[key] = item

    sorted_items = sorted(unique.values(), key=lambda x: x.timestamp, reverse=True)[:limit]
    return NewsResponse(query=query, count=len(sorted_items), items=sorted_items)


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
