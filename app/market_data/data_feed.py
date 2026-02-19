"""
Market Data Feed
=============
Market data management for trading.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MarketData:
    """Market data for a symbol."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume
        }


class MarketDataFeed:
    """
    Market data feed manager.
    """
    
    def __init__(self):
        """Initialize market data feed."""
        self._prices: Dict[str, float] = {}
        self._candles: Dict[str, List[MarketData]] = {}
        self._callbacks: List[Callable] = []
        
        logger.info("Market data feed initialized")
    
    def subscribe(self, callback: Callable):
        """Subscribe to market data updates."""
        self._callbacks.append(callback)
    
    def unsubscribe(self, callback: Callable):
        """Unsubscribe from market data updates."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def update_price(self, symbol: str, price: float):
        """Update price for symbol."""
        self._prices[symbol] = price
        
        # Notify subscribers
        for callback in self._callbacks:
            try:
                callback(symbol, price)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    def get_price(self, symbol: str) -> Optional[float]:
        """Get current price."""
        return self._prices.get(symbol)
    
    def get_all_prices(self) -> Dict[str, float]:
        """Get all prices."""
        return self._prices.copy()
    
    def add_candle(self, candle: MarketData):
        """Add candle data."""
        if candle.symbol not in self._candles:
            self._candles[candle.symbol] = []
        
        self._candles[candle.symbol].append(candle)
        
        # Keep only last 1000 candles
        if len(self._candles[candle.symbol]) > 1000:
            self._candles[candle.symbol] = self._candles[candle.symbol][-1000:]
    
    def get_candles(self, symbol: str, limit: int = 100) -> List[MarketData]:
        """Get candle data."""
        candles = self._candles.get(symbol, [])
        return candles[-limit:]
    
    def get_dataframe(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """Get candle data as DataFrame."""
        candles = self.get_candles(symbol, limit)
        
        if not candles:
            return pd.DataFrame()
        
        data = [c.to_dict() for c in candles]
        df = pd.DataFrame(data)
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        
        return df


# Singleton instance
market_data_feed = MarketDataFeed()

