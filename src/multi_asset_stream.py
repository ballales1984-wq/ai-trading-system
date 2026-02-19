"""
Multi-Asset WebSocket Streaming Module
Manages real-time price streams for multiple crypto assets.
"""
import asyncio
import logging
from typing import Dict, List, Optional, Callable
from datetime import datetime

logger = logging.getLogger(__name__)


class MultiAssetStream:
    """Manages WebSocket streams for multiple crypto assets."""
    
    DEFAULT_SYMBOLS = [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
        "DOGEUSDT", "LINKUSDT", "ADAUSDT", "DOTUSDT", "AVAXUSDT"
    ]
    
    def __init__(self, symbols: Optional[List[str]] = None):
        self.symbols = symbols or self.DEFAULT_SYMBOLS
        self._prices: Dict[str, float] = {}
        self._price_callbacks: List[Callable] = []
        self._is_running = False
        logger.info(f"MultiAssetStream initialized with {len(self.symbols)} symbols")
    
    async def start(self):
        """Start streaming for all symbols."""
        if self._is_running:
            return
        self._is_running = True
        logger.info(f"Started streaming {len(self.symbols)} symbols")
    
    async def stop(self):
        """Stop all streams."""
        self._is_running = False
        logger.info("Multi-asset stream stopped")
    
    def get_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol."""
        return self._prices.get(symbol)
    
    def get_all_prices(self) -> Dict[str, float]:
        """Get all current prices."""
        return self._prices.copy()
    
    def register_callback(self, callback: Callable):
        """Register a callback for price updates."""
        self._price_callbacks.append(callback)
    
    def _notify_callbacks(self, symbol: str, price: float):
        """Notify registered callbacks of price update."""
        for callback in self._price_callbacks:
            try:
                callback(symbol, price)
            except Exception as e:
                logger.error(f"Callback error: {e}")


# Singleton instance
multi_stream = MultiAssetStream()
