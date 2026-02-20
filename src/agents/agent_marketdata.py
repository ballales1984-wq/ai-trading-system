# src/agents/agent_marketdata.py
"""
Market Data Agent
=================
Real-time market data streaming agent.
Fetches and distributes price data from multiple sources.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from src.agents.base_agent import BaseAgent, AgentState
from src.core.event_bus import EventBus, EventType


logger = logging.getLogger(__name__)


class MarketDataAgent(BaseAgent):
    """
    Agent for fetching and distributing real-time market data.
    
    Features:
    - Multi-symbol support
    - Multiple data sources (Binance, CoinGecko, etc.)
    - Price caching and history
    - Automatic reconnection
    - Data normalization
    """
    
    def __init__(
        self,
        name: str,
        event_bus: EventBus,
        state_manager: Any,
        config: Dict[str, Any]
    ):
        """
        Initialize Market Data Agent.
        
        Args:
            name: Agent identifier
            event_bus: Event bus for communication
            state_manager: State manager instance
            config: Configuration dictionary with:
                - symbols: List of trading pairs (e.g., ["BTCUSDT", "ETHUSDT"])
                - interval_sec: Polling interval in seconds
                - sources: List of data sources
                - history_size: Number of price points to keep
        """
        super().__init__(name, event_bus, state_manager, config)
        
        # Configuration
        self.symbols = config.get("symbols", ["BTCUSDT"])
        self.interval_sec = config.get("interval_sec", 5)
        self.sources = config.get("sources", ["binance"])
        self.history_size = config.get("history_size", 1000)
        
        # Price cache
        self._prices: Dict[str, float] = {}
        self._price_history: Dict[str, List[Dict]] = {}
        self._last_update: Dict[str, datetime] = {}
        
        # Data source clients (lazy loaded)
        self._clients: Dict[str, Any] = {}
        
        logger.info(
            f"MarketDataAgent initialized for {len(self.symbols)} symbols"
        )
    
    async def on_start(self):
        """Initialize data source connections."""
        await self._initialize_clients()
        
        # Initialize price history for each symbol
        for symbol in self.symbols:
            self._price_history[symbol] = []
    
    async def _initialize_clients(self):
        """Initialize data source clients."""
        for source in self.sources:
            try:
                if source == "binance":
                    from src.external.market_data_apis import BinanceAPI
                    self._clients["binance"] = BinanceAPI()
                    
                elif source == "coingecko":
                    from src.external.market_data_apis import CoinGeckoAPI
                    self._clients["coingecko"] = CoinGeckoAPI()
                    
                elif source == "alpha_vantage":
                    from src.external.market_data_apis import AlphaVantageAPI
                    self._clients["alpha_vantage"] = AlphaVantageAPI()
                    
                logger.info(f"Initialized {source} client")
                
            except Exception as e:
                logger.warning(f"Failed to initialize {source} client: {e}")
    
    async def run(self):
        """Main agent loop - fetch and distribute market data."""
        while self._running:
            try:
                # Fetch prices for all symbols
                for symbol in self.symbols:
                    price_data = await self._fetch_price(symbol)
                    
                    if price_data:
                        await self._process_price(symbol, price_data)
                
                # Wait for next interval
                await asyncio.sleep(self.interval_sec)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in market data loop: {e}")
                self._metrics.errors += 1
                await asyncio.sleep(self._error_backoff)
    
    async def _fetch_price(self, symbol: str) -> Optional[Dict]:
        """
        Fetch current price for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Price data dictionary or None
        """
        # Try each source in order
        for source in self.sources:
            client = self._clients.get(source)
            if not client:
                continue
            
            try:
                if source == "binance":
                    # Use Binance API
                    data = await self._fetch_binance_price(client, symbol)
                    if data:
                        data["source"] = "binance"
                        return data
                        
                elif source == "coingecko":
                    # Convert symbol for CoinGecko
                    coin_id = self._symbol_to_coingecko_id(symbol)
                    data = await self._fetch_coingecko_price(client, coin_id)
                    if data:
                        data["source"] = "coingecko"
                        return data
                        
            except Exception as e:
                logger.debug(f"Failed to fetch {symbol} from {source}: {e}")
                continue
        
        return None
    
    async def _fetch_binance_price(self, client: Any, symbol: str) -> Optional[Dict]:
        """Fetch price from Binance."""
        try:
            # Check if client has async method
            if hasattr(client, 'get_ticker'):
                ticker = client.get_ticker(symbol)
                if ticker:
                    return {
                        "symbol": symbol,
                        "price": float(ticker.get("lastPrice", 0)),
                        "volume": float(ticker.get("volume", 0)),
                        "quote_volume": float(ticker.get("quoteVolume", 0)),
                        "price_change_percent": float(
                            ticker.get("priceChangePercent", 0)
                        ),
                    }
        except Exception as e:
            logger.debug(f"Binance fetch error: {e}")
        
        return None
    
    async def _fetch_coingecko_price(
        self,
        client: Any,
        coin_id: str
    ) -> Optional[Dict]:
        """Fetch price from CoinGecko."""
        try:
            if hasattr(client, 'get_price'):
                data = client.get_price(coin_id)
                if data and coin_id in data:
                    price_data = data[coin_id]
                    return {
                        "symbol": coin_id.upper(),
                        "price": price_data.get("usd", 0),
                        "market_cap": price_data.get("usd_market_cap", 0),
                        "24h_vol": price_data.get("usd_24h_vol", 0),
                        "24h_change": price_data.get("usd_24h_change", 0),
                    }
        except Exception as e:
            logger.debug(f"CoinGecko fetch error: {e}")
        
        return None
    
    def _symbol_to_coingecko_id(self, symbol: str) -> str:
        """Convert trading pair to CoinGecko coin ID."""
        mapping = {
            "BTCUSDT": "bitcoin",
            "ETHUSDT": "ethereum",
            "BNBUSDT": "binancecoin",
            "SOLUSDT": "solana",
            "XRPUSDT": "ripple",
            "ADAUSDT": "cardano",
            "DOGEUSDT": "dogecoin",
            "DOTUSDT": "polkadot",
            "MATICUSDT": "matic-network",
            "LINKUSDT": "chainlink",
        }
        return mapping.get(symbol, symbol.lower().replace("usdt", ""))
    
    async def _process_price(self, symbol: str, price_data: Dict):
        """
        Process and distribute price update.
        
        Args:
            symbol: Trading pair
            price_data: Price data dictionary
        """
        price = price_data.get("price", 0)
        timestamp = datetime.now()
        
        # Update cache
        self._prices[symbol] = price
        self._last_update[symbol] = timestamp
        
        # Add to history
        history_entry = {
            "price": price,
            "timestamp": timestamp.isoformat(),
            "source": price_data.get("source"),
            "volume": price_data.get("volume", 0),
        }
        self._price_history[symbol].append(history_entry)
        
        # Trim history
        if len(self._price_history[symbol]) > self.history_size:
            self._price_history[symbol].pop(0)
        
        # Update shared state
        self.update_state(f"price:{symbol}", price)
        self.update_state(f"price_data:{symbol}", price_data)
        self.update_state(f"last_update:{symbol}", timestamp.isoformat())
        
        # Emit event
        await self.emit_event(
            EventType.MARKET_DATA,
            {
                "symbol": symbol,
                "price": price,
                "timestamp": timestamp.isoformat(),
                "data": price_data,
            }
        )
        
        # Update metrics
        self._metrics.events_processed += 1
        self._metrics.last_activity = timestamp
        
        logger.debug(f"Price update: {symbol} = {price}")
    
    def get_price(self, symbol: str) -> Optional[float]:
        """
        Get cached price for a symbol.
        
        Args:
            symbol: Trading pair
            
        Returns:
            Current price or None
        """
        return self._prices.get(symbol)
    
    def get_price_history(
        self,
        symbol: str,
        limit: int = 100
    ) -> List[Dict]:
        """
        Get price history for a symbol.
        
        Args:
            symbol: Trading pair
            limit: Maximum number of entries
            
        Returns:
            List of price history entries
        """
        history = self._price_history.get(symbol, [])
        return history[-limit:]
    
    def get_all_prices(self) -> Dict[str, float]:
        """
        Get all cached prices.
        
        Returns:
            Dictionary of symbol -> price
        """
        return self._prices.copy()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent metrics."""
        metrics = super().get_metrics()
        metrics.update({
            "symbols_tracked": len(self.symbols),
            "prices_cached": len(self._prices),
            "sources": self.sources,
        })
        return metrics
