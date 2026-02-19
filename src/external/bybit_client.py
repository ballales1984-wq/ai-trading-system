"""
Bybit API Client
================
Client for Bybit exchange API.
Supports: Spot, Futures, Options, WebSocket

Documentation: https://bybit-exchange.github.io/docs/v5/intro
"""

import asyncio
import hashlib
import hmac
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import aiohttp
import logging
import json

logger = logging.getLogger(__name__)


class BybitEnvironment(Enum):
    """Bybit environments."""
    PRODUCTION = "https://api.bybit.com"
    TESTNET = "https://api-testnet.bybit.com"


class BybitCategory(Enum):
    """Bybit product categories."""
    SPOT = "spot"
    LINEAR = "linear"      # USDT perpetual
    INVERSE = "inverse"    # Inverse perpetual
    OPTION = "option"      # Options
    SPOT_V3 = "spot/v3"


@dataclass
class BybitTicker:
    """Bybit ticker data."""
    symbol: str
    last_price: float
    bid_price: float
    ask_price: float
    volume_24h: float
    turnover_24h: float
    price_change_24h: float
    price_change_pct_24h: float
    high_24h: float
    low_24h: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class BybitKline:
    """Bybit candlestick data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    turnover: float


@dataclass
class BybitOrder:
    """Bybit order data."""
    order_id: str
    symbol: str
    side: str          # Buy, Sell
    order_type: str    # Market, Limit
    price: float
    quantity: float
    filled_quantity: float
    status: str        # Created, New, PartiallyFilled, Filled, Cancelled, Rejected
    created_time: datetime


class BybitClient:
    """
    Bybit API Client
    
    Supports:
    - Market data (tickers, klines, orderbook)
    - Account management
    - Order placement
    - Position management
    - WebSocket streams
    """
    
    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        testnet: bool = False,
        timeout: int = 30,
        max_retries: int = 3
    ):
        """
        Initialize Bybit client.
        
        Args:
            api_key: Bybit API key
            api_secret: Bybit API secret
            testnet: Use testnet (default: False)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Set base URL
        self.base_url = BybitEnvironment.TESTNET.value if testnet else BybitEnvironment.PRODUCTION.value
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        
        # Session
        self._session: Optional[aiohttp.ClientSession] = None
        
        # WebSocket
        self._ws_session: Optional[aiohttp.ClientSession] = None
        self._ws_connections: Dict[str, Any] = {}
        
        logger.info(f"Bybit client initialized (testnet={testnet}, base_url={self.base_url})")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
    
    async def connect(self):
        """Create aiohttp session."""
        if self._session is None:
            # Force use of threaded resolver instead of aiodns (fixes Windows DNS issues)
            try:
                import aiodns
                # If aiodns is installed, force using the threaded resolver
                resolver = aiohttp.ThreadedResolver()
            except ImportError:
                # aiodns not installed, use default
                resolver = aiohttp.DefaultResolver()
            
            connector = aiohttp.TCPConnector(
                limit=100,
                resolver=resolver
            )
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                connector=connector
            )
            logger.debug("Bybit session created")
    
    async def disconnect(self):
        """Close aiohttp session."""
        if self._session:
            await self._session.close()
            self._session = None
        if self._ws_session:
            await self._ws_session.close()
            self._ws_session = None
        logger.debug("Bybit session closed")
    
    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        signed: bool = False
    ) -> Dict:
        """
        Make API request.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            params: Request parameters
            signed: Whether request needs signature
            
        Returns:
            Response data
        """
        # Rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last)
        
        # Build URL
        url = f"{self.base_url}{endpoint}"
        
        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        # Sign request if needed
        if signed and self.api_key and self.api_secret:
            headers["X-BAPI-API-KEY"] = self.api_key
            headers["X-BAPI-TIMESTAMP"] = str(int(time.time() * 1000))
            headers["X-BAPI-SIGN"] = self._generate_signature(endpoint, params or {}, headers["X-BAPI-TIMESTAMP"])
            headers["X-BAPI-SIGN-TYPE"] = "HmacSHA256"
        
        # Make request with retries
        last_error = None
        for attempt in range(self.max_retries):
            try:
                async with self._session.request(
                    method, url, json=params, headers=headers
                ) as response:
                    data = await response.json()
                    
                    if data.get("retCode", 0) == 0:
                        return data.get("result", {})
                    else:
                        error_msg = data.get("retMsg", "Unknown error")
                        logger.warning(f"Bybit API error: {error_msg}")
                        raise Exception(f"API Error: {error_msg}")
                        
            except aiohttp.ClientError as e:
                last_error = e
                logger.warning(f"Request attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(1 * (attempt + 1))
        
        raise last_error or Exception("Request failed")
    
    def _generate_signature(self, endpoint: str, params: Dict, timestamp: str) -> str:
        """Generate HMAC SHA256 signature."""
        # Build query string
        param_str = json.dumps(params) if params else ""
        
        # Create signature payload
        sign_payload = f"{timestamp}{self.api_key}{param_str}"
        
        # Generate signature
        signature = hmac.new(
            self.api_secret.encode(),
            sign_payload.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    # ==================== MARKET DATA ====================
    
    async def get_ticker(self, symbol: str, category: str = "spot") -> BybitTicker:
        """
        Get ticker for symbol.
        
        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            category: Product category (spot, linear, inverse)
            
        Returns:
            BybitTicker object
        """
        params = {
            "category": category,
            "symbol": symbol
        }
        
        data = await self._request("GET", "/v5/market/ticker", params)
        
        if "list" in data and data["list"]:
            t = data["list"][0]
            return BybitTicker(
                symbol=t.get("symbol", symbol),
                last_price=float(t.get("lastPrice", 0)),
                bid_price=float(t.get("bid1Price", 0)),
                ask_price=float(t.get("ask1Price", 0)),
                volume_24h=float(t.get("volume24h", 0)),
                turnover_24h=float(t.get("turnover24h", 0)),
                price_change_24h=float(t.get("priceChange", 0)),
                price_change_pct_24h=float(t.get("priceChange24h", 0)),
                high_24h=float(t.get("highPrice24h", 0)),
                low_24h=float(t.get("lowPrice24h", 0)),
                timestamp=datetime.now()
            )
        
        raise ValueError(f"No ticker data for {symbol}")
    
    async def get_tickers(self, category: str = "spot") -> List[BybitTicker]:
        """
        Get all tickers for category.
        
        Args:
            category: Product category
            
        Returns:
            List of BybitTicker objects
        """
        params = {"category": category}
        data = await self._request("GET", "/v5/market/ticker", params)
        
        tickers = []
        if "list" in data:
            for t in data["list"]:
                tickers.append(BybitTicker(
                    symbol=t.get("symbol", ""),
                    last_price=float(t.get("lastPrice", 0)),
                    bid_price=float(t.get("bid1Price", 0)),
                    ask_price=float(t.get("ask1Price", 0)),
                    volume_24h=float(t.get("volume24h", 0)),
                    turnover_24h=float(t.get("turnover24h", 0)),
                    price_change_24h=float(t.get("priceChange", 0)),
                    price_change_pct_24h=float(t.get("priceChange24h", 0)),
                    high_24h=float(t.get("highPrice24h", 0)),
                    low_24h=float(t.get("lowPrice24h", 0)),
                    timestamp=datetime.now()
                ))
        
        return tickers
    
    async def get_klines(
        self,
        symbol: str,
        interval: str = "15",
        category: str = "spot",
        limit: int = 100,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> List[BybitKline]:
        """
        Get candlestick/kline data.
        
        Args:
            symbol: Trading pair
            interval: Kline interval (1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M)
            category: Product category
            limit: Number of klines (max 1000)
            start_time: Start time in milliseconds
            end_time: End time in milliseconds
            
        Returns:
            List of BybitKline objects
        """
        params = {
            "category": category,
            "symbol": symbol,
            "interval": str(interval),
            "limit": min(limit, 1000)
        }
        
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        
        data = await self._request("GET", "/v5/market/kline", params)
        
        klines = []
        if "list" in data:
            for k in data["list"]:
                # Bybit returns: [startTime, open, high, low, close, volume, turnover]
                klines.append(BybitKline(
                    timestamp=datetime.fromtimestamp(int(k[0]) / 1000),
                    open=float(k[1]),
                    high=float(k[2]),
                    low=float(k[3]),
                    close=float(k[4]),
                    volume=float(k[5]),
                    turnover=float(k[6])
                ))
        
        return klines
    
    async def get_orderbook(self, symbol: str, limit: int = 25) -> Dict:
        """
        Get order book.
        
        Args:
            symbol: Trading pair
            limit: Order book depth (5, 10, 20, 50)
            
        Returns:
            Order book data
        """
        params = {
            "category": "spot",
            "symbol": symbol,
            "limit": limit
        }
        
        return await self._request("GET", "/v5/market/orderbook", params)
    
    async def get_symbols(self, category: str = "spot") -> List[str]:
        """
        Get available symbols.
        
        Args:
            category: Product category
            
        Returns:
            List of trading pairs
        """
        params = {"category": category}
        data = await self._request("GET", "/v5/market/instruments-info", params)
        
        symbols = []
        if "list" in data:
            symbols = [s.get("symbol") for s in data["list"] if s.get("status") == "Trading"]
        
        return symbols
    
    # ==================== ACCOUNT & WALLET ====================
    
    async def get_balance(self, account_type: str = "UNIFIED") -> Dict:
        """
        Get account balance.
        
        Args:
            account_type: Account type (UNIFIED, CONTRACT, SPOT)
            
        Returns:
            Balance information
        """
        params = {"accountType": account_type}
        return await self._request("GET", "/v5/account/wallet-balance", params, signed=True)
    
    async def get_positions(self, category: str = "linear") -> List[Dict]:
        """
        Get open positions.
        
        Args:
            category: Product category
            
        Returns:
            List of positions
        """
        params = {"category": category}
        data = await self._request("GET", "/v5/position/closed-pnl", params, signed=True)
        return data.get("list", [])
    
    # ==================== ORDERS ====================
    
    async def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        category: str = "spot"
    ) -> BybitOrder:
        """
        Place an order.
        
        Args:
            symbol: Trading pair
            side: "Buy" or "Sell"
            order_type: "Market" or "Limit"
            quantity: Order quantity
            price: Limit price (required for Limit orders)
            category: Product category
            
        Returns:
            BybitOrder object
        """
        params = {
            "category": category,
            "symbol": symbol,
            "side": side,
            "orderType": order_type,
            "qty": str(quantity)
        }
        
        if order_type == "Limit" and price:
            params["price"] = str(price)
            params["timeInForce"] = "GTC"
        
        data = await self._request("POST", "/v5/order/create", params, signed=True)
        
        order_data = data.get("list", [{}])[0]
        
        return BybitOrder(
            order_id=order_data.get("orderId", ""),
            symbol=order_data.get("symbol", symbol),
            side=order_data.get("side", side),
            order_type=order_data.get("orderType", order_type),
            price=float(order_data.get("price", 0)),
            quantity=float(order_data.get("qty", 0)),
            filled_quantity=float(order_data.get("cumExecQty", 0)),
            status=order_data.get("orderStatus", "Created"),
            created_time=datetime.fromtimestamp(
                int(order_data.get("createdTime", 0)) / 1000
            )
        )
    
    async def cancel_order(self, symbol: str, order_id: str, category: str = "spot") -> Dict:
        """
        Cancel an order.
        
        Args:
            symbol: Trading pair
            order_id: Order ID
            category: Product category
            
        Returns:
            Cancellation result
        """
        params = {
            "category": category,
            "symbol": symbol,
            "orderId": order_id
        }
        
        return await self._request("POST", "/v5/order/cancel", params, signed=True)
    
    async def get_orders(
        self,
        symbol: Optional[str] = None,
        category: str = "spot",
        limit: int = 50
    ) -> List[BybitOrder]:
        """
        Get order history.
        
        Args:
            symbol: Trading pair (optional)
            category: Product category
            limit: Number of orders
            
        Returns:
            List of BybitOrder objects
        """
        params = {"category": category, "limit": limit}
        if symbol:
            params["symbol"] = symbol
        
        data = await self._request("GET", "/v5/order/history", params, signed=True)
        
        orders = []
        for o in data.get("list", []):
            orders.append(BybitOrder(
                order_id=o.get("orderId", ""),
                symbol=o.get("symbol", ""),
                side=o.get("side", ""),
                order_type=o.get("orderType", ""),
                price=float(o.get("price", 0)),
                quantity=float(o.get("qty", 0)),
                filled_quantity=float(o.get("cumExecQty", 0)),
                status=o.get("orderStatus", ""),
                created_time=datetime.fromtimestamp(
                    int(o.get("createdTime", 0)) / 1000
                )
            ))
        
        return orders
    
    # ==================== WEBSOCKET ====================
    
    async def connect_websocket(
        self,
        streams: List[str],
        callback: Optional[callable] = None
    ) -> str:
        """
        Connect to WebSocket.
        
        Args:
            streams: List of stream names
            callback: Async callback for messages
            
        Returns:
            Connection ID
        """
        if self._ws_session is None:
            self._ws_session = aiohttp.ClientSession()
        
        # Build WebSocket URL
        ws_url = f"{self.base_url.replace('http', 'ws')}/v5/public/spot"
        
        # Subscribe to streams
        subscription = [f"{stream}" for stream in streams]
        
        logger.info(f"Connecting to Bybit WebSocket: {streams}")
        
        # Note: Full WebSocket implementation would require websockets library
        # This is a placeholder
        
        return "ws_connection_id"
    
    async def disconnect_websocket(self, connection_id: str):
        """Disconnect WebSocket."""
        if connection_id in self._ws_connections:
            await self._ws_connections[connection_id].close()
            del self._ws_connections[connection_id]
    
    # ==================== UTILITY METHODS ====================
    
    async def get_price(self, symbol: str) -> float:
        """Get current price for symbol."""
        ticker = await self.get_ticker(symbol)
        return ticker.last_price
    
    async def get_bid_ask(self, symbol: str) -> tuple:
        """Get bid and ask prices."""
        ticker = await self.get_ticker(symbol)
        return ticker.bid_price, ticker.ask_price
    
    async def get_funding_rate(self, symbol: str) -> float:
        """Get current funding rate for perpetual."""
        params = {
            "category": "linear",
            "symbol": symbol
        }
        data = await self._request("GET", "/v5/market/ticker", params)
        
        if "list" in data and data["list"]:
            return float(data["list"][0].get("fundingRate", "0"))
        return 0.0


# ==================== FACTORY FUNCTIONS ====================

def create_bybit_client(
    api_key: str = "",
    api_secret: str = "",
    testnet: bool = False
) -> BybitClient:
    """
    Factory function to create Bybit client.
    
    Args:
        api_key: API key
        api_secret: API secret
        testnet: Use testnet
        
    Returns:
        BybitClient instance
    """
    return BybitClient(
        api_key=api_key,
        api_secret=api_secret,
        testnet=testnet
    )


# ==================== TESTS ====================

async def test_bybit_client():
    """Test Bybit client."""
    print("Testing Bybit API Client...")
    
    # Create client (public, no auth needed for market data)
    client = BybitClient()
    
    try:
        await client.connect()
        
        # Test 1: Get ticker
        print("\n1. Getting BTCUSDT ticker...")
        ticker = await client.get_ticker("BTCUSDT")
        print(f"   Price: ${ticker.last_price:,.2f}")
        print(f"   24h Change: {ticker.price_change_pct_24h:.2f}%")
        
        # Test 2: Get klines
        print("\n2. Getting BTCUSDT klines...")
        klines = await client.get_klines("BTCUSDT", interval="15", limit=5)
        print(f"   Got {len(klines)} klines")
        
        # Test 3: Get orderbook
        print("\n3. Getting orderbook...")
        orderbook = await client.get_orderbook("BTCUSDT", limit=5)
        print(f"   Bids: {len(orderbook.get('b', []))}")
        print(f"   Asks: {len(orderbook.get('a', []))}")
        
        # Test 4: Get symbols
        print("\n4. Getting available symbols...")
        symbols = await client.get_symbols()
        print(f"   Total symbols: {len(symbols)}")
        
        print("\n✅ All tests passed!")
        
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(test_bybit_client())
