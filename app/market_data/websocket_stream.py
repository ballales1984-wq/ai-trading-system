"""
WebSocket Stream
==============
WebSocket streaming for real-time market data.
Supports Binance WebSocket API with auto-reconnect.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class WebSocketStream:
    """
    WebSocket stream manager for real-time data.
    Connects to Binance WebSocket API for live market data.
    """
    
    def __init__(self, url: str = "wss://stream.binance.com:9443/ws"):
        """
        Initialize WebSocket stream.
        
        Args:
            url: WebSocket URL
        """
        self.url = url
        self._ws = None
        self._connected = False
        self._subscriptions: Dict[str, List[Callable]] = {}
        self._reconnect_delay = 1.0
        self._max_reconnect_delay = 60.0
        self._should_reconnect = True
        self._listen_task: Optional[asyncio.Task] = None
        self._subscription_id = 1
        
        logger.info(f"WebSocket stream initialized: {url}")
    
    async def connect(self) -> bool:
        """Connect to WebSocket server."""
        try:
            import websockets
            
            self._ws = await websockets.connect(
                self.url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=5,
                max_size=2**20,  # 1MB max message size
            )
            
            self._connected = True
            self._reconnect_delay = 1.0  # Reset delay on successful connect
            logger.info("WebSocket connected")
            
            # Re-subscribe to existing streams
            if self._subscriptions:
                streams = list(self._subscriptions.keys())
                await self._send_subscribe(streams)
            
            return True
            
        except ImportError:
            logger.warning(
                "websockets library not installed. "
                "Install with: pip install websockets"
            )
            return False
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from WebSocket server."""
        self._should_reconnect = False
        self._connected = False
        
        if self._listen_task and not self._listen_task.done():
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
        
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        
        logger.info("WebSocket disconnected")
    
    async def _send_subscribe(self, streams: List[str]):
        """Send subscription message to WebSocket."""
        if not self._ws or not self._connected:
            return
        
        msg = {
            "method": "SUBSCRIBE",
            "params": streams,
            "id": self._subscription_id
        }
        self._subscription_id += 1
        
        try:
            await self._ws.send(json.dumps(msg))
            logger.debug(f"Sent SUBSCRIBE for {streams}")
        except Exception as e:
            logger.error(f"Failed to send subscribe: {e}")
    
    async def _send_unsubscribe(self, streams: List[str]):
        """Send unsubscription message to WebSocket."""
        if not self._ws or not self._connected:
            return
        
        msg = {
            "method": "UNSUBSCRIBE",
            "params": streams,
            "id": self._subscription_id
        }
        self._subscription_id += 1
        
        try:
            await self._ws.send(json.dumps(msg))
            logger.debug(f"Sent UNSUBSCRIBE for {streams}")
        except Exception as e:
            logger.error(f"Failed to send unsubscribe: {e}")
    
    async def subscribe(
        self,
        streams: List[str],
        callback: Callable
    ):
        """
        Subscribe to streams.
        
        Args:
            streams: List of stream names (e.g. ['btcusdt@trade', 'ethusdt@kline_1m'])
            callback: Callback function that receives parsed message dict
        """
        for stream in streams:
            if stream not in self._subscriptions:
                self._subscriptions[stream] = []
            self._subscriptions[stream].append(callback)
        
        if self._connected:
            await self._send_subscribe(streams)
        
        logger.info(f"Subscribed to streams: {streams}")
    
    async def unsubscribe(self, streams: List[str]):
        """Unsubscribe from streams."""
        for stream in streams:
            if stream in self._subscriptions:
                del self._subscriptions[stream]
        
        if self._connected:
            await self._send_unsubscribe(streams)
        
        logger.info(f"Unsubscribed from streams: {streams}")
    
    async def subscribe_trades(self, symbols: List[str], callback: Callable):
        """Subscribe to trade streams for given symbols."""
        streams = [f"{s.lower()}@trade" for s in symbols]
        await self.subscribe(streams, callback)
    
    async def subscribe_klines(
        self, symbols: List[str], interval: str, callback: Callable
    ):
        """Subscribe to kline/candlestick streams."""
        streams = [f"{s.lower()}@kline_{interval}" for s in symbols]
        await self.subscribe(streams, callback)
    
    async def subscribe_tickers(self, symbols: List[str], callback: Callable):
        """Subscribe to 24hr mini-ticker streams."""
        streams = [f"{s.lower()}@miniTicker" for s in symbols]
        await self.subscribe(streams, callback)
    
    async def subscribe_depth(
        self, symbols: List[str], callback: Callable, levels: int = 5
    ):
        """Subscribe to order book depth streams."""
        streams = [f"{s.lower()}@depth{levels}@100ms" for s in symbols]
        await self.subscribe(streams, callback)
    
    async def subscribe_agg_trades(self, symbols: List[str], callback: Callable):
        """Subscribe to aggregated trade streams."""
        streams = [f"{s.lower()}@aggTrade" for s in symbols]
        await self.subscribe(streams, callback)
    
    async def start_listening(self):
        """Start listening for messages with auto-reconnect."""
        self._should_reconnect = True
        
        while self._should_reconnect:
            if not self._connected:
                connected = await self.connect()
                if not connected:
                    logger.warning(
                        f"Reconnecting in {self._reconnect_delay}s..."
                    )
                    await asyncio.sleep(self._reconnect_delay)
                    self._reconnect_delay = min(
                        self._reconnect_delay * 2,
                        self._max_reconnect_delay
                    )
                    continue
            
            try:
                async for message in self._ws:
                    try:
                        data = json.loads(message)
                        self._handle_message(data)
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON: {message[:100]}")
                    except Exception as e:
                        logger.error(f"Message handling error: {e}")
                        
            except Exception as e:
                logger.warning(f"WebSocket connection lost: {e}")
                self._connected = False
                
                if self._should_reconnect:
                    logger.info(
                        f"Reconnecting in {self._reconnect_delay}s..."
                    )
                    await asyncio.sleep(self._reconnect_delay)
                    self._reconnect_delay = min(
                        self._reconnect_delay * 2,
                        self._max_reconnect_delay
                    )
    
    def _handle_message(self, data: Dict):
        """Handle incoming message and dispatch to callbacks."""
        # Combined stream format: {"stream": "btcusdt@trade", "data": {...}}
        if "stream" in data:
            stream_name = data["stream"]
            payload = data.get("data", data)
        else:
            # Single stream format: {"e": "trade", "s": "BTCUSDT", ...}
            event_type = data.get("e", "")
            symbol = data.get("s", "").lower()
            
            # Try to match stream name
            stream_name = None
            if event_type == "trade":
                stream_name = f"{symbol}@trade"
            elif event_type == "aggTrade":
                stream_name = f"{symbol}@aggTrade"
            elif event_type == "kline":
                interval = data.get("k", {}).get("i", "1m")
                stream_name = f"{symbol}@kline_{interval}"
            elif event_type == "24hrMiniTicker":
                stream_name = f"{symbol}@miniTicker"
            elif event_type == "depthUpdate":
                stream_name = f"{symbol}@depth"
            
            payload = data
        
        if stream_name is None:
            return
        
        # Get callbacks for this stream
        callbacks = self._subscriptions.get(stream_name, [])
        
        # Also check partial matches (e.g. depth5 vs depth)
        if not callbacks:
            for sub_name, sub_callbacks in self._subscriptions.items():
                if stream_name.startswith(sub_name.split("@")[0] + "@"):
                    callbacks = sub_callbacks
                    break
        
        # Notify callbacks
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(payload))
                else:
                    callback(payload)
            except Exception as e:
                logger.error(f"Callback error for {stream_name}: {e}")
    
    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected
    
    @property
    def active_subscriptions(self) -> List[str]:
        """Get list of active subscriptions."""
        return list(self._subscriptions.keys())


# Singleton instance
ws_stream = WebSocketStream()
