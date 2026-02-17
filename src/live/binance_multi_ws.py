"""
Binance WebSocket Multi-Asset Module
Real-time data streaming for multiple assets
"""

import json
import logging
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import websocket

logger = logging.getLogger(__name__)


class BinanceMultiWebSocket:
    """
    WebSocket client for streaming real-time data from multiple Binance symbols.
    """
    
    def __init__(self, symbols: List[str], interval: str = "1m", 
                 testnet: bool = False):
        """
        Initialize the WebSocket client.
        
        Args:
            symbols: List of trading symbols (e.g., ['BTCUSDT', 'ETHUSDT', 'SOLUSDT'])
            interval: Kline interval (1m, 5m, 15m, 1h, 4h, 1d)
            testnet: Use Binance testnet
        """
        self.symbols = [s.upper().replace('/', '') for s in symbols]
        self.interval = interval
        self.testnet = testnet
        
        # Data storage - each symbol has its own DataFrame
        self.data: Dict[str, pd.DataFrame] = {s: pd.DataFrame() for s in self.symbols}
        self.last_prices: Dict[str, float] = {s: 0.0 for s in self.symbols}
        self.ws_connections: Dict[str, websocket.WebSocketApp] = {}
        self.threads: Dict[str, threading.Thread] = {}
        self.running = False
        
        # Lock for thread-safe data updates
        self._lock = threading.Lock()
        
        # Buffer for new candles
        self._buffer: Dict[str, dict] = {}
        
        logger.info(f"Initialized BinanceMultiWebSocket for {symbols}")
    
    def _get_stream_url(self) -> str:
        """Get the WebSocket stream URL."""
        if self.testnet:
            return "wss://testnet.binance.vision/ws"
        return "wss://stream.binance.com:9443/ws"
    
    def _get_kline_url(self) -> str:
        """Get the WebSocket kline stream URL."""
        if self.testnet:
            return "wss://testnet.binance.vision/stream"
        return "wss://stream.binance.com:9443/stream"
    
    def _create_kline_stream(self, symbol: str) -> str:
        """Create the kline stream name for a symbol."""
        return f"{symbol.lower()}@kline_{self.interval}"
    
    def _on_message(self, symbol: str, ws: websocket.WebSocketApp, message: str):
        """Handle incoming WebSocket message."""
        try:
            data = json.loads(message)
            
            # Handle different message types
            if 'data' in data:
                # Combined stream format
                kline = data['data']['k']
            else:
                # Single stream format
                kline = data['k']
            
            # Extract candle data
            candle = {
                'timestamp': pd.to_datetime(kline['t'], unit='ms'),
                'open': float(kline['o']),
                'high': float(kline['h']),
                'low': float(kline['l']),
                'close': float(kline['c']),
                'volume': float(kline['v']),
                'closed': kline['x']  # Is the candle closed?
            }
            
            with self._lock:
                self.last_prices[symbol] = candle['close']
                
                # Update DataFrame
                df = self.data[symbol]
                
                # If candle is closed or it's a new candle, add it
                if len(df) == 0 or candle['timestamp'] > df['timestamp'].iloc[-1]:
                    # New candle
                    df = pd.concat([df, pd.DataFrame([candle])], ignore_index=True)
                    # Keep only last 500 candles
                    if len(df) > 500:
                        df = df.tail(500)
                elif candle['timestamp'] == df['timestamp'].iloc[-1]:
                    # Update existing candle
                    df.iloc[-1] = pd.Series(candle)
                
                self.data[symbol] = df
                
        except Exception as e:
            logger.error(f"Error processing message for {symbol}: {e}")
    
    def _on_error(self, ws: websocket.WebSocketApp, error):
        """Handle WebSocket error."""
        logger.error(f"WebSocket error: {error}")
    
    def _on_close(self, symbol: str, ws: websocket.WebSocketApp, close_status_code: int, close_msg: str):
        """Handle WebSocket close."""
        logger.warning(f"WebSocket closed for {symbol}: {close_status_code} - {close_msg}")
        
        # Attempt to reconnect
        if self.running:
            logger.info(f"Reconnecting {symbol}...")
            time.sleep(5)
            self._start_stream(symbol)
    
    def _on_open(self, symbol: str, ws: websocket.WebSocketApp):
        """Handle WebSocket open."""
        logger.info(f"WebSocket opened for {symbol}")
    
    def _start_stream(self, symbol: str):
        """Start WebSocket stream for a single symbol."""
        stream_name = self._create_kline_stream(symbol)
        url = f"{self._get_kline_url()}/?streams={stream_name}"
        
        ws = websocket.WebSocketApp(
            url,
            on_message=lambda ws, msg: self._on_message(symbol, ws, msg),
            on_error=lambda ws, err: self._on_error(ws, err),
            on_close=lambda ws, code, msg: self._on_close(symbol, ws, code, msg),
            on_open=lambda ws: self._on_open(symbol, ws)
        )
        
        self.ws_connections[symbol] = ws
        
        # Start in a separate thread
        thread = threading.Thread(target=ws.run_forever, daemon=True)
        thread.start()
        self.threads[symbol] = thread
        
        logger.info(f"Started WebSocket stream for {symbol}")
    
    def start(self):
        """Start all WebSocket streams."""
        if self.running:
            logger.warning("WebSocket already running")
            return
        
        self.running = True
        
        # Start streams for all symbols
        for symbol in self.symbols:
            self._start_stream(symbol)
        
        # Wait for initial data
        logger.info("Waiting for initial data...")
        self._wait_for_data()
        
        logger.info("All WebSocket streams started successfully")
    
    def _wait_for_data(self, min_candles: int = 10, timeout: int = 30):
        """Wait for initial data to arrive."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            all_ready = True
            
            for symbol in self.symbols:
                with self._lock:
                    if len(self.data[symbol]) < min_candles:
                        all_ready = False
                        break
            
            if all_ready:
                logger.info(f"Received initial data for all symbols")
                return
            
            time.sleep(1)
        
        logger.warning(f"Timeout waiting for initial data. Some symbols may have no data.")
    
    def stop(self):
        """Stop all WebSocket streams."""
        self.running = False
        
        for symbol, ws in self.ws_connections.items():
            try:
                ws.close()
                logger.info(f"Closed WebSocket for {symbol}")
            except Exception as e:
                logger.error(f"Error closing WebSocket for {symbol}: {e}")
        
        self.ws_connections.clear()
        self.threads.clear()
        
        logger.info("All WebSocket streams stopped")
    
    def get_data(self, symbol: str) -> pd.DataFrame:
        """
        Get the current data for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            DataFrame with OHLCV data
        """
        with self._lock:
            return self.data[symbol].copy()
    
    def get_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Get data for all symbols.
        
        Returns:
            Dictionary of DataFrames
        """
        with self._lock:
            return {s: df.copy() for s, df in self.data.items()}
    
    def get_price(self, symbol: str) -> float:
        """
        Get the latest price for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Latest price
        """
        with self._lock:
            return self.last_prices.get(symbol, 0.0)
    
    def get_all_prices(self) -> Dict[str, float]:
        """
        Get the latest prices for all symbols.
        
        Returns:
            Dictionary of prices
        """
        with self._lock:
            return self.last_prices.copy()
    
    def is_ready(self, min_candles: int = 10) -> bool:
        """
        Check if all streams have received enough data.
        
        Args:
            min_candles: Minimum number of candles required
            
        Returns:
            True if all streams are ready
        """
        with self._lock:
            return all(len(self.data[s]) >= min_candles for s in self.symbols)


class BinanceWebSocketManager:
    """
    Manager class for handling multiple WebSocket connections.
    """
    
    def __init__(self):
        self.streams: Dict[str, BinanceMultiWebSocket] = {}
    
    def add_stream(self, name: str, symbols: List[str], 
                  interval: str = "1m", testnet: bool = False) -> BinanceMultiWebSocket:
        """
        Add a new WebSocket stream.
        
        Args:
            name: Name for this stream
            symbols: List of symbols
            interval: Kline interval
            testnet: Use testnet
            
        Returns:
            The created WebSocket stream
        """
        stream = BinanceMultiWebSocket(symbols, interval, testnet)
        self.streams[name] = stream
        return stream
    
    def start_all(self):
        """Start all streams."""
        for stream in self.streams.values():
            stream.start()
    
    def stop_all(self):
        """Stop all streams."""
        for stream in self.streams.values():
            stream.stop()
    
    def get_stream(self, name: str) -> Optional[BinanceMultiWebSocket]:
        """Get a stream by name."""
        return self.streams.get(name)
