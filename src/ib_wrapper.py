"""
Interactive Brokers Connector Wrapper
======================================
Wrapper for ib_insync with Python 3.14 compatibility fix.

The issue: In Python 3.10+, asyncio.get_event_loop() raises an exception
if no event loop exists in the current thread.

The fix: Create an event loop before importing ib_insync.
"""

import asyncio
import logging
import os
from typing import Optional, Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)

# Create event loop before importing ib_insync (Python 3.14 compatibility)
def _ensure_event_loop():
    """Ensure an event loop exists before importing ib_insync."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop, create a new one
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        except Exception as e:
            logger.warning(f"Could not create event loop: {e}")


# Apply fix before importing
_ensure_event_loop()

# Now try to import ib_insync
try:
    from ib_insync import (
        IB,
        Contract,
        Forex,
        Future,
        LimitOrder,
        MarketOrder,
        Stock,
        StopOrder,
        Trade,
        util,
    )
    IB_AVAILABLE = True
    logger.info("ib_insync imported successfully")
except ImportError as e:
    IB_AVAILABLE = False
    logger.warning(f"ib_insync not available: {e}")
except RuntimeError as e:
    IB_AVAILABLE = False
    logger.warning(f"ib_insync event loop error: {e}")


class IBConnectorWrapper:
    """
    Wrapper for IB Connector with Python 3.14 compatibility.
    
    This wrapper handles the event loop issues in Python 3.14 and provides
    a safe interface to Interactive Brokers.
    """
    
    def __init__(
        self,
        host: str = None,
        port: int = None,
        client_id: int = None
    ):
        """
        Initialize IB Connector Wrapper.
        
        Args:
            host: IB Gateway/TWS host (default from env IB_HOST)
            port: IB Gateway/TWS port (default from env IB_PORT)
            client_id: Client ID (default from env IB_CLIENT_ID)
        """
        self.host = host or os.environ.get('IB_HOST', '127.0.0.1')
        self.port = port or int(os.environ.get('IB_PORT', '7497'))
        self.client_id = client_id or int(os.environ.get('IB_CLIENT_ID', '1'))
        
        self._ib: Optional[IB] = None
        self._connected = False
        
        if IB_AVAILABLE:
            logger.info(f"IBConnectorWrapper initialized (host={self.host}, port={self.port})")
        else:
            logger.warning("IBConnectorWrapper initialized but ib_insync not available")
    
    @property
    def is_available(self) -> bool:
        """Check if ib_insync is available."""
        return IB_AVAILABLE
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to IB."""
        return self._connected and self._ib is not None
    
    async def connect(self) -> bool:
        """
        Connect to IB Gateway/TWS.
        
        Returns:
            True if connected successfully
        """
        if not IB_AVAILABLE:
            logger.error("Cannot connect: ib_insync not available")
            return False
        
        try:
            # Ensure event loop exists
            _ensure_event_loop()
            
            self._ib = IB()
            await self._ib.connectAsync(
                host=self.host,
                port=self.port,
                clientId=self.client_id
            )
            self._connected = True
            logger.info(f"Connected to IB at {self.host}:{self.port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to IB: {e}")
            self._connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from IB Gateway/TWS."""
        if self._ib and self._connected:
            try:
                self._ib.disconnect()
                logger.info("Disconnected from IB")
            except Exception as e:
                logger.error(f"Error disconnecting: {e}")
            finally:
                self._connected = False
                self._ib = None
    
    async def get_positions(self) -> List[Dict]:
        """
        Get current positions.
        
        Returns:
            List of position dictionaries
        """
        if not self.is_connected:
            return []
        
        try:
            positions = []
            for pos in self._ib.positions():
                positions.append({
                    'symbol': pos.contract.symbol,
                    'exchange': pos.contract.exchange,
                    'quantity': pos.position,
                    'avg_cost': pos.avgCost,
                    'market_price': 0.0,  # Would need market data
                    'unrealized_pnl': 0.0
                })
            return positions
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    async def get_account_summary(self) -> Dict:
        """
        Get account summary.
        
        Returns:
            Account summary dictionary
        """
        if not self.is_connected:
            return {}
        
        try:
            summary = await self._ib.accountSummaryAsync()
            result = {}
            for item in summary:
                result[item.tag] = item.value
            return result
        except Exception as e:
            logger.error(f"Error getting account summary: {e}")
            return {}
    
    async def place_order(
        self,
        symbol: str,
        action: str,
        quantity: float,
        order_type: str = 'MKT',
        limit_price: float = None
    ) -> Optional[Dict]:
        """
        Place an order.
        
        Args:
            symbol: Symbol to trade
            action: 'BUY' or 'SELL'
            quantity: Number of shares/contracts
            order_type: 'MKT' or 'LMT'
            limit_price: Limit price (required for LMT orders)
            
        Returns:
            Order result dictionary or None on error
        """
        if not self.is_connected:
            logger.error("Not connected to IB")
            return None
        
        try:
            # Create contract (assuming stock for now)
            contract = Stock(symbol, 'SMART', 'USD')
            
            # Create order
            if order_type == 'MKT':
                order = MarketOrder(action, quantity)
            elif order_type == 'LMT':
                if limit_price is None:
                    raise ValueError("Limit price required for LMT orders")
                order = LimitOrder(action, quantity, limit_price)
            else:
                raise ValueError(f"Unknown order type: {order_type}")
            
            # Place order
            trade = self._ib.placeOrder(contract, order)
            
            return {
                'order_id': trade.order.orderId,
                'status': trade.orderStatus.status,
                'filled': trade.orderStatus.filled,
                'remaining': trade.orderStatus.remaining,
                'avg_fill_price': trade.orderStatus.avgFillPrice
            }
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None
    
    async def get_market_data(self, symbol: str) -> Optional[Dict]:
        """
        Get market data for a symbol.
        
        Args:
            symbol: Symbol to get data for
            
        Returns:
            Market data dictionary or None on error
        """
        if not self.is_connected:
            return None
        
        try:
            contract = Stock(symbol, 'SMART', 'USD')
            self._ib.reqMktData(contract, '', False, False)
            
            # Wait for data
            await asyncio.sleep(2)
            
            ticker = self._ib.ticker(contract)
            
            return {
                'symbol': symbol,
                'bid': ticker.bid,
                'ask': ticker.ask,
                'last': ticker.last,
                'volume': ticker.volume,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return None


def create_ib_connector() -> Optional[IBConnectorWrapper]:
    """
    Factory function to create IB connector.
    
    Returns:
        IBConnectorWrapper instance or None if not available
    """
    if not IB_AVAILABLE:
        logger.warning("ib_insync not available, cannot create connector")
        return None
    
    return IBConnectorWrapper()


# Synchronous wrapper for simple use cases
class IBSyncWrapper:
    """
    Synchronous wrapper for IB operations.
    """
    
    def __init__(self):
        self._wrapper = IBConnectorWrapper()
    
    def connect(self) -> bool:
        """Connect to IB (blocking)."""
        if not self._wrapper.is_available:
            return False
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self._wrapper.connect())
    
    def disconnect(self):
        """Disconnect from IB (blocking)."""
        if not self._wrapper.is_available:
            return
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            return
        
        loop.run_until_complete(self._wrapper.disconnect())
    
    def get_positions(self) -> List[Dict]:
        """Get positions (blocking)."""
        if not self._wrapper.is_available:
            return []
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            return []
        
        return loop.run_until_complete(self._wrapper.get_positions())
