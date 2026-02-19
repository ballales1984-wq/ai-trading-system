"""
app/execution/connectors/ib_connector.py
Interactive Brokers Connector
==============================
Full connector for Interactive Brokers via IB Gateway / TWS.

Requirements:
  pip install ib_insync

Setup:
  1. Install IB Gateway or TWS
  2. Enable API connections in IB Gateway settings
  3. Set environment variables:
     - IB_HOST=127.0.0.1  (or ib-gateway in Docker)
     - IB_PORT=7497        (paper) or 7496 (live)
     - IB_CLIENT_ID=1
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

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
    HAS_IB = True
except ImportError:
    HAS_IB = False
    logger.info("ib_insync not installed. Install with: pip install ib_insync")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class IBOrderResult:
    """Result of an IB order placement."""

    def __init__(
        self,
        order_id: str = "",
        status: str = "PENDING",
        filled_quantity: float = 0.0,
        average_price: float = 0.0,
        commission: float = 0.0,
        error: str = "",
    ):
        self.order_id = order_id
        self.status = status
        self.filled_quantity = filled_quantity
        self.average_price = average_price
        self.commission = commission
        self.error = error


class IBPosition:
    """Represents an IB position."""

    def __init__(
        self,
        symbol: str = "",
        exchange: str = "",
        quantity: float = 0.0,
        avg_cost: float = 0.0,
        market_price: float = 0.0,
        unrealized_pnl: float = 0.0,
        realized_pnl: float = 0.0,
    ):
        self.symbol = symbol
        self.exchange = exchange
        self.quantity = quantity
        self.avg_cost = avg_cost
        self.market_price = market_price
        self.unrealized_pnl = unrealized_pnl
        self.realized_pnl = realized_pnl


# ---------------------------------------------------------------------------
# IB Connector
# ---------------------------------------------------------------------------

class IBConnector:
    """
    Interactive Brokers connector using ib_insync.

    Supports:
      - Stocks (NYSE, NASDAQ, LSE, etc.)
      - Futures (CME, NYMEX, CBOT, etc.)
      - Forex (IDEALPRO)
      - Options (via qualified contracts)

    Usage:
        connector = IBConnector()
        await connector.connect()
        result = await connector.place_order("AAPL", "BUY", 100, order_type="MARKET")
        positions = await connector.get_positions()
        await connector.disconnect()
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        client_id: Optional[int] = None,
        timeout: int = 30,
        readonly: bool = False,
    ):
        self.host = host or os.getenv("IB_HOST", "127.0.0.1")
        self.port = port or int(os.getenv("IB_PORT", "7497"))
        self.client_id = client_id or int(os.getenv("IB_CLIENT_ID", "1"))
        self.timeout = timeout
        self.readonly = readonly
        self._ib: Optional[Any] = None
        self._connected = False

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        """Connect to IB Gateway / TWS."""
        if not HAS_IB:
            logger.error("ib_insync not installed. Run: pip install ib_insync")
            return False

        try:
            self._ib = IB()

            # Set up event handlers
            self._ib.errorEvent += self._on_error
            self._ib.connectedEvent += self._on_connected
            self._ib.disconnectedEvent += self._on_disconnected

            await self._ib.connectAsync(
                host=self.host,
                port=self.port,
                clientId=self.client_id,
                timeout=self.timeout,
                readonly=self.readonly,
            )

            self._connected = True
            logger.info(
                f"Connected to IB Gateway at {self.host}:{self.port} "
                f"(clientId={self.client_id})"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to connect to IB Gateway: {e}")
            self._connected = False
            return False

    async def disconnect(self):
        """Disconnect from IB Gateway."""
        if self._ib and self._connected:
            self._ib.disconnect()
            self._connected = False
            logger.info("Disconnected from IB Gateway")

    def is_connected(self) -> bool:
        """Check if connected to IB."""
        if self._ib:
            return self._ib.isConnected()
        return False

    # ------------------------------------------------------------------
    # Contract creation
    # ------------------------------------------------------------------

    def _create_contract(
        self,
        symbol: str,
        sec_type: str = "STK",
        exchange: str = "SMART",
        currency: str = "USD",
        expiry: str = "",
    ) -> Any:
        """
        Create an IB contract.

        Args:
            symbol: Ticker symbol (e.g., "AAPL", "EURUSD", "ES")
            sec_type: Security type — STK, FUT, CASH, OPT
            exchange: Exchange — SMART, NYSE, NASDAQ, CME, IDEALPRO
            currency: Currency — USD, EUR, GBP
            expiry: Expiry for futures/options (e.g., "202603")
        """
        if not HAS_IB:
            return None

        if sec_type == "STK":
            return Stock(symbol, exchange, currency)
        elif sec_type == "FUT":
            return Future(symbol, expiry, exchange, currency)
        elif sec_type == "CASH":
            # Forex pair: symbol = "EUR", currency = "USD" → EURUSD
            return Forex(symbol + currency)
        else:
            # Generic contract
            contract = Contract()
            contract.symbol = symbol
            contract.secType = sec_type
            contract.exchange = exchange
            contract.currency = currency
            if expiry:
                contract.lastTradeDateOrContractMonth = expiry
            return contract

    async def qualify_contract(self, contract) -> bool:
        """Qualify a contract with IB to get full details."""
        if not self._ib or not self._connected:
            return False
        try:
            qualified = await self._ib.qualifyContractsAsync(contract)
            return len(qualified) > 0
        except Exception as e:
            logger.error(f"Failed to qualify contract: {e}")
            return False

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    async def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "MARKET",
        price: float = 0.0,
        stop_price: float = 0.0,
        sec_type: str = "STK",
        exchange: str = "SMART",
        currency: str = "USD",
        tif: str = "GTC",
    ) -> IBOrderResult:
        """
        Place an order on IB.

        Args:
            symbol: Ticker (e.g., "AAPL")
            side: "BUY" or "SELL"
            quantity: Number of shares/contracts
            order_type: "MARKET", "LIMIT", "STOP"
            price: Limit price (for LIMIT orders)
            stop_price: Stop price (for STOP orders)
            sec_type: "STK", "FUT", "CASH", "OPT"
            exchange: "SMART", "NYSE", "CME", etc.
            currency: "USD", "EUR", etc.
            tif: Time in force — "GTC", "DAY", "IOC"
        """
        if not self._ib or not self._connected:
            return IBOrderResult(error="Not connected to IB Gateway")

        try:
            # Create contract
            contract = self._create_contract(symbol, sec_type, exchange, currency)
            qualified = await self.qualify_contract(contract)
            if not qualified:
                return IBOrderResult(error=f"Could not qualify contract for {symbol}")

            # Create order
            if order_type.upper() == "MARKET":
                ib_order = MarketOrder(side.upper(), quantity)
            elif order_type.upper() == "LIMIT":
                ib_order = LimitOrder(side.upper(), quantity, price)
            elif order_type.upper() == "STOP":
                ib_order = StopOrder(side.upper(), quantity, stop_price)
            else:
                return IBOrderResult(error=f"Unsupported order type: {order_type}")

            ib_order.tif = tif

            # Place order
            trade: Trade = self._ib.placeOrder(contract, ib_order)

            # Wait for fill (with timeout)
            start = datetime.utcnow()
            while not trade.isDone():
                await asyncio.sleep(0.1)
                elapsed = (datetime.utcnow() - start).total_seconds()
                if elapsed > self.timeout:
                    break

            # Build result
            result = IBOrderResult(
                order_id=str(trade.order.orderId),
                status=trade.orderStatus.status,
                filled_quantity=trade.orderStatus.filled,
                average_price=trade.orderStatus.avgFillPrice,
                commission=sum(f.commission for f in trade.fills) if trade.fills else 0.0,
            )

            logger.info(
                f"IB Order {result.order_id}: {side} {quantity} {symbol} "
                f"→ {result.status} @ {result.average_price}"
            )
            return result

        except Exception as e:
            logger.error(f"IB order failed: {e}")
            return IBOrderResult(error=str(e))

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        if not self._ib or not self._connected:
            return False

        try:
            for trade in self._ib.openTrades():
                if str(trade.order.orderId) == order_id:
                    self._ib.cancelOrder(trade.order)
                    logger.info(f"Cancelled IB order {order_id}")
                    return True
            logger.warning(f"Order {order_id} not found in open trades")
            return False
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    async def get_open_orders(self) -> List[Dict[str, Any]]:
        """Get all open orders."""
        if not self._ib or not self._connected:
            return []

        orders = []
        for trade in self._ib.openTrades():
            orders.append({
                "order_id": str(trade.order.orderId),
                "symbol": trade.contract.symbol,
                "side": trade.order.action,
                "quantity": trade.order.totalQuantity,
                "order_type": trade.order.orderType,
                "price": trade.order.lmtPrice,
                "status": trade.orderStatus.status,
                "filled": trade.orderStatus.filled,
            })
        return orders

    # ------------------------------------------------------------------
    # Account & Positions
    # ------------------------------------------------------------------

    async def get_account_summary(self) -> Dict[str, Any]:
        """Get account summary (balance, margin, etc.)."""
        if not self._ib or not self._connected:
            return {}

        try:
            summary = self._ib.accountSummary()
            result = {}
            for item in summary:
                result[item.tag] = {
                    "value": item.value,
                    "currency": item.currency,
                }
            return result
        except Exception as e:
            logger.error(f"Failed to get account summary: {e}")
            return {}

    async def get_balance(self) -> Dict[str, float]:
        """Get account balance."""
        summary = await self.get_account_summary()
        return {
            "net_liquidation": float(summary.get("NetLiquidation", {}).get("value", 0)),
            "total_cash": float(summary.get("TotalCashValue", {}).get("value", 0)),
            "buying_power": float(summary.get("BuyingPower", {}).get("value", 0)),
            "available_funds": float(summary.get("AvailableFunds", {}).get("value", 0)),
            "maintenance_margin": float(summary.get("MaintMarginReq", {}).get("value", 0)),
        }

    async def get_positions(self) -> List[IBPosition]:
        """Get all open positions."""
        if not self._ib or not self._connected:
            return []

        positions = []
        for pos in self._ib.positions():
            positions.append(IBPosition(
                symbol=pos.contract.symbol,
                exchange=pos.contract.exchange,
                quantity=pos.position,
                avg_cost=pos.avgCost,
                market_price=0.0,  # Would need market data subscription
                unrealized_pnl=0.0,
                realized_pnl=0.0,
            ))

        # Try to get PnL data
        try:
            pnl_list = self._ib.pnl()
            for pnl_item in pnl_list:
                for p in positions:
                    if hasattr(pnl_item, 'unrealizedPnL'):
                        p.unrealized_pnl = pnl_item.unrealizedPnL or 0.0
                    if hasattr(pnl_item, 'realizedPnL'):
                        p.realized_pnl = pnl_item.realizedPnL or 0.0
        except Exception:
            pass

        return positions

    # ------------------------------------------------------------------
    # Market Data
    # ------------------------------------------------------------------

    async def get_market_price(
        self,
        symbol: str,
        sec_type: str = "STK",
        exchange: str = "SMART",
        currency: str = "USD",
    ) -> Optional[float]:
        """Get current market price for a symbol."""
        if not self._ib or not self._connected:
            return None

        try:
            contract = self._create_contract(symbol, sec_type, exchange, currency)
            await self.qualify_contract(contract)

            ticker = self._ib.reqMktData(contract, snapshot=True)
            await asyncio.sleep(2)  # Wait for data

            price = ticker.marketPrice()
            self._ib.cancelMktData(contract)

            if price and price > 0:
                return float(price)
            return None

        except Exception as e:
            logger.error(f"Failed to get market price for {symbol}: {e}")
            return None

    async def get_historical_data(
        self,
        symbol: str,
        duration: str = "1 Y",
        bar_size: str = "1 day",
        sec_type: str = "STK",
        exchange: str = "SMART",
        currency: str = "USD",
    ) -> List[Dict[str, Any]]:
        """
        Get historical OHLCV data.

        Args:
            duration: "1 D", "1 W", "1 M", "1 Y"
            bar_size: "1 min", "5 mins", "1 hour", "1 day"
        """
        if not self._ib or not self._connected:
            return []

        try:
            contract = self._create_contract(symbol, sec_type, exchange, currency)
            await self.qualify_contract(contract)

            bars = await self._ib.reqHistoricalDataAsync(
                contract,
                endDateTime="",
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow="TRADES",
                useRTH=True,
            )

            return [
                {
                    "timestamp": bar.date.isoformat() if hasattr(bar.date, 'isoformat') else str(bar.date),
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                }
                for bar in bars
            ]

        except Exception as e:
            logger.error(f"Failed to get historical data for {symbol}: {e}")
            return []

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_error(self, reqId, errorCode, errorString, contract):
        """Handle IB error events."""
        # Codes < 2000 are warnings, not errors
        if errorCode < 2000:
            logger.debug(f"IB Warning {errorCode}: {errorString}")
        else:
            logger.error(f"IB Error {errorCode}: {errorString} (reqId={reqId})")

    def _on_connected(self):
        """Handle connection event."""
        logger.info("IB Gateway connected")

    def _on_disconnected(self):
        """Handle disconnection event."""
        self._connected = False
        logger.warning("IB Gateway disconnected")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_ib_connector(
    host: Optional[str] = None,
    port: Optional[int] = None,
    client_id: Optional[int] = None,
    readonly: bool = False,
) -> IBConnector:
    """Create an IB connector from environment variables."""
    return IBConnector(
        host=host,
        port=port,
        client_id=client_id,
        readonly=readonly,
    )
