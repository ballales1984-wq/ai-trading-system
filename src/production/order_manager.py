# src/production/order_manager.py
"""
Order Manager with Retry Logic
=============================
Manages order execution with automatic retries,
error handling, and safety features for real money trading.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path


logger = logging.getLogger(__name__)


class OrderManagerError(Exception):
    """Base exception for order manager."""
    pass


class MaxRetriesExceeded(OrderManagerError):
    """Raised when max retries are exceeded."""
    pass


class OrderRejected(OrderManagerError):
    """Raised when order is rejected."""
    pass


class InsufficientBalance(OrderManagerError):
    """Raised when balance is insufficient."""
    pass


class RiskViolation(OrderManagerError):
    """Raised when order violates risk limits."""
    pass


class RetryStrategy(Enum):
    """Retry strategy types."""
    IMMEDIATE = "immediate"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"


@dataclass
class RetryConfig:
    """Configuration for retry logic."""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    retry_on: List[str] = field(default_factory=lambda: [
        "timeout",
        "connection_error",
        "rate_limit",
        "unknown"
    ])
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt."""
        if self.strategy == RetryStrategy.IMMEDIATE:
            return 0
        elif self.strategy == RetryStrategy.LINEAR:
            return self.base_delay * attempt
        else:  # EXPONENTIAL
            delay = self.base_delay * (2 ** attempt)
            return min(delay, self.max_delay)


@dataclass
class OrderRequest:
    """Represents an order request."""
    symbol: str
    side: str  # BUY or SELL
    quantity: float
    order_type: str = "MARKET"
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"
    client_order_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'side': self.side,
            'quantity': self.quantity,
            'order_type': self.order_type,
            'price': self.price,
            'stop_price': self.stop_price,
            'time_in_force': self.time_in_force,
            'client_order_id': self.client_order_id,
            'metadata': self.metadata
        }


class RiskManager:
    """
    Risk manager for validating orders before execution.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize risk manager.
        
        Args:
            config: Risk configuration
        """
        self.config = config
        self.max_position_size = config.get('max_position_size', 1.0)  # Max % of portfolio
        self.max_order_size = config.get('max_order_size', 0.2)  # Max % per order
        self.max_daily_loss = config.get('max_daily_loss', 0.05)  # 5% max daily loss
        self.max_leverage = config.get('max_leverage', 3)
        self.min_order_value = config.get('min_order_value', 10)  # Minimum order value
        
    def validate_order(
        self,
        order: OrderRequest,
        account_balance: float,
        current_position: float,
        daily_pnl: float
    ) -> tuple[bool, Optional[str]]:
        """
        Validate order against risk limits.
        
        Args:
            order: Order to validate
            account_balance: Current account balance
            current_position: Current position size
            daily_pnl: Today's PnL
            
        Returns:
            (is_valid, error_message)
        """
        # Check daily loss limit
        daily_loss_pct = abs(daily_pnl) / account_balance if account_balance > 0 else 0
        if daily_loss_pct >= self.max_daily_loss:
            return False, f"Daily loss limit exceeded: {daily_loss_pct:.2%}"
        
        # Check order value
        estimated_value = order.quantity * (order.price or 0)
        if estimated_value < self.min_order_value:
            return False, f"Order value {estimated_value} below minimum {self.min_order_value}"
        
        # Check order size limit
        order_size_pct = estimated_value / account_balance if account_balance > 0 else 0
        if order_size_pct > self.max_order_size:
            return False, f"Order size {order_size_pct:.2%} exceeds max {self.max_order_size:.2%}"
        
        # Check total position limit
        new_position = current_position + order.quantity if order.side == "BUY" else current_position - order.quantity
        position_pct = abs(new_position) / account_balance if account_balance > 0 else 0
        if position_pct > self.max_position_size:
            return False, f"Position size {position_pct:.2%} exceeds max {self.max_position_size:.2%}"
        
        return True, None
    
    def check_emergency_exit(
        self,
        account_balance: float,
        initial_balance: float,
        daily_pnl: float
    ) -> tuple[bool, str]:
        """
        Check if emergency exit is needed.
        
        Args:
            account_balance: Current account balance
            initial_balance: Starting balance for the day
            daily_pnl: Today's PnL
            
        Returns:
            (should_exit, reason)
        """
        # Check daily loss
        daily_loss_pct = (initial_balance - account_balance) / initial_balance
        if daily_loss_pct >= self.max_daily_loss:
            return True, f"Emergency exit: daily loss {daily_loss_pct:.2%} exceeded"
        
        # Check balance drawdown
        drawdown = (initial_balance - account_balance) / initial_balance
        if drawdown >= 0.15:  # 15% drawdown
            return True, f"Emergency exit: drawdown {drawdown:.2%} too high"
        
        return False, ""


class OrderManager:
    """
    Production order manager with retry logic and error handling.
    """
    
    def __init__(
        self,
        broker,
        risk_manager: RiskManager,
        retry_config: Optional[RetryConfig] = None,
        on_order_filled: Optional[Callable] = None,
        on_order_cancelled: Optional[Callable] = None,
        on_error: Optional[Callable] = None
    ):
        """
        Initialize order manager.
        
        Args:
            broker: Broker interface
            risk_manager: Risk manager instance
            retry_config: Retry configuration
            on_order_filled: Callback when order is filled
            on_order_cancelled: Callback when order is cancelled
            on_error: Callback on error
        """
        self.broker = broker
        self.risk_manager = risk_manager
        self.retry_config = retry_config or RetryConfig()
        self.on_order_filled = on_order_filled
        self.on_order_cancelled = on_order_cancelled
        self.on_error = on_error
        
        # State tracking
        self.pending_orders: Dict[str, OrderRequest] = {}
        self.order_history: List[Dict] = []
        self.daily_stats = {
            'orders_placed': 0,
            'orders_filled': 0,
            'orders_cancelled': 0,
            'orders_rejected': 0,
            'total_volume': 0.0,
            'start_time': datetime.now()
        }
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
        
    async def place_order(
        self,
        order_request: OrderRequest,
        validate: bool = True
    ) -> Dict:
        """
        Place an order with retry logic.
        
        Args:
            order_request: Order to place
            validate: Whether to validate order
            
        Returns:
            Order result dictionary
        """
        async with self._lock:
            # Generate client order ID if not provided
            if not order_request.client_order_id:
                order_request.client_order_id = self._generate_order_id()
            
            self.pending_orders[order_request.client_order_id] = order_request
            
            # Log attempt
            self._log_order_event("ORDER_PLACED", order_request)
            
            # Validate if requested
            if validate:
                is_valid, error = await self._validate_order(order_request)
                if not is_valid:
                    await self._handle_rejection(order_request, error)
                    return {
                        'success': False,
                        'error': error,
                        'order_id': order_request.client_order_id
                    }
            
            # Execute with retry
            result = await self._execute_with_retry(order_request)
            
            return result
    
    async def _validate_order(self, order: OrderRequest) -> tuple[bool, Optional[str]]:
        """Validate order against risk limits."""
        try:
            balance = await self.broker.get_balance()
            positions = await self.broker.get_positions()
            
            current_position = 0
            for pos in positions:
                if pos.symbol == order.symbol:
                    current_position = pos.quantity
                    break
            
            daily_pnl = balance.unrealized_pnl + balance.realized_pnl
            
            return self.risk_manager.validate_order(
                order,
                balance.total_equity,
                current_position,
                daily_pnl
            )
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return True, None  # Allow if validation fails
    
    async def _execute_with_retry(self, order: OrderRequest) -> Dict:
        """Execute order with retry logic."""
        attempt = 0
        last_error = None
        
        while attempt <= self.retry_config.max_retries:
            try:
                # Attempt to place order
                result = await self._place_order_internal(order)
                
                if result.get('success'):
                    self.daily_stats['orders_filled'] += 1
                    self.daily_stats['orders_placed'] += 1
                    self._log_order_event("ORDER_FILLED", order, result)
                    
                    # Call callback
                    if self.on_order_filled:
                        await self.on_order_filled(result)
                    
                    return result
                else:
                    error = result.get('error', 'Unknown error')
                    
                    # Check if error is retryable
                    if not self._is_retryable(error):
                        self.daily_stats['orders_rejected'] += 1
                        await self._handle_rejection(order, error)
                        return result
                    
                    last_error = error
                    
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Order attempt {attempt + 1} failed: {e}")
            
            # Wait before retry
            if attempt < self.retry_config.max_retries:
                delay = self.retry_config.get_delay(attempt)
                logger.info(f"Retrying order in {delay:.1f}s (attempt {attempt + 1})")
                await asyncio.sleep(delay)
            
            attempt += 1
        
        # Max retries exceeded
        self.daily_stats['orders_rejected'] += 1
        error_msg = f"Max retries exceeded: {last_error}"
        await self._handle_rejection(order, error_msg)
        
        return {
            'success': False,
            'error': error_msg,
            'order_id': order.client_order_id
        }
    
    async def _place_order_internal(self, order: OrderRequest) -> Dict:
        """Internal order placement."""
        try:
            # Import Order class from broker_interface
            from src.production.broker_interface import (
                Order, OrderSide, OrderType, TimeInForce
            )
            
            # Convert to broker order
            broker_order = Order(
                symbol=order.symbol,
                side=OrderSide[order.side],
                order_type=OrderType[order.order_type],
                quantity=order.quantity,
                price=order.price,
                stop_price=order.stop_price,
                time_in_force=TimeInForce[order.time_in_force],
                client_order_id=order.client_order_id
            )
            
            # Place order through broker
            result_order = await self.broker.place_order(broker_order)
            
            return {
                'success': True,
                'order_id': result_order.order_id,
                'filled_quantity': result_order.filled_quantity,
                'avg_fill_price': result_order.avg_fill_price,
                'commission': result_order.commission,
                'status': result_order.status.value
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _is_retryable(self, error: str) -> bool:
        """Check if error is retryable."""
        error_lower = error.lower()
        for retryable in self.retry_config.retry_on:
            if retryable in error_lower:
                return True
        return False
    
    async def _handle_rejection(self, order: OrderRequest, error: str):
        """Handle order rejection."""
        logger.error(f"Order rejected: {error}")
        
        self._log_order_event("ORDER_REJECTED", order, {'error': error})
        
        if self.on_error:
            await self.on_error({
                'type': 'order_rejected',
                'order': order.to_dict(),
                'error': error
            })
    
    def _log_order_event(self, event: str, order: OrderRequest, extra: Dict = None):
        """Log order event to file."""
        log_dir = Path("logs/orders")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"order_manager_{datetime.now().strftime('%Y%m%d')}.jsonl"
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event': event,
            'order': order.to_dict(),
            'extra': extra or {}
        }
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        # Also log to main logger
        logger.info(f"Order event: {event} - {order.symbol} {order.side} {order.quantity}")
    
    def _generate_order_id(self) -> str:
        """Generate unique order ID."""
        import uuid
        return f"ord_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: Order ID to cancel
            symbol: Trading symbol
            
        Returns:
            True if successful, False otherwise
        """
        try:
            result = await self.broker.cancel_order(order_id, symbol)
            
            if result:
                self.daily_stats['orders_cancelled'] += 1
                
                if self.on_order_cancelled:
                    await self.on_order_cancelled({
                        'order_id': order_id,
                        'symbol': symbol
                    })
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    async def get_daily_stats(self) -> Dict:
        """Get daily statistics."""
        runtime = (datetime.now() - self.daily_stats['start_time']).total_seconds()
        
        return {
            **self.daily_stats,
            'runtime_seconds': runtime,
            'orders_per_minute': self.daily_stats['orders_placed'] / (runtime / 60) if runtime > 0 else 0
        }
    
    async def check_emergency_conditions(self) -> tuple[bool, str]:
        """Check if emergency conditions are met."""
        try:
            balance = await self.broker.get_balance()
            
            should_exit, reason = self.risk_manager.check_emergency_exit(
                balance.total_equity,
                self.daily_stats.get('initial_balance', balance.total_equity),
                balance.unrealized_pnl + balance.realized_pnl
            )
            
            if should_exit:
                logger.critical(f"EMERGENCY: {reason}")
                
                # Log emergency
                self._log_order_event("EMERGENCY_EXIT", OrderRequest(
                    symbol="ALL",
                    side="SELL",
                    quantity=0,
                    metadata={'reason': reason}
                ))
            
            return should_exit, reason
            
        except Exception as e:
            logger.error(f"Failed to check emergency conditions: {e}")
            return False, ""
    
    async def close_all_positions(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        Close all positions (emergency function).
        
        Args:
            symbol: Optional symbol to close, or all if None
            
        Returns:
            List of close results
        """
        logger.critical(f"Closing all positions: {symbol or 'ALL'}")
        
        results = []
        
        try:
            positions = await self.broker.get_positions()
            
            for position in positions:
                if symbol and position.symbol != symbol:
                    continue
                
                if position.quantity == 0:
                    continue
                
                # Create close order
                side = "SELL" if position.quantity > 0 else "BUY"
                
                close_order = OrderRequest(
                    symbol=position.symbol,
                    side=side,
                    quantity=abs(position.quantity),
                    order_type="MARKET",
                    metadata={'reason': 'emergency_close'}
                )
                
                result = await self.place_order(close_order, validate=False)
                results.append(result)
                
                logger.info(f"Closed position: {position.symbol} {position.quantity}")
            
        except Exception as e:
            logger.error(f"Error closing positions: {e}")
        
        return results
