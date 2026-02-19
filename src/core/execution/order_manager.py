# src/core/execution/order_manager.py
"""
Order Manager
=============
Manages order execution with retry logic, risk validation, 
and comprehensive error handling.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid


logger = logging.getLogger(__name__)


class OrderManagerState(Enum):
    """Order manager states."""
    IDLE = "idle"
    EXECUTING = "executing"
    RETRYING = "retrying"
    PAUSED = "paused"


@dataclass
class RetryConfig:
    """Retry configuration."""
    max_retries: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 30.0  # seconds
    exponential_base: float = 2.0
    jitter: bool = True
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for attempt."""
        delay = min(
            self.base_delay * (self.exponential_base ** attempt),
            self.max_delay
        )
        
        if self.jitter:
            import random
            delay *= (0.5 + random.random())
        
        return delay


@dataclass
class OrderRequest:
    """Order request."""
    symbol: str
    side: str  # buy/sell
    quantity: float
    order_type: str = "market"  # market/limit
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"
    reduce_only: bool = False
    
    # Metadata
    signal_id: Optional[str] = None
    confidence: float = 1.0
    strategy: str = "default"
    tags: List[str] = field(default_factory=list)


@dataclass
class ExecutionResult:
    """Order execution result."""
    success: bool
    order_id: Optional[str] = None
    message: str = ""
    filled_quantity: float = 0.0
    avg_price: float = 0.0
    commission: float = 0.0
    attempts: int = 1
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'success': self.success,
            'order_id': self.order_id,
            'message': self.message,
            'filled_quantity': self.filled_quantity,
            'avg_price': self.avg_price,
            'commission': self.commission,
            'attempts': self.attempts,
            'error': self.error
        }


class OrderManager:
    """
    Manages order execution with retry logic and risk validation.
    """
    
    def __init__(
        self,
        broker,
        risk_engine=None,
        retry_config: Optional[RetryConfig] = None
    ):
        """
        Initialize order manager.
        
        Args:
            broker: Broker instance
            risk_engine: Risk engine for validation
            retry_config: Retry configuration
        """
        self.broker = broker
        self.risk_engine = risk_engine
        self.retry_config = retry_config or RetryConfig()
        
        # State
        self.state = OrderManagerState.IDLE
        
        # Order tracking
        self.pending_orders: Dict[str, OrderRequest] = {}
        self.order_history: List[Dict] = []
        
        # Callbacks
        self.on_order_placed: Optional[Callable] = None
        self.on_order_filled: Optional[Callable] = None
        self.on_order_failed: Optional[Callable] = None
        self.on_retry: Optional[Callable] = None
        
        # Statistics
        self.stats = {
            'total_orders': 0,
            'successful_orders': 0,
            'failed_orders': 0,
            'total_retries': 0,
            'avg_execution_time': 0.0
        }
        
        logger.info("Order manager initialized")
    
    async def execute_order(
        self,
        request: OrderRequest,
        validate_risk: bool = True
    ) -> ExecutionResult:
        """
        Execute order with retry logic.
        
        Args:
            request: Order request
            validate_risk: Whether to validate with risk engine
            
        Returns:
            ExecutionResult
        """
        start_time = datetime.now()
        
        # Validate with risk engine
        if validate_risk and self.risk_engine:
            check_result = self.risk_engine.check_order({
                'symbol': request.symbol,
                'quantity': request.quantity,
                'price': request.price or await self.broker.get_market_price(request.symbol),
                'side': request.side
            })
            
            if not check_result.passed:
                logger.warning(f"Order rejected by risk: {check_result.reason}")
                return ExecutionResult(
                    success=False,
                    message=check_result.reason,
                    error='risk_rejected'
                )
        
        # Execute with retries
        last_error = None
        attempt = 0
        
        while attempt <= self.retry_config.max_retries:
            attempt += 1
            
            try:
                result = await self._execute_single(request, attempt)
                
                if result.success:
                    # Success
                    self._record_success(result, start_time)
                    
                    if self.on_order_filled:
                        await self.on_order_filled(result.to_dict())
                    
                    return result
                
                else:
                    # Order failed
                    last_error = result.error
                    
                    if attempt <= self.retry_config.max_retries:
                        # Retry
                        delay = self.retry_config.get_delay(attempt - 1)
                        logger.warning(
                            f"Order failed (attempt {attempt}), retrying in {delay:.1f}s: {last_error}"
                        )
                        
                        self.stats['total_retries'] += 1
                        
                        if self.on_retry:
                            await self.on_retry({
                                'attempt': attempt,
                                'delay': delay,
                                'error': last_error,
                                'request': request.__dict__
                            })
                        
                        await asyncio.sleep(delay)
                    else:
                        # Max retries reached
                        logger.error(f"Order failed after {attempt} attempts: {last_error}")
            
            except Exception as e:
                last_error = str(e)
                logger.error(f"Order execution exception: {e}")
                
                if attempt <= self.retry_config.max_retries:
                    delay = self.retry_config.get_delay(attempt - 1)
                    await asyncio.sleep(delay)
                else:
                    break
        
        # Failed
        self._record_failure(last_error, start_time)
        
        result = ExecutionResult(
            success=False,
            message=f"Failed after {attempt} attempts",
            error=last_error,
            attempts=attempt
        )
        
        if self.on_order_failed:
            await self.on_order_failed(result.to_dict())
        
        return result
    
    async def _execute_single(
        self,
        request: OrderRequest,
        attempt: int
    ) -> ExecutionResult:
        """Execute single order attempt."""
        from src.core.execution.broker_interface import (
            Order, OrderSide, OrderType, OrderStatus
        )
        
        self.state = OrderManagerState.EXECUTING
        
        # Create order
        order = Order(
            order_id=f"ORD_{uuid.uuid4().hex[:12]}",
            symbol=request.symbol,
            side=OrderSide.BUY if request.side.lower() == 'buy' else OrderSide.SELL,
            order_type=OrderType(request.order_type.lower()),
            quantity=request.quantity,
            price=request.price,
            stop_price=request.stop_price,
            time_in_force=request.time_in_force,
            reduce_only=request.reduce_only
        )
        
        try:
            # Place order through broker
            filled_order = await self.broker.place_order(order)
            
            # Check if filled
            if filled_order.status == OrderStatus.FILLED:
                return ExecutionResult(
                    success=True,
                    order_id=filled_order.order_id,
                    message="Order filled",
                    filled_quantity=filled_order.filled_quantity,
                    avg_price=filled_order.avg_fill_price,
                    commission=filled_order.commission,
                    attempts=attempt
                )
            elif filled_order.status == OrderStatus.PARTIALLY_FILLED:
                return ExecutionResult(
                    success=True,
                    order_id=filled_order.order_id,
                    message="Order partially filled",
                    filled_quantity=filled_order.filled_quantity,
                    avg_price=filled_order.avg_fill_price,
                    commission=filled_order.commission,
                    attempts=attempt
                )
            else:
                return ExecutionResult(
                    success=False,
                    order_id=filled_order.order_id,
                    message=f"Order status: {filled_order.status.value}",
                    error=filled_order.status.value,
                    attempts=attempt
                )
        
        except Exception as e:
            return ExecutionResult(
                success=False,
                message=str(e),
                error='execution_error',
                attempts=attempt
            )
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order."""
        try:
            success = await self.broker.cancel_order(order_id)
            
            if success:
                logger.info(f"Order cancelled: {order_id}")
                
                # Remove from pending
                for req_id, req in list(self.pending_orders.items()):
                    if hasattr(req, 'order_id') and req.order_id == order_id:
                        del self.pending_orders[req_id]
                        break
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to cancel order: {e}")
            return False
    
    async def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """Cancel all pending orders."""
        cancelled = 0
        
        try:
            # Get open orders
            open_orders = await self.broker.get_open_orders()
            
            for order in open_orders:
                if symbol is None or order.symbol == symbol:
                    if await self.cancel_order(order.order_id):
                        cancelled += 1
            
            logger.info(f"Cancelled {cancelled} orders" + (f" for {symbol}" if symbol else ""))
            
        except Exception as e:
            logger.error(f"Failed to cancel orders: {e}")
        
        return cancelled
    
    async def close_position(
        self,
        symbol: str,
        quantity: Optional[float] = None,
        order_type: str = "market"
    ) -> ExecutionResult:
        """Close position completely or partially."""
        try:
            # Get current position
            position = await self.broker.get_position(symbol)
            
            if not position or position.quantity == 0:
                return ExecutionResult(
                    success=False,
                    message="No open position"
                )
            
            # Determine close quantity
            close_qty = quantity if quantity else abs(position.quantity)
            side = "sell" if position.quantity > 0 else "buy"
            
            # Create order request
            request = OrderRequest(
                symbol=symbol,
                side=side,
                quantity=close_qty,
                order_type=order_type,
                price=position.current_price if order_type == "limit" else None,
                reduce_only=True,
                strategy="close_position"
            )
            
            return await self.execute_order(request)
            
        except Exception as e:
            logger.error(f"Failed to close position: {e}")
            return ExecutionResult(
                success=False,
                message=str(e),
                error='close_position_error'
            )
    
    async def close_all_positions(self) -> Dict:
        """Close all open positions."""
        results = {
            'closed': [],
            'failed': []
        }
        
        try:
            positions = await self.broker.get_positions()
            
            for position in positions:
                if position.quantity != 0:
                    result = await self.close_position(position.symbol)
                    
                    if result.success:
                        results['closed'].append({
                            'symbol': position.symbol,
                            'quantity': result.filled_quantity
                        })
                    else:
                        results['failed'].append({
                            'symbol': position.symbol,
                            'error': result.error
                        })
            
            logger.info(
                f"Close positions: {len(results['closed'])} closed, "
                f"{len(results['failed'])} failed"
            )
            
        except Exception as e:
            logger.error(f"Failed to close positions: {e}")
        
        return results
    
    def _record_success(self, result: ExecutionResult, start_time: datetime):
        """Record successful execution."""
        self.stats['successful_orders'] += 1
        self.stats['total_orders'] += 1
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Update average
        total = self.stats['successful_orders']
        old_avg = self.stats['avg_execution_time']
        self.stats['avg_execution_time'] = (
            (old_avg * (total - 1) + execution_time) / total
        )
        
        # Record in history
        self.order_history.append({
            'order_id': result.order_id,
            'success': True,
            'execution_time': execution_time,
            'attempts': result.attempts,
            'timestamp': datetime.now().isoformat()
        })
    
    def _record_failure(self, error: str, start_time: datetime):
        """Record failed execution."""
        self.stats['failed_orders'] += 1
        self.stats['total_orders'] += 1
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Record in history
        self.order_history.append({
            'success': False,
            'error': error,
            'execution_time': execution_time,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_stats(self) -> Dict:
        """Get order manager statistics."""
        return {
            **self.stats,
            'pending_orders': len(self.pending_orders),
            'success_rate': (
                self.stats['successful_orders'] / self.stats['total_orders']
                if self.stats['total_orders'] > 0 else 0
            )
        }
    
    def get_order_history(
        self,
        limit: int = 100,
        symbol: Optional[str] = None
    ) -> List[Dict]:
        """Get order history."""
        history = self.order_history[-limit:]
        
        if symbol:
            # Filter by symbol from stored order history
            history = [
                h for h in history
                if h.get('symbol', '') == symbol
            ]
        
        return history
    
    def pause(self):
        """Pause order execution."""
        self.state = OrderManagerState.PAUSED
        logger.warning("Order manager paused")
    
    def resume(self):
        """Resume order execution."""
        self.state = OrderManagerState.IDLE
        logger.info("Order manager resumed")
    
    async def get_open_orders(self) -> List[Dict]:
        """Get all open orders."""
        try:
            orders = await self.broker.get_open_orders()
            return [o.to_dict() for o in orders]
        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            return []


class EmergencyExit:
    """Emergency exit handler for critical situations."""
    
    def __init__(self, order_manager: OrderManager):
        self.order_manager = order_manager
    
    async def execute_emergency_exit(self, reason: str) -> Dict:
        """
        Execute emergency exit - close all positions and cancel orders.
        
        Args:
            reason: Reason for emergency exit
            
        Returns:
            Exit result
        """
        logger.critical(f"EMERGENCY EXIT: {reason}")
        
        result = {
            'reason': reason,
            'timestamp': datetime.now().isoformat(),
            'cancelled_orders': 0,
            'closed_positions': [],
            'errors': []
        }
        
        try:
            # Cancel all pending orders
            result['cancelled_orders'] = await self.order_manager.cancel_all_orders()
            
            # Close all positions
            close_result = await self.order_manager.close_all_positions()
            result['closed_positions'] = close_result.get('closed', [])
            result['errors'] = close_result.get('failed', [])
            
            logger.critical(
                f"Emergency exit complete: {result['cancelled_orders']} orders cancelled, "
                f"{len(result['closed_positions'])} positions closed"
            )
            
        except Exception as e:
            logger.error(f"Emergency exit failed: {e}")
            result['errors'].append(str(e))
        
        return result
