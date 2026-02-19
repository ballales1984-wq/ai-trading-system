"""
Execution Engine
=============
Orchestrates order execution with risk management.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid

from app.execution.broker_connector import (
    BrokerConnector, Order, OrderSide, OrderType, OrderStatus
)


logger = logging.getLogger(__name__)


class ExecutionState(Enum):
    """Execution engine states."""
    IDLE = "idle"
    EXECUTING = "executing"
    RETRYING = "retrying"
    PAUSED = "paused"


@dataclass
class RetryConfig:
    """Retry configuration."""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0
    jitter: bool = True


@dataclass
class ExecutionResult:
    """Execution result."""
    success: bool
    order_id: Optional[str] = None
    message: str = ""
    filled_quantity: float = 0.0
    avg_price: float = 0.0
    commission: float = 0.0
    attempts: int = 1
    error: Optional[str] = None


class ExecutionEngine:
    """
    Manages order execution with retry logic and risk validation.
    """
    
    def __init__(
        self,
        broker: BrokerConnector,
        risk_engine=None,
        retry_config: Optional[RetryConfig] = None
    ):
        """
        Initialize execution engine.
        
        Args:
            broker: Broker connector
            risk_engine: Risk engine for validation
            retry_config: Retry configuration
        """
        self.broker = broker
        self.risk_engine = risk_engine
        self.retry_config = retry_config or RetryConfig()
        
        # State
        self.state = ExecutionState.IDLE
        
        # Order tracking
        self.pending_orders: Dict[str, Order] = {}
        self.order_history: List[Dict] = []
        
        # Callbacks
        self.on_order_placed: Optional[Callable] = None
        self.on_order_filled: Optional[Callable] = None
        self.on_order_failed: Optional[Callable] = None
        
        # Statistics
        self.stats = {
            'total_orders': 0,
            'successful_orders': 0,
            'failed_orders': 0,
            'total_retries': 0,
            'avg_execution_time': 0.0
        }
        
        logger.info("Execution engine initialized")
    
    async def execute_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "market",
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        validate_risk: bool = True,
        **kwargs
    ) -> ExecutionResult:
        """
        Execute order with retry logic.
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            quantity: Order quantity
            order_type: Order type
            price: Limit price
            stop_price: Stop price
            validate_risk: Whether to validate with risk engine
            
        Returns:
            ExecutionResult
        """
        start_time = datetime.now()
        
        # Validate with risk engine
        if validate_risk and self.risk_engine:
            risk_check = await self.risk_engine.check_order({
                'symbol': symbol,
                'quantity': quantity,
                'price': price or await self.broker.get_market_price(symbol),
                'side': side
            })
            
            if not risk_check.passed:
                logger.warning(f"Order rejected by risk: {risk_check.reason}")
                return ExecutionResult(
                    success=False,
                    message=risk_check.reason,
                    error='risk_rejected'
                )
        
        # Create order
        order = Order(
            order_id=f"EXEC_{uuid.uuid4().hex[:12]}",
            symbol=symbol,
            side=OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL,
            order_type=OrderType(order_type.lower()),
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            **kwargs
        )
        
        # Execute with retries
        last_error = None
        attempt = 0
        
        while attempt <= self.retry_config.max_retries:
            attempt += 1
            
            try:
                result = await self._execute_single(order, attempt)
                
                if result.success:
                    self._record_success(result, start_time)
                    
                    if self.on_order_filled:
                        await self.on_order_filled(result)
                    
                    return result
                
                else:
                    last_error = result.error
                    
                    if attempt <= self.retry_config.max_retries:
                        delay = self._get_delay(attempt - 1)
                        logger.warning(
                            f"Order failed (attempt {attempt}), retrying in {delay:.1f}s"
                        )
                        
                        self.stats['total_retries'] += 1
                        await asyncio.sleep(delay)
            
            except Exception as e:
                last_error = str(e)
                logger.error(f"Order execution exception: {e}")
                
                if attempt <= self.retry_config.max_retries:
                    delay = self._get_delay(attempt - 1)
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
            await self.on_order_failed(result)
        
        return result
    
    async def _execute_single(
        self,
        order: Order,
        attempt: int
    ) -> ExecutionResult:
        """Execute single order attempt."""
        self.state = ExecutionState.EXECUTING
        
        try:
            # Place order
            filled_order = await self.broker.place_order(order)
            
            # Check status
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
    
    def _get_delay(self, attempt: int) -> float:
        """Calculate retry delay."""
        delay = min(
            self.retry_config.base_delay * (self.retry_config.exponential_base ** attempt),
            self.retry_config.max_delay
        )
        
        if self.retry_config.jitter:
            import random
            delay *= (0.5 + random.random())
        
        return delay
    
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
        
        self.order_history.append({
            'success': False,
            'error': error,
            'execution_time': execution_time,
            'timestamp': datetime.now().isoformat()
        })
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order."""
        try:
            return await self.broker.cancel_order(order_id)
        except Exception as e:
            logger.error(f"Failed to cancel order: {e}")
            return False
    
    async def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """Cancel all pending orders."""
        try:
            orders = await self.broker.get_open_orders()
            cancelled = 0
            
            for order in orders:
                if symbol is None or order.symbol == symbol:
                    if await self.cancel_order(order.order_id):
                        cancelled += 1
            
            return cancelled
            
        except Exception as e:
            logger.error(f"Failed to cancel orders: {e}")
            return 0
    
    async def close_position(
        self,
        symbol: str,
        quantity: Optional[float] = None
    ) -> ExecutionResult:
        """Close position."""
        try:
            return await self.broker.close_position(symbol, quantity)
        except Exception as e:
            logger.error(f"Failed to close position: {e}")
            return ExecutionResult(success=False, error=str(e))
    
    async def close_all_positions(self) -> List[ExecutionResult]:
        """Close all positions."""
        return await self.broker.close_all_positions()
    
    def get_stats(self) -> Dict:
        """Get execution statistics."""
        return {
            **self.stats,
            'pending_orders': len(self.pending_orders),
            'success_rate': (
                self.stats['successful_orders'] / self.stats['total_orders']
                if self.stats['total_orders'] > 0 else 0
            )
        }
    
    def pause(self):
        """Pause execution."""
        self.state = ExecutionState.PAUSED
        logger.warning("Execution engine paused")
    
    def resume(self):
        """Resume execution."""
        self.state = ExecutionState.IDLE
        logger.info("Execution engine resumed")

