"""
Algorithmic Execution Engine
============================
Implementation of TWAP, VWAP, and other professional execution algorithms.
"""

import asyncio
import logging
import math
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from app.execution.broker_connector import BrokerConnector, BrokerOrder, OrderStatus, OrderSide

logger = logging.getLogger(__name__)

@dataclass
class AlgoConfig:
    """Configuration for algorithmic execution."""
    symbol: str
    side: OrderSide
    total_quantity: float
    duration_minutes: int
    num_chunks: int = 10
    limit_price: Optional[float] = None
    min_chunk_size: float = 0.001

class AlgoExecutor(ABC):
    """Base class for algorithmic execution engines."""
    
    def __init__(self, broker: BrokerConnector, config: AlgoConfig):
        self.broker = broker
        self.config = config
        self.is_running = False
        self.executed_quantity = 0.0
        self.orders: List[BrokerOrder] = []
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

    @abstractmethod
    async def execute(self):
        """Main execution loop."""
        pass

    def get_progress(self) -> float:
        """Get execution progress (0.0 to 1.0)."""
        if self.config.total_quantity <= 0:
            return 1.0
        return self.executed_quantity / self.config.total_quantity

    def get_status(self) -> Dict[str, Any]:
        """Get technical status of the execution."""
        return {
            "symbol": self.config.symbol,
            "side": self.config.side.value,
            "progress": self.get_progress(),
            "executed_qty": self.executed_quantity,
            "total_qty": self.config.total_quantity,
            "is_running": self.is_running,
            "avg_price": self._calculate_avg_price()
        }

    def _calculate_avg_price(self) -> float:
        """Calculate weighted average price of executed orders."""
        filled_orders = [o for o in self.orders if o.status == OrderStatus.FILLED]
        if not filled_orders:
            return 0.0
        
        total_value = sum(o.filled_quantity * (o.average_price or 0) for o in filled_orders)
        total_qty = sum(o.filled_quantity for o in filled_orders)
        
        return total_value / total_qty if total_qty > 0 else 0.0

class TWAPExecutor(AlgoExecutor):
    """
    Time-Weighted Average Price (TWAP)
    ----------------------------------
    Executes an order by splitting it into equal time intervals.
    """
    
    async def execute(self):
        self.is_running = True
        self.start_time = datetime.now()
        self.end_time = self.start_time + timedelta(minutes=self.config.duration_minutes)
        
        chunk_size = self.config.total_quantity / self.config.num_chunks
        interval_seconds = (self.config.duration_minutes * 60) / self.config.num_chunks
        
        logger.info(f"Starting TWAP for {self.config.symbol}: {self.config.total_quantity} qty over {self.config.duration_minutes}m")
        
        for i in range(self.config.num_chunks):
            if not self.is_running:
                break
                
            # Calculate remaining quantity to ensure precision
            if i == self.config.num_chunks - 1:
                current_chunk = self.config.total_quantity - self.executed_quantity
            else:
                current_chunk = chunk_size
                
            if current_chunk < self.config.min_chunk_size:
                continue

            # Place small order
            order = BrokerOrder(
                symbol=self.config.symbol,
                side=self.config.side.value,
                order_type="MARKET" if not self.config.limit_price else "LIMIT",
                quantity=current_chunk,
                price=self.config.limit_price
            )
            
            try:
                executed_order = await self.broker.place_order(order)
                self.orders.append(executed_order)
                
                if executed_order.status == OrderStatus.FILLED:
                    self.executed_quantity += executed_order.filled_quantity
                    logger.debug(f"TWAP chunk {i+1}/{self.config.num_chunks} filled: {executed_order.filled_quantity}")
                
            except Exception as e:
                logger.error(f"TWAP chunk {i+1} failed: {e}")
            
            # Wait for next interval
            if i < self.config.num_chunks - 1:
                await asyncio.sleep(interval_seconds)
        
        self.is_running = False
        logger.info(f"TWAP completed for {self.config.symbol}. Executed: {self.executed_quantity}")

class VWAPExecutor(AlgoExecutor):
    """
    Volume-Weighted Average Price (VWAP)
    -----------------------------------
    Executes orders based on a volume profile. 
    In this version, it uses a predicted volume distribution (U-curve).
    """
    
    def _get_volume_profile(self, num_chunks: int) -> List[float]:
        """Simulates a typical U-shaped volume curve (higher at start/end of day)."""
        # simplified U-curve using a parabola
        profile = []
        for i in range(num_chunks):
            x = (i / (num_chunks - 1)) * 2 - 1  # -1 to 1
            weight = x*x + 0.5  # U-shape
            profile.append(weight)
        
        # Normalize
        total_weight = sum(profile)
        return [w / total_weight for w in profile]

    async def execute(self):
        self.is_running = True
        self.start_time = datetime.now()
        
        profile = self._get_volume_profile(self.config.num_chunks)
        interval_seconds = (self.config.duration_minutes * 60) / self.config.num_chunks
        
        logger.info(f"Starting VWAP (Simulated Profile) for {self.config.symbol}")
        
        for i, weight in enumerate(profile):
            if not self.is_running:
                break
            
            current_chunk = self.config.total_quantity * weight
            if current_chunk < self.config.min_chunk_size:
                continue
                
            order = BrokerOrder(
                symbol=self.config.symbol,
                side=self.config.side.value,
                order_type="MARKET",
                quantity=current_chunk
            )
            
            try:
                executed_order = await self.broker.place_order(order)
                self.orders.append(executed_order)
                self.executed_quantity += executed_order.filled_quantity
            except Exception as e:
                logger.error(f"VWAP chunk {i+1} failed: {e}")
                
            if i < self.config.num_chunks - 1:
                await asyncio.sleep(interval_seconds)
                
        self.is_running = False
        logger.info(f"VWAP completed for {self.config.symbol}")
