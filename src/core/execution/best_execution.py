# src/core/execution/best_execution.py
"""
Best Execution Engine
====================
Smart order routing and execution algorithms:
- TWAP (Time-Weighted Average Price)
- VWAP (Volume-Weighted Average Price)
- POV (Percentage of Volume)
- Adaptive execution strategies
- Smart order router interface

Author: AI Trading System
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class ExecutionStrategy(Enum):
    """Execution strategy types."""
    MARKET = "market"
    LIMIT = "limit"
    TWAP = "twap"
    VWAP = "vwap"
    POV = "pov"
    IS = "implementation_shortfall"
    ADAPTIVE = "adaptive"


class ExecutionStatus(Enum):
    """Execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class ExecutionSlice:
    """Individual execution slice."""
    slice_id: int
    timestamp: datetime
    quantity: float
    price: Optional[float] = None
    status: str = "pending"
    filled_quantity: float = 0.0
    commission: float = 0.0
    
    @property
    def fill_rate(self) -> float:
        """Fill rate for this slice."""
        return self.filled_quantity / self.quantity if self.quantity > 0 else 0


@dataclass
class ExecutionPlan:
    """Complete execution plan."""
    strategy: ExecutionStrategy
    symbol: str
    side: str  # 'buy' or 'sell'
    total_quantity: float
    start_time: datetime
    end_time: datetime
    slices: List[ExecutionSlice] = field(default_factory=list)
    status: ExecutionStatus = ExecutionStatus.PENDING
    
    @property
    def total_filled(self) -> float:
        """Total filled quantity."""
        return sum(s.filled_quantity for s in self.slices)
    
    @property
    def fill_percentage(self) -> float:
        """Percentage filled."""
        return (self.total_filled / self.total_quantity * 100) if self.total_quantity > 0 else 0
    
    @property
    def remaining_quantity(self) -> float:
        """Remaining quantity to fill."""
        return self.total_quantity - self.total_filled


@dataclass
class MarketDataSnapshot:
    """Market data for execution decisions."""
    timestamp: datetime
    symbol: str
    last_price: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    mid: float = 0.0
    volume: float = 0.0
    vwap: float = 0.0
    volatility: float = 0.02
    bid_ask_spread: float = 0.0
    market_volume_1m: float = 0.0  # 1-minute volume
    market_volume_5m: float = 0.0  # 5-minute volume
    
    @property
    def spread_bps(self) -> float:
        """Spread in basis points."""
        return (self.bid_ask_spread / self.mid * 10000) if self.mid > 0 else 0


@dataclass 
class ExecutionConfig:
    """Configuration for execution algorithm."""
    strategy: ExecutionStrategy = ExecutionStrategy.VWAP
    
    # Timing
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: float = 300.0  # Default 5 minutes
    
    # Participation rates
    target_participation_rate: float = 0.10  # 10% of volume
    max_participation_rate: float = 0.25  # 25% max
    
    # Order types
    use_limit_orders: bool = True
    limit_offset_bps: float = 2.0  # Offset from market for limit orders
    
    # Risk controls
    max_slippage_bps: float = 10.0  # Max slippage tolerance
    auto_cancel_threshold_bps: float = 20.0  # Cancel if slippage exceeds
    
    # Adaptive parameters
    adaptation_enabled: bool = True
    volatility_adjustment: bool = True
    liquidity_adjustment: bool = True
    
    # Execution style
    urgency: float = 0.5  # 0 = patient, 1 = aggressive
    

class BaseExecutionAlgorithm(ABC):
    """Base class for execution algorithms."""
    
    def __init__(self, config: ExecutionConfig):
        """Initialize algorithm."""
        self.config = config
        self.plan: Optional[ExecutionPlan] = None
        
    @abstractmethod
    def create_execution_plan(
        self,
        symbol: str,
        side: str,
        quantity: float,
        market_data: MarketDataSnapshot,
    ) -> ExecutionPlan:
        """Create execution plan."""
        pass
    
    @abstractmethod
    def calculate_slice_size(
        self,
        slice_num: int,
        total_slices: int,
        remaining_quantity: float,
        market_data: MarketDataSnapshot,
    ) -> float:
        """Calculate size for next slice."""
        pass
    
    @abstractmethod
    def should_execute(
        self,
        current_slice: ExecutionSlice,
        market_data: MarketDataSnapshot,
    ) -> bool:
        """Determine if should execute now."""
        pass


class TWAPAlgorithm(BaseExecutionAlgorithm):
    """
    TWAP (Time-Weighted Average Price) Algorithm
    ==============================================
    Executes orders in equal time intervals.
    """
    
    def create_execution_plan(
        self,
        symbol: str,
        side: str,
        quantity: float,
        market_data: MarketDataSnapshot,
    ) -> ExecutionPlan:
        """Create TWAP execution plan."""
        # Determine number of slices based on duration
        slice_interval = 30  # 30 seconds between slices
        num_slices = max(1, int(self.config.duration_seconds / slice_interval))
        
        # Create slices
        now = market_data.timestamp
        slices = []
        
        for i in range(num_slices):
            slice_time = now + timedelta(seconds=i * slice_interval)
            slice_qty = quantity / num_slices
            
            slices.append(ExecutionSlice(
                slice_id=i,
                timestamp=slice_time,
                quantity=slice_qty,
            ))
        
        return ExecutionPlan(
            strategy=ExecutionStrategy.TWAP,
            symbol=symbol,
            side=side,
            total_quantity=quantity,
            start_time=now,
            end_time=now + timedelta(seconds=self.config.duration_seconds),
            slices=slices,
        )
    
    def calculate_slice_size(
        self,
        slice_num: int,
        total_slices: int,
        remaining_quantity: float,
        market_data: MarketDataSnapshot,
    ) -> float:
        """TWAP uses equal slice sizes."""
        return remaining_quantity / (total_slices - slice_num)
    
    def should_execute(
        self,
        current_slice: ExecutionSlice,
        market_data: MarketDataSnapshot,
    ) -> bool:
        """Execute at scheduled time regardless of market conditions."""
        return True


class VWAPAlgorithm(BaseExecutionAlgorithm):
    """
    VWAP (Volume-Weighted Average Price) Algorithm
    ================================================
    Matches historical volume profile for execution.
    """
    
    def __init__(self, config: ExecutionConfig):
        """Initialize VWAP algorithm."""
        super().__init__(config)
        # Volume profile (can be calibrated from historical data)
        # Default: uniform distribution
        self.volume_profile: List[float] = []
        
    def set_volume_profile(self, profile: List[float]):
        """Set historical volume profile."""
        # Normalize profile
        total = sum(profile)
        if total > 0:
            self.volume_profile = [p / total for p in profile]
        else:
            self.volume_profile = []
    
    def create_execution_plan(
        self,
        symbol: str,
        side: str,
        quantity: float,
        market_data: MarketDataSnapshot,
    ) -> ExecutionPlan:
        """Create VWAP execution plan."""
        # Use volume profile if available, otherwise use time-based
        if self.volume_profile:
            profile = self.volume_profile
        else:
            # Default: uniform profile
            slice_interval = 30
            num_slices = max(1, int(self.config.duration_seconds / slice_interval))
            profile = [1.0 / num_slices] * num_slices
        
        # Create slices based on profile
        now = market_data.timestamp
        slice_interval = 30
        slices = []
        
        for i, portion in enumerate(profile):
            slice_time = now + timedelta(seconds=i * slice_interval)
            slice_qty = quantity * portion
            
            slices.append(ExecutionSlice(
                slice_id=i,
                timestamp=slice_time,
                quantity=slice_qty,
            ))
        
        return ExecutionPlan(
            strategy=ExecutionStrategy.VWAP,
            symbol=symbol,
            side=side,
            total_quantity=quantity,
            start_time=now,
            end_time=now + timedelta(seconds=len(profile) * slice_interval),
            slices=slices,
        )
    
    def calculate_slice_size(
        self,
        slice_num: int,
        total_slices: int,
        remaining_quantity: float,
        market_data: MarketDataSnapshot,
    ) -> float:
        """VWAP uses volume profile for slice sizes."""
        if slice_num < len(self.volume_profile):
            target_portion = self.volume_profile[slice_num]
        else:
            target_portion = 1.0 / total_slices
            
        return remaining_quantity * target_portion
    
    def should_execute(
        self,
        current_slice: ExecutionSlice,
        market_data: MarketDataSnapshot,
    ) -> bool:
        """Execute based on volume participation target."""
        # Check if current volume justifies execution
        if market_data.market_volume_1m > 0:
            participation = current_slice.quantity / market_data.market_volume_1m
            
            # Execute if within participation bounds
            if participation <= self.config.max_participation_rate:
                return True
                
        return False


class POVAlgorithm(BaseExecutionAlgorithm):
    """
    POV (Percentage of Volume) Algorithm
    =====================================
    Executes at a fixed percentage of market volume.
    """
    
    def create_execution_plan(
        self,
        symbol: str,
        side: str,
        quantity: float,
        market_data: MarketDataSnapshot,
    ) -> ExecutionPlan:
        """Create POV execution plan."""
        # Estimate number of slices needed
        target_rate = self.config.target_participation_rate
        
        # If we expect volume V over duration D, and want to trade at rate r,
        # we need approximately 1/(r*D) slices
        if market_data.market_volume_1m > 0 and self.config.duration_seconds > 0:
            expected_volume = market_data.market_volume_1m * (self.config.duration_seconds / 60)
            num_slices = max(1, int(quantity / (expected_volume * target_rate)))
        else:
            num_slices = max(1, int(self.config.duration_seconds / 30))
        
        # Create initial plan (will be adjusted dynamically)
        now = market_data.timestamp
        slices = []
        
        for i in range(num_slices):
            slice_time = now + timedelta(seconds=i * 30)
            # Initial slice size will be adjusted based on actual volume
            slice_qty = quantity / num_slices
            
            slices.append(ExecutionSlice(
                slice_id=i,
                timestamp=slice_time,
                quantity=slice_qty,
            ))
        
        return ExecutionPlan(
            strategy=ExecutionStrategy.POV,
            symbol=symbol,
            side=side,
            total_quantity=quantity,
            start_time=now,
            end_time=now + timedelta(seconds=self.config.duration_seconds),
            slices=slices,
        )
    
    def calculate_slice_size(
        self,
        slice_num: int,
        total_slices: int,
        remaining_quantity: float,
        market_data: MarketDataSnapshot,
    ) -> float:
        """POV calculates size based on target participation rate."""
        # Get expected volume for next interval
        expected_volume = market_data.market_volume_1m
        
        if expected_volume > 0:
            target_quantity = expected_volume * self.config.target_participation_rate
        else:
            # Fallback to equal sizing
            target_quantity = remaining_quantity / (total_slices - slice_num)
        
        # Cap at remaining quantity
        return min(target_quantity, remaining_quantity)
    
    def should_execute(
        self,
        current_slice: ExecutionSlice,
        market_data: MarketDataSnapshot,
    ) -> bool:
        """Execute if market volume is sufficient."""
        min_volume_threshold = current_slice.quantity / self.config.target_participation_rate
        
        if market_data.market_volume_1m >= min_volume_threshold * 0.5:  # At least 50% of expected
            return True
            
        return False


class AdaptiveAlgorithm(BaseExecutionAlgorithm):
    """
    Adaptive Execution Algorithm
    ============================
    Adjusts execution strategy based on market conditions.
    """
    
    def __init__(self, config: ExecutionConfig):
        """Initialize adaptive algorithm."""
        super().__init__(config)
        
        # Track market conditions
        self.market_regime = "normal"  # 'normal', 'volatile', 'illiquid'
        self.recent_slippage: List[float] = []
        self.recent_imbalance: List[float] = []
        
    def create_execution_plan(
        self,
        symbol: str,
        side: str,
        quantity: float,
        market_data: MarketDataSnapshot,
    ) -> ExecutionPlan:
        """Create adaptive execution plan."""
        # Determine market regime
        self._update_market_regime(market_data)
        
        # Adjust parameters based on regime
        if self.market_regime == "volatile":
            # More aggressive in volatile markets
            urgency = min(1.0, self.config.urgency * 1.5)
            participation = min(0.25, self.config.target_participation_rate * 1.5)
            duration = max(60, self.config.duration_seconds * 0.7)
        elif self.market_regime == "illiquid":
            # More patient in illiquid markets
            urgency = max(0.0, self.config.urgency * 0.7)
            participation = max(0.05, self.config.target_participation_rate * 0.7)
            duration = self.config.duration_seconds * 1.5
        else:
            urgency = self.config.urgency
            participation = self.config.target_participation_rate
            duration = self.config.duration_seconds
        
        # Create plan with adjusted parameters
        slice_interval = 30
        num_slices = max(1, int(duration / slice_interval))
        
        now = market_data.timestamp
        slices = []
        
        for i in range(num_slices):
            slice_time = now + timedelta(seconds=i * slice_interval)
            slice_qty = quantity / num_slices
            
            slices.append(ExecutionSlice(
                slice_id=i,
                timestamp=slice_time,
                quantity=slice_qty,
            ))
        
        return ExecutionPlan(
            strategy=ExecutionStrategy.ADAPTIVE,
            symbol=symbol,
            side=side,
            total_quantity=quantity,
            start_time=now,
            end_time=now + timedelta(seconds=duration),
            slices=slices,
        )
    
    def _update_market_regime(self, market_data: MarketDataSnapshot):
        """Update market regime based on conditions."""
        # Check volatility
        if market_data.volatility > 0.03:  # > 3% volatility
            self.market_regime = "volatile"
        # Check liquidity (wide spread = illiquid)
        elif market_data.spread_bps > 10:  # > 10 bps spread
            self.market_regime = "illiquid"
        else:
            self.market_regime = "normal"
    
    def calculate_slice_size(
        self,
        slice_num: int,
        total_slices: int,
        remaining_quantity: float,
        market_data: MarketDataSnapshot,
    ) -> float:
        """Adaptive slice sizing based on recent execution quality."""
        # Base size
        base_size = remaining_quantity / (total_slices - slice_num)
        
        # Adjust based on recent slippage
        if self.recent_slippage:
            avg_slippage = np.mean(self.recent_slippage)
            
            if avg_slippage > self.config.max_slippage_bps:
                # Too much slippage - reduce size
                adjustment = 0.7
            elif avg_slippage < self.config.max_slippage_bps * 0.3:
                # Low slippage - can be more aggressive
                adjustment = 1.2
            else:
                adjustment = 1.0
        else:
            adjustment = 1.0
            
        # Adjust based on market regime
        if self.market_regime == "volatile":
            adjustment *= 0.8  # Reduce size in volatile markets
        elif self.market_regime == "illiquid":
            adjustment *= 0.6  # Significantly reduce in illiquid markets
            
        return base_size * adjustment
    
    def should_execute(
        self,
        current_slice: ExecutionSlice,
        market_data: MarketDataSnapshot,
    ) -> bool:
        """Adaptive execution decision."""
        # Check slippage tolerance
        if market_data.bid_ask_spread > 0:
            # Estimate potential slippage
            potential_slippage = market_data.bid_ask_spread / 2
            
            if potential_slippage > self.config.auto_cancel_threshold_bps:
                return False  # Too expensive
        
        # In volatile markets, be more selective
        if self.market_regime == "volatile":
            # Only execute if volume is high
            return market_data.market_volume_1m > 0
            
        return True
    
    def record_execution_result(
        self,
        slice_filled: float,
        slice_quantity: float,
        execution_price: float,
        arrival_price: float,
    ):
        """Record execution result for adaptation."""
        if slice_quantity > 0 and arrival_price > 0:
            slippage = abs(execution_price - arrival_price) / arrival_price * 10000
            self.recent_slippage.append(slippage)
            
            # Keep only recent history
            if len(self.recent_slippage) > 10:
                self.recent_slippage = self.recent_slippage[-10:]


class BestExecutionEngine:
    """
    Best Execution Engine
    ====================
    Orchestrates execution algorithms and smart order routing.
    """
    
    def __init__(self, config: Optional[ExecutionConfig] = None):
        """
        Initialize best execution engine.
        
        Args:
            config: Execution configuration
        """
        self.config = config or ExecutionConfig()
        
        # Initialize algorithms
        self.algorithms: Dict[ExecutionStrategy, BaseExecutionAlgorithm] = {
            ExecutionStrategy.TWAP: TWAPAlgorithm(self.config),
            ExecutionStrategy.VWAP: VWAPAlgorithm(self.config),
            ExecutionStrategy.POV: POVAlgorithm(self.config),
            ExecutionStrategy.ADAPTIVE: AdaptiveAlgorithm(self.config),
        }
        
        # Active execution plans
        self.active_plans: Dict[str, ExecutionPlan] = {}
        
        # Execution callbacks
        self.execute_callback: Optional[Callable] = None
        
    def create_execution_order(
        self,
        strategy: ExecutionStrategy,
        symbol: str,
        side: str,
        quantity: float,
        market_data: MarketDataSnapshot,
    ) -> ExecutionPlan:
        """
        Create an execution order with the specified strategy.
        
        Args:
            strategy: Execution strategy to use
            symbol: Trading symbol
            side: 'buy' or 'sell'
            quantity: Order quantity
            market_data: Current market data
            
        Returns:
            ExecutionPlan
        """
        # Get algorithm
        algorithm = self.algorithms.get(strategy)
        
        if algorithm is None:
            # Default to VWAP
            algorithm = self.algorithms[ExecutionStrategy.VWAP]
            
        # Create plan
        plan = algorithm.create_execution_plan(symbol, side, quantity, market_data)
        
        # Store plan
        plan_id = f"{symbol}_{datetime.now().timestamp()}"
        self.active_plans[plan_id] = plan
        
        logger.info(f"Created {strategy.value} execution plan for {symbol}: {quantity}")
        
        return plan
    
    def get_next_slice(
        self,
        plan_id: str,
        market_data: MarketDataSnapshot,
    ) -> Optional[ExecutionSlice]:
        """
        Get the next execution slice.
        
        Args:
            plan_id: Execution plan ID
            market_data: Current market data
            
        Returns:
            Next ExecutionSlice to execute, or None
        """
        plan = self.active_plans.get(plan_id)
        
        if plan is None or plan.status == ExecutionStatus.COMPLETED:
            return None
            
        # Get next pending slice
        for slice_ in plan.slices:
            if slice_.status == "pending":
                # Check if algorithm says to execute
                algorithm = self.algorithms.get(plan.strategy)
                
                if algorithm and algorithm.should_execute(slice_, market_data):
                    return slice_
                    
        return None
    
    def update_slice_filled(
        self,
        plan_id: str,
        slice_id: int,
        filled_quantity: float,
        execution_price: float,
        commission: float = 0.0,
    ):
        """
        Update slice with fill information.
        
        Args:
            plan_id: Execution plan ID
            slice_id: Slice ID
            filled_quantity: Actual filled quantity
            execution_price: Execution price
            commission: Commission paid
        """
        plan = self.active_plans.get(plan_id)
        
        if plan is None:
            return
            
        # Find and update slice
        for slice_ in plan.slices:
            if slice_.slice_id == slice_id:
                slice_.filled_quantity = filled_quantity
                slice_.price = execution_price
                slice_.commission = commission
                slice_.status = "filled"
                break
                
        # Check if plan is complete
        if plan.total_filled >= plan.total_quantity * 0.99:  # 99% filled
            plan.status = ExecutionStatus.COMPLETED
            
        # Record for adaptive algorithm
        if plan.strategy == ExecutionStrategy.ADAPTIVE and execution_price > 0:
            algorithm = self.algorithms[ExecutionStrategy.ADAPTIVE]
            if hasattr(algorithm, 'record_execution_result'):
                algorithm.record_execution_result(
                    filled_quantity,
                    sum(s.quantity for s in plan.slices),
                    execution_price,
                    market_data.mid if hasattr(self, 'market_data') else execution_price
                )
    
    def calculate_expected_cost(
        self,
        plan: ExecutionPlan,
        market_data: MarketDataSnapshot,
    ) -> Dict[str, float]:
        """
        Calculate expected execution cost.
        
        Args:
            plan: Execution plan
            market_data: Current market data
            
        Returns:
            Expected cost breakdown
        """
        # Spread cost
        spread_cost = market_data.bid_ask_spread / 2
        
        # Market impact estimate (simplified)
        participation = plan.total_quantity / market_data.market_volume_1m if market_data.market_volume_1m > 0 else 0.1
        impact_coef = 0.0001
        market_impact = impact_coef * (participation ** 0.5) * market_data.mid
        
        # Total expected cost
        expected_cost = spread_cost + market_impact
        expected_cost_bps = (expected_cost / market_data.mid * 10000)
        
        return {
            "spread_cost": spread_cost,
            "market_impact": market_impact,
            "total_cost": expected_cost,
            "total_cost_bps": expected_cost_bps,
            "expected_slippage_bps": expected_cost_bps,
        }
    
    def select_best_strategy(
        self,
        symbol: str,
        quantity: float,
        market_data: MarketDataSnapshot,
    ) -> Tuple[ExecutionStrategy, Dict[str, Any]]:
        """
        Select the best execution strategy based on conditions.
        
        Args:
            symbol: Trading symbol
            quantity: Order quantity
            market_data: Current market data
            
        Returns:
            Tuple of (selected strategy, analysis)
        """
        # Analyze market conditions
        conditions = self._analyze_market_conditions(market_data)
        
        # Decision logic
        if conditions["is_illiquid"]:
            # Illiquid: use more patient strategy
            strategy = ExecutionStrategy.TWAP
            reason = "Illiquid market - using patient TWAP"
        elif conditions["is_volatile"]:
            # Volatile: use adaptive
            strategy = ExecutionStrategy.ADAPTIVE
            reason = "Volatile market - using adaptive strategy"
        elif conditions["high_volume"]:
            # High volume environment: use POV
            strategy = ExecutionStrategy.POV
            reason = "High volume - using POV for optimal fill"
        else:
            # Normal conditions: use VWAP
            strategy = ExecutionStrategy.VWAP
            reason = "Normal conditions - using VWAP"
            
        return strategy, {
            "strategy": strategy.value,
            "reason": reason,
            "conditions": conditions,
        }
    
    def _analyze_market_conditions(
        self,
        market_data: MarketDataSnapshot,
    ) -> Dict[str, bool]:
        """Analyze current market conditions."""
        # High volatility
        is_volatile = market_data.volatility > 0.025
        
        # Illiquid (wide spread or low volume)
        is_illiquid = market_data.spread_bps > 15 or market_data.market_volume_1m < 1000
        
        # High volume
        high_volume = market_data.market_volume_1m > 10000
        
        return {
            "is_volatile": is_volatile,
            "is_illiquid": is_illiquid,
            "high_volume": high_volume,
        }


def create_execution_engine(
    strategy: str = "vwap",
    **kwargs,
) -> BestExecutionEngine:
    """
    Factory function to create execution engine.
    
    Args:
        strategy: Strategy name ('twap', 'vwap', 'pov', 'adaptive')
        **kwargs: Additional configuration options
        
    Returns:
        Configured BestExecutionEngine
    """
    strategy_map = {
        "twap": ExecutionStrategy.TWAP,
        "vwap": ExecutionStrategy.VWAP,
        "pov": ExecutionStrategy.POV,
        "adaptive": ExecutionStrategy.ADAPTIVE,
        "is": ExecutionStrategy.IS,
        "market": ExecutionStrategy.MARKET,
    }
    
    strat = strategy_map.get(strategy.lower(), ExecutionStrategy.VWAP)
    
    config = ExecutionConfig(strategy=strat, **kwargs)
    
    return BestExecutionEngine(config)
