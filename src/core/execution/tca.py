# src/core/execution/tca.py
"""
Transaction Cost Analysis (TCA)
================================
Institutional-grade execution quality analysis including:
- Implementation Shortfall (IS)
- Market Impact Modeling (Quadratic/Linear)
- Spread Cost Analysis
- Slippage Measurement
- VWAP/Arrival Price comparison

Author: AI Trading System
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MarketImpactModel(Enum):
    """Market impact model types."""
    LINEAR = "linear"
    QUADRATIC = "quadratic"
    SQRT = "sqrt"
    EXPONENTIAL = "exponential"


class ExecutionAlgorithm(Enum):
    """Execution algorithm types."""
    MARKET = "market"
    LIMIT = "limit"
    TWAP = "twap"
    VWAP = "vwap"
    POV = "pov"
    IS = "implementation_shortfall"
    ADAPTIVE = "adaptive"


@dataclass
class ExecutionQuality:
    """Execution quality metrics."""
    # Implementation Shortfall components
    delay_cost: float = 0.0  # Cost due to execution delay
    market_impact_cost: float = 0.0  # Cost due to market impact
    spread_cost: float = 0.0  # Cost due to bid-ask spread
    price_improvement: float = 0.0  # Price improvement vs arrival
    
    # Overall metrics
    total_implementation_shortfall: float = 0.0  # Total IS in basis points
    execution_price: float = 0.0
    arrival_price: float = 0.0
    vwap: float = 0.0
    twap: float = 0.0
    
    # Slippage metrics
    slippage: float = 0.0  # Slippage in basis points
    expected_slippage: float = 0.0
    slippage_variance: float = 0.0
    
    # Volume metrics
    participation_rate: float = 0.0  # Actual participation rate
    volume_traded: float = 0.0
    avg_trade_size: float = 0.0
    
    # Timing metrics
    execution_time: float = 0.0  # Total execution time in seconds
    order_count: int = 0
    
    # Market conditions
    realized_volatility: float = 0.0
    bid_ask_spread: float = 0.0
    market_impact_coefficient: float = 0.0


@dataclass
class TradeRecord:
    """Individual trade record for TCA."""
    timestamp: datetime
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    commission: float = 0.0
    order_id: str = ""
    venue: str = ""
    
    @property
    def notional_value(self) -> float:
        """Calculate notional value."""
        return self.quantity * self.price


@dataclass
class OrderSnapshot:
    """Order state snapshot for analysis."""
    timestamp: datetime
    symbol: str
    side: str
    order_type: str
    quantity: float
    filled_quantity: float = 0.0
    limit_price: Optional[float] = None
    arrival_price: Optional[float] = None
    best_bid: float = 0.0
    best_ask: float = 0.0
    mid_price: float = 0.0
    vwap: float = 0.0
    volume_traded: float = 0.0
    market_volume: float = 0.0


class TransactionCostAnalyzer:
    """
    Institutional-Grade Transaction Cost Analysis
    ==============================================
    Provides comprehensive execution quality analysis.
    """
    
    def __init__(
        self,
        impact_model: MarketImpactModel = MarketImpactModel.SQRT,
        default_commission: float = 0.0002,  # 2 bps
        default_slippage_bps: float = 1.0,    # 1 basis point
    ):
        """
        Initialize TCA.
        
        Args:
            impact_model: Market impact model to use
            default_commission: Default commission rate (fraction)
            default_slippage_bps: Default slippage in basis points
        """
        self.impact_model = impact_model
        self.default_commission = default_commission
        self.default_slippage_bps = default_slippage_bps
        
        # Historical data for model calibration
        self.calibration_data: List[Dict] = []
        
        # Model parameters (can be calibrated)
        self.impact_coefficients: Dict[str, float] = {}
        
    def calculate_implementation_shortfall(
        self,
        trades: List[TradeRecord],
        order_snapshot: OrderSnapshot,
        commission_rate: Optional[float] = None,
    ) -> ExecutionQuality:
        """
        Calculate Implementation Shortfall (IS).
        
        IS = (Execution Price - Arrival Price) * Direction + Commission + Slippage
        
        Args:
            trades: List of executed trades
            order_snapshot: Initial order snapshot
            commission_rate: Commission rate (fraction)
            
        Returns:
            ExecutionQuality with IS metrics
        """
        if not trades or order_snapshot.arrival_price is None:
            return ExecutionQuality()
            
        commission_rate = commission_rate or self.default_commission
        
        # Calculate metrics
        total_notional = sum(t.notional_value for t in trades)
        total_quantity = sum(t.quantity for t in trades)
        total_commission = sum(t.commission for t in trades)
        
        if total_quantity == 0:
            return ExecutionQuality()
            
        # Weighted average execution price
        execution_price = total_notional / total_quantity if total_quantity > 0 else 0
        
        # Direction: +1 for buy, -1 for sell
        direction = 1 if order_snapshot.side.lower() == 'buy' else -1
        
        # Calculate IS components
        arrival_price = order_snapshot.arrival_price
        
        # Price difference (signed)
        price_diff = (execution_price - arrival_price) * direction
        
        # Delay cost (simplified - can be enhanced with actual timing)
        delay_cost = 0.0
        
        # Market impact cost (estimated)
        market_impact = self.estimate_market_impact(
            quantity=total_quantity,
            order_value=total_notional,
            market_volume=order_snapshot.market_volume,
            volatility=self._estimate_volatility(trades),
        )
        market_impact_cost = market_impact * direction
        
        # Spread cost (half of spread at arrival)
        spread = order_snapshot.best_ask - order_snapshot.best_bid
        spread_cost = spread / 2 * direction
        
        # Price improvement (negative means improvement)
        # Compare to mid-price
        price_improvement = (execution_price - order_snapshot.mid_price) * direction
        
        # Total IS in currency
        total_is = price_diff + total_commission + market_impact + spread_cost
        
        # Convert to basis points
        is_bps = (total_is / arrival_price * 10000) if arrival_price > 0 else 0
        
        # Calculate slippage
        expected_price = order_snapshot.mid_price
        slippage = (execution_price - expected_price) * direction
        slippage_bps = (slippage / expected_price * 10000) if expected_price > 0 else 0
        
        return ExecutionQuality(
            delay_cost=delay_cost,
            market_impact_cost=market_impact_cost,
            spread_cost=spread_cost,
            price_improvement=price_improvement,
            total_implementation_shortfall=is_bps,
            execution_price=execution_price,
            arrival_price=arrival_price,
            vwap=order_snapshot.vwap,
            slippage=slippage_bps,
            volume_traded=total_quantity,
            avg_trade_size=total_quantity / len(trades) if trades else 0,
            bid_ask_spread=spread,
        )
    
    def estimate_market_impact(
        self,
        quantity: float,
        order_value: float,
        market_volume: float,
        volatility: float = 0.02,
        participation_rate: float = 0.01,
    ) -> float:
        """
        Estimate market impact cost.
        
        Uses multiple models:
        - Linear: impact = alpha * participation_rate
        - Quadratic: impact = alpha * participation_rate^2
        - Sqrt: impact = alpha * sqrt(participation_rate)
        
        Args:
            quantity: Order quantity
            order_value: Total order value
            market_volume: Expected market volume
            volatility: Expected volatility
            participation_rate: Order size as fraction of volume
            
        Returns:
            Estimated market impact (in price units)
        """
        if market_volume <= 0 or participation_rate <= 0:
            return 0.0
            
        # Base impact coefficient (can be calibrated)
        alpha = self._get_impact_coefficient(volatility)
        
        # Model-specific impact calculation
        if self.impact_model == MarketImpactModel.LINEAR:
            impact = alpha * participation_rate
        elif self.impact_model == MarketImpactModel.QUADRATIC:
            impact = alpha * (participation_rate ** 2)
        elif self.impact_model == MarketImpactModel.SQRT:
            impact = alpha * np.sqrt(participation_rate)
        elif self.impact_model == MarketImpactModel.EXPONENTIAL:
            impact = alpha * (np.exp(participation_rate) - 1)
        else:
            impact = alpha * np.sqrt(participation_rate)
            
        # Scale by order value
        return impact * order_value
    
    def _get_impact_coefficient(self, volatility: float) -> float:
        """
        Get impact coefficient based on volatility.
        
        Args:
            volatility: Expected volatility
            
        Returns:
            Impact coefficient
        """
        # Base coefficient (empirically derived)
        base_coef = 0.0001
        
        # Scale by volatility
        return base_coef * (volatility / 0.02)
    
    def _estimate_volatility(self, trades: List[TradeRecord]) -> float:
        """Estimate realized volatility from trades."""
        if len(trades) < 2:
            return 0.02  # Default 2% volatility
            
        prices = [t.price for t in trades]
        returns = np.diff(prices) / prices[:-1]
        
        if len(returns) == 0:
            return 0.02
            
        return float(np.std(returns)) if len(returns) > 0 else 0.02
    
    def calculate_vwap_comparison(
        self,
        trades: List[TradeRecord],
        order_snapshot: OrderSnapshot,
    ) -> Dict[str, float]:
        """
        Compare execution to VWAP.
        
        Args:
            trades: Executed trades
            order_snapshot: Order snapshot with VWAP data
            
        Returns:
            Dictionary with VWAP comparison metrics
        """
        if not trades:
            return {}
            
        total_notional = sum(t.notional_value for t in trades)
        total_quantity = sum(t.quantity for t in trades)
        
        execution_vwap = total_notional / total_quantity if total_quantity > 0 else 0
        benchmark_vwap = order_snapshot.vwap if order_snapshot.vwap > 0 else order_snapshot.mid_price
        
        # Direction for comparison
        direction = 1 if order_snapshot.side.lower() == 'buy' else -1
        
        # VWAP slippage (execution vs benchmark)
        vwap_diff = (execution_vwap - benchmark_vwap) * direction
        vwap_slippage_bps = (vwap_diff / benchmark_vwap * 10000) if benchmark_vwap > 0 else 0
        
        return {
            "execution_vwap": execution_vwap,
            "benchmark_vwap": benchmark_vwap,
            "vwap_difference": vwap_diff,
            "vwap_slippage_bps": vwap_slippage_bps,
            "execution_quality": "good" if vwap_slippage_bps < 0 else "poor",
        }
    
    def calculate_arrival_price_comparison(
        self,
        trades: List[TradeRecord],
        order_snapshot: OrderSnapshot,
    ) -> Dict[str, float]:
        """
        Compare execution to arrival price.
        
        Args:
            trades: Executed trades
            order_snapshot: Order snapshot with arrival price
            
        Returns:
            Dictionary with arrival price comparison
        """
        if not trades or order_snapshot.arrival_price is None:
            return {}
            
        total_notional = sum(t.notional_value for t in trades)
        total_quantity = sum(t.quantity for t in trades)
        
        execution_price = total_notional / total_quantity if total_quantity > 0 else 0
        arrival_price = order_snapshot.arrival_price
        
        direction = 1 if order_snapshot.side.lower() == 'buy' else -1
        
        # Price improvement (negative is good for buys)
        price_diff = (execution_price - arrival_price) * direction
        price_improvement_bps = (price_diff / arrival_price * 10000) if arrival_price > 0 else 0
        
        return {
            "execution_price": execution_price,
            "arrival_price": arrival_price,
            "price_difference": price_diff,
            "price_improvement_bps": price_improvement_bps,
            "quality": "improved" if price_improvement_bps < 0 else "worse",
        }
    
    def analyze_spread_cost(
        self,
        order_snapshot: OrderSnapshot,
    ) -> Dict[str, float]:
        """
        Analyze spread cost.
        
        Args:
            order_snapshot: Order snapshot
            
        Returns:
            Spread cost metrics
        """
        bid = order_snapshot.best_bid
        ask = order_snapshot.best_ask
        mid = order_snapshot.mid_price
        
        if bid <= 0 or ask <= 0:
            return {}
            
        spread = ask - bid
        spread_bps = (spread / mid * 10000) if mid > 0 else 0
        
        # Half spread (entry cost)
        half_spread = spread / 2
        half_spread_bps = spread_bps / 2
        
        return {
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "spread": spread,
            "spread_bps": spread_bps,
            "half_spread": half_spread,
            "half_spread_bps": half_spread_bps,
        }
    
    def calibrate_impact_model(
        self,
        historical_executions: List[Tuple[float, float, float, float]],
    ):
        """
        Calibrate impact model coefficients using historical data.
        
        Args:
            historical_executions: List of (participation_rate, volatility, market_impact, volume) tuples
        """
        if len(historical_executions) < 10:
            logger.warning("Insufficient data for calibration")
            return
            
        # Extract features
        participation_rates = np.array([x[0] for x in historical_executions])
        volatilities = np.array([x[1] for x in historical_executions])
        impacts = np.array([x[2] for x in historical_executions])
        
        # Simple linear regression for coefficient
        # impact = alpha * sqrt(participation_rate) * (volatility / 0.02)
        X = np.sqrt(participation_rates) * (volatilities / 0.02)
        
        # Solve for alpha using least squares
        alpha = np.mean(impacts / X) if np.mean(X) != 0 else 0.0001
        
        # Store calibrated coefficient
        self.impact_coefficients['calibrated'] = alpha
        
        logger.info(f"Calibrated impact coefficient: {alpha:.6f}")
    
    def generate_tca_report(
        self,
        trades: List[TradeRecord],
        order_snapshot: OrderSnapshot,
        commission_rate: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive TCA report.
        
        Args:
            trades: Executed trades
            order_snapshot: Initial order snapshot
            commission_rate: Commission rate
            
        Returns:
            Comprehensive TCA report
        """
        # Calculate all metrics
        quality = self.calculate_implementation_shortfall(
            trades, order_snapshot, commission_rate
        )
        
        vwap_analysis = self.calculate_vwap_comparison(trades, order_snapshot)
        arrival_analysis = self.calculate_arrival_price_comparison(trades, order_snapshot)
        spread_analysis = self.analyze_spread_cost(order_snapshot)
        
        # Build comprehensive report
        report = {
            "timestamp": datetime.now().isoformat(),
            "symbol": order_snapshot.symbol,
            "side": order_snapshot.side,
            "quantity": order_snapshot.quantity,
            "filled_quantity": sum(t.quantity for t in trades),
            
            # Implementation Shortfall
            "implementation_shortfall": {
                "total_bps": quality.total_implementation_shortfall,
                "delay_cost": quality.delay_cost,
                "market_impact_cost": quality.market_impact_cost,
                "spread_cost": quality.spread_cost,
                "price_improvement": quality.price_improvement,
            },
            
            # Price analysis
            "price_analysis": {
                "execution_price": quality.execution_price,
                "arrival_price": quality.arrival_price,
                "vwap": quality.vwap,
                "twap": quality.twap,
                "mid_price": order_snapshot.mid_price,
                "best_bid": order_snapshot.best_bid,
                "best_ask": order_snapshot.best_ask,
            },
            
            # VWAP comparison
            "vwap_comparison": vwap_analysis,
            
            # Arrival price comparison
            "arrival_comparison": arrival_analysis,
            
            # Spread analysis
            "spread_analysis": spread_analysis,
            
            # Slippage
            "slippage": {
                "actual_bps": quality.slippage,
                "expected_bps": quality.expected_slippage,
                "variance": quality.slippage_variance,
            },
            
            # Volume analysis
            "volume": {
                "traded": quality.volume_traded,
                "avg_trade_size": quality.avg_trade_size,
                "participation_rate": quality.participation_rate,
            },
            
            # Overall assessment
            "overall_quality": self._assess_execution_quality(quality),
        }
        
        return report
    
    def _assess_execution_quality(self, quality: ExecutionQuality) -> str:
        """Assess overall execution quality."""
        is_bps = quality.total_implementation_shortfall
        
        if is_bps < 5:
            return "excellent"
        elif is_bps < 10:
            return "good"
        elif is_bps < 20:
            return "fair"
        else:
            return "poor"


class SlippageModel:
    """
    Slippage Modeling
    ==================
    Realistic cost estimation based on:
    - Volume-based slippage
    - Volatility-based slippage
    - Liquidity-adjusted slippage
    - Market impact decay functions
    """
    
    def __init__(
        self,
        base_slippage_bps: float = 1.0,
        volatility_scalar: float = 2.0,
        liquidity_scalar: float = 0.5,
    ):
        """
        Initialize slippage model.
        
        Args:
            base_slippage_bps: Base slippage in basis points
            volatility_scalar: Volatility multiplier
            liquidity_scalar: Liquidity multiplier
        """
        self.base_slippage_bps = base_slippage_bps
        self.volatility_scalar = volatility_scalar
        self.liquidity_scalar = liquidity_scalar
        
    def estimate_slippage(
        self,
        order_size: float,
        market_volume: float,
        volatility: float,
        order_type: str = "market",
        urgency: float = 0.5,
    ) -> Dict[str, float]:
        """
        Estimate slippage for an order.
        
        Args:
            order_size: Order size
            market_volume: Expected market volume
            volatility: Expected volatility (annualized)
            order_type: Type of order
            urgency: Execution urgency (0-1, higher = more urgent)
            
        Returns:
            Slippage estimates
        """
        if market_volume <= 0:
            return {"slippage_bps": self.base_slippage_bps, "slippage_cost": 0}
            
        # Participation rate
        participation = order_size / market_volume
        
        # Volume-based slippage (non-linear)
        volume_slippage = self._volume_based_slippage(participation)
        
        # Volatility-based slippage
        volatility_slippage = self._volatility_based_slippage(volatility)
        
        # Liquidity adjustment
        liquidity_factor = self._liquidity_factor(participation)
        
        # Combine factors
        total_slippage_bps = (
            self.base_slippage_bps * 
            volume_slippage * 
            volatility_slippage * 
            liquidity_factor *
            (1 + urgency * 0.5)  # Urgency premium
        )
        
        # Calculate cost
        slippage_cost = order_size * (total_slippage_bps / 10000)
        
        return {
            "slippage_bps": total_slippage_bps,
            "slippage_cost": slippage_cost,
            "volume_component": volume_slippage,
            "volatility_component": volatility_slippage,
            "liquidity_component": liquidity_factor,
            "participation_rate": participation,
        }
    
    def _volume_based_slippage(self, participation: float) -> float:
        """Calculate volume-based slippage multiplier."""
        # Non-linear relationship: small orders have minimal impact,
        # larger orders experience increasing slippage
        return 1 + (participation ** 1.5) * 10
    
    def _volatility_based_slippage(self, volatility: float) -> float:
        """Calculate volatility-based slippage multiplier."""
        # Higher volatility = higher slippage
        # Use annualized volatility
        return 1 + (volatility * self.volatility_scalar * 10)
    
    def _liquidity_factor(self, participation: float) -> float:
        """Calculate liquidity adjustment factor."""
        # Lower participation = better liquidity = lower slippage
        if participation < 0.01:  # < 1% participation
            return 0.5
        elif participation < 0.05:  # < 5%
            return 1.0
        elif participation < 0.10:  # < 10%
            return 1.5
        else:
            return 2.0
    
    def estimate_market_impact_decay(
        self,
        time_horizon: float,
        initial_impact: float,
        decay_half_life: float = 300.0,
    ) -> float:
        """
        Estimate market impact decay over time.
        
        Args:
            time_horizon: Time since order (seconds)
            initial_impact: Initial market impact
            decay_half_life: Half-life of decay (seconds)
            
        Returns:
            Remaining impact
        """
        # Exponential decay
        decay_rate = np.log(2) / decay_half_life
        return initial_impact * np.exp(-decay_rate * time_horizon)
    
    def calculate_iceberg_impact(
        self,
        visible_size: float,
        hidden_size: float,
        market_volume: float,
    ) -> Dict[str, float]:
        """
        Calculate market impact for iceberg orders.
        
        Args:
            visible_size: Visible order size
            hidden_size: Hidden order size
            market_volume: Expected market volume
            
        Returns:
            Impact analysis
        """
        total_size = visible_size + hidden_size
        participation = total_size / market_volume if market_volume > 0 else 0
        
        # Total impact (using total size)
        total_impact = self._volume_based_slippage(participation)
        
        # Visible portion impact
        visible_participation = visible_size / market_volume if market_volume > 0 else 0
        visible_impact = self._volume_based_slippage(visible_participation)
        
        # Hidden portion impact
        hidden_participation = hidden_size / market_volume if market_volume > 0 else 0
        hidden_impact = self._volume_based_slippage(hidden_participation)
        
        return {
            "total_impact": total_impact,
            "visible_impact": visible_impact,
            "hidden_impact": hidden_impact,
            "hidden_ratio": hidden_size / total_size if total_size > 0 else 0,
            "impact_saved": total_impact - visible_impact,
        }


def create_tca_analyzer(
    impact_model: str = "sqrt",
    commission: float = 0.0002,
) -> TransactionCostAnalyzer:
    """
    Factory function to create TCA analyzer.
    
    Args:
        impact_model: Impact model type ('linear', 'quadratic', 'sqrt', 'exponential')
        commission: Default commission rate
        
    Returns:
        Configured TCA analyzer
    """
    model_map = {
        "linear": MarketImpactModel.LINEAR,
        "quadratic": MarketImpactModel.QUADRATIC,
        "sqrt": MarketImpactModel.SQRT,
        "exponential": MarketImpactModel.EXPONENTIAL,
    }
    
    model = model_map.get(impact_model.lower(), MarketImpactModel.SQRT)
    
    return TransactionCostAnalyzer(
        impact_model=model,
        default_commission=commission,
    )
