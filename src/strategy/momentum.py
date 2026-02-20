# src/strategy/momentum.py
"""
Momentum Strategy Module
========================
Momentum-based trading strategy.
Generates signals based on price momentum and trend strength.
"""

import logging
from typing import Dict, Any, Optional
import numpy as np

from src.strategy.base_strategy import (
    BaseStrategy,
    TradingSignal,
    SignalType,
    StrategyContext,
)


logger = logging.getLogger(__name__)


class MomentumStrategy(BaseStrategy):
    """
    Momentum trading strategy.
    
    Generates signals based on:
    - Price momentum (rate of change)
    - Volume confirmation
    - Trend strength (ADX)
    - Moving average crossovers
    
    Parameters:
        momentum_period: Lookback period for momentum calculation
        momentum_threshold: Minimum momentum for signal
        volume_factor: Volume confirmation factor
        use_ma_filter: Use moving average as trend filter
        ma_period: Moving average period
        adx_threshold: Minimum ADX for trend strength
    """
    
    def __init__(self, name: str = "Momentum", config: Dict[str, Any] = None):
        """
        Initialize Momentum Strategy.
        
        Args:
            name: Strategy identifier
            config: Configuration dictionary
        """
        config = config or {}
        super().__init__(name, config)
        
        # Strategy parameters
        self.momentum_period = self.get_param("momentum_period", 10)
        self.momentum_threshold = self.get_param("momentum_threshold", 0.02)
        self.volume_factor = self.get_param("volume_factor", 1.5)
        self.use_ma_filter = self.get_param("use_ma_filter", True)
        self.ma_period = self.get_param("ma_period", 20)
        self.adx_threshold = self.get_param("adx_threshold", 25)
        
        logger.info(
            f"MomentumStrategy initialized: period={self.momentum_period}, "
            f"threshold={self.momentum_threshold}"
        )
    
    def generate_signal(self, context: StrategyContext) -> Optional[TradingSignal]:
        """
        Generate momentum-based trading signal.
        
        Args:
            context: Strategy context with market data
            
        Returns:
            TradingSignal or None
        """
        if not self.enabled:
            return None
        
        prices = context.prices
        volumes = context.volumes
        
        if len(prices) < max(self.momentum_period, self.ma_period) + 1:
            return None
        
        # Calculate momentum
        momentum = self._calculate_momentum(prices)
        
        if momentum is None:
            return None
        
        # Calculate confidence based on multiple factors
        confidence = self._calculate_confidence(context, momentum)
        
        # Determine signal type
        signal_type = self._determine_signal_type(momentum, context)
        
        if signal_type == SignalType.HOLD:
            return None
        
        # Create signal
        signal = self.create_signal(
            symbol=context.symbol,
            signal_type=signal_type,
            confidence=confidence,
            price=float(prices[-1]),
            metadata={
                "momentum": float(momentum),
                "momentum_period": self.momentum_period,
            }
        )
        
        # Validate signal
        if not self.validate_signal(signal, context):
            return None
        
        logger.debug(
            f"Momentum signal: {signal.signal_type.value} "
            f"{context.symbol} @ {signal.price} "
            f"(confidence: {confidence:.2f}, momentum: {momentum:.4f})"
        )
        
        return signal
    
    def _calculate_momentum(self, prices: np.ndarray) -> Optional[float]:
        """
        Calculate price momentum (rate of change).
        
        Args:
            prices: Price array
            
        Returns:
            Momentum value or None
        """
        if len(prices) < self.momentum_period + 1:
            return None
        
        # Rate of change
        current_price = prices[-1]
        past_price = prices[-self.momentum_period - 1]
        
        if past_price == 0:
            return None
        
        momentum = (current_price - past_price) / past_price
        
        return momentum
    
    def _calculate_confidence(
        self,
        context: StrategyContext,
        momentum: float
    ) -> float:
        """
        Calculate signal confidence based on multiple factors.
        
        Args:
            context: Strategy context
            momentum: Calculated momentum
            
        Returns:
            Confidence value (0-1)
        """
        confidence_factors = []
        
        # 1. Momentum strength factor
        momentum_strength = min(abs(momentum) / self.momentum_threshold, 1.0)
        confidence_factors.append(momentum_strength)
        
        # 2. Volume confirmation factor
        volume_conf = self._check_volume_confirmation(context)
        confidence_factors.append(volume_conf)
        
        # 3. Trend filter factor (MA)
        if self.use_ma_filter:
            ma_conf = self._check_ma_filter(context)
            confidence_factors.append(ma_conf)
        
        # 4. ADX trend strength factor
        adx_conf = self._check_adx(context)
        confidence_factors.append(adx_conf)
        
        # 5. Monte Carlo probability factor
        if context.mc_probability_up is not None:
            mc_conf = self._check_monte_carlo(context, momentum)
            confidence_factors.append(mc_conf)
        
        # Weighted average
        weights = [0.3, 0.2, 0.15, 0.15, 0.2][:len(confidence_factors)]
        total_weight = sum(weights)
        
        confidence = sum(
            f * w for f, w in zip(confidence_factors, weights)
        ) / total_weight
        
        return min(max(confidence, 0.0), 1.0)
    
    def _check_volume_confirmation(self, context: StrategyContext) -> float:
        """Check if volume confirms the move."""
        volumes = context.volumes
        
        if len(volumes) < self.momentum_period:
            return 0.5
        
        avg_volume = np.mean(volumes[-self.momentum_period:])
        current_volume = volumes[-1]
        
        if avg_volume == 0:
            return 0.5
        
        volume_ratio = current_volume / avg_volume
        
        # Higher volume = higher confidence
        if volume_ratio >= self.volume_factor:
            return 1.0
        elif volume_ratio >= 1.0:
            return 0.7
        else:
            return 0.3
    
    def _check_ma_filter(self, context: StrategyContext) -> float:
        """Check moving average trend filter."""
        prices = context.prices
        
        if len(prices) < self.ma_period:
            return 0.5
        
        ma = np.mean(prices[-self.ma_period:])
        current_price = prices[-1]
        
        # Price above MA = bullish
        price_position = (current_price - ma) / ma if ma > 0 else 0
        
        if abs(price_position) > 0.02:
            return 1.0
        elif abs(price_position) > 0.01:
            return 0.7
        else:
            return 0.5
    
    def _check_adx(self, context: StrategyContext) -> float:
        """Check ADX trend strength."""
        indicators = context.indicators
        
        adx = indicators.get("adx")
        
        if adx is None:
            return 0.5
        
        adx_value = adx[-1] if isinstance(adx, np.ndarray) else adx
        
        if adx_value >= self.adx_threshold + 10:
            return 1.0
        elif adx_value >= self.adx_threshold:
            return 0.8
        elif adx_value >= self.adx_threshold - 5:
            return 0.5
        else:
            return 0.3
    
    def _check_monte_carlo(
        self,
        context: StrategyContext,
        momentum: float
    ) -> float:
        """Check Monte Carlo probability alignment."""
        prob_up = context.mc_probability_up
        
        if prob_up is None:
            return 0.5
        
        # Momentum positive should align with probability up
        if momentum > 0 and prob_up > 0.5:
            alignment = prob_up
        elif momentum < 0 and prob_up < 0.5:
            alignment = 1 - prob_up
        else:
            alignment = 0.3
        
        return alignment
    
    def _determine_signal_type(
        self,
        momentum: float,
        context: StrategyContext
    ) -> SignalType:
        """
        Determine signal type based on momentum.
        
        Args:
            momentum: Calculated momentum
            context: Strategy context
            
        Returns:
            SignalType enum value
        """
        # Check if momentum exceeds threshold
        if momentum > self.momentum_threshold:
            # Positive momentum - BUY signal
            # Check if we have existing position
            if context.current_position:
                pos_side = context.current_position.get("side")
                if pos_side == "short":
                    return SignalType.CLOSE_SHORT
            
            return SignalType.BUY
        
        elif momentum < -self.momentum_threshold:
            # Negative momentum - SELL signal
            if context.current_position:
                pos_side = context.current_position.get("side")
                if pos_side == "long":
                    return SignalType.CLOSE_LONG
            
            return SignalType.SELL
        
        else:
            # Momentum within threshold - HOLD
            return SignalType.HOLD
    
    def update_parameters(
        self,
        momentum_period: int = None,
        momentum_threshold: float = None,
        volume_factor: float = None,
    ):
        """
        Update strategy parameters.
        
        Args:
            momentum_period: New momentum period
            momentum_threshold: New momentum threshold
            volume_factor: New volume factor
        """
        if momentum_period is not None:
            self.momentum_period = momentum_period
            self.set_param("momentum_period", momentum_period)
        
        if momentum_threshold is not None:
            self.momentum_threshold = momentum_threshold
            self.set_param("momentum_threshold", momentum_threshold)
        
        if volume_factor is not None:
            self.volume_factor = volume_factor
            self.set_param("volume_factor", volume_factor)
        
        logger.info(
            f"MomentumStrategy params updated: "
            f"period={self.momentum_period}, "
            f"threshold={self.momentum_threshold}, "
            f"volume_factor={self.volume_factor}"
        )
