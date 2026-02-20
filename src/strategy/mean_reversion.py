# src/strategy/mean_reversion.py
"""
Mean Reversion Strategy Module
==============================
Mean reversion trading strategy.
Generates signals when price deviates from statistical mean.
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


class MeanReversionStrategy(BaseStrategy):
    """
    Mean reversion trading strategy.
    
    Generates signals based on:
    - Bollinger Bands
    - Z-score deviation
    - RSI overbought/oversold
    - Statistical arbitrage
    
    Parameters:
        lookback_period: Period for mean calculation
        entry_z_score: Z-score threshold for entry
        exit_z_score: Z-score threshold for exit
        use_bollinger: Use Bollinger Bands
        bollinger_std: Standard deviations for Bollinger
        rsi_oversold: RSI oversold threshold
        rsi_overbought: RSI overbought threshold
    """
    
    def __init__(self, name: str = "MeanReversion", config: Dict[str, Any] = None):
        """
        Initialize Mean Reversion Strategy.
        
        Args:
            name: Strategy identifier
            config: Configuration dictionary
        """
        config = config or {}
        super().__init__(name, config)
        
        # Strategy parameters
        self.lookback_period = self.get_param("lookback_period", 20)
        self.entry_z_score = self.get_param("entry_z_score", 2.0)
        self.exit_z_score = self.get_param("exit_z_score", 0.5)
        self.use_bollinger = self.get_param("use_bollinger", True)
        self.bollinger_std = self.get_param("bollinger_std", 2.0)
        self.rsi_oversold = self.get_param("rsi_oversold", 30)
        self.rsi_overbought = self.get_param("rsi_overbought", 70)
        
        logger.info(
            f"MeanReversionStrategy initialized: lookback={self.lookback_period}, "
            f"entry_z={self.entry_z_score}"
        )
    
    def generate_signal(self, context: StrategyContext) -> Optional[TradingSignal]:
        """
        Generate mean reversion trading signal.
        
        Args:
            context: Strategy context with market data
            
        Returns:
            TradingSignal or None
        """
        if not self.enabled:
            return None
        
        prices = context.prices
        
        if len(prices) < self.lookback_period + 1:
            return None
        
        # Calculate z-score
        z_score = self._calculate_z_score(prices)
        
        if z_score is None:
            return None
        
        # Calculate confidence
        confidence = self._calculate_confidence(context, z_score)
        
        # Determine signal type
        signal_type = self._determine_signal_type(z_score, context)
        
        if signal_type == SignalType.HOLD:
            return None
        
        # Create signal
        signal = self.create_signal(
            symbol=context.symbol,
            signal_type=signal_type,
            confidence=confidence,
            price=float(prices[-1]),
            metadata={
                "z_score": float(z_score),
                "lookback_period": self.lookback_period,
            }
        )
        
        # Validate signal
        if not self.validate_signal(signal, context):
            return None
        
        logger.debug(
            f"MeanReversion signal: {signal.signal_type.value} "
            f"{context.symbol} @ {signal.price} "
            f"(confidence: {confidence:.2f}, z_score: {z_score:.2f})"
        )
        
        return signal
    
    def _calculate_z_score(self, prices: np.ndarray) -> Optional[float]:
        """
        Calculate z-score of current price.
        
        Args:
            prices: Price array
            
        Returns:
            Z-score value or None
        """
        if len(prices) < self.lookback_period:
            return None
        
        window = prices[-self.lookback_period:]
        mean = np.mean(window)
        std = np.std(window)
        
        if std == 0:
            return None
        
        z_score = (prices[-1] - mean) / std
        
        return z_score
    
    def _calculate_confidence(
        self,
        context: StrategyContext,
        z_score: float
    ) -> float:
        """
        Calculate signal confidence.
        
        Args:
            context: Strategy context
            z_score: Calculated z-score
            
        Returns:
            Confidence value (0-1)
        """
        confidence_factors = []
        
        # 1. Z-score magnitude factor
        z_magnitude = min(abs(z_score) / self.entry_z_score, 1.5) / 1.5
        confidence_factors.append(z_magnitude)
        
        # 2. RSI confirmation
        rsi_conf = self._check_rsi(context, z_score)
        confidence_factors.append(rsi_conf)
        
        # 3. Bollinger Band position
        if self.use_bollinger:
            bb_conf = self._check_bollinger(context)
            confidence_factors.append(bb_conf)
        
        # 4. Monte Carlo probability
        if context.mc_probability_up is not None:
            mc_conf = self._check_monte_carlo(context, z_score)
            confidence_factors.append(mc_conf)
        
        # Weighted average
        weights = [0.35, 0.25, 0.2, 0.2][:len(confidence_factors)]
        total_weight = sum(weights)
        
        confidence = sum(
            f * w for f, w in zip(confidence_factors, weights)
        ) / total_weight
        
        return min(max(confidence, 0.0), 1.0)
    
    def _check_rsi(self, context: StrategyContext, z_score: float) -> float:
        """Check RSI confirmation."""
        indicators = context.indicators
        rsi = indicators.get("rsi")
        
        if rsi is None:
            return 0.5
        
        rsi_value = rsi[-1] if isinstance(rsi, np.ndarray) else rsi
        
        # Z-score negative (oversold) should align with low RSI
        if z_score < -self.entry_z_score:
            if rsi_value <= self.rsi_oversold:
                return 1.0
            elif rsi_value <= 40:
                return 0.7
            else:
                return 0.3
        
        # Z-score positive (overbought) should align with high RSI
        elif z_score > self.entry_z_score:
            if rsi_value >= self.rsi_overbought:
                return 1.0
            elif rsi_value >= 60:
                return 0.7
            else:
                return 0.3
        
        return 0.5
    
    def _check_bollinger(self, context: StrategyContext) -> float:
        """Check Bollinger Band position."""
        prices = context.prices
        
        if len(prices) < self.lookback_period:
            return 0.5
        
        window = prices[-self.lookback_period:]
        mean = np.mean(window)
        std = np.std(window)
        
        if std == 0:
            return 0.5
        
        upper = mean + self.bollinger_std * std
        lower = mean - self.bollinger_std * std
        current = prices[-1]
        
        # Price near or beyond bands
        if current <= lower:
            return 1.0  # Strong buy signal
        elif current >= upper:
            return 1.0  # Strong sell signal
        elif current <= mean - std:
            return 0.7
        elif current >= mean + std:
            return 0.7
        else:
            return 0.3
    
    def _check_monte_carlo(
        self,
        context: StrategyContext,
        z_score: float
    ) -> float:
        """Check Monte Carlo probability alignment."""
        prob_up = context.mc_probability_up
        
        if prob_up is None:
            return 0.5
        
        # Mean reversion: negative z-score should mean price likely to go up
        if z_score < -self.entry_z_score:
            # Expect price to revert up
            return prob_up
        elif z_score > self.entry_z_score:
            # Expect price to revert down
            return 1 - prob_up
        
        return 0.5
    
    def _determine_signal_type(
        self,
        z_score: float,
        context: StrategyContext
    ) -> SignalType:
        """
        Determine signal type based on z-score.
        
        Args:
            z_score: Calculated z-score
            context: Strategy context
            
        Returns:
            SignalType enum value
        """
        current_position = context.current_position
        
        # Check for exit signals first
        if current_position:
            pos_side = current_position.get("side")
            
            # Exit long if z-score returns to mean
            if pos_side == "long" and z_score > -self.exit_z_score:
                return SignalType.CLOSE_LONG
            
            # Exit short if z-score returns to mean
            if pos_side == "short" and z_score < self.exit_z_score:
                return SignalType.CLOSE_SHORT
        
        # Entry signals
        if z_score < -self.entry_z_score:
            # Price significantly below mean - BUY
            return SignalType.BUY
        
        elif z_score > self.entry_z_score:
            # Price significantly above mean - SELL
            return SignalType.SELL
        
        return SignalType.HOLD
    
    def update_parameters(
        self,
        lookback_period: int = None,
        entry_z_score: float = None,
        exit_z_score: float = None,
    ):
        """
        Update strategy parameters.
        
        Args:
            lookback_period: New lookback period
            entry_z_score: New entry z-score threshold
            exit_z_score: New exit z-score threshold
        """
        if lookback_period is not None:
            self.lookback_period = lookback_period
            self.set_param("lookback_period", lookback_period)
        
        if entry_z_score is not None:
            self.entry_z_score = entry_z_score
            self.set_param("entry_z_score", entry_z_score)
        
        if exit_z_score is not None:
            self.exit_z_score = exit_z_score
            self.set_param("exit_z_score", exit_z_score)
        
        logger.info(
            f"MeanReversionStrategy params updated: "
            f"lookback={self.lookback_period}, "
            f"entry_z={self.entry_z_score}, "
            f"exit_z={self.exit_z_score}"
        )
