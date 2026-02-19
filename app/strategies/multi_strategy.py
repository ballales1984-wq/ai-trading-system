"""
Multi Strategy Manager
====================
Manages multiple trading strategies with signal aggregation.
"""

from typing import Dict, List, Optional
import logging
from dataclasses import dataclass, field

import pandas as pd

from app.strategies.base_strategy import (
    BaseStrategy, StrategyConfig, TradingSignal, SignalDirection
)
from app.strategies.momentum import MomentumStrategy
from app.strategies.mean_reversion import MeanReversionStrategy


logger = logging.getLogger(__name__)


@dataclass
class StrategySignal:
    """Signal from a single strategy."""
    strategy_name: str
    signal: TradingSignal
    weight: float = 1.0


class MultiStrategyManager:
    """
    Manages multiple trading strategies.
    
    Aggregates signals from multiple strategies with weighting.
    """
    
    def __init__(self):
        """Initialize multi-strategy manager."""
        self.strategies: Dict[str, BaseStrategy] = {}
        self.strategy_weights: Dict[str, float] = {}
        self.enabled_strategies: set = set()
        
        # Default strategies
        self._register_default_strategies()
        
        logger.info("Multi-strategy manager initialized")
    
    def _register_default_strategies(self):
        """Register default strategies."""
        # Momentum strategy
        momentum_config = StrategyConfig(
            name="momentum",
            enabled=True,
            parameters={
                "lookback_period": 20,
                "threshold": 0.02,
                "rsi_oversold": 30,
                "rsi_overbought": 70,
                "stop_loss_pct": 0.02,
                "take_profit_pct": 0.05
            }
        )
        self.register_strategy(MomentumStrategy(momentum_config), weight=1.0)
        
        # Mean reversion strategy
        mean_reversion_config = StrategyConfig(
            name="mean_reversion",
            enabled=True,
            parameters={
                "lookback_period": 50,
                "bb_std": 2.0,
                "zscore_threshold": 2.0,
                "stop_loss_pct": 0.03,
                "take_profit_pct": 0.02
            }
        )
        self.register_strategy(MeanReversionStrategy(mean_reversion_config), weight=1.0)
    
    def register_strategy(
        self, 
        strategy: BaseStrategy, 
        weight: float = 1.0
    ):
        """
        Register a strategy.
        
        Args:
            strategy: Strategy instance
            weight: Strategy weight (0-1)
        """
        self.strategies[strategy.name] = strategy
        self.strategy_weights[strategy.name] = weight
        
        if strategy.enabled:
            self.enabled_strategies.add(strategy.name)
        
        logger.info(f"Registered strategy: {strategy.name} (weight: {weight})")
    
    def enable_strategy(self, strategy_name: str):
        """Enable a strategy."""
        if strategy_name in self.strategies:
            self.strategies[strategy_name].enabled = True
            self.enabled_strategies.add(strategy_name)
            logger.info(f"Enabled strategy: {strategy_name}")
    
    def disable_strategy(self, strategy_name: str):
        """Disable a strategy."""
        if strategy_name in self.strategies:
            self.strategies[strategy_name].enabled = False
            self.enabled_strategies.discard(strategy_name)
            logger.info(f"Disabled strategy: {strategy_name}")
    
    def set_weight(self, strategy_name: str, weight: float):
        """Set strategy weight."""
        if strategy_name in self.strategies:
            self.strategy_weights[strategy_name] = weight
            logger.info(f"Set weight for {strategy_name}: {weight}")
    
    def generate_signals(
        self, 
        data: pd.DataFrame
    ) -> List[StrategySignal]:
        """
        Generate signals from all enabled strategies.
        
        Args:
            data: OHLCV data
            
        Returns:
            List of strategy signals
        """
        signals = []
        
        for name in self.enabled_strategies:
            strategy = self.strategies.get(name)
            
            if strategy and strategy.enabled:
                try:
                    signal = strategy.generate_signal(data)
                    
                    if signal:
                        strategy_signal = StrategySignal(
                            strategy_name=name,
                            signal=signal,
                            weight=self.strategy_weights.get(name, 1.0)
                        )
                        signals.append(strategy_signal)
                        
                except Exception as e:
                    logger.error(f"Error generating signal from {name}: {e}")
        
        return signals
    
    def aggregate_signals(
        self, 
        signals: List[StrategySignal]
    ) -> Optional[TradingSignal]:
        """
        Aggregate signals from multiple strategies.
        
        Args:
            signals: List of strategy signals
            
        Returns:
            Aggregated signal or None
        """
        if not signals:
            return None
        
        # Filter non-FLAT signals
        actionable_signals = [
            s for s in signals 
            if s.signal.direction != SignalDirection.FLAT
        ]
        
        if not actionable_signals:
            # Return first FLAT signal if all are FLAT
            flat_signals = [s for s in signals if s.signal.direction == SignalDirection.FLAT]
            if flat_signals:
                return flat_signals[0].signal
            return None
        
        # Calculate weighted confidence
        total_weight = sum(s.weight for s in actionable_signals)
        weighted_confidence = sum(
            s.signal.confidence * s.weight 
            for s in actionable_signals
        ) / total_weight
        
        # Determine direction (majority vote weighted by confidence)
        long_count = sum(
            s.weight * s.signal.confidence 
            for s in actionable_signals 
            if s.signal.direction == SignalDirection.LONG
        )
        short_count = sum(
            s.weight * s.signal.confidence 
            for s in actionable_signals 
            if s.signal.direction == SignalDirection.SHORT
        )
        
        if long_count > short_count:
            direction = SignalDirection.LONG
        elif short_count > long_count:
            direction = SignalDirection.SHORT
        else:
            direction = SignalDirection.FLAT
        
        # Use average of entry prices
        entry_prices = [
            s.signal.entry_price 
            for s in actionable_signals 
            if s.signal.entry_price
        ]
        avg_entry = sum(entry_prices) / len(entry_prices) if entry_prices else None
        
        # Use weighted average of stop loss
        stop_losses = [
            s.signal.stop_loss 
            for s in actionable_signals 
            if s.signal.stop_loss
        ]
        avg_stop = sum(stop_losses) / len(stop_losses) if stop_losses else None
        
        # Use weighted average of take profit
        take_profits = [
            s.signal.take_profit 
            for s in actionable_signals 
            if s.signal.take_profit
        ]
        avg_tp = sum(take_profits) / len(take_profits) if take_profits else None
        
        return TradingSignal(
            symbol=actionable_signals[0].signal.symbol,
            direction=direction,
            confidence=weighted_confidence,
            entry_price=avg_entry,
            stop_loss=avg_stop,
            take_profit=avg_tp,
            strategy_name="multi_strategy",
            metadata={
                "strategies": [s.strategy_name for s in actionable_signals],
                "long_score": long_count,
                "short_score": short_count
            }
        )
    
    def get_signals(
        self, 
        data: pd.DataFrame
    ) -> Optional[TradingSignal]:
        """
        Get aggregated signals from all strategies.
        
        Args:
            data: OHLCV data
            
        Returns:
            Aggregated signal or None
        """
        signals = self.generate_signals(data)
        return self.aggregate_signals(signals)
    
    def get_all_strategies(self) -> List[Dict]:
        """Get all registered strategies."""
        return [
            {
                "name": name,
                "enabled": name in self.enabled_strategies,
                "weight": self.strategy_weights.get(name, 1.0),
                "performance": strategy.get_performance()
            }
            for name, strategy in self.strategies.items()
        ]
    
    def get_strategy(self, name: str) -> Optional[BaseStrategy]:
        """Get strategy by name."""
        return self.strategies.get(name)
    
    def reset_all(self):
        """Reset all strategies."""
        for strategy in self.strategies.values():
            strategy.reset()
        
        logger.info("All strategies reset")

