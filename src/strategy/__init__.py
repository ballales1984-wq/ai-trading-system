# src/strategy/__init__.py
"""
Strategy Modules for AI Trading System
======================================
Plug-in strategy architecture for trading signals.

Strategies:
- BaseStrategy: Abstract base class for all strategies
- MomentumStrategy: Momentum-based trading
- MeanReversionStrategy: Mean reversion trading
- BreakoutStrategy: Breakout detection
- TrendFollowingStrategy: Trend following
"""

from src.strategy.base_strategy import (
    BaseStrategy,
    TradingSignal,
    SignalType,
    SignalStrength,
    StrategyContext,
)

# Aliases for backwards compatibility
Signal = TradingSignal
SignalAction = SignalType

from src.strategy.momentum import MomentumStrategy

# Optional import for mean reversion
try:
    from src.strategy.mean_reversion import MeanReversionStrategy
except ImportError:
    MeanReversionStrategy = None

__all__ = [
    "BaseStrategy",
    "TradingSignal",
    "Signal",
    "SignalType",
    "SignalAction",
    "SignalStrength",
    "StrategyContext",
    "MomentumStrategy",
    "MeanReversionStrategy",
]
