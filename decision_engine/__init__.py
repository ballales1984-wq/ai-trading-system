"""
Decision Engine Package
Generates probabilistic trading signals by combining all analysis types

Modules:
- core: Core data structures and base classes
- signals: Signal generation logic
- monte_carlo: Monte Carlo simulation engine
- five_question: 5-Question decision framework
- external: External API integration

Usage:
    from decision_engine import DecisionEngine, TradingSignal
    
    engine = DecisionEngine()
    signals = engine.generate_signals()
"""

from .core import TradingSignal, PortfolioState, DecisionEngine
from .signals import SignalGenerator
from .monte_carlo import MonteCarloEngine
from .five_question import FiveQuestionFramework
from .external import ExternalDataFetcher

__all__ = [
    'TradingSignal',
    'PortfolioState', 
    'DecisionEngine',
    'SignalGenerator',
    'MonteCarloEngine',
    'FiveQuestionFramework',
    'ExternalDataFetcher',
]

__version__ = '2.0.0'

