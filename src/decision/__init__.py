"""
Decision Module - Monte Carlo Simulation and Decision Engine

This module provides tools for evaluating trading opportunities using:
- Monte Carlo simulations for risk assessment
- Semantic analysis (sentiment, events, trends)
- Numeric analysis (RSI, MACD, volatility)
- Combined scoring and order generation
"""

from .decision_montecarlo import MonteCarloSimulator, OpportunityFilter, DecisionEngine

__all__ = ['MonteCarloSimulator', 'OpportunityFilter', 'DecisionEngine']
