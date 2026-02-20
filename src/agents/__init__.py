# src/agents/__init__.py
"""
Agent Modules for AI Trading System
====================================
Multi-agent architecture for trading system components.

Agents:
- BaseAgent: Abstract base class for all agents
- MarketDataAgent: Real-time market data streaming
- MonteCarloAgent: Monte Carlo simulation engine
- RiskAgent: Risk metrics calculation (VaR, CVaR)
- SupervisorAgent: Orchestrates all agents
"""

from src.agents.base_agent import BaseAgent
from src.agents.agent_marketdata import MarketDataAgent
from src.agents.agent_montecarlo import MonteCarloAgent
from src.agents.agent_risk import RiskAgent
from src.agents.agent_supervisor import SupervisorAgent

__all__ = [
    "BaseAgent",
    "MarketDataAgent",
    "MonteCarloAgent",
    "RiskAgent",
    "SupervisorAgent",
]
