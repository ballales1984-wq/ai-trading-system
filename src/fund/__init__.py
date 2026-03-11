"""
AI Trading System - Fund Management Module
Provides fund structure simulation and investor management
"""

from .fund_manager import (
    FundManager,
    FundStatus,
    FeeType,
    FeeStructure,
    Investor,
    InvestorStatus,
    NAV,
    Subscription,
    Redemption
)

from .performance import (
    PerformanceAnalyzer,
    PerformanceMetrics,
    BenchmarkComparison,
    AttributionFactor,
    calculate_var,
    calculate_cvar
)

__all__ = [
    'FundManager',
    'FundStatus',
    'FeeType',
    'FeeStructure',
    'Investor',
    'InvestorStatus',
    'NAV',
    'Subscription',
    'Redemption',
    'PerformanceAnalyzer',
    'PerformanceMetrics',
    'BenchmarkComparison',
    'AttributionFactor',
    'calculate_var',
    'calculate_cvar'
]
