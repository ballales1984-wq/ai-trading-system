"""
Performance Attribution Module
=============================
Analyzes portfolio performance by attributing returns to different factors:
- Asset allocation effects
- Security selection effects
- Interaction effects
- Timing effects

Author: AI Trading System
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class AttributionResult:
    """Container for attribution analysis results"""
    total_return: float
    asset_allocation_return: float
    security_selection_return: float
    interaction_return: float
    timing_return: float
    benchmark_return: float
    active_return: float
    holdings_attribution: Dict[str, float] = field(default_factory=dict)
    sector_attribution: Dict[str, float] = field(default_factory=dict)
    factor_contributions: Dict[str, float] = field(default_factory=dict)


@dataclass
class BrinsonResult:
    """Brinson-Fachler attribution result"""
    allocation_effect: float
    selection_effect: float
    interaction_effect: float
    total_effect: float


class PerformanceAttribution:
    """
    Performance Attribution Analysis
    
    Implements Brinson-Fachler and Brinson-Hood-Beebower models
    to decompose portfolio returns into contributing factors.
    """
    
    def __init__(
        self,
        portfolio_weights: Optional[pd.DataFrame] = None,
        benchmark_weights: Optional[pd.DataFrame] = None,
        portfolio_returns: Optional[pd.DataFrame] = None,
        benchmark_returns: Optional[pd.DataFrame] = None
    ):
        """
        Initialize Performance Attribution.
        
        Args:
            portfolio_weights: DataFrame with columns [asset, weight]
            benchmark_weights: DataFrame with columns [asset, weight]
            portfolio_returns: DataFrame with columns [asset, return]
            benchmark_returns: DataFrame with columns [asset, return]
        """
        self.portfolio_weights = portfolio_weights
        self.benchmark_weights = benchmark_weights
        self.portfolio_returns = portfolio_returns
        self.benchmark_returns = benchmark_returns
        
    def brinson_fachler(
        self,
        portfolio_weights: pd.DataFrame,
        benchmark_weights: pd.DataFrame,
        portfolio_returns: pd.DataFrame,
        benchmark_returns: pd.DataFrame
    ) -> BrinsonResult:
        """
        Brinson-Fachler Attribution Model.
        
        Formula:
        - Allocation Effect = (Rp - Rb) * Bw
        - Selection Effect = Bw * (Rp - Rb)
        - Interaction Effect = (Rw - Bw) * (Rp - Rb)
        
        Where:
        - Rp = Portfolio return for sector
        - Rb = Benchmark return for sector
        - Rw = Portfolio weight
        - Bw = Benchmark weight
        
        Args:
            portfolio_weights: Weights in portfolio
            benchmark_weights: Weights in benchmark
            portfolio_returns: Returns from portfolio
            benchmark_returns: Returns from benchmark
            
        Returns:
            BrinsonResult with attribution components
        """
        # Merge weights and returns
        df = pd.merge(
            portfolio_weights,
            benchmark_weights,
            on='asset',
            suffixes=('_portfolio', '_benchmark')
        )
        df = pd.merge(
            df,
            portfolio_returns,
            on='asset'
        )
        df = pd.merge(
            df,
            benchmark_returns,
            on='asset',
            suffixes=('_portfolio', '_benchmark')
        )
        
        # Calculate allocation effect: (Rp - Rb) * Bw
        df['allocation_effect'] = (
            df['return_portfolio'] - df['return_benchmark']
        ) * df['weight_benchmark']
        
        # Calculate selection effect: Bw * (Rp - Rb)
        df['selection_effect'] = (
            df['weight_benchmark'] * 
            (df['return_portfolio'] - df['return_benchmark'])
        )
        
        # Calculate interaction effect: (Rw - Bw) * (Rp - Rb)
        df['interaction_effect'] = (
            (df['weight_portfolio'] - df['weight_benchmark']) * 
            (df['return_portfolio'] - df['return_benchmark'])
        )
        
        # Sum effects
        allocation_effect = df['allocation_effect'].sum()
        selection_effect = df['selection_effect'].sum()
        interaction_effect = df['interaction_effect'].sum()
        total_effect = allocation_effect + selection_effect + interaction_effect
        
        return BrinsonResult(
            allocation_effect=allocation_effect,
            selection_effect=selection_effect,
            interaction_effect=interaction_effect,
            total_effect=total_effect
        )
    
    def brinson_hood_beebower(
        self,
        portfolio_weights: pd.DataFrame,
        benchmark_weights: pd.DataFrame,
        portfolio_returns: pd.DataFrame,
        benchmark_returns: pd.DataFrame
    ) -> BrinsonResult:
        """
        Brinson-Hood-Beebower Attribution Model.
        
        Difference from Brinson-Fachler:
        - Allocation Effect = (Rw - Bw) * Rb
        - Selection Effect = Rw * (Rp - Rb)
        
        Args:
            portfolio_weights: Weights in portfolio
            benchmark_weights: Weights in benchmark
            portfolio_returns: Returns from portfolio
            benchmark_returns: Returns from benchmark
            
        Returns:
            BrinsonResult with attribution components
        """
        # Merge all data
        df = pd.merge(
            portfolio_weights,
            benchmark_weights,
            on='asset',
            suffixes=('_portfolio', '_benchmark')
        )
        df = pd.merge(
            df,
            portfolio_returns,
            on='asset'
        )
        df = pd.merge(
            df,
            benchmark_returns,
            on='asset',
            suffixes=('_portfolio', '_benchmark')
        )
        
        # Calculate allocation effect: (Rw - Bw) * Rb
        df['allocation_effect'] = (
            df['weight_portfolio'] - df['weight_benchmark']
        ) * df['return_benchmark']
        
        # Calculate selection effect: Rw * (Rp - Rb)
        df['selection_effect'] = (
            df['weight_portfolio'] * 
            (df['return_portfolio'] - df['return_benchmark'])
        )
        
        # Calculate interaction effect: (Rw - Bw) * (Rp - Rb)
        df['interaction_effect'] = (
            (df['weight_portfolio'] - df['weight_benchmark']) * 
            (df['return_portfolio'] - df['return_benchmark'])
        )
        
        allocation_effect = df['allocation_effect'].sum()
        selection_effect = df['selection_effect'].sum()
        interaction_effect = df['interaction_effect'].sum()
        total_effect = allocation_effect + selection_effect + interaction_effect
        
        return BrinsonResult(
            allocation_effect=allocation_effect,
            selection_effect=selection_effect,
            interaction_effect=interaction_effect,
            total_effect=total_effect
        )
    
    def factor_attribution(
        self,
        portfolio_returns: pd.Series,
        factor_returns: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Factor-based attribution using regression.
        
        Decomposes returns into factor contributions using
        multi-factor regression analysis.
        
        Args:
            portfolio_returns: Series of portfolio returns
            factor_returns: DataFrame of factor returns (columns: factor names)
            
        Returns:
            Dictionary of factor contributions
        """
        from numpy.linalg import lstsq
        
        # Remove NaN values
        valid_idx = portfolio_returns.notna() & factor_returns.notna().all(axis=1)
        y = portfolio_returns[valid_idx].values
        X = factor_returns[valid_idx].values
        
        # Add constant for intercept
        X = np.column_stack([np.ones(len(y)), X])
        
        # Run regression
        coefficients, residuals, rank, s = lstsq(X, y, rcond=None)
        
        # Calculate factor contributions
        contributions = {}
        factor_names = ['intercept'] + list(factor_returns.columns)
        
        for i, name in enumerate(factor_names):
            contributions[name] = coefficients[i] * factor_returns[name].mean() if name != 'intercept' else coefficients[i]
        
        return contributions
    
    def timing_attribution(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        lookback_period: int = 20
    ) -> float:
        """
        Calculate timing contribution to returns.
        
        Measures the value added by tactical asset allocation
        (timing) vs. passive holding.
        
        Args:
            portfolio_returns: Portfolio return series
            benchmark_returns: Benchmark return series
            lookback_period: Period for calculating moving average
            
        Returns:
            Timing contribution as decimal
        """
        # Calculate active returns (portfolio - benchmark)
        active_returns = portfolio_returns - benchmark_returns
        
        # Calculate benchmark momentum (rolling mean)
        benchmark_momentum = benchmark_returns.rolling(lookback_period).mean()
        
        # Timing value = correlation between active returns and benchmark momentum
        # Positive correlation = good timing
        timing_value = active_returns.corr(benchmark_momentum)
        
        return timing_value if not np.isnan(timing_value) else 0.0
    
    def full_attribution(
        self,
        portfolio_weights: pd.DataFrame,
        benchmark_weights: pd.DataFrame,
        portfolio_returns: pd.DataFrame,
        benchmark_returns: pd.DataFrame,
        factor_returns: Optional[pd.DataFrame] = None
    ) -> AttributionResult:
        """
        Run full attribution analysis combining all methods.
        
        Args:
            portfolio_weights: Portfolio asset weights
            benchmark_weights: Benchmark asset weights
            portfolio_returns: Portfolio asset returns
            benchmark_returns: Benchmark asset returns
            factor_returns: Optional factor returns for factor attribution
            
        Returns:
            AttributionResult with all components
        """
        # Get total returns
        portfolio_total_return = (
            portfolio_weights['weight'] * portfolio_returns['return']
        ).sum()
        
        benchmark_total_return = (
            benchmark_weights['weight'] * benchmark_returns['return']
        ).sum()
        
        # Run Brinson attribution
        brinson = self.brinson_fachler(
            portfolio_weights,
            benchmark_weights,
            portfolio_returns,
            benchmark_returns
        )
        
        # Calculate active return
        active_return = portfolio_total_return - benchmark_total_return
        
        # Factor attribution (if provided)
        factor_contributions = {}
        if factor_returns is not None:
            factor_contributions = self.factor_attribution(
                portfolio_returns['return'],
                factor_returns
            )
        
        # Timing attribution
        timing_return = self.timing_attribution(
            portfolio_returns['return'],
            benchmark_returns['return']
        )
        
        return AttributionResult(
            total_return=portfolio_total_return,
            asset_allocation_return=brinson.allocation_effect,
            security_selection_return=brinson.selection_effect,
            interaction_return=brinson.interaction_effect,
            timing_return=timing_return,
            benchmark_return=benchmark_total_return,
            active_return=active_return,
            holdings_attribution={},
            sector_attribution={},
            factor_contributions=factor_contributions
        )
    
    def generate_report(self, result: AttributionResult) -> str:
        """
        Generate human-readable attribution report.
        
        Args:
            result: AttributionResult from analysis
            
        Returns:
            Formatted string report
        """
        report = []
        report.append("=" * 60)
        report.append("PERFORMANCE ATTRIBUTION REPORT")
        report.append("=" * 60)
        report.append("")
        
        report.append("RETURNS SUMMARY")
        report.append("-" * 40)
        report.append(f"Total Portfolio Return:     {result.total_return:>10.2%}")
        report.append(f"Benchmark Return:           {result.benchmark_return:>10.2%}")
        report.append(f"Active Return:              {result.active_return:>10.2%}")
        report.append("")
        
        report.append("ATTRIBUTION BREAKDOWN")
        report.append("-" * 40)
        report.append(f"Asset Allocation Effect:    {result.asset_allocation_return:>10.2%}")
        report.append(f"Security Selection Effect:  {result.security_selection_return:>10.2%}")
        report.append(f"Interaction Effect:         {result.interaction_return:>10.2%}")
        report.append(f"Timing Effect:              {result.timing_return:>10.2%}")
        report.append("")
        
        total_attributed = (
            result.asset_allocation_return + 
            result.security_selection_return + 
            result.interaction_return
        )
        report.append(f"Total Attributed:            {total_attributed:>10.2%}")
        report.append(f"Unexplained:                 {result.active_return - total_attributed:>10.2%}")
        
        if result.factor_contributions:
            report.append("")
            report.append("FACTOR CONTRIBUTIONS")
            report.append("-" * 40)
            for factor, contribution in result.factor_contributions.items():
                report.append(f"{factor:25s}: {contribution:>10.4f}")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)


def run_demo():
    """Demo function showing how to use the attribution module."""
    # Create sample data
    assets = ['BTC', 'ETH', 'SOL', 'ADA', 'DOT']
    
    # Portfolio weights
    portfolio_weights = pd.DataFrame({
        'asset': assets,
        'weight': [0.40, 0.25, 0.15, 0.12, 0.08]
    })
    
    # Benchmark weights (e.g., market cap weighted)
    benchmark_weights = pd.DataFrame({
        'asset': assets,
        'weight': [0.35, 0.30, 0.15, 0.10, 0.10]
    })
    
    # Portfolio returns (annualized)
    portfolio_returns = pd.DataFrame({
        'asset': assets,
        'return': [0.45, 0.35, 0.55, 0.25, 0.30]
    })
    
    # Benchmark returns
    benchmark_returns = pd.DataFrame({
        'asset': assets,
        'return': [0.40, 0.30, 0.50, 0.20, 0.25]
    })
    
    # Create factor returns (optional)
    factor_returns = pd.DataFrame({
        'momentum': [0.02, 0.01, 0.03, -0.01, 0.02],
        'value': [0.01, 0.02, -0.01, 0.03, 0.01],
        'size': [-0.01, 0.01, 0.02, 0.01, -0.01]
    }, index=assets)
    
    # Run attribution
    pa = PerformanceAttribution()
    result = pa.full_attribution(
        portfolio_weights,
        benchmark_weights,
        portfolio_returns,
        benchmark_returns,
        factor_returns
    )
    
    # Print report
    print(pa.generate_report(result))
    
    return result


if __name__ == "__main__":
    run_demo()
