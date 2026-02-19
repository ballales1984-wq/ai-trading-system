# src/risk_optimizer.py
"""
Risk Parameter Optimization
===========================
Optimizes risk management parameters for trading strategies.

Features:
- Parameter grid search
- Walk-forward optimization
- Risk-adjusted performance metrics
- Monte Carlo simulation for stress testing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import logging
from itertools import product

logger = logging.getLogger(__name__)


@dataclass
class RiskParameters:
    """Container for risk parameters."""
    var_confidence: float = 0.95
    max_drawdown: float = 0.20
    max_position_pct: float = 0.10
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.10
    trailing_stop_pct: float = 0.03
    kelly_fraction: float = 0.25
    
    def to_dict(self) -> Dict:
        return {
            'var_confidence': self.var_confidence,
            'max_drawdown': self.max_drawdown,
            'max_position_pct': self.max_position_pct,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct,
            'trailing_stop_pct': self.trailing_stop_pct,
            'kelly_fraction': self.kelly_fraction
        }


@dataclass
class OptimizationResult:
    """Container for optimization results."""
    best_params: RiskParameters
    best_score: float
    all_results: List[Dict]
    metrics: Dict[str, float]
    optimization_time: float
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class RiskOptimizer:
    """
    Optimizes risk parameters using historical data.
    
    Uses walk-forward optimization to find optimal risk parameters
    that maximize risk-adjusted returns.
    """
    
    # Default parameter ranges for optimization
    PARAM_RANGES = {
        'var_confidence': [0.90, 0.95, 0.99],
        'max_drawdown': [0.10, 0.15, 0.20, 0.25],
        'max_position_pct': [0.05, 0.10, 0.15, 0.20],
        'stop_loss_pct': [0.02, 0.03, 0.05, 0.07],
        'take_profit_pct': [0.05, 0.10, 0.15, 0.20],
        'trailing_stop_pct': [0.02, 0.03, 0.05],
        'kelly_fraction': [0.10, 0.25, 0.50]
    }
    
    def __init__(
        self,
        n_splits: int = 5,
        min_trades: int = 10,
        risk_free_rate: float = 0.02
    ):
        """
        Initialize Risk Optimizer.
        
        Args:
            n_splits: Number of walk-forward splits
            min_trades: Minimum trades required for valid optimization
            risk_free_rate: Annual risk-free rate for Sharpe calculation
        """
        self.n_splits = n_splits
        self.min_trades = min_trades
        self.risk_free_rate = risk_free_rate
        self.results: List[OptimizationResult] = []
    
    def calculate_sharpe_ratio(
        self,
        returns: pd.Series,
        periods_per_year: int = 252
    ) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        excess_returns = returns.mean() - self.risk_free_rate / periods_per_year
        sharpe = excess_returns / returns.std() * np.sqrt(periods_per_year)
        return sharpe
    
    def calculate_sortino_ratio(
        self,
        returns: pd.Series,
        periods_per_year: int = 252
    ) -> float:
        """Calculate Sortino ratio (downside deviation only)."""
        if len(returns) == 0:
            return 0.0
        
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return float('inf')
        
        downside_std = downside_returns.std()
        if downside_std == 0:
            return 0.0
        
        excess_returns = returns.mean() - self.risk_free_rate / periods_per_year
        sortino = excess_returns / downside_std * np.sqrt(periods_per_year)
        return sortino
    
    def calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if len(equity_curve) == 0:
            return 0.0
        
        peak = equity_curve.expanding(min_periods=1).max()
        drawdown = (equity_curve - peak) / peak
        return abs(drawdown.min())
    
    def calculate_calmar_ratio(
        self,
        returns: pd.Series,
        periods_per_year: int = 252
    ) -> float:
        """Calculate Calmar ratio (return / max drawdown)."""
        if len(returns) == 0:
            return 0.0
        
        equity_curve = (1 + returns).cumprod()
        max_dd = self.calculate_max_drawdown(equity_curve)
        
        if max_dd == 0:
            return float('inf')
        
        annual_return = returns.mean() * periods_per_year
        return annual_return / max_dd
    
    def simulate_trades(
        self,
        returns: pd.Series,
        params: RiskParameters
    ) -> Tuple[pd.Series, int]:
        """
        Simulate trades with risk parameters applied.
        
        Args:
            returns: Series of trade returns
            params: Risk parameters to apply
            
        Returns:
            Tuple of (adjusted_returns, num_trades)
        """
        adjusted_returns = []
        equity = 1.0
        peak_equity = 1.0
        trades = 0
        
        for ret in returns:
            # Apply position sizing
            position_size = min(params.max_position_pct, params.kelly_fraction)
            sized_ret = ret * position_size
            
            # Apply stop loss
            if sized_ret < -params.stop_loss_pct:
                sized_ret = -params.stop_loss_pct
            
            # Apply take profit
            if sized_ret > params.take_profit_pct:
                sized_ret = params.take_profit_pct
            
            # Update equity
            equity *= (1 + sized_ret)
            
            # Trailing stop check
            if equity > peak_equity:
                peak_equity = equity
            
            if (peak_equity - equity) / peak_equity > params.trailing_stop_pct:
                # Trailing stop triggered - reduce position
                equity = peak_equity * (1 - params.trailing_stop_pct)
            
            # Max drawdown check
            if (peak_equity - equity) / peak_equity > params.max_drawdown:
                # Stop trading
                break
            
            adjusted_returns.append(sized_ret)
            trades += 1
        
        return pd.Series(adjusted_returns), trades
    
    def evaluate_params(
        self,
        returns: pd.Series,
        params: RiskParameters
    ) -> Dict[str, float]:
        """
        Evaluate risk parameters on returns.
        
        Args:
            returns: Historical returns
            params: Risk parameters to evaluate
            
        Returns:
            Dictionary of evaluation metrics
        """
        adjusted_returns, trades = self.simulate_trades(returns, params)
        
        if trades < self.min_trades:
            return {
                'sharpe': -999,
                'sortino': -999,
                'calmar': -999,
                'total_return': -999,
                'max_drawdown': 1.0,
                'trades': trades
            }
        
        equity_curve = (1 + adjusted_returns).cumprod()
        
        return {
            'sharpe': self.calculate_sharpe_ratio(adjusted_returns),
            'sortino': self.calculate_sortino_ratio(adjusted_returns),
            'calmar': self.calculate_calmar_ratio(adjusted_returns),
            'total_return': equity_curve.iloc[-1] - 1 if len(equity_curve) > 0 else 0,
            'max_drawdown': self.calculate_max_drawdown(equity_curve),
            'trades': trades
        }
    
    def optimize(
        self,
        returns: pd.Series,
        param_ranges: Optional[Dict[str, List]] = None,
        metric: str = 'sharpe'
    ) -> OptimizationResult:
        """
        Optimize risk parameters using grid search.
        
        Args:
            returns: Historical returns series
            param_ranges: Custom parameter ranges (optional)
            metric: Metric to optimize ('sharpe', 'sortino', 'calmar')
            
        Returns:
            OptimizationResult with best parameters
        """
        import time
        start_time = time.time()
        
        if param_ranges is None:
            param_ranges = self.PARAM_RANGES
        
        # Generate all parameter combinations
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        combinations = list(product(*param_values))
        
        logger.info(f"Testing {len(combinations)} parameter combinations...")
        
        all_results = []
        best_score = -float('inf')
        best_params = None
        
        for combo in combinations:
            params = RiskParameters(**dict(zip(param_names, combo)))
            metrics = self.evaluate_params(returns, params)
            
            result = {
                'params': params.to_dict(),
                'metrics': metrics
            }
            all_results.append(result)
            
            score = metrics.get(metric, -999)
            if score > best_score:
                best_score = score
                best_params = params
        
        optimization_time = time.time() - start_time
        
        result = OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_results=all_results,
            metrics=self.evaluate_params(returns, best_params) if best_params else {},
            optimization_time=optimization_time
        )
        
        self.results.append(result)
        
        logger.info(
            f"Optimization complete: best_{metric}={best_score:.4f}, "
            f"time={optimization_time:.2f}s"
        )
        
        return result
    
    def walk_forward_optimize(
        self,
        returns: pd.Series,
        train_size: int = 252,
        test_size: int = 63,
        param_ranges: Optional[Dict[str, List]] = None,
        metric: str = 'sharpe'
    ) -> List[OptimizationResult]:
        """
        Walk-forward optimization.
        
        Args:
            returns: Full historical returns
            train_size: Training window size (days)
            test_size: Test window size (days)
            param_ranges: Custom parameter ranges
            metric: Metric to optimize
            
        Returns:
            List of OptimizationResult for each fold
        """
        results = []
        n = len(returns)
        
        for i in range(0, n - train_size - test_size, test_size):
            train = returns.iloc[i:i + train_size]
            test = returns.iloc[i + train_size:i + train_size + test_size]
            
            # Optimize on training data
            opt_result = self.optimize(train, param_ranges, metric)
            
            # Evaluate on test data
            test_metrics = self.evaluate_params(test, opt_result.best_params)
            
            results.append({
                'fold': len(results) + 1,
                'train_start': i,
                'train_metrics': opt_result.metrics,
                'test_metrics': test_metrics,
                'best_params': opt_result.best_params.to_dict()
            })
        
        logger.info(f"Completed {len(results)} walk-forward folds")
        return results
    
    def monte_carlo_stress_test(
        self,
        returns: pd.Series,
        params: RiskParameters,
        n_simulations: int = 1000,
        n_days: int = 252
    ) -> Dict[str, float]:
        """
        Monte Carlo stress test for risk parameters.
        
        Args:
            returns: Historical returns for parameter estimation
            params: Risk parameters to test
            n_simulations: Number of Monte Carlo simulations
            n_days: Number of days to simulate
            
        Returns:
            Dictionary of stress test results
        """
        # Estimate return distribution
        mu = returns.mean()
        sigma = returns.std()
        
        final_values = []
        max_drawdowns = []
        
        for _ in range(n_simulations):
            # Generate random returns
            sim_returns = np.random.normal(mu, sigma, n_days)
            sim_series = pd.Series(sim_returns)
            
            # Apply risk parameters
            adjusted, _ = self.simulate_trades(sim_series, params)
            
            if len(adjusted) > 0:
                equity = (1 + adjusted).cumprod()
                final_values.append(equity.iloc[-1])
                max_drawdowns.append(self.calculate_max_drawdown(equity))
        
        return {
            'mean_final_value': np.mean(final_values),
            'median_final_value': np.median(final_values),
            'worst_case': np.percentile(final_values, 5),
            'best_case': np.percentile(final_values, 95),
            'prob_loss': sum(1 for v in final_values if v < 1) / len(final_values),
            'avg_max_drawdown': np.mean(max_drawdowns),
            'worst_max_drawdown': np.percentile(max_drawdowns, 95)
        }


def optimize_risk_for_strategy(
    returns: pd.Series,
    metric: str = 'sharpe'
) -> Tuple[RiskParameters, Dict]:
    """
    Convenience function to optimize risk parameters.
    
    Args:
        returns: Strategy returns series
        metric: Metric to optimize
        
    Returns:
        Tuple of (best_params, metrics)
    """
    optimizer = RiskOptimizer(n_splits=5)
    result = optimizer.optimize(returns, metric=metric)
    return result.best_params, result.metrics


if __name__ == "__main__":
    print("Risk Parameter Optimization Module")
    print("=" * 50)
    print("\nAvailable parameters to optimize:")
    for param, values in RiskOptimizer.PARAM_RANGES.items():
        print(f"  - {param}: {values}")
    print("\nUsage:")
    print("  optimizer = RiskOptimizer()")
    print("  result = optimizer.optimize(returns)")
    print("  best_params = result.best_params")
