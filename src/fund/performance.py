"""
Performance Attribution Module - AI Trading System
Calcola e analizza le performance del fondo
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from decimal import Decimal
import math


@dataclass
class PerformanceMetrics:
    """Metriche di performance del fondo"""
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_return: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "volatility": self.volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "calmar_ratio": self.calmar_ratio,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_trade_return": self.avg_trade_return
        }


@dataclass
class BenchmarkComparison:
    """Confronto con benchmark"""
    benchmark_return: float = 0.0
    alpha: float = 0.0
    beta: float = 1.0
    correlation: float = 0.0
    tracking_error: float = 0.0
    information_ratio: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "benchmark_return": self.benchmark_return,
            "alpha": self.alpha,
            "beta": self.beta,
            "correlation": self.correlation,
            "tracking_error": self.tracking_error,
            "information_ratio": self.information_ratio
        }


@dataclass
class AttributionFactor:
    """Fattori di attribuzione della performance"""
    asset_allocation: float = 0.0
    security_selection: float = 0.0
    interaction_effect: float = 0.0
    currency_effect: float = 0.0
    timing_effect: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "asset_allocation": self.asset_allocation,
            "security_selection": self.security_selection,
            "interaction_effect": self.interaction_effect,
            "currency_effect": self.currency_effect,
            "timing_effect": self.timing_effect
        }


class PerformanceAnalyzer:
    """
    Analizza le performance del fondo con attribuzione
    """
    
    def __init__(self):
        self.returns: List[float] = []
        self.benchmark_returns: List[float] = []
        self.trade_returns: List[float] = []
        self.daily_values: List[Decimal] = []
    
    def add_return(self, return_pct: float):
        """Aggiunge un rendimento"""
        self.returns.append(return_pct)
    
    def add_benchmark_return(self, return_pct: float):
        """Aggiunge un rendimento del benchmark"""
        self.benchmark_returns.append(return_pct)
    
    def add_trade_return(self, return_pct: float):
        """Aggiunge un rendimento di trade"""
        self.trade_returns.append(return_pct)
    
    def add_daily_value(self, value: Decimal):
        """Aggiunge il valore giornaliero del portfolio"""
        self.daily_values.append(value)
    
    def calculate_metrics(self, risk_free_rate: float = 0.02) -> PerformanceMetrics:
        """
        Calcola le metriche di performance
        """
        metrics = PerformanceMetrics()
        
        if not self.returns:
            return metrics
        
        # Total return
        metrics.total_return = sum(self.returns)
        
        # Annualized return (assumendo 252 trading days)
        years = len(self.returns) / 252
        if years > 0 and metrics.total_return > -1:
            metrics.annualized_return = ((1 + metrics.total_return / 100) ** (1 / years) - 1) * 100
        
        # Volatility (annualized)
        metrics.volatility = self._calculate_volatility() * math.sqrt(252) * 100
        
        # Sharpe Ratio
        if metrics.volatility > 0:
            metrics.sharpe_ratio = (metrics.annualized_return - risk_free_rate * 100) / metrics.volatility
        
        # Sortino Ratio (downside deviation)
        downside = self._calculate_downside_deviation()
        if downside > 0:
            metrics.sortino_ratio = (metrics.annualized_return - risk_free_rate * 100) / (downside * math.sqrt(252) * 100)
        
        # Max Drawdown
        metrics.max_drawdown = self._calculate_max_drawdown()
        
        # Calmar Ratio
        if metrics.max_drawdown > 0:
            metrics.calmar_ratio = metrics.annualized_return / metrics.max_drawdown
        
        # Win Rate
        if self.trade_returns:
            winning_trades = [r for r in self.trade_returns if r > 0]
            metrics.win_rate = len(winning_trades) / len(self.trade_returns) * 100 if self.trade_returns else 0
        
        # Profit Factor
        if self.trade_returns:
            gross_profit = sum(r for r in self.trade_returns if r > 0)
            gross_loss = abs(sum(r for r in self.trade_returns if r < 0))
            metrics.profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Avg Trade Return
        if self.trade_returns:
            metrics.avg_trade_return = sum(self.trade_returns) / len(self.trade_returns)
        
        return metrics
    
    def _calculate_volatility(self) -> float:
        """Calcola la deviazione standard"""
        if len(self.returns) < 2:
            return 0.0
        
        mean = sum(self.returns) / len(self.returns)
        variance = sum((r - mean) ** 2 for r in self.returns) / (len(self.returns) - 1)
        return math.sqrt(variance) / 100
    
    def _calculate_downside_deviation(self) -> float:
        """Calcola la downside deviation"""
        if not self.returns:
            return 0.0
        
        negative_returns = [r for r in self.returns if r < 0]
        if not negative_returns:
            return 0.0
        
        mean = sum(negative_returns) / len(negative_returns)
        variance = sum((r - mean) ** 2 for r in negative_returns) / len(negative_returns)
        return math.sqrt(variance) / 100
    
    def _calculate_max_drawdown(self) -> float:
        """Calcola il max drawdown"""
        if not self.daily_values:
            return 0.0
        
        peak = float(self.daily_values[0])
        max_dd = 0.0
        
        for value in self.daily_values:
            val = float(value)
            if val > peak:
                peak = val
            
            dd = (peak - val) / peak * 100
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    def compare_with_benchmark(self) -> BenchmarkComparison:
        """
        Confronta le performance con un benchmark
        """
        comparison = BenchmarkComparison()
        
        if not self.returns or not self.benchmark_returns:
            return comparison
        
        # Assumiamo stessa lunghezza
        n = min(len(self.returns), len(self.benchmark_returns))
        
        # Benchmark return
        comparison.benchmark_return = sum(self.benchmark_returns[:n])
        
        # Alpha (intercept della regressione)
        comparison.alpha = self._calculate_alpha(
            self.returns[:n], 
            self.benchmark_returns[:n]
        )
        
        # Beta (slope della regressione)
        comparison.beta = self._calculate_beta(
            self.returns[:n],
            self.benchmark_returns[:n]
        )
        
        # Correlation
        comparison.correlation = self._calculate_correlation(
            self.returns[:n],
            self.benchmark_returns[:n]
        )
        
        # Tracking Error
        comparison.tracking_error = self._calculate_tracking_error(
            self.returns[:n],
            self.benchmark_returns[:n]
        )
        
        # Information Ratio
        if comparison.tracking_error > 0:
            comparison.information_ratio = comparison.alpha / comparison.tracking_error
        
        return comparison
    
    def _calculate_alpha(self, returns: List[float], benchmark: List[float]) -> float:
        """Calcola Alpha"""
        if len(returns) < 2:
            return 0.0
        
        mean_ret = sum(returns) / len(returns)
        mean_bench = sum(benchmark) / len(benchmark)
        beta = self._calculate_beta(returns, benchmark)
        
        # Alpha = Mean Return - (Risk Free + Beta * (Benchmark - Risk Free))
        return mean_ret - beta * mean_bench
    
    def _calculate_beta(self, returns: List[float], benchmark: List[float]) -> float:
        """Calcola Beta"""
        if len(returns) < 2:
            return 1.0
        
        n = len(returns)
        mean_ret = sum(returns) / n
        mean_bench = sum(benchmark) / n
        
        covariance = sum((returns[i] - mean_ret) * (benchmark[i] - mean_bench) 
                        for i in range(n)) / (n - 1)
        variance = sum((b - mean_bench) ** 2 for b in benchmark) / (n - 1)
        
        return covariance / variance if variance > 0 else 1.0
    
    def _calculate_correlation(self, returns: List[float], benchmark: List[float]) -> float:
        """Calcola la correlazione"""
        if len(returns) < 2:
            return 0.0
        
        n = len(returns)
        mean_ret = sum(returns) / n
        mean_bench = sum(benchmark) / n
        
        covariance = sum((returns[i] - mean_ret) * (benchmark[i] - mean_bench) 
                        for i in range(n)) / (n - 1)
        
        std_ret = math.sqrt(sum((r - mean_ret) ** 2 for r in returns) / (n - 1))
        std_bench = math.sqrt(sum((b - mean_bench) ** 2 for b in benchmark) / (n - 1))
        
        if std_ret == 0 or std_bench == 0:
            return 0.0
        
        return covariance / (std_ret * std_bench)
    
    def _calculate_tracking_error(self, returns: List[float], benchmark: List[float]) -> float:
        """Calcola il tracking error"""
        if len(returns) < 2:
            return 0.0
        
        differences = [returns[i] - benchmark[i] for i in range(len(returns))]
        mean_diff = sum(differences) / len(differences)
        variance = sum((d - mean_diff) ** 2 for d in differences) / (len(differences) - 1)
        
        return math.sqrt(variance) * math.sqrt(252)
    
    def calculate_attribution(self, 
                            portfolio_weights: Dict[str, float],
                            sector_returns: Dict[str, float],
                            benchmark_weights: Dict[str, float]) -> AttributionFactor:
        """
        Calcola l'attribuzione della performance
        """
        attribution = AttributionFactor()
        
        # Asset Allocation Effect
        for sector in portfolio_weights:
            if sector in benchmark_weights:
                w_p = portfolio_weights[sector]
                w_b = benchmark_weights.get(sector, 0)
                r_b = sector_returns.get(sector, 0)
                attribution.asset_allocation += (w_p - w_b) * r_b
        
        # Security Selection Effect
        for sector in sector_returns:
            if sector in portfolio_weights:
                w_p = portfolio_weights.get(sector, 0)
                r_p = sector_returns[sector]
                r_b = sector_returns.get(sector, 0)
                attribution.security_selection += w_p * (r_p - r_b)
        
        return attribution
    
    def get_rolling_metrics(self, window: int = 30) -> List[Dict]:
        """
        Calcola metriche rolling
        """
        results = []
        
        for i in range(window, len(self.returns) + 1):
            window_returns = self.returns[i - window:i]
            
            metrics = {
                "period_start": i - window,
                "period_end": i - 1,
                "return": sum(window_returns),
                "volatility": 0.0,
                "max_dd": 0.0
            }
            
            # Volatility
            if len(window_returns) > 1:
                mean = sum(window_returns) / len(window_returns)
                variance = sum((r - mean) ** 2 for r in window_returns) / (len(window_returns) - 1)
                metrics["volatility"] = math.sqrt(variance) * math.sqrt(252) * 100
            
            # Max DD (se abbiamo valori)
            if len(self.daily_values) >= i:
                window_values = self.daily_values[i - window:i]
                peak = float(window_values[0])
                max_dd = 0.0
                for v in window_values:
                    val = float(v)
                    if val > peak:
                        peak = val
                    dd = (peak - val) / peak * 100
                    if dd > max_dd:
                        max_dd = dd
                metrics["max_dd"] = max_dd
            
            results.append(metrics)
        
        return results
    
    def generate_performance_report(self) -> Dict:
        """
        Genera un report completo delle performance
        """
        metrics = self.calculate_metrics()
        benchmark = self.compare_with_benchmark()
        
        return {
            "metrics": metrics.to_dict(),
            "benchmark_comparison": benchmark.to_dict(),
            "summary": {
                "total_days": len(self.returns),
                "total_trades": len(self.trade_returns),
                "positive_days": len([r for r in self.returns if r > 0]),
                "negative_days": len([r for r in self.returns if r < 0]),
                "best_day": max(self.returns) if self.returns else 0,
                "worst_day": min(self.returns) if self.returns else 0
            }
        }


def calculate_var(returns: List[float], confidence: float = 0.95) -> float:
    """
    Calcola il Value at Risk
    """
    if not returns:
        return 0.0
    
    sorted_returns = sorted(returns)
    index = int((1 - confidence) * len(sorted_returns))
    
    return abs(sorted_returns[index]) if index < len(sorted_returns) else 0.0


def calculate_cvar(returns: List[float], confidence: float = 0.95) -> float:
    """
    Calcola il Conditional Value at Risk (Expected Shortfall)
    """
    if not returns:
        return 0.0
    
    sorted_returns = sorted(returns)
    index = int((1 - confidence) * len(sorted_returns))
    
    tail_returns = sorted_returns[:index + 1] if index < len(sorted_returns) else sorted_returns
    
    return abs(sum(tail_returns) / len(tail_returns)) if tail_returns else 0.0
