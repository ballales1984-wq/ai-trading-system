"""
Alpha Lab Module
===============
Research environment for discovering and testing trading alphas.

Author: AI Trading System
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
import hashlib
from scipy import stats
from collections import defaultdict
import warnings

logger = logging.getLogger(__name__)


class AlphaType(Enum):
    """Alpha types."""
    PRICE = "price"
    VOLUME = "volume"
    TECHNICAL = "technical"
    SENTIMENT = "sentiment"
    MACRO = "fundamental"
    CROSS_ASSET = "cross_asset"
    PATTERN = "pattern"
    REGIME = "regime"


class SignalDirection(Enum):
    """Signal direction."""
    LONG = 1
    SHORT = -1
    NEUTRAL = 0


@dataclass
class AlphaSignal:
    """Alpha signal."""
    alpha_id: str
    name: str
    alpha_type: AlphaType
    timestamp: datetime
    direction: SignalDirection
    strength: float  # 0-1
    features: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "alpha_id": self.alpha_id,
            "name": self.name,
            "type": self.alpha_type.value,
            "timestamp": self.timestamp.isoformat(),
            "direction": self.direction.name,
            "strength": self.strength,
            "features": self.features,
            "metadata": self.metadata,
        }


@dataclass
class AlphaPerformance:
    """Alpha performance metrics."""
    alpha_id: str
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    trade_count: int
    holding_period: float  # Average days
    turnover: float  # Average daily turnover
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "alpha_id": self.alpha_id,
            "total_return": self.total_return,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "trade_count": self.trade_count,
            "holding_period": self.holding_period,
            "turnover": self.turnover,
        }


class AlphaTemplate:
    """
    Alpha Template
    =============
    Predefined alpha patterns for research.
    """
    
    @staticmethod
    def momentum(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
        """Momentum alpha."""
        return df["close"].pct_change(lookback)
    
    @staticmethod
    def mean_reversion(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
        """Mean reversion alpha."""
        ma = df["close"].rolling(lookback).mean()
        return ( - ma) / ma
    
    @staticmethod
    def volume_price_correlation(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
        """Volume-price correlation alpha."""
        returns = df["close"].pct_change()
        volume_change = df["volume"].pct_change()
        return returns.rolling(lookback).corr(volume_change)
    
    @staticmethod
    def volatility_ratio(df: pd.DataFrame, short_window: int = 5, long_window: int = 20) -> pd.Series:
        """Volatility ratio alpha."""
        short_vol = df["close"].rolling(short_window).std()
        long_vol = df["close"].rolling(long_window).std()
        return short_vol / long_vol - 1
    
    @staticmethod
    def rsi_alpha(df: pd.DataFrame, window: int = 14) -> pd.Series:
        """RSI-based alpha."""
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return (rsi - 50) / 100  # Normalize to -0.5 to 0.5
    
    @staticmethod
    def Bollinger_Band_alpha(df: pd.DataFrame, window: int = 20, num_std: float = 2) -> pd.Series:
        """Bollinger Band alpha."""
        ma = df["close"].rolling(window).mean()
        std = df["close"].rolling(window).std()
        upper = ma + num_std * std
        lower = ma - num_std * std
        return (df["close"] - ma) / (upper - lower)
    
    @staticmethod
    def volume_spike(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
        """Volume spike alpha."""
        vol_ma = df["volume"].rolling(lookback).mean()
        return (df["volume"] - vol_ma) / vol_ma
    
    @staticmethod
    def price_volume_trend(df: pd.DataFrame) -> pd.Series:
        """Price-volume trend alpha."""
        pvt = (df["close"].pct_change() * df["volume"]).cumsum()
        return pvt.pct_change()
    
    @staticmethod
    def MACD_cross(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """MACD crossover alpha."""
        ema_fast = df["close"].ewm(span=fast).mean()
        ema_slow = df["close"].ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        return macd - signal_line
    
    @staticmethod
    def gap_up(df: pd.DataFrame) -> pd.Series:
        """Gap up alpha."""
        return (df["open"] - df["close"].shift(1)) / df["close"].shift(1)
    
    @staticmethod
    def high_low_range(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
        """High-low range alpha."""
        high = df["high"].rolling(lookback).max()
        low = df["low"].rolling(lookback).min()
        return (df["close"] - low) / (high - low + 1e-10) - 0.5


class AlphaResearcher:
    """
    Alpha Researcher
    ===============
    Research and discover new alphas.
    """
    
    def __init__(self):
        """Initialize alpha researcher."""
        self._alphas: Dict[str, Callable] = {}
        self._research_results: Dict[str, AlphaPerformance] = {}
        self._register_default_alphas()
        
        logger.info("Alpha Researcher initialized")
    
    def _register_default_alphas(self):
        """Register default alpha templates."""
        self._alphas["momentum"] = AlphaTemplate.momentum
        self._alphas["mean_reversion"] = AlphaTemplate.mean_reversion
        self._alphas["volume_price_corr"] = AlphaTemplate.volume_price_correlation
        self._alphas["volatility_ratio"] = AlphaTemplate.volatility_ratio
        self._alphas["rsi_alpha"] = AlphaTemplate.rsi_alpha
        self._alphas["bollinger_alpha"] = AlphaTemplate.Bollinger_Band_alpha
        self._alphas["volume_spike"] = AlphaTemplate.volume_spike
        self._alphas["pvt"] = AlphaTemplate.price_volume_trend
        self._alphas["macd_cross"] = AlphaTemplate.MACD_cross
        self._alphas["gap_up"] = AlphaTemplate.gap_up
        self._alphas["high_low_range"] = AlphaTemplate.high_low_range
    
    def register_alpha(self, name: str, fn: Callable):
        """Register custom alpha."""
        self._alphas[name] = fn
        logger.info(f"Registered alpha: {name}")
    
    def compute_alpha(
        self,
        alpha_name: str,
        df: pd.DataFrame,
        **params,
    ) -> pd.Series:
        """Compute alpha signal."""
        alpha_fn = self._alphas.get(alpha_name)
        
        if alpha_fn is None:
            raise ValueError(f"Unknown alpha: {alpha_name}")
        
        try:
            result = alpha_fn(df, **params)
            return result
        except Exception as e:
            logger.error(f"Error computing alpha {alpha_name}: {e}")
            return pd.Series(0, index=df.index)
    
    def compute_all_alphas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all registered alphas."""
        result = pd.DataFrame(index=df.index)
        
        for alpha_name in self._alphas.keys():
            try:
                result[alpha_name] = self.compute_alpha(alpha_name, df)
            except Exception as e:
                logger.warning(f"Alpha {alpha_name} failed: {e}")
                result[alpha_name] = 0
        
        return result
    
    def backtest_alpha(
        self,
        alpha_signals: pd.Series,
        prices: pd.Series,
        costs: float = 0.001,
    ) -> AlphaPerformance:
        """Backtest alpha signal."""
        # Generate signals (normalize to -1, 0, 1)
        signals = alpha_signals.copy()
        signals[signals > 0] = 1
        signals[signals < 0] = -1
        signals[signals == 0] = 0
        
        # Calculate returns
        returns = prices.pct_change()
        strategy_returns = signals.shift(1) * returns - np.abs(signals.diff()) * costs
        
        # Calculate metrics
        total_return = (1 + strategy_returns).prod() - 1
        
        # Sharpe ratio
        if strategy_returns.std() > 0:
            sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
        else:
            sharpe = 0
        
        # Max drawdown
        cumulative = (1 + strategy_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        
        # Win rate
        trades = signals.diff() != 0
        winning_trades = (strategy_returns > 0).sum()
        total_trades = trades.sum()
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Profit factor
        gross_profit = strategy_returns[strategy_returns > 0].sum()
        gross_loss = abs(strategy_returns[strategy_returns < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Holding period (average)
        position_changes = signals.diff() != 0
        holding_period = 1.0  # Simplified
        
        # Turnover
        turnover = signals.diff().abs().mean()
        
        return AlphaPerformance(
            alpha_id=hashlib.md5(str(alpha_signals.name).encode()).hexdigest()[:8],
            total_return=total_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            profit_factor=profit_factor,
            trade_count=int(total_trades),
            holding_period=holding_period,
            turnover=turnover,
        )
    
    def rank_alphas(self, df: pd.DataFrame, prices: pd.Series) -> List[AlphaPerformance]:
        """Rank all alphas by performance."""
        results = []
        
        for alpha_name in self._alphas.keys():
            try:
                signals = self.compute_alpha(alpha_name, df)
                perf = self.backtest_alpha(signals, prices)
                perf.alpha_id = alpha_name
                results.append(perf)
            except Exception as e:
                logger.warning(f"Alpha {alpha_name} backtest failed: {e}")
        
        # Sort by Sharpe ratio
        results.sort(key=lambda x: x.sharpe_ratio, reverse=True)
        
        return results
    
    def find_optimal_parameters(
        self,
        alpha_name: str,
        df: pd.DataFrame,
        prices: pd.Series,
        param_grid: Dict[str, List[Any]],
    ) -> Dict[str, Any]:
        """Find optimal parameters for alpha."""
        best_sharpe = -np.inf
        best_params = {}
        results = []
        
        # Simple grid search
        param_combinations = self._generate_param_combinations(param_grid)
        
        for params in param_combinations:
            try:
                signals = self.compute_alpha(alpha_name, df, **params)
                perf = self.backtest_alpha(signals, prices)
                
                results.append({
                    "params": params,
                    "sharpe": perf.sharpe_ratio,
                    "return": perf.total_return,
                })
                
                if perf.sharpe_ratio > best_sharpe:
                    best_sharpe = perf.sharpe_ratio
                    best_params = params
                    
            except Exception as e:
                logger.warning(f"Parameter combination failed: {e}")
        
        return {
            "best_params": best_params,
            "best_sharpe": best_sharpe,
            "all_results": results,
        }
    
    def _generate_param_combinations(self, param_grid: Dict) -> List[Dict]:
        """Generate all parameter combinations."""
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        combinations = []
        for combination in self._cartesian_product(values):
            combinations.append(dict(zip(keys, combination)))
        
        return combinations
    
    def _cartesian_product(self, arrays: List) -> List[Tuple]:
        """Generate cartesian product of arrays."""
        if not arrays:
            return [()]
        
        result = [[]]
        for array in arrays:
            result = [x + [y] for x in result for y in array]
        
        return [tuple(x) for x in result]
    
    def create_ensemble(
        self,
        alpha_signals: List[pd.Series],
        weights: Optional[List[float]] = None,
    ) -> pd.Series:
        """Create ensemble alpha from multiple signals."""
        if weights is None:
            weights = [1.0 / len(alpha_signals)] * len(alpha_signals)
        
        # Normalize weights
        total = sum(weights)
        weights = [w / total for w in weights]
        
        # Combine signals
        ensemble = sum(s * w for s, w in zip(alpha_signals, weights))
        
        return ensemble
    
    def analyze_alpha_correlation(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Analyze correlation between alphas."""
        signals = self.compute_all_alphas(df)
        return signals.corr()


class AlphaLab:
    """
    Alpha Lab
    =========
    Complete research environment for alpha discovery.
    """
    
    def __init__(self):
        """Initialize Alpha Lab."""
        self.researcher = AlphaResearcher()
        self._experiments: Dict[str, Dict] = {}
        
        logger.info("Alpha Lab initialized")
    
    def run_experiment(
        self,
        name: str,
        df: pd.DataFrame,
        prices: pd.Series,
        alpha_names: Optional[List[str]] = None,
    ) -> Dict:
        """Run research experiment."""
        experiment_id = hashlib.md5(f"{name}_{datetime.now()}".encode()).hexdigest()[:8]
        
        results = {
            "experiment_id": experiment_id,
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "data_info": {
                "rows": len(df),
                "start": df.index[0].isoformat() if len(df) > 0 else None,
                "end": df.index[-1].isoformat() if len(df) > 0 else None,
            },
        }
        
        # Compute individual alphas
        if alpha_names is None:
            alpha_names = list(self.researcher._alphas.keys())
        
        alpha_results = []
        for alpha_name in alpha_names:
            try:
                signals = self.researcher.compute_alpha(alpha_name, df)
                perf = self.researcher.backtest_alpha(signals, prices)
                perf.alpha_id = alpha_name
                alpha_results.append(perf.to_dict())
            except Exception as e:
                logger.warning(f"Alpha {alpha_name} failed: {e}")
        
        results["alphas"] = alpha_results
        
        # Find top alphas
        alpha_results.sort(key=lambda x: x["sharpe_ratio"], reverse=True)
        results["top_alphas"] = alpha_results[:5]
        
        # Correlation analysis
        try:
            correlation = self.researcher.analyze_alpha_correlation(df)
            results["correlation_summary"] = {
                "mean_correlation": correlation.mean().mean(),
                "high_correlation_pairs": self._find_high_correlations(correlation),
            }
        except Exception as e:
            logger.warning(f"Correlation analysis failed: {e}")
        
        # Ensemble
        if len(alpha_results) >= 2:
            top_signals = [
                self.researcher.compute_alpha(a["alpha_id"], df)
                for a in alpha_results[:3]
            ]
            ensemble = self.researcher.create_ensemble(top_signals)
            ensemble_perf = self.researcher.backtest_alpha(ensemble, prices)
            results["ensemble"] = ensemble_perf.to_dict()
        
        # Store experiment
        self._experiments[experiment_id] = results
        
        logger.info(f"Experiment {experiment_id} completed with {len(alpha_results)} alphas")
        
        return results
    
    def _find_high_correlations(self, corr_matrix: pd.DataFrame, threshold: float = 0.7) -> List:
        """Find highly correlated alpha pairs."""
        high_corr = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    high_corr.append({
                        "alpha1": corr_matrix.columns[i],
                        "alpha2": corr_matrix.columns[j],
                        "correlation": corr_matrix.iloc[i, j],
                    })
        
        return high_corr
    
    def get_experiment(self, experiment_id: str) -> Optional[Dict]:
        """Get experiment by ID."""
        return self._experiments.get(experiment_id)
    
    def list_experiments(self) -> List[Dict]:
        """List all experiments."""
        return [
            {
                "experiment_id": exp_id,
                "name": exp["name"],
                "timestamp": exp["timestamp"],
                "num_alphas": len(exp.get("alphas", [])),
            }
            for exp_id, exp in self._experiments.items()
        ]


# Default alpha lab instance
default_alpha_lab = AlphaLab()

