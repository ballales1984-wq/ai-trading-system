"""
app/portfolio/performance.py
Portfolio performance tracking and metrics calculation.

Provides:
  - Real-time PnL tracking (realized + unrealized)
  - Risk-adjusted metrics (Sharpe, Sortino, Calmar)
  - Drawdown analysis (max drawdown, recovery time)
  - Trade-level analytics (win rate, profit factor, avg trade)
  - Benchmark comparison (alpha, beta, information ratio)
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PerformanceMetrics:
    """Snapshot of portfolio performance metrics."""

    # Returns
    total_return: float = 0.0
    annualized_return: float = 0.0
    daily_returns_mean: float = 0.0
    daily_returns_std: float = 0.0

    # Risk-adjusted
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    information_ratio: float = 0.0

    # Drawdown
    max_drawdown: float = 0.0
    max_drawdown_duration_days: int = 0
    current_drawdown: float = 0.0

    # Trade stats
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_trade_pnl: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    avg_holding_period_hours: float = 0.0

    # Portfolio
    total_pnl: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_fees: float = 0.0
    net_pnl: float = 0.0

    # Benchmark
    alpha: float = 0.0
    beta: float = 0.0

    # Timestamps
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    last_updated: Optional[str] = None


@dataclass
class TradeRecord:
    """Record of a single trade."""

    trade_id: str = ""
    symbol: str = ""
    side: str = ""  # BUY / SELL
    entry_price: float = 0.0
    exit_price: float = 0.0
    quantity: float = 0.0
    pnl: float = 0.0
    fees: float = 0.0
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    strategy: str = ""


# ---------------------------------------------------------------------------
# Portfolio Performance Tracker
# ---------------------------------------------------------------------------

class PortfolioPerformance:
    """
    Tracks portfolio equity curve, trades, and computes performance metrics.

    Usage:
        perf = PortfolioPerformance(initial_capital=100_000)
        perf.record_equity(100_500)
        perf.record_trade(TradeRecord(...))
        metrics = perf.compute_metrics()
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        risk_free_rate: float = 0.04,
        benchmark_returns: Optional[List[float]] = None,
    ):
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        self.benchmark_returns = benchmark_returns or []

        # Equity curve
        self._equity_curve: List[Tuple[datetime, float]] = [
            (datetime.utcnow(), initial_capital)
        ]
        self._daily_returns: List[float] = []

        # Trades
        self._trades: List[TradeRecord] = []
        self._open_positions: Dict[str, Dict[str, Any]] = {}

        # Fees
        self._total_fees: float = 0.0

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_equity(self, equity: float, timestamp: Optional[datetime] = None):
        """Record a new equity value."""
        ts = timestamp or datetime.utcnow()
        self._equity_curve.append((ts, equity))

        if len(self._equity_curve) >= 2:
            prev = self._equity_curve[-2][1]
            if prev > 0:
                ret = (equity - prev) / prev
                self._daily_returns.append(ret)

    def record_trade(self, trade: TradeRecord):
        """Record a completed trade."""
        self._trades.append(trade)
        self._total_fees += trade.fees

    def record_open_position(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        quantity: float,
        entry_time: Optional[datetime] = None,
    ):
        """Track an open position for unrealized PnL."""
        self._open_positions[symbol] = {
            "side": side,
            "entry_price": entry_price,
            "quantity": quantity,
            "entry_time": entry_time or datetime.utcnow(),
        }

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        fees: float = 0.0,
        exit_time: Optional[datetime] = None,
    ) -> Optional[TradeRecord]:
        """Close an open position and record the trade."""
        pos = self._open_positions.pop(symbol, None)
        if pos is None:
            logger.warning(f"No open position for {symbol}")
            return None

        pnl = (exit_price - pos["entry_price"]) * pos["quantity"]
        if pos["side"] == "SELL":
            pnl = -pnl

        trade = TradeRecord(
            trade_id=f"{symbol}_{datetime.utcnow():%Y%m%d%H%M%S}",
            symbol=symbol,
            side=pos["side"],
            entry_price=pos["entry_price"],
            exit_price=exit_price,
            quantity=pos["quantity"],
            pnl=pnl,
            fees=fees,
            entry_time=pos["entry_time"],
            exit_time=exit_time or datetime.utcnow(),
        )
        self.record_trade(trade)
        return trade

    # ------------------------------------------------------------------
    # Metrics computation
    # ------------------------------------------------------------------

    def compute_metrics(
        self,
        current_prices: Optional[Dict[str, float]] = None,
    ) -> PerformanceMetrics:
        """Compute all performance metrics."""
        m = PerformanceMetrics()
        m.last_updated = datetime.utcnow().isoformat()

        if len(self._equity_curve) < 2:
            return m

        # Dates
        m.start_date = self._equity_curve[0][0].isoformat()
        m.end_date = self._equity_curve[-1][0].isoformat()

        # Returns
        equity_start = self._equity_curve[0][1]
        equity_end = self._equity_curve[-1][1]
        m.total_return = (equity_end - equity_start) / equity_start if equity_start else 0

        days = max(
            (self._equity_curve[-1][0] - self._equity_curve[0][0]).days, 1
        )
        m.annualized_return = (1 + m.total_return) ** (365 / days) - 1

        returns = np.array(self._daily_returns) if self._daily_returns else np.array([0.0])
        m.daily_returns_mean = float(np.mean(returns))
        m.daily_returns_std = float(np.std(returns))

        # Risk-adjusted ratios
        daily_rf = self.risk_free_rate / 252
        excess = returns - daily_rf

        if m.daily_returns_std > 0:
            m.sharpe_ratio = float(np.mean(excess) / np.std(excess) * np.sqrt(252))

        downside = returns[returns < 0]
        downside_std = float(np.std(downside)) if len(downside) > 0 else 1e-10
        m.sortino_ratio = float(np.mean(excess) / downside_std * np.sqrt(252))

        # Drawdown
        equities = np.array([e[1] for e in self._equity_curve])
        peak = np.maximum.accumulate(equities)
        drawdowns = (equities - peak) / peak
        m.max_drawdown = float(np.min(drawdowns))
        m.current_drawdown = float(drawdowns[-1])

        # Max drawdown duration
        in_dd = drawdowns < 0
        if np.any(in_dd):
            dd_starts = np.where(np.diff(in_dd.astype(int)) == 1)[0]
            dd_ends = np.where(np.diff(in_dd.astype(int)) == -1)[0]
            if len(dd_starts) > 0 and len(dd_ends) > 0:
                max_dur = 0
                for s in dd_starts:
                    ends_after = dd_ends[dd_ends > s]
                    if len(ends_after) > 0:
                        dur = int(ends_after[0] - s)
                        max_dur = max(max_dur, dur)
                m.max_drawdown_duration_days = max_dur

        # Calmar
        if abs(m.max_drawdown) > 0:
            m.calmar_ratio = m.annualized_return / abs(m.max_drawdown)

        # Trade stats
        m.total_trades = len(self._trades)
        if m.total_trades > 0:
            pnls = [t.pnl for t in self._trades]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p <= 0]

            m.winning_trades = len(wins)
            m.losing_trades = len(losses)
            m.win_rate = m.winning_trades / m.total_trades
            m.avg_trade_pnl = float(np.mean(pnls))
            m.best_trade = float(max(pnls))
            m.worst_trade = float(min(pnls))

            m.avg_win = float(np.mean(wins)) if wins else 0.0
            m.avg_loss = float(np.mean(losses)) if losses else 0.0

            gross_profit = sum(wins)
            gross_loss = abs(sum(losses))
            m.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

            # Average holding period
            holding_hours = []
            for t in self._trades:
                if t.entry_time and t.exit_time:
                    delta = (t.exit_time - t.entry_time).total_seconds() / 3600
                    holding_hours.append(delta)
            m.avg_holding_period_hours = float(np.mean(holding_hours)) if holding_hours else 0.0

        # PnL
        m.realized_pnl = sum(t.pnl for t in self._trades)
        m.total_fees = self._total_fees
        m.net_pnl = m.realized_pnl - m.total_fees

        # Unrealized PnL
        if current_prices:
            for sym, pos in self._open_positions.items():
                price = current_prices.get(sym, pos["entry_price"])
                upnl = (price - pos["entry_price"]) * pos["quantity"]
                if pos["side"] == "SELL":
                    upnl = -upnl
                m.unrealized_pnl += upnl

        m.total_pnl = m.net_pnl + m.unrealized_pnl

        # Benchmark comparison
        if self.benchmark_returns and len(self.benchmark_returns) >= 2:
            bench = np.array(self.benchmark_returns[: len(returns)])
            port = returns[: len(bench)]
            if len(bench) == len(port) and len(bench) > 1:
                cov = np.cov(port, bench)
                bench_var = np.var(bench)
                if bench_var > 0:
                    m.beta = float(cov[0, 1] / bench_var)
                    m.alpha = float(np.mean(port) - m.beta * np.mean(bench)) * 252

                tracking_error = float(np.std(port - bench))
                if tracking_error > 0:
                    m.information_ratio = float(
                        (np.mean(port) - np.mean(bench)) / tracking_error * np.sqrt(252)
                    )

        return m

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_equity_curve(self) -> List[Tuple[datetime, float]]:
        """Return the full equity curve."""
        return list(self._equity_curve)

    def get_trades(self) -> List[TradeRecord]:
        """Return all completed trades."""
        return list(self._trades)

    def get_open_positions(self) -> Dict[str, Dict[str, Any]]:
        """Return current open positions."""
        return dict(self._open_positions)

    def get_daily_returns(self) -> List[float]:
        """Return daily returns series."""
        return list(self._daily_returns)

    def to_dict(self, current_prices: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Export metrics as dictionary."""
        metrics = self.compute_metrics(current_prices)
        return {
            k: v
            for k, v in metrics.__dict__.items()
            if v is not None
        }


# Alias for backward compatibility
PerformanceTracker = PortfolioPerformance


# ============================================================================
# ADVANCED RISK METRICS (NEW - No external dependencies)
# ============================================================================

def calculate_value_at_risk(returns: np.ndarray, confidence: float = 0.95) -> float:
    """
    Calculate Value at Risk (VaR) using historical method.
    
    Args:
        returns: Array of returns
        confidence: Confidence level (default 0.95 for 95%)
    
    Returns:
        VaR as a positive number (loss)
    """
    if len(returns) < 2:
        return 0.0
    var = np.percentile(returns, (1 - confidence) * 100)
    return abs(var) if var < 0 else 0.0


def calculate_conditional_var(returns: np.ndarray, confidence: float = 0.95) -> float:
    """
    Calculate Conditional Value at Risk (CVaR) / Expected Shortfall.
    
    Args:
        returns: Array of returns
        confidence: Confidence level
    
    Returns:
        CVaR as a positive number (expected loss beyond VaR)
    """
    if len(returns) < 2:
        return 0.0
    var = np.percentile(returns, (1 - confidence) * 100)
    cvar = returns[returns <= var].mean()
    return abs(cvar) if cvar < 0 else 0.0


def calculate_tail_ratio(returns: np.ndarray) -> float:
    """
    Calculate Tail Ratio - Ratio of right tail to left tail.
    
    A value > 1 means more positive extreme returns than negative.
    """
    if len(returns) < 20:
        return 1.0
    
    right_tail = np.percentile(returns, 95)
    left_tail = np.percentile(returns, 5)
    
    if left_tail == 0:
        return 1.0
    
    return abs(right_tail / left_tail) if left_tail < 0 else 1.0


def calculate_skewness(returns: np.ndarray) -> float:
    """
    Calculate skewness of returns distribution.
    
    - Negative: Left-skewed (more negative outliers)
    - Positive: Right-skewed (more positive outliers)
    - Near 0: Symmetric
    """
    if len(returns) < 3:
        return 0.0
    
    mean = np.mean(returns)
    std = np.std(returns)
    
    if std == 0:
        return 0.0
    
    n = len(returns)
    skew = (n / ((n - 1) * (n - 2))) * np.sum(((returns - mean) / std) ** 3)
    
    return float(skew)


def calculate_kurtosis(returns: np.ndarray) -> float:
    """
    Calculate excess kurtosis of returns distribution.
    
    - Positive: Fat tails (more outliers than normal)
    - Negative: Thin tails (fewer outliers)
    """
    if len(returns) < 4:
        return 0.0
    
    mean = np.mean(returns)
    std = np.std(returns)
    
    if std == 0:
        return 0.0
    
    n = len(returns)
    kurt = (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))) * \
           np.sum(((returns - mean) / std) ** 4) - \
           (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))
    
    return float(kurt)


def calculate_kelly_criterion(returns: np.ndarray) -> float:
    """
    Calculate Kelly Criterion for optimal bet sizing.
    
    Returns the optimal fraction of capital to risk.
    
    Returns:
        Kelly percentage (0 to 1, where 1 = 100%)
    """
    if len(returns) < 10:
        return 0.0
    
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    
    if len(wins) == 0 or len(losses) == 0:
        return 0.0
    
    win_rate = len(wins) / len(returns)
    avg_win = np.mean(wins)
    avg_loss = abs(np.mean(losses))
    
    if avg_loss == 0:
        return 0.0
    
    win_loss_ratio = avg_win / avg_loss
    kelly = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
    
    # Cap Kelly at reasonable levels (half-Kelly is safer)
    return max(0.0, min(kelly, 1.0))


def calculate_omega_ratio(returns: np.ndarray, threshold: float = 0.0) -> float:
    """
    Calculate Omega Ratio - Probability-weighted ratio of gains vs losses.
    
    Args:
        returns: Array of returns
        threshold: Minimum acceptable return (default 0)
    
    Returns:
        Omega ratio (higher is better)
    """
    if len(returns) < 2:
        return 1.0
    
    gains = returns[returns > threshold] - threshold
    losses = threshold - returns[returns < threshold]
    
    total_gains = np.sum(gains)
    total_losses = np.sum(losses)
    
    if total_losses == 0:
        return float('inf') if total_gains > 0 else 1.0
    
    return total_gains / total_losses


def calculate_ulcer_index(equity_curve: np.ndarray, period: int = 14) -> float:
    """
    Calculate Ulcer Index - Measures downside volatility in terms of depth and duration.
    """
    if len(equity_curve) < 2:
        return 0.0
    
    # Calculate drawdown percentage
    rolling_max = np.maximum.accumulate(equity_curve)
    drawdown = ((equity_curve - rolling_max) / rolling_max) * 100
    
    # Ulcer Index = Square root of mean squared drawdown
    ulcer = np.sqrt(np.mean(drawdown ** 2))
    
    return float(ulcer)


def calculate_gain_to_pain_ratio(returns: np.ndarray) -> float:
    """
    Calculate Gain to Pain Ratio - Sum of returns / sum of absolute losses.
    """
    if len(returns) < 2:
        return 0.0
    
    total_return = np.sum(returns)
    pain = np.sum(np.abs(returns[returns < 0]))
    
    if pain == 0:
        return float('inf') if total_return > 0 else 0.0
    
    return total_return / pain


def calculate_rolling_sharpe(returns: np.ndarray, window: int = 30) -> np.ndarray:
    """
    Calculate rolling Sharpe ratio over a window.
    
    Returns:
        Array of rolling Sharpe ratios
    """
    if len(returns) < window:
        return np.array([0.0])
    
    rolling_mean = np.convolve(returns, np.ones(window)/window, mode='valid')
    rolling_std = np.array([np.std(returns[i:i+window]) for i in range(len(returns)-window+1)])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rolling_sharpe = rolling_mean / rolling_std * np.sqrt(252)
        rolling_sharpe = np.nan_to_num(rolling_sharpe, nan=0.0, posinf=0.0, neginf=0.0)
    
    return rolling_sharpe


class AdvancedRiskMetrics:
    """
    Advanced risk metrics calculator.
    
    Usage:
        risk = AdvancedRiskMetrics()
        metrics = risk.calculate_all(returns, equity_curve)
    """
    
    @staticmethod
    def calculate_all(
        returns: np.ndarray,
        equity_curve: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate all advanced risk metrics.
        
        Args:
            returns: Array of returns
            equity_curve: Optional array of equity values
        
        Returns:
            Dictionary of risk metrics
        """
        metrics = {}
        
        # VaR and CVaR
        metrics['var_95'] = calculate_value_at_risk(returns, 0.95)
        metrics['var_99'] = calculate_value_at_risk(returns, 0.99)
        metrics['cvar_95'] = calculate_conditional_var(returns, 0.95)
        metrics['cvar_99'] = calculate_conditional_var(returns, 0.99)
        
        # Distribution metrics
        metrics['tail_ratio'] = calculate_tail_ratio(returns)
        metrics['skewness'] = calculate_skewness(returns)
        metrics['kurtosis'] = calculate_kurtosis(returns)
        
        # Risk-adjusted
        metrics['kelly_criterion'] = calculate_kelly_criterion(returns)
        metrics['kelly_fraction'] = calculate_kelly_criterion(returns) * 0.5  # Half-Kelly
        metrics['omega_ratio'] = calculate_omega_ratio(returns)
        
        if equity_curve is not None:
            metrics['ulcer_index'] = calculate_ulcer_index(equity_curve)
        
        metrics['gain_to_pain'] = calculate_gain_to_pain_ratio(returns)
        
        return metrics
