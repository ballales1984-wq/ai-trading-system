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
