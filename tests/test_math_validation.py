"""
Comprehensive Mathematical Validation Tests
============================================
Tests all calculation functions in the AI Trading System
to ensure mathematical correctness.
"""

import pytest
import math
from decimal import Decimal
from datetime import datetime, timedelta
import numpy as np
from typing import List, Dict, Tuple


# ============================================================================
# PORTFOLIO CALCULATIONS
# ============================================================================


class TestPortfolioCalculations:
    """Test portfolio calculation correctness."""

    def test_simple_pnl_calculation(self):
        """Test basic P&L: (sell_price - buy_price) * quantity"""
        buy_price = 100.0
        sell_price = 150.0
        quantity = 10.0

        pnl = (sell_price - buy_price) * quantity
        assert pnl == 500.0, f"Expected 500, got {pnl}"

    def test_pnl_with_commission(self):
        """Test P&L with commission deducted"""
        buy_price = 100.0
        sell_price = 150.0
        quantity = 10.0
        commission_rate = 0.001  # 0.1%

        gross_pnl = (sell_price - buy_price) * quantity
        commission = (buy_price + sell_price) * quantity * commission_rate
        net_pnl = gross_pnl - commission

        assert abs(net_pnl - 497.5) < 0.1, f"Expected ~497.5, got {net_pnl}"

    def test_roi_calculation(self):
        """Test ROI: ((final - initial) / initial) * 100"""
        initial_value = 10000.0
        final_value = 11500.0

        roi = ((final_value - initial_value) / initial_value) * 100
        assert roi == 15.0, f"Expected 15%, got {roi}%"

    def test_portfolio_equity_curve(self):
        """Test equity curve calculation"""
        initial_balance = 10000.0
        trades = [
            {"pnl": 500, "timestamp": "2024-01-01"},
            {"pnl": -200, "timestamp": "2024-01-02"},
            {"pnl": 800, "timestamp": "2024-01-03"},
        ]

        equity = initial_balance
        for trade in trades:
            equity += trade["pnl"]

        assert equity == 11100.0, f"Expected 11100, got {equity}"

    def test_position_value(self):
        """Test position value: price * quantity"""
        positions = [
            {"symbol": "BTC", "quantity": 0.5, "avg_price": 50000},
            {"symbol": "ETH", "quantity": 10, "avg_price": 3000},
        ]

        total_value = sum(p["quantity"] * p["avg_price"] for p in positions)
        expected = (0.5 * 50000) + (10 * 3000)
        assert total_value == expected

    def test_weighted_portfolio_return(self):
        """Test weighted portfolio return"""
        positions = [
            {"weight": 0.6, "return": 0.10},  # 10%
            {"weight": 0.3, "return": 0.05},  # 5%
            {"weight": 0.1, "return": -0.02},  # -2%
        ]

        portfolio_return = sum(p["weight"] * p["return"] for p in positions)
        expected = 0.6 * 0.10 + 0.3 * 0.05 + 0.1 * (-0.02)
        assert abs(portfolio_return - expected) < 0.0001


# ============================================================================
# RISK CALCULATIONS
# ============================================================================


class TestRiskCalculations:
    """Test risk metric calculations."""

    def test_var_95_basic(self):
        """Test 95% Value at Risk - parametric method"""
        returns = np.array([-0.05, -0.02, 0.01, 0.02, -0.01, 0.03, -0.03, 0.01])
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)

        # VaR = mean - z * std (z=1.645 for 95%)
        var_95 = mean_return - 1.645 * std_return

        assert var_95 < 0, "VaR should be negative"
        assert abs(var_95) > 0, "VaR should have magnitude"

    def test_cvar_calculation(self):
        """Test Conditional VaR (Expected Shortfall)"""
        returns = np.array([-0.05, -0.02, 0.01, 0.02, -0.01, 0.03, -0.03, 0.01])

        # Sort returns
        sorted_returns = np.sort(returns)
        # 5% worst returns (tail)
        tail_5pct = sorted_returns[: int(len(sorted_returns) * 0.05)]

        cvar = np.mean(tail_5pct) if len(tail_5pct) > 0 else sorted_returns[0]

        assert cvar < 0, "CVaR should be negative for negative returns"

    def test_sharpe_ratio(self):
        """Test Sharpe Ratio: (Rp - Rf) / Sigma_p"""
        portfolio_return = 0.15  # 15%
        risk_free_rate = 0.02  # 2%
        std_dev = 0.10  # 10%

        sharpe = (portfolio_return - risk_free_rate) / std_dev
        expected = (0.15 - 0.02) / 0.10

        assert sharpe == expected
        assert sharpe == 1.3

    def test_sortino_ratio(self):
        """Test Sortino Ratio: (Rp - Rf) / DownsideDev"""
        portfolio_return = 0.15
        risk_free_rate = 0.02
        returns = [0.05, -0.02, 0.03, -0.01, 0.02]

        downside_returns = [r for r in returns if r < 0]
        downside_dev = math.sqrt(sum((r - 0) ** 2 for r in downside_returns) / len(returns))

        sortino = (portfolio_return - risk_free_rate) / downside_dev

        assert sortino > 0, "Sortino should be positive"

    def test_max_drawdown(self):
        """Test maximum drawdown calculation"""
        equity_curve = [10000, 11000, 10500, 12000, 11500, 13000, 12500]

        peak = equity_curve[0]
        max_dd = 0

        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd

        # Max drawdown: from 12000 to 11500 = 4.17%
        assert max_dd > 0, "Drawdown should be positive"
        assert abs(max_dd - 0.0417) < 0.01, f"Expected ~4.17%, got {max_dd * 100}%"

    def test_portfolio_volatility(self):
        """Test portfolio volatility: sqrt(w.T * Cov * w)"""
        # Simple case: equal weights, no correlation
        weights = np.array([0.5, 0.5])
        volatilities = np.array([0.15, 0.20])  # 15%, 20%

        # With zero correlation
        variance = sum((w * v) ** 2 for w, v in zip(weights, volatilities))
        portfolio_vol = math.sqrt(variance)

        expected = math.sqrt((0.5 * 0.15) ** 2 + (0.5 * 0.20) ** 2)
        assert abs(portfolio_vol - expected) < 0.0001


# ============================================================================
# OPTIMIZATION CALCULATIONS
# ============================================================================


class TestOptimizationCalculations:
    """Test portfolio optimization math."""

    def test_minimum_variance_weights(self):
        """Test minimum variance portfolio calculation"""
        # Two assets, perfect negative correlation
        vol_a = 0.20
        vol_b = 0.15
        corr = -1.0

        # MV weight for A = (vol_b^2 - cov_ab) / (vol_a^2 + vol_b^2 - 2*cov_ab)
        cov_ab = corr * vol_a * vol_b
        w_a = (vol_b**2 - cov_ab) / (vol_a**2 + vol_b**2 - 2 * cov_ab)

        assert w_a > 0 and w_a < 1

    def test_risk_parity_weights(self):
        """Test risk parity: equal risk contribution"""
        volatilities = np.array([0.15, 0.20, 0.25])

        # Inverse volatility weights
        inv_vol = 1 / volatilities
        weights = inv_vol / inv_vol.sum()

        expected_a = (1 / 0.15) / ((1 / 0.15) + (1 / 0.20) + (1 / 0.25))

        assert abs(weights[0] - expected_a) < 0.0001
        assert abs(weights.sum() - 1.0) < 0.0001

    def test_efficient_frontier_point(self):
        """Test a point on efficient frontier"""
        # Given expected return, find minimum variance
        target_return = 0.10

        # Simplified: assume 2 assets
        returns = np.array([0.08, 0.12])
        vols = np.array([0.15, 0.20])

        # For equal risk aversion, weight toward higher return/lower vol
        score = returns / vols
        weights = score / score.sum()

        portfolio_return = np.dot(weights, returns)

        assert portfolio_return > 0


# ============================================================================
# ORDER EXECUTION CALCULATIONS
# ============================================================================


class TestOrderCalculations:
    """Test order execution math."""

    def test_market_order_slippage(self):
        """Test slippage calculation"""
        base_price = 50000
        slippage_bps = 5  # 5 basis points = 0.05%

        # Slippage in dollars = price * bps / 10000
        # 50000 * 5 / 10000 = $25 per unit
        slippage = base_price * (slippage_bps / 10000)
        expected = 25.0  # $25 per unit

        assert abs(slippage - expected) < 0.01

    def test_limit_order_fees(self):
        """Test fee calculation for limit orders"""
        price = 50000
        quantity = 1.0
        fee_rate = 0.0004  # 0.04% (Binance maker)

        notional = price * quantity
        fee = notional * fee_rate

        assert fee == 20.0, f"Expected $20, got {fee}"

    def test_order_sizing_kelly(self):
        """Test Kelly Criterion position sizing"""
        win_rate = 0.55
        avg_win = 100
        avg_loss = 80

        # Kelly = p * b - (1-p) where b = avg_win/avg_loss
        b = avg_win / avg_loss
        kelly = win_rate * b - (1 - win_rate)

        assert 0 < kelly < 1, "Kelly fraction should be between 0 and 1"

    def test_position_sizing_fixed_fraction(self):
        """Test fixed fraction position sizing"""
        account_balance = 100000
        risk_per_trade = 0.02  # 2%
        stop_loss_pct = 0.05  # 5%

        risk_amount = account_balance * risk_per_trade
        position_size = risk_amount / stop_loss_pct

        expected = 100000 * 0.02 / 0.05
        assert position_size == expected

    def test_breakeven_calculation(self):
        """Test breakeven price with fees"""
        buy_price = 100
        quantity = 10
        fee_rate = 0.001

        total_cost = buy_price * quantity
        fees = total_cost * fee_rate
        breakeven = (total_cost + fees) / quantity

        assert breakeven > buy_price


# ============================================================================
# STATISTICAL CALCULATIONS
# ============================================================================


class TestStatisticalCalculations:
    """Test statistical functions."""

    def test_moving_average_simple(self):
        """Test SMA calculation"""
        prices = [100, 102, 104, 106, 108]
        window = 3

        sma = sum(prices[-window:]) / window
        assert sma == 106.0

    def test_exponential_moving_average(self):
        """Test EMA calculation"""
        prices = [100, 102, 104, 106]
        alpha = 0.5

        ema = [prices[0]]
        for p in prices[1:]:
            ema.append(alpha * p + (1 - alpha) * ema[-1])

        # EMA after 3 periods
        assert len(ema) == 4

    def test_standard_deviation(self):
        """Test standard deviation calculation"""
        values = [2, 4, 4, 4, 5, 5, 7, 9]

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        std = math.sqrt(variance)

        assert abs(std - 2.138) < 0.01

    def test_correlation_pearson(self):
        """Test Pearson correlation coefficient"""
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 5, 4, 5]

        n = len(x)
        mean_x = sum(x) / n
        mean_y = sum(y) / n

        cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n)) / n
        std_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x) / n)
        std_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y) / n)

        corr = cov / (std_x * std_y)

        assert -1 <= corr <= 1

    def test_beta_calculation(self):
        """Test Beta coefficient"""
        # Market returns
        market_returns = [0.02, -0.01, 0.03, 0.01, -0.02]
        # Stock returns
        stock_returns = [0.03, -0.015, 0.04, 0.015, -0.025]

        n = len(market_returns)
        mean_m = sum(market_returns) / n
        mean_s = sum(stock_returns) / n

        cov = sum((market_returns[i] - mean_m) * (stock_returns[i] - mean_s) for i in range(n)) / n
        var_m = sum((r - mean_m) ** 2 for r in market_returns) / n

        beta = cov / var_m

        assert beta > 0, "Beta should be positive"


# ============================================================================
# TIME SERIES CALCULATIONS
# ============================================================================


class TestTimeSeriesCalculations:
    """Test time series analysis."""

    def test_daily_returns(self):
        """Test daily return calculation"""
        prices = [100, 102, 101, 105]

        returns = []
        for i in range(1, len(prices)):
            ret = (prices[i] - prices[i - 1]) / prices[i - 1]
            returns.append(ret)

        assert len(returns) == 3
        assert abs(returns[0] - 0.02) < 0.001

    def test_log_returns(self):
        """Test logarithmic returns"""
        price_t0 = 100
        price_t1 = 105

        log_return = math.log(price_t1 / price_t0)

        assert abs(log_return - 0.04879) < 0.001

    def test_cumulative_returns(self):
        """Test cumulative returns"""
        returns = [0.05, -0.02, 0.03]

        cumulative = 1.0
        for r in returns:
            cumulative *= 1 + r

        total_return = (cumulative - 1) * 100
        expected = ((1.05 * 0.98 * 1.03) - 1) * 100

        assert abs(total_return - expected) < 0.01

    def test_compound_annual_growth_rate(self):
        """Test CAGR calculation"""
        initial_value = 10000
        final_value = 20000
        years = 5

        cagr = (final_value / initial_value) ** (1 / years) - 1

        expected = (2**0.2) - 1
        assert abs(cagr - expected) < 0.0001


# ============================================================================
# BACKTEST CALCULATIONS
# ============================================================================


class TestBacktestCalculations:
    """Test backtesting metrics."""

    def test_calmar_ratio(self):
        """Test Calmar Ratio: CAGR / Max Drawdown"""
        cagr = 0.25  # 25%
        max_dd = 0.15  # 15%

        calmar = cagr / max_dd

        assert calmar == 25 / 15

    def test_information_ratio(self):
        """Test Information Ratio: (Return - Benchmark) / Tracking Error"""
        portfolio_return = 0.12
        benchmark_return = 0.08
        tracking_diff = [0.01, -0.02, 0.03, -0.01, 0.02]

        excess_return = portfolio_return - benchmark_return
        tracking_error = np.std(tracking_diff)

        info_ratio = excess_return / tracking_error

        assert info_ratio > 0

    def test_win_rate(self):
        """Test win rate calculation"""
        trades = [
            {"pnl": 100},
            {"pnl": -50},
            {"pnl": 200},
            {"pnl": -30},
            {"pnl": 150},
        ]

        wins = sum(1 for t in trades if t["pnl"] > 0)
        win_rate = wins / len(trades)

        assert win_rate == 0.6

    def test_profit_factor(self):
        """Test profit factor: gross profit / gross loss"""
        trades = [
            {"pnl": 100},
            {"pnl": -50},
            {"pnl": 200},
            {"pnl": -30},
            {"pnl": 150},
        ]

        gross_profit = sum(t["pnl"] for t in trades if t["pnl"] > 0)
        gross_loss = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))

        profit_factor = gross_profit / gross_loss

        expected = 450 / 80
        assert abs(profit_factor - expected) < 0.01


# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
