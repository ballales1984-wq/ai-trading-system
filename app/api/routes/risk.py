"""
Risk Management Routes
======================
REST API for institutional risk management.
"""

from typing import List, Dict, Any
from uuid import uuid4
import numpy as np

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field


router = APIRouter()


# ============================================================================
# DATA MODELS
# ============================================================================


class RiskMetrics(BaseModel):
    """Portfolio risk metrics."""

    var_1d: float = Field(description="Value at Risk (1-day, 95%)")
    var_5d: float = Field(description="Value at Risk (5-day, 95%)")
    cvar_1d: float = Field(description="Conditional VaR (1-day, 95%)")
    cvar_5d: float = Field(description="Conditional VaR (5-day, 95%)")
    volatility: float = Field(description="Portfolio volatility (annualized)")
    beta: float = Field(description="Portfolio beta")
    correlation_to_btc: float = Field(description="Correlation to BTC")
    max_drawdown: float = Field(description="Current max drawdown")
    sharpe_ratio: float = Field(description="Sharpe ratio")
    leverage: float = Field(description="Current leverage")
    margin_utilization: float = Field(description="Margin utilization %")


class PositionRisk(BaseModel):
    """Risk metrics for a single position."""

    symbol: str
    position_size: float
    market_value: float
    var_contribution: float = Field(description="VaR contribution")
    beta_weighted_exposure: float
    correlation_to_portfolio: float
    concentration_risk: float


class RiskLimit(BaseModel):
    """Risk limit configuration."""

    limit_id: str
    limit_type: str
    limit_value: float
    current_value: float
    limit_percentage: float
    is_breached: bool
    severity: str


class OrderRiskCheckRequest(BaseModel):
    """Request model for order risk check."""

    symbol: str = Field(..., min_length=1, max_length=20, description="Trading symbol")
    side: str = Field(..., pattern="^(BUY|SELL)$", description="Order side")
    quantity: float = Field(..., gt=0, le=1e9, description="Order quantity")
    price: float = Field(..., gt=0, le=1e9, description="Order price")


class OrderRiskCheck(BaseModel):
    """Risk check result for an order."""

    order_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    estimated_impact: float
    risk_score: float
    approved: bool
    rejection_reasons: List[str] = []
    warnings: List[str] = []


# ============================================================================
# ROUTES
# ============================================================================


@router.get("/metrics", response_model=RiskMetrics)
async def get_risk_metrics() -> RiskMetrics:
    """
    Get current portfolio risk metrics.

    Returns VaR, CVaR, volatility, and other risk measures.
    """
    # Simulated risk metrics - in production, calculate from actual data
    return RiskMetrics(
        var_1d=12500.0,
        var_5d=28000.0,
        cvar_1d=18750.0,
        cvar_5d=42000.0,
        volatility=0.25,
        beta=1.15,
        correlation_to_btc=0.85,
        max_drawdown=-8.5,
        sharpe_ratio=1.85,
        leverage=1.2,
        margin_utilization=0.45,
    )


@router.get("/limits", response_model=List[RiskLimit])
async def get_risk_limits() -> List[RiskLimit]:
    """
    Get current risk limits and their status.
    """
    return [
        RiskLimit(
            limit_id="var_limit",
            limit_type="var",
            limit_value=20000.0,
            current_value=12500.0,
            limit_percentage=62.5,
            is_breached=False,
            severity="green",
        ),
        RiskLimit(
            limit_id="cvar_limit",
            limit_type="cvar",
            limit_value=30000.0,
            current_value=18750.0,
            limit_percentage=62.5,
            is_breached=False,
            severity="green",
        ),
        RiskLimit(
            limit_id="drawdown_limit",
            limit_type="drawdown",
            limit_value=100000.0,
            current_value=8500.0,
            limit_percentage=8.5,
            is_breached=False,
            severity="green",
        ),
        RiskLimit(
            limit_id="exposure_limit",
            limit_type="exposure",
            limit_value=0.3,
            current_value=0.25,
            limit_percentage=83.3,
            is_breached=False,
            severity="yellow",
        ),
    ]


@router.get("/positions", response_model=List[PositionRisk])
async def get_position_risks() -> List[PositionRisk]:
    """
    Get risk metrics for all positions.
    """
    return [
        PositionRisk(
            symbol="BTCUSDT",
            position_size=1.5,
            market_value=65250.0,
            var_contribution=8500.0,
            beta_weighted_exposure=1.2,
            correlation_to_portfolio=0.9,
            concentration_risk=0.65,
        ),
        PositionRisk(
            symbol="ETHUSDT",
            position_size=15.0,
            market_value=35250.0,
            var_contribution=4000.0,
            beta_weighted_exposure=1.1,
            correlation_to_portfolio=0.85,
            concentration_risk=0.35,
        ),
    ]


@router.post("/check_order", response_model=OrderRiskCheck)
async def check_order_risk(request: OrderRiskCheckRequest) -> OrderRiskCheck:
    """
    Check if an order passes risk limits before execution.

    This is the heart of the risk engine - every order must be validated.
    """
    order_id = str(uuid4())
    market_value = request.quantity * request.price

    # Calculate risk metrics
    risk_score = min(100, (market_value / 100000) * 50)
    estimated_impact = market_value * 0.001

    # Check limits
    rejection_reasons = []
    warnings = []
    approved = True

    # Check VaR limit
    if market_value > 50000:
        risk_score += 20
        warnings.append("Large order - may impact VaR")

    # Check concentration
    if market_value > 100000:
        approved = False
        rejection_reasons.append("Concentration limit exceeded")

    # Check leverage
    if risk_score > 80:
        approved = False
        rejection_reasons.append("Risk score exceeds threshold")

    return OrderRiskCheck(
        order_id=order_id,
        symbol=request.symbol,
        side=request.side,
        quantity=request.quantity,
        price=request.price,
        estimated_impact=estimated_impact,
        risk_score=risk_score,
        approved=approved,
        rejection_reasons=rejection_reasons,
        warnings=warnings,
    )


@router.get("/var/monte_carlo")
async def run_monte_carlo(
    simulations: int = Query(10000, ge=1000, le=100000),
    confidence: float = Query(0.95, ge=0.9, le=0.99),
) -> Dict[str, Any]:
    """
    Run Monte Carlo simulation for VaR calculation.

    Uses historical returns to simulate potential portfolio outcomes.
    """
    # Generate sample returns
    np.random.seed(42)
    daily_returns = np.random.normal(0.001, 0.02, simulations)

    # Calculate portfolio value changes
    portfolio_value = 1000000.0
    portfolio_changes = portfolio_value * daily_returns

    # Calculate VaR
    var = -np.percentile(portfolio_changes, (1 - confidence) * 100)
    cvar = -portfolio_changes[portfolio_changes <= -var].mean()

    return {
        "simulations": simulations,
        "confidence": confidence,
        "var_1d": var,
        "cvar_1d": cvar,
        "worst_case": float(np.min(portfolio_changes)),
        "best_case": float(np.max(portfolio_changes)),
        "mean_outcome": float(np.mean(portfolio_changes)),
        "percentile_5": float(np.percentile(portfolio_changes, 5)),
        "percentile_95": float(np.percentile(portfolio_changes, 95)),
    }


@router.get("/stress_test")
async def run_stress_test() -> Dict[str, Any]:
    """
    Run stress test scenarios.

    Tests portfolio under adverse market conditions.
    """
    scenarios = [
        {
            "name": "Crypto Crash",
            "btc_change": -50.0,
            "eth_change": -60.0,
            "volatility_multiplier": 2.5,
        },
        {
            "name": "Flash Crash",
            "btc_change": -20.0,
            "eth_change": -25.0,
            "volatility_multiplier": 3.0,
        },
        {
            "name": "Market Correction",
            "btc_change": -15.0,
            "eth_change": -18.0,
            "volatility_multiplier": 1.5,
        },
        {
            "name": "Liquidity Crisis",
            "btc_change": -30.0,
            "eth_change": -35.0,
            "volatility_multiplier": 4.0,
        },
    ]

    results = []
    portfolio_value = 1000000.0

    for scenario in scenarios:
        loss = portfolio_value * (scenario["btc_change"] / 100)
        results.append(
            {
                "scenario": scenario["name"],
                "projected_loss": loss,
                "projected_loss_pct": scenario["btc_change"],
                "remaining_value": portfolio_value - loss,
            }
        )

    return {"scenarios": results}


@router.get("/correlation")
async def get_correlation_matrix() -> Dict[str, Any]:
    """
    Get portfolio asset correlation matrix.
    """
    return {
        "assets": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "EURUSD"],
        "matrix": [
            [1.0, 0.85, 0.72, 0.15],
            [0.85, 1.0, 0.78, 0.12],
            [0.72, 0.78, 1.0, 0.08],
            [0.15, 0.12, 0.08, 1.0],
        ],
    }


@router.get("/rolling-sharpe")
async def get_rolling_sharpe(window: int = Query(30, ge=7, le=180)) -> List[Dict[str, Any]]:
    """
    Get rolling Sharpe ratio over time.
    """
    import random

    # Generate sample rolling sharpe data
    np.random.seed(42)
    dates = []
    base_date = "2024-01-01"

    for i in range(90):
        dates.append(f"2024-{((i // 30) + 1):02d}-{((i % 30) + 1):02d}")

    rolling_sharpe = []
    current = 1.5
    for _ in range(90):
        current += np.random.normal(0, 0.3)
        current = max(-2, min(4, current))
        rolling_sharpe.append(round(current, 2))

    return [{"date": date, "rolling_sharpe": sharpe} for date, sharpe in zip(dates, rolling_sharpe)]


@router.get("/drawdown")
async def get_drawdown() -> List[Dict[str, Any]]:
    """
    Get drawdown data over time with equity curve.
    """
    np.random.seed(42)

    dates = []
    base_value = 100000
    equity = base_value
    max_equity = base_value

    data = []

    for i in range(90):
        date = f"2024-{((i // 30) + 1):02d}-{((i % 30) + 1):02d}"
        dates.append(date)

        # Random walk
        change = np.random.normal(0.002, 0.03)
        equity = equity * (1 + change)

        # Track max equity
        if equity > max_equity:
            max_equity = equity

        # Calculate drawdown
        drawdown = ((equity - max_equity) / max_equity) * 100

        data.append({"date": date, "drawdown": round(drawdown, 2), "equity": round(equity, 2)})

    return data


@router.get("/monte-carlo")
async def get_monte_carlo_distribution(
    simulations: int = Query(1000, ge=100, le=10000),
) -> List[Dict[str, Any]]:
    """
    Get Monte Carlo simulation distribution percentiles.
    """
    np.random.seed(42)

    # Simulate final portfolio values
    daily_return = 0.001  # Mean daily return
    daily_vol = 0.02  # Daily volatility

    final_values = []
    for _ in range(simulations):
        # Simulate 30 days of trading
        portfolio_value = 100000
        for _ in range(30):
            portfolio_value *= 1 + np.random.normal(daily_return, daily_vol)
        final_values.append(portfolio_value)

    final_values.sort()

    percentiles = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]

    return [
        {"percentile": f"P{p}", "value": round(final_values[int(len(final_values) * p / 100)], 2)}
        for p in percentiles
    ]


@router.get("/risk-return")
async def get_risk_return_scatter() -> List[Dict[str, Any]]:
    """
    Get risk/return data for scatter plot (assets vs portfolio).
    """
    return [
        {"name": "BTC", "risk": 45.2, "return": 28.5},
        {"name": "ETH", "risk": 52.8, "return": 22.1},
        {"name": "SOL", "risk": 68.5, "return": 35.2},
        {"name": "SP500", "risk": 15.2, "return": 12.5},
        {"name": "Gold", "risk": 12.5, "return": 8.2},
        {"name": "BTC+ETH", "risk": 38.5, "return": 24.8},
        {"name": "Portfolio", "risk": 22.5, "return": 18.2},
        {"name": "Algo Strategy", "risk": 18.2, "return": 22.5},
    ]
