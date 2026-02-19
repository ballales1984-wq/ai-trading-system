"""
Strategy Routes
===============
REST API for trading strategy management.
"""

from datetime import datetime
from typing import List, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, status, Query
from pydantic import BaseModel, Field


router = APIRouter()


# ============================================================================
# DATA MODELS
# ============================================================================

class Signal(BaseModel):
    """Trading signal model."""
    signal_id: str = Field(default_factory=lambda: str(uuid4()))
    strategy_id: str
    symbol: str
    direction: str = Field(..., description="LONG, SHORT, or CLOSE")
    confidence: float = Field(..., ge=0.0, le=1.0)
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict = Field(default_factory=dict)


class Strategy(BaseModel):
    """Trading strategy model."""
    strategy_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str
    strategy_type: str = Field(..., description="momentum, mean_reversion, ml, etc.")
    asset_classes: List[str] = Field(default=["crypto"])
    parameters: dict = Field(default_factory=dict)
    enabled: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    performance: dict = Field(default_factory=dict)


class StrategyPerformance(BaseModel):
    """Strategy performance metrics."""
    strategy_id: str
    strategy_name: str
    total_return: float
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    num_signals: int
    num_trades: int
    avg_trade_pnl: float


# ============================================================================
# IN-MEMORY STRATEGY STORE
# ============================================================================

strategies_db = {
    "strat_001": {
        "strategy_id": "strat_001",
        "name": "Momentum BTC",
        "description": "Trend-following momentum strategy for Bitcoin",
        "strategy_type": "momentum",
        "asset_classes": ["crypto"],
        "parameters": {
            "lookback_period": 20,
            "threshold": 0.02,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.05,
        },
        "enabled": True,
        "created_at": "2026-01-01T00:00:00",
        "updated_at": "2026-02-18T00:00:00",
    },
    "strat_002": {
        "strategy_id": "strat_002",
        "name": "Mean Reversion ETH",
        "description": "Mean reversion strategy for Ethereum",
        "strategy_type": "mean_reversion",
        "asset_classes": ["crypto"],
        "parameters": {
            "lookback_period": 50,
            "std_dev_threshold": 2.0,
            "stop_loss_pct": 0.03,
        },
        "enabled": True,
        "created_at": "2026-01-15T00:00:00",
        "updated_at": "2026-02-18T00:00:00",
    },
    "strat_003": {
        "strategy_id": "strat_003",
        "name": "ML Signal",
        "description": "Machine learning based signal generation",
        "strategy_type": "ml",
        "asset_classes": ["crypto", "forex"],
        "parameters": {
            "model_type": "xgboost",
            "features": ["rsi", "macd", "bbands", "volume"],
            "prediction_horizon": 24,
        },
        "enabled": False,
        "created_at": "2026-02-01T00:00:00",
        "updated_at": "2026-02-10T00:00:00",
    },
}


# ============================================================================
# ROUTES
# ============================================================================

@router.get("/", response_model=List[Strategy])
async def list_strategies(
    strategy_type: Optional[str] = Query(None, description="Filter by type"),
    enabled_only: bool = Query(False, description="Only enabled strategies"),
) -> List[Strategy]:
    """List all trading strategies."""
    strategies = list(strategies_db.values())
    
    if strategy_type:
        strategies = [s for s in strategies if s["strategy_type"] == strategy_type]
    if enabled_only:
        strategies = [s for s in strategies if s["enabled"]]
    
    return [Strategy(**s) for s in strategies]


@router.get("/{strategy_id}", response_model=Strategy)
async def get_strategy(strategy_id: str) -> Strategy:
    """Get strategy by ID."""
    if strategy_id not in strategies_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy {strategy_id} not found"
        )
    return Strategy(**strategies_db[strategy_id])


@router.post("/", response_model=Strategy, status_code=status.HTTP_201_CREATED)
async def create_strategy(strategy: Strategy) -> Strategy:
    """Create a new trading strategy."""
    strategy_id = str(uuid4())
    strategy.strategy_id = strategy_id
    strategy.created_at = datetime.utcnow()
    strategy.updated_at = datetime.utcnow()
    strategies_db[strategy_id] = strategy.model_dump()
    return strategy


@router.patch("/{strategy_id}", response_model=Strategy)
async def update_strategy(
    strategy_id: str,
    enabled: Optional[bool] = None,
    parameters: Optional[dict] = None,
) -> Strategy:
    """Update strategy settings."""
    if strategy_id not in strategies_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy {strategy_id} not found"
        )
    
    strategy = strategies_db[strategy_id]
    if enabled is not None:
        strategy["enabled"] = enabled
    if parameters is not None:
        strategy["parameters"].update(parameters)
    strategy["updated_at"] = datetime.utcnow().isoformat()
    return Strategy(**strategy)


@router.delete("/{strategy_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_strategy(strategy_id: str) -> None:
    """Delete a strategy."""
    if strategy_id not in strategies_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy {strategy_id} not found"
        )
    del strategies_db[strategy_id]


@router.get("/{strategy_id}/signals", response_model=List[Signal])
async def get_strategy_signals(
    strategy_id: str,
    limit: int = Query(50, ge=1, le=500),
) -> List[Signal]:
    """Get recent signals from a strategy."""
    if strategy_id not in strategies_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy {strategy_id} not found"
        )
    
    signals = []
    for i in range(min(limit, 10)):
        signals.append(Signal(
            strategy_id=strategy_id,
            symbol="BTCUSDT" if i % 2 == 0 else "ETHUSDT",
            direction="LONG" if i % 3 != 0 else "SHORT",
            confidence=0.6 + (i * 0.03),
            timestamp=datetime.utcnow(),
        ))
    return signals


@router.get("/{strategy_id}/performance", response_model=StrategyPerformance)
async def get_strategy_performance(strategy_id: str) -> StrategyPerformance:
    """Get strategy performance metrics."""
    if strategy_id not in strategies_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy {strategy_id} not found"
        )
    
    strategy = strategies_db[strategy_id]
    return StrategyPerformance(
        strategy_id=strategy_id,
        strategy_name=strategy["name"],
        total_return=12500.0,
        total_return_pct=12.5,
        sharpe_ratio=1.95,
        max_drawdown=-3200.0,
        win_rate=0.65,
        num_signals=156,
        num_trades=89,
        avg_trade_pnl=140.45,
    )


@router.post("/{strategy_id}/run", status_code=status.HTTP_200_OK)
async def run_strategy(strategy_id: str) -> dict:
    """Trigger strategy execution."""
    if strategy_id not in strategies_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy {strategy_id} not found"
        )
    
    strategy = strategies_db[strategy_id]
    if not strategy["enabled"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Strategy is disabled"
        )
    
    return {
        "status": "running",
        "strategy_id": strategy_id,
        "message": f"Strategy {strategy['name']} started",
    }
