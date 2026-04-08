"""
Autonomous Agent API Routes
==========================
API endpoints for the Level 5 Autonomous Quant Agent.

Routes:
    GET  /agents/autonomous/report/{symbol} - Get daily report
    GET  /agents/autonomous/proposals/{symbol} - Get action proposals
    GET  /agents/autonomous/portfolio - Get portfolio status
    POST /agents/autonomous/execute - Execute approved action
"""

from typing import Optional, List
from datetime import datetime
from fastapi import APIRouter, HTTPException, Path
from pydantic import BaseModel, Field
import logging

from src.agents.autonomous_quant_agent import (
    AutonomousQuantAgent,
    AgentConfig,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agents/autonomous", tags=["Autonomous Agent"])

# Global agent instance
_agent: Optional[AutonomousQuantAgent] = None


def get_agent() -> AutonomousQuantAgent:
    """Get or create the global agent instance."""
    global _agent
    if _agent is None:
        config = AgentConfig(
            default_symbols=["BTCUSDT", "ETHUSDT"],
            max_position_pct=0.10,
            max_drawdown_pct=0.05,
        )
        _agent = AutonomousQuantAgent(config)
    return _agent


# Request/Response Models
class ExecuteRequest(BaseModel):
    """Request to execute an action."""

    symbol: str = Field(..., min_length=1, max_length=20, description="Trading symbol")
    action: str = Field(..., pattern="^(buy|sell|hold)$", description="Action: buy, sell, hold")
    size: float = Field(..., ge=0, le=1e9, description="Position size")
    reason: str = Field(default="", max_length=1000, description="Reason for action")


class AgentReportResponse(BaseModel):
    """Response containing agent report."""

    timestamp: str
    trading_mode: str
    regime: dict
    monte_carlo: dict
    portfolio: dict
    risk: dict
    models: dict


class ProposalsResponse(BaseModel):
    """Response containing action proposals."""

    proposals: List[dict]
    timestamp: str


class PortfolioResponse(BaseModel):
    """Response containing portfolio status."""

    positions: List[dict]
    equity: float
    pnl: float
    pnl_pct: float
    position_count: int
    timestamp: str


class ExecuteResponse(BaseModel):
    """Response from action execution."""

    success: bool
    message: str
    order_id: Optional[str] = None


# Routes
@router.get("/report/{symbol}", response_model=AgentReportResponse)
async def get_agent_report(
    symbol: str = Path(..., description="Trading symbol (e.g., BTCUSDT)"),
) -> AgentReportResponse:
    """
    Get the daily report for a symbol.

    Returns comprehensive analysis including:
    - Market regime (HMM)
    - Monte Carlo simulation
    - Portfolio status
    - Risk metrics
    - Model information
    """
    try:
        agent = get_agent()
        report = agent.daily_report(symbol)

        return AgentReportResponse(
            timestamp=report.get("timestamp", datetime.now().isoformat()),
            trading_mode=report.get("trading_mode", "active"),
            regime=report.get("regime", {}),
            monte_carlo=report.get("monte_carlo", {}),
            portfolio=report.get("portfolio", {}),
            risk=report.get("risk", {}),
            models=report.get("models", {}),
        )
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/proposals/{symbol}", response_model=ProposalsResponse)
async def get_action_proposals(
    symbol: str = Path(..., description="Trading symbol"),
) -> ProposalsResponse:
    """
    Get action proposals for a symbol.

    Returns a list of recommended actions based on:
    - Market regime analysis
    - Volatility forecasts
    - Monte Carlo simulations
    - Risk constraints
    """
    try:
        agent = get_agent()
        proposals = agent.propose_actions(symbol)

        return ProposalsResponse(
            proposals=proposals,
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        logger.error(f"Error generating proposals: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/portfolio", response_model=PortfolioResponse)
async def get_portfolio_status() -> PortfolioResponse:
    """
    Get current portfolio status.

    Returns:
    - Current positions
    - Equity and P&L
    - Risk metrics
    """
    try:
        agent = get_agent()
        status = agent.get_portfolio_status()

        return PortfolioResponse(
            positions=status.get("positions", []),
            equity=status.get("equity", 0),
            pnl=status.get("pnl", 0),
            pnl_pct=status.get("pnl_pct", 0),
            position_count=status.get("position_count", 0),
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        logger.error(f"Error getting portfolio status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/execute", response_model=ExecuteResponse)
async def execute_action(request: ExecuteRequest) -> ExecuteResponse:
    """
    Execute an approved action.

    Note: This is a mock execution for demonstration.
    In production, this would connect to the execution engine.
    """
    try:
        agent = get_agent()

        # Validate action
        if request.action not in ["buy", "sell", "hold"]:
            raise HTTPException(status_code=400, detail="Invalid action")

        # Mock execution - in production, connect to execution engine
        order_id = f"ORD-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        logger.info(f"Executing: {request.action} {request.size} {request.symbol}")

        # Update position in risk book
        if request.action in ["buy", "sell"]:
            agent.update_position(
                symbol=request.symbol,
                side=request.action,
                quantity=request.size,
                avg_price=0,  # Would get from market data
            )

        return ExecuteResponse(
            success=True,
            message=f"{request.action.upper()} order placed for {request.symbol}",
            order_id=order_id,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing action: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset")
async def reset_agent() -> dict:
    """Reset the agent to initial state."""
    global _agent
    _agent = None
    return {"message": "Agent reset successfully"}


@router.get("/health")
async def agent_health() -> dict:
    """Health check for the autonomous agent."""
    try:
        agent = get_agent()
        return {
            "status": "healthy",
            "trading_mode": agent.trading_mode,
            "last_update": agent.last_update.isoformat() if agent.last_update else None,
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
        }
