"""
Autonomous Quant Agent (Level 5)
=================================
Self-contained quant agent that orchestrates all system components.

This is the highest level of abstraction, combining:
- OpenClaw skills (HMM, GARCH, Monte Carlo, Portfolio)
- Risk Book management
- Model Registry for ML model selection
- Decision making based on regime and risk

Usage:
    from src.agents.autonomous_quant_agent import AutonomousQuantAgent
    
    agent = AutonomousQuantAgent()
    
    # Get daily report
    report = agent.daily_report("BTCUSDT")
    
    # Get action proposals
    actions = agent.propose_actions("ETHUSDT")
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

from openclaw_skills.intent_router import route_intent
from src.risk.risk_book import RiskBook, RiskLimits
from src.research.model_registry import ModelRegistry, ModelMeta


@dataclass
class AgentConfig:
    """Configuration for the Autonomous Agent."""
    # Risk limits
    max_position_pct: float = 0.10
    max_drawdown_pct: float = 0.05
    
    # Trading parameters
    default_symbols: List[str] = None
    regime_confidence_threshold: float = 0.7
    
    # Monte Carlo defaults
    mc_simulations: int = 5000
    mc_days_ahead: int = 30
    
    def __post_init__(self):
        if self.default_symbols is None:
            self.default_symbols = ["BTCUSDT", "ETHUSDT"]


@dataclass
class RegimeSignal:
    """Regime detection signal."""
    symbol: str
    regime: str  # "bull", "bear", "sideways"
    confidence: float
    volatility: float
    timestamp: datetime


@dataclass
class RiskAssessment:
    """Risk assessment result."""
    var_95: float
    cvar_95: float
    drawdown_pct: float
    exposure_pct: float
    position_count: int
    within_limits: bool


class AutonomousQuantAgent:
    """
    Level 5 Autonomous Quant Agent.
    
    This agent orchestrates all system components to make
    autonomous trading decisions based on:
    - Market regime (HMM)
    - Volatility forecasting (GARCH)
    - Risk simulation (Monte Carlo)
    - Portfolio optimization (MPT)
    - Risk management (Risk Book)
    - Model selection (Model Registry)
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        """
        Initialize the Autonomous Quant Agent.
        
        Args:
            config: Agent configuration (uses defaults if not provided)
        """
        self.config = config or AgentConfig()
        
        # Initialize components
        self._init_risk_book()
        self._init_model_registry()
        
        # State
        self.last_update: Optional[datetime] = None
        self.trading_mode: str = "active"  # active, close_only, paused
    
    def _init_risk_book(self) -> None:
        """Initialize Risk Book with configured limits."""
        limits = RiskLimits(
            max_position_pct=self.config.max_position_pct,
            max_daily_drawdown_pct=self.config.max_drawdown_pct,
            var_95_limit=0.08,
            cvar_95_limit=0.10,
        )
        self.risk_book = RiskBook(limits)
        # Initialize with default equity
        self.risk_book.register_equity(100000.0)
    
    def _init_model_registry(self) -> None:
        """Initialize Model Registry."""
        self.model_registry = ModelRegistry()
    
    # =========================================================================
    # Market Analysis (OpenClaw Skills)
    # =========================================================================
    
    def analyze_regime(self, symbol: str) -> RegimeSignal:
        """
        Analyze market regime using HMM.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            RegimeSignal with regime and confidence
        """
        try:
            result = route_intent("regime_analysis", {"symbol": symbol})
            
            return RegimeSignal(
                symbol=symbol,
                regime=result.get("current_state", "sideways"),
                confidence=result.get("confidence", 0.0),
                volatility=result.get("volatility", 0.03),
                timestamp=datetime.utcnow(),
            )
        except Exception as e:
            # Return neutral signal on error
            return RegimeSignal(
                symbol=symbol,
                regime="sideways",
                confidence=0.0,
                volatility=0.03,
                timestamp=datetime.utcnow(),
            )
    
    def analyze_volatility(self, symbol: str) -> Dict[str, Any]:
        """
        Analyze volatility using GARCH.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Volatility analysis results
        """
        try:
            return route_intent("volatility_analysis", {
                "symbol": symbol,
                "forecast_horizon": 5,
            })
        except Exception as e:
            return {"error": str(e)}
    
    def run_monte_carlo(
        self,
        price: float,
        volatility: Optional[float] = None,
        expected_return: float = 0.001,
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation.
        
        Args:
            price: Current price
            volatility: Optional volatility (uses default if not provided)
            expected_return: Expected daily return
            
        Returns:
            Monte Carlo simulation results
        """
        vol = volatility or 0.03
        
        return route_intent("simulate_paths", {
            "initial_price": price,
            "expected_return": expected_return,
            "volatility": vol,
            "n_paths": self.config.mc_simulations,
            "days_ahead": self.config.mc_days_ahead,
        })
    
    def optimize_portfolio(
        self,
        assets: List[str],
        objective: str = "max_sharpe",
    ) -> Dict[str, Any]:
        """
        Optimize portfolio using MPT.
        
        Args:
            assets: List of asset symbols
            objective: Optimization objective
            
        Returns:
            Portfolio optimization results
        """
        return route_intent("portfolio_optimization", {
            "assets": assets,
            "objective": objective,
        })
    
    # =========================================================================
    # Risk Management
    # =========================================================================
    
    def assess_risk(self) -> RiskAssessment:
        """
        Assess current portfolio risk.
        
        Returns:
            RiskAssessment with current risk metrics
        """
        # Get current positions
        positions = self.risk_book.positions
        
        # Check drawdown
        drawdown_pct = self.risk_book.daily_drawdown_pct()
        
        # Estimate exposure (simplified)
        exposure_pct = 0.0
        if positions:
            # This would use real prices in production
            mock_prices = {"BTCUSDT": 50000, "ETHUSDT": 3000}
            exposure = sum(
                abs(pos.quantity * mock_prices.get(pos.symbol, pos.avg_price))
                for pos in positions.values()
            )
            exposure_pct = exposure / self.risk_book.equity if self.risk_book.equity > 0 else 0
        
        # Check limits
        within_limits = (
            drawdown_pct <= self.config.max_drawdown_pct and
            exposure_pct <= self.config.max_position_pct
        )
        
        return RiskAssessment(
            var_95=0.05,  # Would be calculated from MC
            cvar_95=0.08,  # Would be calculated from MC
            drawdown_pct=drawdown_pct,
            exposure_pct=exposure_pct,
            position_count=len(positions),
            within_limits=within_limits,
        )
    
    def update_position(
        self,
        symbol: str,
        quantity: float,
        price: float,
        side: str,
    ) -> bool:
        """
        Update position in Risk Book.
        
        Args:
            symbol: Trading symbol
            quantity: Position size
            price: Entry price
            side: "long" or "short"
            
        Returns:
            True if update successful
        """
        from src.risk.risk_book import Position
        
        # Create position
        pos = Position(
            symbol=symbol,
            quantity=quantity,
            avg_price=price,
            side=side,
        )
        
        # Check limits before update
        prices = {symbol: price}
        equity = self.risk_book.equity
        
        if not self.risk_book.check_position_limit(symbol, prices, equity):
            return False
        
        self.risk_book.update_position(pos)
        return True
    
    # =========================================================================
    # Decision Making
    # =========================================================================
    
    def propose_actions(self, symbol: str) -> Dict[str, Any]:
        """
        Propose trading actions based on regime analysis.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with proposed actions
        """
        # Get regime
        regime = self.analyze_regime(symbol)
        
        # Get risk status
        risk = self.assess_risk()
        
        # Determine action based on regime
        if not risk.within_limits:
            action = "hold"
            reason = "Risk limits breached"
            confidence = 1.0
        elif regime.regime == "bull" and regime.confidence > self.config.regime_confidence_threshold:
            action = "consider_buy"
            reason = f"Bull regime ({regime.confidence:.0%} confidence)"
            confidence = regime.confidence
        elif regime.regime == "bear":
            action = "consider_sell"
            reason = "Bear regime - reduce exposure"
            confidence = regime.confidence
        else:
            action = "hold"
            reason = f"Sideways regime ({regime.confidence:.0%} confidence)"
            confidence = regime.confidence
        
        return {
            "symbol": symbol,
            "action": action,
            "reason": reason,
            "confidence": confidence,
            "regime": regime.regime,
            "risk_within_limits": risk.within_limits,
            "current_drawdown": risk.drawdown_pct,
        }
    
    # =========================================================================
    # Reporting
    # =========================================================================
    
    def daily_report(self, symbol: str = "BTCUSDT") -> Dict[str, Any]:
        """
        Generate comprehensive daily report.
        
        Args:
            symbol: Primary symbol to analyze
            
        Returns:
            Complete daily analysis report
        """
        # Market analysis
        regime = self.analyze_regime(symbol)
        volatility = self.analyze_volatility(symbol)
        
        # Risk simulation
        mc_result = self.run_monte_carlo(
            price=50000,  # Would get from market data
            volatility=regime.volatility,
        )
        
        # Portfolio optimization (if multiple assets)
        portfolio = self.optimize_portfolio(["BTCUSDT", "ETHUSDT", "SOLUSDT"])
        
        # Risk assessment
        risk = self.assess_risk()
        
        # Actions
        actions = self.propose_actions(symbol)
        
        # Model status
        models = self.model_registry.list_models()
        
        return {
            "generated_at": datetime.utcnow().isoformat(),
            "agent_version": "1.0.0",
            "trading_mode": self.trading_mode,
            "symbol": symbol,
            "market": {
                "regime": regime.regime,
                "regime_confidence": regime.confidence,
                "volatility": regime.volatility,
                "garch_forecast": volatility.get("forecasted_volatility"),
            },
            "monte_carlo": {
                "percentiles": mc_result.get("percentiles", {}),
                "var_95": mc_result.get("var", {}).get("95%"),
                "probability_profit": mc_result.get("probability_profit", 0),
            },
            "portfolio": {
                "weights": portfolio.get("weights", {}),
                "expected_return": portfolio.get("expected_return"),
                "sharpe_ratio": portfolio.get("sharpe_ratio"),
            },
            "risk": {
                "drawdown_pct": risk.drawdown_pct,
                "exposure_pct": risk.exposure_pct,
                "position_count": risk.position_count,
                "within_limits": risk.within_limits,
            },
            "proposed_actions": actions,
            "models": [
                {"name": m.name, "version": m.version, "status": m.status}
                for m in models[:5]  # Top 5
            ],
        }
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get agent status summary.
        
        Returns:
            Status dictionary
        """
        risk = self.assess_risk()
        
        return {
            "trading_mode": self.trading_mode,
            "equity": self.risk_book.equity,
            "risk": {
                "within_limits": risk.within_limits,
                "drawdown_pct": risk.drawdown_pct,
                "exposure_pct": risk.exposure_pct,
                "positions": len(self.risk_book.positions),
            },
            "models": len(self.model_registry.list_models()),
            "last_update": self.last_update.isoformat() if self.last_update else None,
        }
    
    def set_trading_mode(self, mode: str) -> None:
        """
        Set trading mode.
        
        Args:
            mode: "active", "close_only", or "paused"
        """
        valid_modes = ["active", "close_only", "paused"]
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode: {mode}. Must be one of {valid_modes}")
        
        self.trading_mode = mode


# Convenience function
def create_agent(
    max_position_pct: float = 0.10,
    max_drawdown_pct: float = 0.05,
    symbols: List[str] = None,
) -> AutonomousQuantAgent:
    """
    Create a configured Autonomous Quant Agent.
    
    Args:
        max_position_pct: Maximum position size
        max_drawdown_pct: Maximum drawdown
        symbols: Default symbols to trade
        
    Returns:
        Configured AutonomousQuantAgent
    """
    config = AgentConfig(
        max_position_pct=max_position_pct,
        max_drawdown_pct=max_drawdown_pct,
        default_symbols=symbols or ["BTCUSDT", "ETHUSDT"],
    )
    
    return AutonomousQuantAgent(config)
