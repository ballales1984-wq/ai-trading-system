"""
Unified Decision Engine
======================
Single source of truth for all trading decisions.
Integrates: Risk Book, Model Registry, OpenClaw Skills, ML Predictor.

This is the central brain of the trading system used by:
- Scripts (main_auto_trader.py, live_multi_asset.py)
- API routes
- OpenClaw (via wrapper)

Usage:
    from src.decision.unified_engine import UnifiedDecisionEngine
    
    engine = UnifiedDecisionEngine()
    
    # Make a decision
    decision = engine.decide(
        symbol="BTCUSDT",
        current_price=50000,
        signals={"technical": 0.7, "sentiment": 0.6}
    )
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

# Risk Management
from app.risk.risk_book import RiskBook, RiskLimits, Position, PositionSide

# Model Registry
from src.research.model_registry import ModelRegistry, ModelMeta

# OpenClaw Skills
try:
    from openclaw_skills.intent_router import route_intent
    OPENCLAW_AVAILABLE = True
except ImportError:
    OPENCLAW_AVAILABLE = False
    logging.warning("OpenClaw not available - running in limited mode")

# ML Predictor
try:
    from ml_predictor import PricePredictor
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logging.warning("ML Predictor not available")

logger = logging.getLogger(__name__)


class Decision(str, Enum):
    """Trading decision enumeration."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    REJECT = "reject"


@dataclass
class DecisionResult:
    """Result of a trading decision."""
    decision: Decision
    confidence: float
    position_size: float
    risk_score: float
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class UnifiedEngineConfig:
    """Configuration for the Unified Decision Engine."""
    # Risk limits
    max_position_pct: float = 0.10
    max_daily_drawdown_pct: float = 0.05
    var_95_limit: float = 0.08
    cvar_95_limit: float = 0.10
    
    # Trading parameters
    min_confidence: float = 0.6
    max_risk_per_trade: float = 0.02
    default_equity: float = 100000.0
    
    # Model selection
    use_champion_model: bool = True
    
    # OpenClaw integration
    use_openclaw_skills: bool = True


class UnifiedDecisionEngine:
    """
    Unified Decision Engine - Single Source of Truth
    
    This engine orchestrates all components to make trading decisions:
    - Risk Book for position/limit management
    - Model Registry for ML model selection
    - OpenClaw skills for advanced analysis
    - ML Predictor for price predictions
    
    All scripts and API routes should use this engine.
    """
    
    def __init__(self, config: Optional[UnifiedEngineConfig] = None):
        """
        Initialize the Unified Decision Engine.
        
        Args:
            config: Engine configuration (uses defaults if not provided)
        """
        self.config = config or UnifiedEngineConfig()
        
        # Initialize components
        self._init_risk_book()
        self._init_model_registry()
        self._init_ml_predictor()
        
        # State
        self.last_update: Optional[datetime] = None
        self.decision_count: int = 0
        
        logger.info("UnifiedDecisionEngine initialized")
    
    def _init_risk_book(self) -> None:
        """Initialize Risk Book with configured limits."""
        limits = RiskLimits(
            max_position_pct=self.config.max_position_pct,
            max_daily_drawdown_pct=self.config.max_daily_drawdown_pct,
            var_95_limit=self.config.var_95_limit,
            cvar_95_limit=self.config.cvar_95_limit,
        )
        self.risk_book = RiskBook(limits)
        self.risk_book.register_equity(self.config.default_equity)
        logger.info("RiskBook initialized")
    
    def _init_model_registry(self) -> None:
        """Initialize Model Registry."""
        self.model_registry = ModelRegistry()
        logger.info("ModelRegistry initialized")
    
    def _init_ml_predictor(self) -> None:
        """Initialize ML Predictor."""
        if ML_AVAILABLE:
            self.ml_predictor = PricePredictor()
            self.ml_model_loaded = False
            logger.info("ML Predictor initialized")
        else:
            self.ml_predictor = None
            self.ml_model_loaded = False
    
    def load_champion_model(self) -> bool:
        """
        Load the champion model from Model Registry.
        
        Returns:
            True if successful, False otherwise
        """
        if not ML_AVAILABLE or not self.model_registry:
            return False
        
        champion = self.model_registry.get_champion("price_prediction")
        if champion:
            logger.info(f"Champion model loaded: {champion.name}")
            self.ml_model_loaded = True
            return True
        
        logger.warning("No champion model found in registry")
        return False
    
    def get_ml_prediction(self, symbol: str, df: pd.DataFrame) -> Dict[str, float]:
        """
        Get ML prediction for a symbol.
        
        Args:
            symbol: Trading symbol
            df: OHLCV dataframe
            
        Returns:
            Dictionary with prediction scores
        """
        if not self.ml_predictor:
            return {"prediction": 0.5, "confidence": 0.0}
        
        try:
            features = self.ml_predictor.prepare_features(df)
            prediction = self.ml_predictor.predict(features.iloc[-1:])
            return {
                "prediction": prediction.get("probability", 0.5),
                "confidence": prediction.get("confidence", 0.0)
            }
        except Exception as e:
            logger.error(f"ML prediction error: {e}")
            return {"prediction": 0.5, "confidence": 0.0}
    
    def get_regime_analysis(self, symbol: str) -> Dict[str, Any]:
        """
        Get market regime analysis using OpenClaw HMM skill.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Regime analysis result
        """
        if not OPENCLAW_AVAILABLE or not self.config.use_openclaw_skills:
            return {"regime": "unknown", "confidence": 0.0}
        
        try:
            result = route_intent("regime_analysis", {"symbol": symbol})
            return result
        except Exception as e:
            logger.error(f"Regime analysis error: {e}")
            return {"regime": "unknown", "confidence": 0.0}
    
    def get_volatility_forecast(self, symbol: str) -> Dict[str, Any]:
        """
        Get volatility forecast using OpenClaw GARCH skill.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Volatility forecast result
        """
        if not OPENCLAW_AVAILABLE or not self.config.use_openclaw_skills:
            return {"volatility": 0.03, "forecast": None}
        
        try:
            result = route_intent("volatility_analysis", {"symbol": symbol})
            return result
        except Exception as e:
            logger.error(f"Volatility forecast error: {e}")
            return {"volatility": 0.03, "forecast": None}
    
    def get_monte_carlo_simulation(
        self,
        symbol: str,
        initial_price: float,
        expected_return: float = 0.0,
        volatility: float = 0.03,
        n_paths: int = 5000,
        days_ahead: int = 30
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation using OpenClaw skill.
        
        Args:
            symbol: Trading symbol
            initial_price: Starting price
            expected_return: Expected daily return
            volatility: Volatility
            n_paths: Number of simulation paths
            days_ahead: Forecast horizon
            
        Returns:
            Monte Carlo simulation result
        """
        if not OPENCLAW_AVAILABLE or not self.config.use_openclaw_skills:
            return {"paths": [], "mean": initial_price, "percentiles": {}}
        
        try:
            result = route_intent(
                "simulate_paths",
                {
                    "initial_price": initial_price,
                    "expected_return": expected_return,
                    "volatility": volatility,
                    "n_paths": n_paths,
                    "days_ahead": days_ahead,
                }
            )
            return result
        except Exception as e:
            logger.error(f"Monte Carlo simulation error: {e}")
            return {"paths": [], "mean": initial_price, "percentiles": {}}
    
    def check_risk_limits(
        self,
        symbol: str,
        side: str,
        size: float,
        price: float,
        equity: float
    ) -> Dict[str, Any]:
        """
        Check if a trade is within risk limits.
        
        Args:
            symbol: Trading symbol
            side: "buy" or "sell"
            size: Position size
            price: Entry price
            equity: Current equity
            
        Returns:
            Risk check result with approval status
        """
        position_side = PositionSide.LONG if side == "buy" else PositionSide.SHORT
        
        # Create temporary position
        temp_position = Position(
            symbol=symbol,
            quantity=size,
            avg_price=price,
            side=position_side
        )
        
        # Check position limit
        prices = {symbol: price}
        position_ok = self.risk_book.check_position_limit(symbol, prices, equity)
        
        # Check daily drawdown
        drawdown_ok = self.risk_book.daily_drawdown_ok(equity)
        
        # Calculate position percentage
        position_value = size * price
        position_pct = position_value / equity if equity > 0 else 0
        
        # Calculate risk score (simplified)
        risk_score = min(position_pct / self.config.max_position_pct, 1.0)
        
        approved = position_ok and drawdown_ok
        
        return {
            "approved": approved,
            "reason": "Approved" if approved else "Risk limit breach",
            "risk_score": risk_score,
            "position_pct": position_pct,
            "drawdown_pct": abs(equity - self.risk_book.initial_equity) / self.risk_book.initial_equity,
            "position_ok": position_ok,
            "drawdown_ok": drawdown_ok,
        }
    
    def decide(
        self,
        symbol: str,
        current_price: float,
        signals: Dict[str, float],
        equity: Optional[float] = None,
        use_openclaw: bool = True,
    ) -> DecisionResult:
        """
        Make a trading decision for a symbol.
        
        This is the main entry point for making trading decisions.
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            signals: Dictionary of signal scores (technical, sentiment, ml, etc.)
            equity: Current equity (uses default if not provided)
            use_openclaw: Whether to use OpenClaw skills
            
        Returns:
            DecisionResult with the decision and metadata
        """
        self.decision_count += 1
        self.last_update = datetime.now()
        
        # Use default equity if not provided
        if equity is None:
            equity = self.config.default_equity
        
        # Aggregate signals
        signal_values = list(signals.values())
        avg_signal = np.mean(signal_values) if signal_values else 0.5
        max_signal = max(signal_values) if signal_values else 0.5
        
        # Get confidence
        confidence = max_signal
        
        # Check minimum confidence threshold
        if confidence < self.config.min_confidence:
            return DecisionResult(
                decision=Decision.HOLD,
                confidence=confidence,
                position_size=0.0,
                risk_score=0.0,
                reason=f"Confidence {confidence:.2f} below threshold {self.config.min_confidence}",
                metadata={"signals": signals}
            )
        
        # Determine decision based on signals
        if avg_signal > 0.65:
            base_decision = Decision.BUY
        elif avg_signal < 0.35:
            base_decision = Decision.SELL
        else:
            base_decision = Decision.HOLD
        
        # If BUY, calculate position size and check risk
        if base_decision == Decision.BUY:
            position_size = self._calculate_position_size(
                symbol=symbol,
                price=current_price,
                confidence=confidence,
                equity=equity
            )
            
            # Check risk limits
            risk_check = self.check_risk_limits(
                symbol=symbol,
                side="buy",
                size=position_size,
                price=current_price,
                equity=equity
            )
            
            if not risk_check["approved"]:
                return DecisionResult(
                    decision=Decision.REJECT,
                    confidence=confidence,
                    position_size=0.0,
                    risk_score=risk_check["risk_score"],
                    reason=f"Rejected: {risk_check['reason']}",
                    metadata={"risk_check": risk_check, "signals": signals}
                )
            
            decision = Decision.BUY
            reason = f"BUY signal accepted (confidence: {confidence:.2f})"
            
        elif base_decision == Decision.SELL:
            # Check if we have a position to sell
            position = self.risk_book.get_position(symbol)
            if position:
                position_size = position.quantity
                reason = f"SELL signal for existing position"
            else:
                position_size = 0.0
                reason = "SELL signal but no position to close"
            
            decision = Decision.SELL if position_size > 0 else Decision.HOLD
            
        else:
            position_size = 0.0
            reason = f"HOLD (signal: {avg_signal:.2f})"
        
        # Calculate risk score
        risk_score = 1.0 - confidence
        
        return DecisionResult(
            decision=decision,
            confidence=confidence,
            position_size=position_size,
            risk_score=risk_score,
            reason=reason,
            metadata={
                "signals": signals,
                "avg_signal": avg_signal,
                "symbol": symbol,
                "price": current_price,
                "equity": equity,
            }
        )
    
    def _calculate_position_size(
        self,
        symbol: str,
        price: float,
        confidence: float,
        equity: float
    ) -> float:
        """
        Calculate position size based on confidence and risk limits.
        
        Args:
            symbol: Trading symbol
            price: Entry price
            confidence: Signal confidence
            equity: Current equity
            
        Returns:
            Position size in base currency
        """
        # Base position size as percentage of equity
        base_pct = self.config.max_position_pct * confidence
        
        # Apply risk per trade limit
        max_pct = min(base_pct, self.config.max_risk_per_trade * 10)
        
        # Calculate position value
        position_value = equity * max_pct
        
        # Convert to quantity
        position_size = position_value / price
        
        return position_size
    
    def get_portfolio_status(self) -> Dict[str, Any]:
        """
        Get current portfolio status including positions and risk metrics.
        
        Returns:
            Dictionary with portfolio status
        """
        positions = self.risk_book.get_all_positions()
        equity = self.risk_book.current_equity
        initial_equity = self.risk_book.initial_equity
        
        return {
            "positions": [p.__dict__ for p in positions],
            "position_count": len(positions),
            "equity": equity,
            "initial_equity": initial_equity,
            "total_pnl": equity - initial_equity,
            "pnl_pct": ((equity - initial_equity) / initial_equity * 100) if initial_equity > 0 else 0,
            "decision_count": self.decision_count,
            "last_update": self.last_update.isoformat() if self.last_update else None,
        }
    
    def update_position(
        self,
        symbol: str,
        side: str,
        quantity: float,
        avg_price: float
    ) -> None:
        """
        Update a position in the Risk Book.
        
        Args:
            symbol: Trading symbol
            side: "buy" or "sell"
            quantity: Position quantity
            avg_price: Average entry price
        """
        position_side = PositionSide.LONG if side == "buy" else PositionSide.SHORT
        
        position = Position(
            symbol=symbol,
            quantity=quantity,
            avg_price=avg_price,
            side=position_side
        )
        
        self.risk_book.update_position(position)
        logger.info(f"Position updated: {side} {quantity} {symbol} @ {avg_price}")


# Global instance for convenience
_engine: Optional[UnifiedDecisionEngine] = None


def get_engine(config: Optional[UnifiedEngineConfig] = None) -> UnifiedDecisionEngine:
    """
    Get or create the global UnifiedDecisionEngine instance.
    
    Args:
        config: Optional configuration
        
    Returns:
        UnifiedDecisionEngine instance
    """
    global _engine
    if _engine is None:
        _engine = UnifiedDecisionEngine(config)
    return _engine
