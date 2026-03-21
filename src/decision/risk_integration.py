"""
Risk Integration Module
=====================
Integration layer between Decision Engine and Risk Book.

This module provides:
- RiskBook integration with DecisionEngine
- Signal validation through risk checks
- Position limit enforcement

Usage:
    from src.decision.risk_integration import RiskIntegratedDecisionEngine
    
    engine = RiskIntegratedDecisionEngine(risk_book=risk_book)
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from app.risk.risk_book import RiskBook, RiskLimits, Position

try:
    from sentiment_concept_bridge import SentimentConceptBridge
except ImportError:
    SentimentConceptBridge = None


@dataclass
class TradingSignal:
    """Trading signal with risk assessment."""
    symbol: str
    side: str  # "buy" / "sell"
    size: float
    reason: str
    confidence: float = 0.0
    risk_score: float = 0.0


@dataclass
class RiskCheckResult:
    """Result of risk validation."""
    approved: bool
    reason: str
    risk_score: float
    position_pct: float = 0.0
    drawdown_pct: float = 0.0


class RiskIntegratedDecisionEngine:
    """
    Decision Engine with integrated Risk Book checks.
    
    Extends the base DecisionEngine with:
    - Pre-trade risk validation
    - Position limit enforcement
    - Drawdown monitoring
    - Real-time exposure tracking
    """
    
    def __init__(
        self,
        risk_book: Optional[RiskBook] = None,
        sentiment_bridge: Optional['SentimentConceptBridge'] = None,
        portfolio_balance: float = 100000,
        threshold_confidence: float = 0.6,
        max_risk_per_trade: float = 0.02,
    ):
        """
        Initialize risk-integrated decision engine.
        
        Args:
            risk_book: RiskBook instance (creates default if None)
            portfolio_balance: Initial portfolio balance
            threshold_confidence: Minimum confidence for signals
            max_risk_per_trade: Maximum risk per trade
        """
        # Initialize or use provided RiskBook
        if risk_book is None:
            limits = RiskLimits(
                max_position_pct=0.10,  # 10% max per position
                max_daily_drawdown_pct=0.05,  # 5% max drawdown
                var_95_limit=0.08,
                cvar_95_limit=0.10,
            )
            self.risk_book = RiskBook(limits)
        else:
            self.risk_book = risk_book
        
        self.sentiment_bridge = sentiment_bridge
        self.portfolio_balance = portfolio_balance
        self.threshold_confidence = threshold_confidence
        self.max_risk_per_trade = max_risk_per_trade
        
        # Initialize equity
        self.risk_book.register_equity(portfolio_balance)
    
    def _get_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Get current prices for symbols.
        
        In production, this would connect to market data.
        For now, returns mock prices.
        """
        # This would be replaced with actual market data client
        # For now, return placeholder
        mock_prices = {
            "BTCUSDT": 50000.0,
            "ETHUSDT": 3000.0,
            "SOLUSDT": 100.0,
        }
        return {s: mock_prices.get(s, 1000.0) for s in symbols}
    
    def validate_signal(
        self,
        signal: TradingSignal,
        current_price: Optional[float] = None
    ) -> RiskCheckResult:
        """
        Validate a trading signal against risk limits.
        
        Args:
            signal: TradingSignal to validate
            current_price: Current market price (fetched if not provided)
            
        Returns:
            RiskCheckResult with approval status
        """
        # Get current price
        if current_price is None:
            prices = self._get_prices([signal.symbol])
            current_price = prices.get(signal.symbol, 0.0)
        
        equity = self.risk_book.equity
        
        # Calculate position percentage
        position_value = signal.size * current_price
        position_pct = position_value / equity if equity > 0 else 0
        
        # Get current drawdown
        drawdown_pct = self.risk_book.daily_drawdown_pct()
        
        # Dynamic Limits based on Sentiment
        dynamic_max_pos_pct = self.risk_book.limits.max_position_pct
        dynamic_threshold_conf = self.threshold_confidence
        sentiment_note = ""
        
        if self.sentiment_bridge and signal.symbol:
            try:
                # Fetch concept-aware sentiment
                sentiment = self.sentiment_bridge.analyze_asset_sentiment_with_concepts(signal.symbol)
                
                if sentiment.confidence > 0.4:
                    if sentiment.sentiment_score < -0.3:
                        # Bearish: Halve max position, require higher confidence
                        dynamic_max_pos_pct *= 0.5
                        dynamic_threshold_conf = min(0.95, dynamic_threshold_conf + 0.15)
                        sentiment_note = f" (Bearish sentiment {sentiment.sentiment_score:.2f}: strict limits applied)"
                    elif sentiment.sentiment_score > 0.3:
                        # Bullish: slightly relax confidence threshold
                        dynamic_threshold_conf = max(0.4, dynamic_threshold_conf - 0.1)
                        sentiment_note = f" (Bullish sentiment {sentiment.sentiment_score:.2f}: relaxed limits)"
            except Exception as e:
                pass # Ignore sentiment errors and use defaults
        
        # Calculate risk score (0-100)
        risk_factors = []
        
        # Check position limit
        if position_pct > dynamic_max_pos_pct:
            risk_factors.append(f"Position {position_pct:.1%} exceeds limit {dynamic_max_pos_pct:.1%}{sentiment_note}")
        
        # Check drawdown
        if drawdown_pct > self.risk_book.limits.max_daily_drawdown_pct:
            risk_factors.append(f"Drawdown {drawdown_pct:.1%} exceeds limit {self.risk_book.limits.max_daily_drawdown_pct:.1%}")
        
        # Check confidence
        if signal.confidence < dynamic_threshold_conf:
            risk_factors.append(f"Confidence {signal.confidence:.2f} below threshold {dynamic_threshold_conf:.2f}{sentiment_note}")
        
        # Calculate risk score
        risk_score = len(risk_factors) * 33.33  # 0, 33, 66, 100
        
        # Determine approval
        approved = len(risk_factors) == 0 and risk_score < 50
        
        reason = "; ".join(risk_factors) if risk_factors else "Approved"
        
        return RiskCheckResult(
            approved=approved,
            reason=reason,
            risk_score=risk_score,
            position_pct=position_pct,
            drawdown_pct=drawdown_pct,
        )
    
    def execute_signal(
        self,
        signal: TradingSignal,
        current_price: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Execute a signal with risk validation.
        
        Args:
            signal: TradingSignal to execute
            current_price: Optional current price
            
        Returns:
            Execution result with risk assessment
        """
        # Validate signal
        risk_check = self.validate_signal(signal, current_price)
        
        if not risk_check.approved:
            return {
                "success": False,
                "reason": risk_check.reason,
                "risk_score": risk_check.risk_score,
                "signal": signal,
            }
        
        # Update position in RiskBook
        current_pos = self.risk_book.get_position(signal.symbol)
        
        if signal.side == "buy":
            new_qty = (current_pos.quantity if current_pos else 0) + signal.size
        else:
            new_qty = (current_pos.quantity if current_pos else 0) - signal.size
        
        # Create/update position
        new_pos = Position(
            symbol=signal.symbol,
            quantity=new_qty,
            avg_price=current_price or 0,
            side="long" if new_qty >= 0 else "short",
        )
        
        self.risk_book.update_position(new_pos)
        
        return {
            "success": True,
            "reason": "Signal executed with risk approval",
            "risk_score": risk_check.risk_score,
            "position": new_pos,
            "risk_check": risk_check,
        }
    
    def get_current_risk_status(self) -> Dict[str, Any]:
        """
        Get current risk status of the portfolio.
        
        Returns:
            Dictionary with risk metrics
        """
        symbols = list(self.risk_book.positions.keys())
        prices = self._get_prices(symbols) if symbols else {}
        
        metrics = self.risk_book.get_metrics(prices)
        
        return {
            "equity": self.risk_book.equity,
            "total_exposure": metrics.total_exposure,
            "exposure_pct": metrics.exposure_pct,
            "daily_drawdown_pct": metrics.daily_drawdown_pct,
            "positions": self.risk_book.get_positions_summary(),
            "limits": {
                "max_position_pct": self.risk_book.limits.max_position_pct,
                "max_daily_drawdown_pct": self.risk_book.limits.max_daily_drawdown_pct,
            },
        }
    
    def check_drawdown_alert(self) -> bool:
        """
        Check if drawdown exceeds warning threshold.
        
        Returns:
            True if drawdown is near limit (80% of max)
        """
        dd = self.risk_book.daily_drawdown_pct()
        threshold = self.risk_book.limits.max_daily_drawdown_pct * 0.8
        
        return dd >= threshold


# Convenience function for creating integrated engine
def create_risk_integrated_engine(
    max_position_pct: float = 0.10,
    max_drawdown_pct: float = 0.05,
    portfolio_balance: float = 100000,
) -> RiskIntegratedDecisionEngine:
    """
    Create a risk-integrated decision engine with custom limits.
    
    Args:
        max_position_pct: Maximum position size as % of equity
        max_drawdown_pct: Maximum daily drawdown as % 
        portfolio_balance: Initial portfolio balance
        
    Returns:
        Configured RiskIntegratedDecisionEngine
    """
    limits = RiskLimits(
        max_position_pct=max_position_pct,
        max_daily_drawdown_pct=max_drawdown_pct,
        var_95_limit=0.08,
        cvar_95_limit=0.10,
    )
    
    risk_book = RiskBook(limits)
    risk_book.register_equity(portfolio_balance)
    
    return RiskIntegratedDecisionEngine(
        risk_book=risk_book,
        portfolio_balance=portfolio_balance,
    )
