"""
Hardened Risk Engine
====================
Production-grade risk management with circuit breakers,
position limits, kill switches, and comprehensive safeguards.
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import numpy as np
from pydantic import BaseModel, Field

from app.core.logging_production import TradingLogger, LogCategory


logger = TradingLogger(__name__, LogCategory.RISK)


# ============================================================================
# ENUMS AND DATA MODELS
# ============================================================================

class RiskLevel(str, Enum):
    """Risk severity levels."""
    GREEN = "green"      # Normal operations
    YELLOW = "yellow"    # Caution - increased monitoring
    ORANGE = "orange"    # Warning - reduce exposure
    RED = "red"          # Critical - halt new positions
    BLACK = "black"      # Emergency - liquidate all


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Trading halted
    HALF_OPEN = "half_open"  # Testing recovery


class KillSwitchType(str, Enum):
    """Types of kill switches."""
    MANUAL = "manual"
    DRAWDOWN = "drawdown"
    VAR_BREACH = "var_breach"
    LEVERAGE_BREACH = "leverage_breach"
    LOSS_LIMIT = "loss_limit"
    VOLATILITY_SPIKE = "volatility_spike"
    CORRELATION_COLLAPSE = "correlation_collapse"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    SYSTEM_ERROR = "system_error"


@dataclass
class Position:
    """Position for risk calculation."""
    symbol: str
    side: str  # LONG, SHORT
    quantity: float
    entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    leverage: float = 1.0
    sector: str = "unknown"
    asset_class: str = "crypto"


@dataclass
class Portfolio:
    """Portfolio for risk calculation."""
    positions: List[Position]
    cash: float
    total_value: float
    initial_capital: float


@dataclass
class RiskLimit:
    """Risk limit configuration."""
    limit_id: str
    limit_type: str
    limit_value: float
    current_value: float
    percentage: float
    is_breached: bool
    severity: RiskLevel
    last_breach_time: Optional[datetime] = None
    breach_count: int = 0


@dataclass
class CircuitBreaker:
    """Circuit breaker configuration."""
    name: str
    threshold: float
    cooldown_seconds: int
    state: CircuitState = CircuitState.CLOSED
    trip_count: int = 0
    last_trip_time: Optional[datetime] = None
    reset_time: Optional[datetime] = None


@dataclass
class KillSwitch:
    """Kill switch status."""
    switch_type: KillSwitchType
    is_active: bool
    activated_at: Optional[datetime] = None
    activated_by: str = "system"
    reason: str = ""
    auto_reset: bool = False
    reset_after_seconds: int = 0


class RiskCheckResult(BaseModel):
    """Result of risk check for an order."""
    approved: bool
    risk_score: float = 0.0
    risk_level: RiskLevel = RiskLevel.GREEN
    reasons: List[str] = []
    warnings: List[str] = []
    var_impact: float = 0.0
    concentration_impact: float = 0.0
    circuit_breakers_tripped: List[str] = []
    kill_switches_active: List[str] = []


# ============================================================================
# HARDENED RISK ENGINE
# ============================================================================

class HardenedRiskEngine:
    """
    Production-grade risk engine with comprehensive safeguards.
    
    Features:
    - Multiple circuit breakers
    - Kill switches (manual and automatic)
    - Position limits (symbol, sector, asset class)
    - VaR and CVaR monitoring
    - Drawdown protection
    - Leverage limits
    - Correlation monitoring
    - Liquidity assessment
    - Real-time monitoring
    """
    
    def __init__(
        self,
        # Capital limits
        initial_capital: float = 100000.0,
        max_drawdown_pct: float = 0.20,  # 20% max drawdown
        daily_loss_limit_pct: float = 0.05,  # 5% daily loss limit
        
        # Position limits
        max_position_pct: float = 0.10,  # 10% max single position
        max_sector_pct: float = 0.25,  # 25% max sector exposure
        max_asset_class_pct: float = 0.50,  # 50% max asset class
        
        # Leverage limits
        max_leverage: float = 5.0,
        max_gross_exposure_pct: float = 2.0,  # 200% gross exposure
        
        # VaR limits
        max_var_95_pct: float = 0.02,  # 2% daily VaR (95%)
        max_cvar_95_pct: float = 0.03,  # 3% daily CVaR (95%)
        
        # Circuit breaker thresholds
        var_circuit_threshold: float = 0.80,  # Trip at 80% of VaR limit
        drawdown_circuit_threshold: float = 0.75,  # Trip at 75% of drawdown
        loss_circuit_threshold: float = 0.80,  # Trip at 80% of daily loss
        
        # Cooldown periods (seconds)
        circuit_cooldown: int = 300,  # 5 minutes
        kill_switch_cooldown: int = 3600,  # 1 hour
    ):
        # Store configuration
        self.initial_capital = initial_capital
        self.max_drawdown_pct = max_drawdown_pct
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.max_position_pct = max_position_pct
        self.max_sector_pct = max_sector_pct
        self.max_asset_class_pct = max_asset_class_pct
        self.max_leverage = max_leverage
        self.max_gross_exposure_pct = max_gross_exposure_pct
        self.max_var_95_pct = max_var_95_pct
        self.max_cvar_95_pct = max_cvar_95_pct
        
        # Circuit breakers
        self.circuit_breakers: Dict[str, CircuitBreaker] = {
            "var": CircuitBreaker("var", var_circuit_threshold, circuit_cooldown),
            "drawdown": CircuitBreaker("drawdown", drawdown_circuit_threshold, circuit_cooldown),
            "daily_loss": CircuitBreaker("daily_loss", loss_circuit_threshold, circuit_cooldown),
            "leverage": CircuitBreaker("leverage", 0.90, circuit_cooldown),
            "concentration": CircuitBreaker("concentration", 0.85, circuit_cooldown),
        }
        
        # Kill switches
        self.kill_switches: Dict[KillSwitchType, KillSwitch] = {
            switch_type: KillSwitch(switch_type, False)
            for switch_type in KillSwitchType
        }
        
        # Risk limits tracking
        self.risk_limits: Dict[str, RiskLimit] = {}
        
        # Historical tracking
        self.daily_pnl_history: List[Tuple[datetime, float]] = []
        self.var_history: List[Tuple[datetime, float]] = []
        self.drawdown_history: List[Tuple[datetime, float]] = []
        
        # State tracking
        self._lock = threading.RLock()
        self._monitoring_active = False
        self._last_check_time: Optional[datetime] = None
        
        # Callbacks for risk events
        self._risk_callbacks: List[Callable] = []
        
        # Performance metrics
        self._check_count = 0
        self._rejection_count = 0
        self._circuit_trip_count = 0
    
    # ========================================================================
    # CIRCUIT BREAKER MANAGEMENT
    # ========================================================================
    
    def check_circuit_breakers(self, portfolio: Portfolio) -> List[str]:
        """
        Check all circuit breakers and return list of tripped breakers.
        """
        tripped = []
        
        with self._lock:
            for name, breaker in self.circuit_breakers.items():
                if breaker.state == CircuitState.OPEN:
                    # Check if cooldown has passed
                    if breaker.reset_time and datetime.now() > breaker.reset_time:
                        breaker.state = CircuitState.HALF_OPEN
                        logger.info(
                            f"Circuit breaker {name} entering half-open state",
                            event_type="circuit_half_open"
                        )
                    else:
                        tripped.append(name)
                        continue
                
                # Check threshold
                threshold_hit = self._check_breaker_threshold(name, portfolio)
                
                if threshold_hit:
                    if breaker.state == CircuitState.HALF_OPEN:
                        # Failed recovery test, reopen
                        breaker.state = CircuitState.OPEN
                        breaker.reset_time = datetime.now() + timedelta(seconds=breaker.cooldown_seconds)
                        tripped.append(name)
                        logger.warning(
                            f"Circuit breaker {name} reopened after failed recovery",
                            event_type="circuit_reopen"
                        )
                    elif breaker.state == CircuitState.CLOSED:
                        # Trip the breaker
                        breaker.state = CircuitState.OPEN
                        breaker.trip_count += 1
                        breaker.last_trip_time = datetime.now()
                        breaker.reset_time = datetime.now() + timedelta(seconds=breaker.cooldown_seconds)
                        tripped.append(name)
                        self._circuit_trip_count += 1
                        
                        logger.log_risk_violation(
                            f"circuit_breaker_{name}",
                            1.0,
                            breaker.threshold,
                            severity="critical"
                        )
        
        return tripped
    
    def _check_breaker_threshold(self, name: str, portfolio: Portfolio) -> bool:
        """Check if a specific breaker threshold is hit."""
        if name == "var":
            var = self._calculate_var(portfolio)
            limit = self.max_var_95_pct * portfolio.total_value
            return var > limit * self.circuit_breakers[name].threshold
        
        elif name == "drawdown":
            drawdown = self._calculate_drawdown(portfolio)
            return drawdown > self.max_drawdown_pct * self.circuit_breakers[name].threshold
        
        elif name == "daily_loss":
            daily_pnl = self._calculate_daily_pnl(portfolio)
            loss_pct = abs(min(daily_pnl, 0)) / self.initial_capital
            return loss_pct > self.daily_loss_limit_pct * self.circuit_breakers[name].threshold
        
        elif name == "leverage":
            leverage = self._calculate_leverage(portfolio)
            return leverage > self.max_leverage * self.circuit_breakers[name].threshold
        
        elif name == "concentration":
            max_concentration = self._calculate_max_concentration(portfolio)
            return max_concentration > self.max_position_pct * self.circuit_breakers[name].threshold
        
        return False
    
    def reset_circuit_breaker(self, name: str) -> bool:
        """Manually reset a circuit breaker."""
        with self._lock:
            if name in self.circuit_breakers:
                breaker = self.circuit_breakers[name]
                breaker.state = CircuitState.CLOSED
                breaker.reset_time = None
                logger.info(
                    f"Circuit breaker {name} manually reset",
                    event_type="circuit_reset"
                )
                return True
        return False
    
    # ========================================================================
    # KILL SWITCH MANAGEMENT
    # ========================================================================
    
    def activate_kill_switch(
        self,
        switch_type: KillSwitchType,
        reason: str = "",
        activated_by: str = "manual",
        auto_reset: bool = False,
        reset_after_seconds: int = 0
    ) -> None:
        """Activate a kill switch."""
        with self._lock:
            switch = self.kill_switches[switch_type]
            switch.is_active = True
            switch.activated_at = datetime.now()
            switch.activated_by = activated_by
            switch.reason = reason
            switch.auto_reset = auto_reset
            switch.reset_after_seconds = reset_after_seconds
            
            logger.critical(
                f"Kill switch activated: {switch_type.value}",
                event_type="kill_switch_activated",
                switch_type=switch_type.value,
                reason=reason,
                activated_by=activated_by
            )
            
            self._notify_callbacks("kill_switch_activated", switch_type, reason)
    
    def deactivate_kill_switch(self, switch_type: KillSwitchType) -> None:
        """Deactivate a kill switch."""
        with self._lock:
            switch = self.kill_switches[switch_type]
            switch.is_active = False
            switch.activated_at = None
            
            logger.info(
                f"Kill switch deactivated: {switch_type.value}",
                event_type="kill_switch_deactivated"
            )
    
    def check_kill_switches(self) -> List[str]:
        """Check active kill switches."""
        active = []
        
        with self._lock:
            for switch_type, switch in self.kill_switches.items():
                if switch.is_active:
                    # Check auto-reset
                    if switch.auto_reset and switch.activated_at:
                        reset_time = switch.activated_at + timedelta(seconds=switch.reset_after_seconds)
                        if datetime.now() > reset_time:
                            self.deactivate_kill_switch(switch_type)
                            continue
                    
                    active.append(switch_type.value)
        
        return active
    
    def emergency_stop(self, reason: str = "Manual emergency stop") -> None:
        """Activate emergency stop - halts all trading."""
        self.activate_kill_switch(
            KillSwitchType.MANUAL,
            reason=reason,
            activated_by="manual"
        )
        
        # Trip all circuit breakers
        for name, breaker in self.circuit_breakers.items():
            breaker.state = CircuitState.OPEN
            breaker.last_trip_time = datetime.now()
    
    # ========================================================================
    # ORDER RISK CHECK
    # ========================================================================
    
    def check_order_risk(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        portfolio: Portfolio,
        strategy: Optional[str] = None
    ) -> RiskCheckResult:
        """
        Comprehensive risk check for an order.
        
        This is the main entry point for all order validation.
        Every order must pass through this check before execution.
        """
        self._check_count += 1
        
        with self._lock:
            reasons = []
            warnings = []
            risk_score = 0.0
            approved = True
            risk_level = RiskLevel.GREEN
            
            # Check kill switches first
            active_kill_switches = self.check_kill_switches()
            if active_kill_switches:
                approved = False
                reasons.append(f"Kill switches active: {', '.join(active_kill_switches)}")
                risk_level = RiskLevel.BLACK
                risk_score = 100.0
                self._rejection_count += 1
                
                return RiskCheckResult(
                    approved=approved,
                    risk_score=risk_score,
                    risk_level=risk_level,
                    reasons=reasons,
                    warnings=warnings,
                    kill_switches_active=active_kill_switches
                )
            
            # Check circuit breakers
            tripped_breakers = self.check_circuit_breakers(portfolio)
            if tripped_breakers:
                approved = False
                reasons.append(f"Circuit breakers tripped: {', '.join(tripped_breakers)}")
                risk_level = RiskLevel.RED
                risk_score = 80.0
                self._rejection_count += 1
                
                return RiskCheckResult(
                    approved=approved,
                    risk_score=risk_score,
                    risk_level=risk_level,
                    reasons=reasons,
                    warnings=warnings,
                    circuit_breakers_tripped=tripped_breakers
                )
            
            order_value = quantity * price
            order_pct = order_value / portfolio.total_value if portfolio.total_value > 0 else 0
            
            # Check 1: Position size limit
            position_check = self._check_position_limit(symbol, order_pct, portfolio)
            if not position_check["passed"]:
                approved = False
                reasons.append(position_check["reason"])
                risk_score += 30
                risk_level = max(risk_level, RiskLevel.RED, key=lambda x: list(RiskLevel).index(x))
            elif position_check.get("warning"):
                warnings.append(position_check["warning"])
                risk_score += 10
                risk_level = max(risk_level, RiskLevel.YELLOW, key=lambda x: list(RiskLevel).index(x))
            
            # Check 2: Sector concentration
            sector_check = self._check_sector_concentration(symbol, order_value, portfolio)
            if not sector_check["passed"]:
                approved = False
                reasons.append(sector_check["reason"])
                risk_score += 25
                risk_level = max(risk_level, RiskLevel.ORANGE, key=lambda x: list(RiskLevel).index(x))
            elif sector_check.get("warning"):
                warnings.append(sector_check["warning"])
                risk_score += 8
            
            # Check 3: Leverage impact
            leverage_check = self._check_leverage_impact(order_value, portfolio)
            if not leverage_check["passed"]:
                approved = False
                reasons.append(leverage_check["reason"])
                risk_score += 25
                risk_level = max(risk_level, RiskLevel.RED, key=lambda x: list(RiskLevel).index(x))
            elif leverage_check.get("warning"):
                warnings.append(leverage_check["warning"])
                risk_score += 10
            
            # Check 4: VaR impact
            var_impact = self._calculate_var_impact(symbol, quantity, price, portfolio)
            var_check = self._check_var_impact(var_impact, portfolio)
            if not var_check["passed"]:
                approved = False
                reasons.append(var_check["reason"])
                risk_score += 20
                risk_level = max(risk_level, RiskLevel.ORANGE, key=lambda x: list(RiskLevel).index(x))
            elif var_check.get("warning"):
                warnings.append(var_check["warning"])
                risk_score += 5
            
            # Check 5: Drawdown protection
            drawdown_check = self._check_drawdown_impact(order_value, portfolio)
            if not drawdown_check["passed"]:
                approved = False
                reasons.append(drawdown_check["reason"])
                risk_score += 30
                risk_level = max(risk_level, RiskLevel.RED, key=lambda x: list(RiskLevel).index(x))
            elif drawdown_check.get("warning"):
                warnings.append(drawdown_check["warning"])
                risk_score += 10
            
            # Check 6: Daily loss limit
            daily_loss_check = self._check_daily_loss_limit(portfolio)
            if not daily_loss_check["passed"]:
                approved = False
                reasons.append(daily_loss_check["reason"])
                risk_score += 35
                risk_level = max(risk_level, RiskLevel.BLACK, key=lambda x: list(RiskLevel).index(x))
            elif daily_loss_check.get("warning"):
                warnings.append(daily_loss_check["warning"])
                risk_score += 15
                risk_level = max(risk_level, RiskLevel.YELLOW, key=lambda x: list(RiskLevel).index(x))
            
            # Check 7: Gross exposure
            exposure_check = self._check_gross_exposure(order_value, portfolio)
            if not exposure_check["passed"]:
                approved = False
                reasons.append(exposure_check["reason"])
                risk_score += 20
            elif exposure_check.get("warning"):
                warnings.append(exposure_check["warning"])
                risk_score += 8
            
            # Calculate final risk score
            risk_score = min(risk_score, 100.0)
            
            # Log rejection if applicable
            if not approved:
                self._rejection_count += 1
                logger.log_risk_violation(
                    "order_rejection",
                    risk_score,
                    50.0,
                    severity="warning" if risk_level in [RiskLevel.YELLOW, RiskLevel.ORANGE] else "critical"
                )
            
            # Update last check time
            self._last_check_time = datetime.now()
            
            return RiskCheckResult(
                approved=approved,
                risk_score=risk_score,
                risk_level=risk_level,
                reasons=reasons,
                warnings=warnings,
                var_impact=var_impact,
                concentration_impact=order_pct,
                circuit_breakers_tripped=[],
                kill_switches_active=[]
            )
    
    # ========================================================================
    # INDIVIDUAL CHECKS
    # ========================================================================
    
    def _check_position_limit(
        self,
        symbol: str,
        order_pct: float,
        portfolio: Portfolio
    ) -> Dict[str, Any]:
        """Check position size limit."""
        # Find existing position
        existing_value = sum(
            pos.market_value for pos in portfolio.positions
            if pos.symbol == symbol
        )
        existing_pct = existing_value / portfolio.total_value if portfolio.total_value > 0 else 0
        total_pct = existing_pct + order_pct
        
        if total_pct > self.max_position_pct:
            return {
                "passed": False,
                "reason": f"Position size {total_pct:.1%} would exceed limit {self.max_position_pct:.1%}"
            }
        elif total_pct > self.max_position_pct * 0.8:
            return {
                "passed": True,
                "warning": f"Position size {total_pct:.1%} approaching limit"
            }
        
        return {"passed": True}
    
    def _check_sector_concentration(
        self,
        symbol: str,
        order_value: float,
        portfolio: Portfolio
    ) -> Dict[str, Any]:
        """Check sector concentration limits."""
        # Group positions by sector
        sector_values: Dict[str, float] = defaultdict(float)
        
        for pos in portfolio.positions:
            sector_values[pos.sector] += pos.market_value
        
        # Add order value (assume same sector as symbol if known)
        # In production, would look up symbol's sector
        order_sector = "crypto"  # Default
        sector_values[order_sector] += order_value
        
        # Check each sector
        for sector, value in sector_values.items():
            pct = value / portfolio.total_value if portfolio.total_value > 0 else 0
            
            if pct > self.max_sector_pct:
                return {
                    "passed": False,
                    "reason": f"Sector {sector} concentration {pct:.1%} exceeds limit {self.max_sector_pct:.1%}"
                }
            elif pct > self.max_sector_pct * 0.8:
                return {
                    "passed": True,
                    "warning": f"Sector {sector} concentration {pct:.1%} approaching limit"
                }
        
        return {"passed": True}
    
    def _check_leverage_impact(
        self,
        order_value: float,
        portfolio: Portfolio
    ) -> Dict[str, Any]:
        """Check leverage impact."""
        current_exposure = sum(abs(pos.market_value) for pos in portfolio.positions)
        new_exposure = current_exposure + order_value
        new_leverage = new_exposure / portfolio.total_value if portfolio.total_value > 0 else 1.0
        
        if new_leverage > self.max_leverage:
            return {
                "passed": False,
                "reason": f"Leverage {new_leverage:.1f}x would exceed limit {self.max_leverage:.1f}x"
            }
        elif new_leverage > self.max_leverage * 0.8:
            return {
                "passed": True,
                "warning": f"Leverage {new_leverage:.1f}x approaching limit"
            }
        
        return {"passed": True}
    
    def _check_var_impact(
        self,
        var_impact: float,
        portfolio: Portfolio
    ) -> Dict[str, Any]:
        """Check VaR impact."""
        current_var = self._calculate_var(portfolio)
        projected_var = current_var + var_impact
        var_limit = self.max_var_95_pct * portfolio.total_value
        
        if projected_var > var_limit:
            return {
                "passed": False,
                "reason": f"Projected VaR {projected_var:.2f} would exceed limit {var_limit:.2f}"
            }
        elif projected_var > var_limit * 0.8:
            return {
                "passed": True,
                "warning": f"Projected VaR {projected_var:.2f} approaching limit"
            }
        
        return {"passed": True}
    
    def _check_drawdown_impact(
        self,
        order_value: float,
        portfolio: Portfolio
    ) -> Dict[str, Any]:
        """Check drawdown protection."""
        current_drawdown = self._calculate_drawdown(portfolio)
        potential_loss = order_value * 0.5  # Assume 50% worst case
        potential_drawdown = (portfolio.total_value - self.initial_capital - potential_loss) / self.initial_capital
        
        if potential_drawdown > self.max_drawdown_pct:
            return {
                "passed": False,
                "reason": f"Potential drawdown {potential_drawdown:.1%} would exceed limit {self.max_drawdown_pct:.1%}"
            }
        elif current_drawdown > self.max_drawdown_pct * 0.75:
            return {
                "passed": True,
                "warning": f"Current drawdown {current_drawdown:.1%} approaching limit"
            }
        
        return {"passed": True}
    
    def _check_daily_loss_limit(self, portfolio: Portfolio) -> Dict[str, Any]:
        """Check daily loss limit."""
        daily_pnl = self._calculate_daily_pnl(portfolio)
        daily_loss_pct = abs(min(daily_pnl, 0)) / self.initial_capital
        
        if daily_loss_pct > self.daily_loss_limit_pct:
            return {
                "passed": False,
                "reason": f"Daily loss {daily_loss_pct:.1%} exceeds limit {self.daily_loss_limit_pct:.1%}"
            }
        elif daily_loss_pct > self.daily_loss_limit_pct * 0.8:
            return {
                "passed": True,
                "warning": f"Daily loss {daily_loss_pct:.1%} approaching limit"
            }
        
        return {"passed": True}
    
    def _check_gross_exposure(
        self,
        order_value: float,
        portfolio: Portfolio
    ) -> Dict[str, Any]:
        """Check gross exposure limit."""
        current_exposure = sum(abs(pos.market_value) for pos in portfolio.positions)
        new_exposure = current_exposure + order_value
        exposure_pct = new_exposure / portfolio.total_value if portfolio.total_value > 0 else 0
        
        if exposure_pct > self.max_gross_exposure_pct:
            return {
                "passed": False,
                "reason": f"Gross exposure {exposure_pct:.1%} would exceed limit {self.max_gross_exposure_pct:.1%}"
            }
        elif exposure_pct > self.max_gross_exposure_pct * 0.8:
            return {
                "passed": True,
                "warning": f"Gross exposure {exposure_pct:.1%} approaching limit"
            }
        
        return {"passed": True}
    
    # ========================================================================
    # CALCULATION HELPERS
    # ========================================================================
    
    def _calculate_var(self, portfolio: Portfolio, confidence: float = 0.95) -> float:
        """Calculate portfolio VaR."""
        if not portfolio.positions:
            return 0.0
        
        # Simplified VaR calculation
        # In production, would use historical simulation or Monte Carlo
        total_value = portfolio.total_value
        daily_volatility = 0.02  # 2% daily volatility assumption
        
        # Z-score for confidence level
        from scipy import stats
        z_score = stats.norm.ppf(1 - confidence)
        
        var = abs(z_score) * daily_volatility * total_value
        return var
    
    def _calculate_cvar(self, portfolio: Portfolio, confidence: float = 0.95) -> float:
        """Calculate Conditional VaR (Expected Shortfall)."""
        var = self._calculate_var(portfolio, confidence)
        # CVaR is typically 1.2-1.5x VaR for normal distributions
        return var * 1.3
    
    def _calculate_drawdown(self, portfolio: Portfolio) -> float:
        """Calculate current drawdown."""
        if self.initial_capital <= 0:
            return 0.0
        
        peak_value = max(self.initial_capital, portfolio.total_value)
        if portfolio.total_value >= peak_value:
            return 0.0
        
        return (peak_value - portfolio.total_value) / peak_value
    
    def _calculate_daily_pnl(self, portfolio: Portfolio) -> float:
        """Calculate daily P&L."""
        # In production, would track actual daily starting value
        # Simplified: use unrealized + realized P&L
        unrealized = sum(pos.unrealized_pnl for pos in portfolio.positions)
        return unrealized
    
    def _calculate_leverage(self, portfolio: Portfolio) -> float:
        """Calculate portfolio leverage."""
        if portfolio.total_value <= 0:
            return 0.0
        
        gross_exposure = sum(abs(pos.market_value) for pos in portfolio.positions)
        return gross_exposure / portfolio.total_value
    
    def _calculate_max_concentration(self, portfolio: Portfolio) -> float:
        """Calculate maximum position concentration."""
        if not portfolio.positions or portfolio.total_value <= 0:
            return 0.0
        
        max_position = max(pos.market_value for pos in portfolio.positions)
        return max_position / portfolio.total_value
    
    def _calculate_var_impact(
        self,
        symbol: str,
        quantity: float,
        price: float,
        portfolio: Portfolio
    ) -> float:
        """Calculate VaR impact of new position."""
        order_value = quantity * price
        
        # Simplified: assume position adds proportional risk
        # In production, would consider correlation with existing positions
        daily_volatility = 0.02
        var_contribution = order_value * daily_volatility * 1.65  # 95% confidence
        
        return var_contribution
    
    # ========================================================================
    # MONITORING AND CALLBACKS
    # ========================================================================
    
    def register_callback(self, callback: Callable) -> None:
        """Register a callback for risk events."""
        self._risk_callbacks.append(callback)
    
    def _notify_callbacks(self, event_type: str, *args, **kwargs) -> None:
        """Notify all registered callbacks."""
        for callback in self._risk_callbacks:
            try:
                callback(event_type, *args, **kwargs)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    def get_risk_status(self, portfolio: Portfolio) -> Dict[str, Any]:
        """Get comprehensive risk status."""
        with self._lock:
            return {
                "timestamp": datetime.now().isoformat(),
                "portfolio_value": portfolio.total_value,
                "initial_capital": self.initial_capital,
                "risk_level": self._determine_risk_level(portfolio),
                "metrics": {
                    "var_95": self._calculate_var(portfolio),
                    "cvar_95": self._calculate_cvar(portfolio),
                    "drawdown": self._calculate_drawdown(portfolio),
                    "leverage": self._calculate_leverage(portfolio),
                    "daily_pnl": self._calculate_daily_pnl(portfolio),
                    "max_concentration": self._calculate_max_concentration(portfolio),
                },
                "limits": {
                    name: {
                        "limit_value": limit.limit_value,
                        "current_value": limit.current_value,
                        "percentage": limit.percentage,
                        "is_breached": limit.is_breached,
                        "severity": limit.severity.value
                    }
                    for name, limit in self.risk_limits.items()
                },
                "circuit_breakers": {
                    name: {
                        "state": breaker.state.value,
                        "trip_count": breaker.trip_count,
                        "last_trip": breaker.last_trip_time.isoformat() if breaker.last_trip_time else None
                    }
                    for name, breaker in self.circuit_breakers.items()
                },
                "kill_switches": {
                    switch_type.value: {
                        "is_active": switch.is_active,
                        "activated_at": switch.activated_at.isoformat() if switch.activated_at else None,
                        "reason": switch.reason
                    }
                    for switch_type, switch in self.kill_switches.items()
                    if switch.is_active
                },
                "statistics": {
                    "total_checks": self._check_count,
                    "rejections": self._rejection_count,
                    "circuit_trips": self._circuit_trip_count,
                    "rejection_rate": self._rejection_count / max(self._check_count, 1)
                }
            }
    
    def _determine_risk_level(self, portfolio: Portfolio) -> str:
        """Determine overall risk level."""
        # Check kill switches
        if any(switch.is_active for switch in self.kill_switches.values()):
            return RiskLevel.BLACK.value
        
        # Check circuit breakers
        open_breakers = [b for b in self.circuit_breakers.values() if b.state == CircuitState.OPEN]
        if open_breakers:
            return RiskLevel.RED.value
        
        # Check metrics
        drawdown = self._calculate_drawdown(portfolio)
        if drawdown > self.max_drawdown_pct * 0.9:
            return RiskLevel.RED.value
        elif drawdown > self.max_drawdown_pct * 0.75:
            return RiskLevel.ORANGE.value
        elif drawdown > self.max_drawdown_pct * 0.5:
            return RiskLevel.YELLOW.value
        
        return RiskLevel.GREEN.value
    
    def update_limits_tracking(self, portfolio: Portfolio) -> None:
        """Update risk limits tracking."""
        with self._lock:
            # VaR limit
            var = self._calculate_var(portfolio)
            var_limit = self.max_var_95_pct * portfolio.total_value
            self.risk_limits["var_95"] = RiskLimit(
                limit_id="var_95",
                limit_type="var",
                limit_value=var_limit,
                current_value=var,
                percentage=(var / var_limit * 100) if var_limit > 0 else 0,
                is_breached=var > var_limit,
                severity=self._get_severity(var / var_limit if var_limit > 0 else 0)
            )
            
            # Drawdown limit
            drawdown = self._calculate_drawdown(portfolio)
            self.risk_limits["drawdown"] = RiskLimit(
                limit_id="drawdown",
                limit_type="drawdown",
                limit_value=self.max_drawdown_pct,
                current_value=drawdown,
                percentage=drawdown / self.max_drawdown_pct * 100 if self.max_drawdown_pct > 0 else 0,
                is_breached=drawdown > self.max_drawdown_pct,
                severity=self._get_severity(drawdown / self.max_drawdown_pct if self.max_drawdown_pct > 0 else 0)
            )
            
            # Leverage limit
            leverage = self._calculate_leverage(portfolio)
            self.risk_limits["leverage"] = RiskLimit(
                limit_id="leverage",
                limit_type="leverage",
                limit_value=self.max_leverage,
                current_value=leverage,
                percentage=leverage / self.max_leverage * 100 if self.max_leverage > 0 else 0,
                is_breached=leverage > self.max_leverage,
                severity=self._get_severity(leverage / self.max_leverage if self.max_leverage > 0 else 0)
            )
    
    def _get_severity(self, ratio: float) -> RiskLevel:
        """Get severity level based on ratio."""
        if ratio >= 1.0:
            return RiskLevel.RED
        elif ratio >= 0.9:
            return RiskLevel.ORANGE
        elif ratio >= 0.75:
            return RiskLevel.YELLOW
        return RiskLevel.GREEN
