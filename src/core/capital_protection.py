"""
Capital Protection Layer
========================
Institutional-grade capital protection system.

This module provides:
- Hard risk limits with automatic enforcement
- Daily loss protection
- Drawdown protection
- Correlation exposure monitoring
- API failure auto-shutdown
- Emergency kill switch
- Real-time capital monitoring

Author: AI Trading System
Version: 1.0.0
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import numpy as np
import json
from pathlib import Path

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND DATA MODELS
# =============================================================================

class ProtectionLevel(str, Enum):
    """Protection severity levels."""
    NORMAL = "normal"           # All systems operational
    ELEVATED = "elevated"       # Increased monitoring
    RESTRICTED = "restricted"   # New positions blocked
    FROZEN = "frozen"           # All trading halted
    LIQUIDATION = "liquidation" # Emergency liquidation


class TriggerType(str, Enum):
    """Types of protection triggers."""
    DAILY_LOSS = "daily_loss"
    DRAWDOWN = "drawdown"
    POSITION_SIZE = "position_size"
    CORRELATION = "correlation"
    API_FAILURE = "api_failure"
    MANUAL = "manual"
    VOLATILITY = "volatility"
    LIQUIDITY = "liquidity"
    LEVERAGE = "leverage"


@dataclass
class ProtectionConfig:
    """Configuration for capital protection."""
    # Daily Loss Protection
    max_daily_loss_pct: float = 0.03          # 3% max daily loss
    daily_loss_warning_pct: float = 0.02      # Warning at 2%
    
    # Drawdown Protection
    max_drawdown_pct: float = 0.10            # 10% max drawdown
    drawdown_warning_pct: float = 0.07        # Warning at 7%
    
    # Position Limits
    max_single_position_pct: float = 0.15     # 15% max single position
    max_sector_exposure_pct: float = 0.30     # 30% max sector
    max_correlated_assets_pct: float = 0.40   # 40% max correlated cluster
    
    # Correlation Monitoring
    correlation_threshold: float = 0.70       # High correlation threshold
    correlation_lookback_days: int = 30       # Days for correlation calc
    
    # API Failure Protection
    max_api_failures_per_hour: int = 10       # Max failures before shutdown
    max_api_failures_per_day: int = 50        # Daily limit
    api_recovery_attempts: int = 3            # Recovery attempts
    
    # Leverage Limits
    max_portfolio_leverage: float = 3.0       # Max overall leverage
    leverage_warning_level: float = 2.5       # Warning level
    
    # Volatility Protection
    max_volatility_regime: float = 0.50       # Max vol regime (annualized)
    vol_spike_multiplier: float = 3.0         # Vol spike detection
    
    # Recovery Settings
    auto_recovery_enabled: bool = True
    recovery_cooldown_minutes: int = 60       # Cooldown before recovery
    manual_reset_required: bool = False       # Require manual reset after freeze


@dataclass
class ProtectionState:
    """Current state of capital protection."""
    level: ProtectionLevel = ProtectionLevel.NORMAL
    active_triggers: List[TriggerType] = field(default_factory=list)
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    current_drawdown: float = 0.0
    peak_capital: float = 0.0
    current_capital: float = 0.0
    api_failures_hour: int = 0
    api_failures_day: int = 0
    last_reset_time: datetime = field(default_factory=datetime.now)
    frozen_at: Optional[datetime] = None
    frozen_reason: str = ""
    warnings: List[str] = field(default_factory=list)


@dataclass
class PositionRisk:
    """Risk metrics for a position."""
    symbol: str
    market_value: float
    weight_pct: float
    sector: str
    correlation_cluster: int
    var_contribution: float
    leverage: float


# =============================================================================
# CAPITAL PROTECTION ENGINE
# =============================================================================

class CapitalProtectionEngine:
    """
    Institutional-grade capital protection system.
    
    Features:
    - Hard daily loss limit with automatic trading halt
    - Maximum drawdown protection
    - Position size limits
    - Correlation exposure monitoring
    - API failure detection and auto-shutdown
    - Manual emergency kill switch
    - Real-time monitoring and alerts
    """
    
    def __init__(
        self,
        config: Optional[ProtectionConfig] = None,
        initial_capital: float = 100000.0,
        state_file: Optional[str] = None
    ):
        self.config = config or ProtectionConfig()
        self.initial_capital = initial_capital
        self.state_file = state_file or "data/capital_protection_state.json"
        
        # Initialize state
        self.state = ProtectionState(
            peak_capital=initial_capital,
            current_capital=initial_capital
        )
        
        # Position tracking
        self.positions: Dict[str, PositionRisk] = {}
        self.correlation_matrix: np.ndarray = np.array([])
        self.symbols: List[str] = []
        
        # API failure tracking
        self._api_failures: deque = deque(maxlen=1000)  # Rolling window
        self._last_hour_check: datetime = datetime.now()
        
        # Threading
        self._lock = threading.RLock()
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        
        # Callbacks
        self._protection_callbacks: List[Callable] = []
        self._alert_callbacks: List[Callable] = []
        
        # Load saved state
        self._load_state()
        
        logger.info(
            f"Capital Protection Engine initialized with "
            f"max_daily_loss={self.config.max_daily_loss_pct*100}%, "
            f"max_drawdown={self.config.max_drawdown_pct*100}%"
        )
    
    # =========================================================================
    # CORE PROTECTION CHECKS
    # =========================================================================
    
    def check_all_protections(
        self,
        portfolio_value: float,
        positions: Optional[Dict[str, PositionRisk]] = None,
        daily_pnl: float = 0.0
    ) -> Tuple[ProtectionLevel, List[TriggerType], List[str]]:
        """
        Run all protection checks and return current status.
        
        Returns:
            Tuple of (ProtectionLevel, active triggers, warnings)
        """
        with self._lock:
            triggers = []
            warnings = []
            
            # Update state
            self.state.current_capital = portfolio_value
            self.state.daily_pnl = daily_pnl
            self.state.daily_pnl_pct = daily_pnl / self.initial_capital
            
            # Update peak capital
            if portfolio_value > self.state.peak_capital:
                self.state.peak_capital = portfolio_value
            
            # Calculate drawdown
            self.state.current_drawdown = (
                (self.state.peak_capital - portfolio_value) / self.state.peak_capital
                if self.state.peak_capital > 0 else 0
            )
            
            # Update positions if provided
            if positions:
                self.positions = positions
            
            # Run all checks
            self._check_daily_loss(triggers, warnings)
            self._check_drawdown(triggers, warnings)
            self._check_position_limits(triggers, warnings)
            self._check_correlation_exposure(triggers, warnings)
            self._check_api_failures(triggers, warnings)
            self._check_leverage(triggers, warnings)
            
            # Determine protection level
            level = self._determine_level(triggers)
            
            # Update state
            self.state.level = level
            self.state.active_triggers = triggers
            self.state.warnings = warnings
            
            # Save state
            self._save_state()
            
            # Trigger callbacks if level changed
            if triggers:
                self._trigger_callbacks(level, triggers, warnings)
            
            return level, triggers, warnings
    
    def _check_daily_loss(
        self,
        triggers: List[TriggerType],
        warnings: List[str]
    ) -> None:
        """Check daily loss limit."""
        loss_pct = abs(min(self.state.daily_pnl_pct, 0))
        
        if loss_pct >= self.config.max_daily_loss_pct:
            triggers.append(TriggerType.DAILY_LOSS)
            logger.critical(
                f"DAILY LOSS LIMIT BREACHED: {loss_pct*100:.2f}% >= "
                f"{self.config.max_daily_loss_pct*100:.2f}%"
            )
        elif loss_pct >= self.config.daily_loss_warning_pct:
            warnings.append(
                f"Daily loss at {loss_pct*100:.2f}% - approaching limit"
            )
            logger.warning(f"Daily loss warning: {loss_pct*100:.2f}%")
    
    def _check_drawdown(
        self,
        triggers: List[TriggerType],
        warnings: List[str]
    ) -> None:
        """Check maximum drawdown."""
        if self.state.current_drawdown >= self.config.max_drawdown_pct:
            triggers.append(TriggerType.DRAWDOWN)
            logger.critical(
                f"DRAWDOWN LIMIT BREACHED: {self.state.current_drawdown*100:.2f}% >= "
                f"{self.config.max_drawdown_pct*100:.2f}%"
            )
        elif self.state.current_drawdown >= self.config.drawdown_warning_pct:
            warnings.append(
                f"Drawdown at {self.state.current_drawdown*100:.2f}% - approaching limit"
            )
            logger.warning(f"Drawdown warning: {self.state.current_drawdown*100:.2f}%")
    
    def _check_position_limits(
        self,
        triggers: List[TriggerType],
        warnings: List[str]
    ) -> None:
        """Check position size limits."""
        for symbol, pos in self.positions.items():
            if pos.weight_pct > self.config.max_single_position_pct:
                triggers.append(TriggerType.POSITION_SIZE)
                logger.critical(
                    f"POSITION SIZE LIMIT BREACHED: {symbol} at "
                    f"{pos.weight_pct*100:.2f}% > {self.config.max_single_position_pct*100:.2f}%"
                )
        
        # Check sector exposure
        sector_exposure: Dict[str, float] = defaultdict(float)
        for pos in self.positions.values():
            sector_exposure[pos.sector] += pos.weight_pct
        
        for sector, exposure in sector_exposure.items():
            if exposure > self.config.max_sector_exposure_pct:
                warnings.append(
                    f"Sector {sector} exposure at {exposure*100:.2f}% - over limit"
                )
    
    def _check_correlation_exposure(
        self,
        triggers: List[TriggerType],
        warnings: List[str]
    ) -> None:
        """Check correlation cluster exposure."""
        if not self.positions:
            return
        
        # Group by correlation cluster
        cluster_exposure: Dict[int, float] = defaultdict(float)
        for pos in self.positions.values():
            cluster_exposure[pos.correlation_cluster] += pos.weight_pct
        
        for cluster, exposure in cluster_exposure.items():
            if exposure > self.config.max_correlated_assets_pct:
                triggers.append(TriggerType.CORRELATION)
                logger.critical(
                    f"CORRELATION EXPOSURE BREACHED: Cluster {cluster} at "
                    f"{exposure*100:.2f}% > {self.config.max_correlated_assets_pct*100:.2f}%"
                )
    
    def _check_api_failures(
        self,
        triggers: List[TriggerType],
        warnings: List[str]
    ) -> None:
        """Check API failure rate."""
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        
        # Count failures in last hour
        recent_failures = sum(
            1 for f in self._api_failures if f > hour_ago
        )
        self.state.api_failures_hour = recent_failures
        
        if recent_failures >= self.config.max_api_failures_per_hour:
            triggers.append(TriggerType.API_FAILURE)
            logger.critical(
                f"API FAILURE LIMIT BREACHED: {recent_failures} failures in last hour"
            )
        elif recent_failures >= self.config.max_api_failures_per_hour * 0.7:
            warnings.append(
                f"API failures elevated: {recent_failures} in last hour"
            )
    
    def _check_leverage(
        self,
        triggers: List[TriggerType],
        warnings: List[str]
    ) -> None:
        """Check portfolio leverage."""
        if not self.positions:
            return
        
        total_leverage = sum(
            pos.leverage * pos.weight_pct for pos in self.positions.values()
        )
        
        if total_leverage > self.config.max_portfolio_leverage:
            triggers.append(TriggerType.LEVERAGE)
            logger.critical(
                f"LEVERAGE LIMIT BREACHED: {total_leverage:.2f}x > "
                f"{self.config.max_portfolio_leverage:.2f}x"
            )
        elif total_leverage > self.config.leverage_warning_level:
            warnings.append(
                f"Leverage elevated: {total_leverage:.2f}x"
            )
    
    def _determine_level(self, triggers: List[TriggerType]) -> ProtectionLevel:
        """Determine protection level based on triggers."""
        if not triggers:
            return ProtectionLevel.NORMAL
        
        # Critical triggers that cause immediate freeze
        critical_triggers = {
            TriggerType.DAILY_LOSS,
            TriggerType.DRAWDOWN,
            TriggerType.API_FAILURE,
            TriggerType.MANUAL
        }
        
        if any(t in critical_triggers for t in triggers):
            return ProtectionLevel.FROZEN
        
        # Other triggers cause restricted mode
        return ProtectionLevel.RESTRICTED
    
    # =========================================================================
    # API FAILURE TRACKING
    # =========================================================================
    
    def record_api_failure(self, api_name: str, error: str) -> None:
        """Record an API failure for monitoring."""
        with self._lock:
            self._api_failures.append(datetime.now())
            self.state.api_failures_day += 1
            
            logger.warning(
                f"API failure recorded: {api_name} - {error}. "
                f"Hourly: {self.state.api_failures_hour}, "
                f"Daily: {self.state.api_failures_day}"
            )
            
            # Check if we need to trigger protection
            if self.state.api_failures_day >= self.config.max_api_failures_per_day:
                self.activate_emergency_shutdown(
                    TriggerType.API_FAILURE,
                    f"Daily API failure limit reached: {self.state.api_failures_day}"
                )
    
    def reset_api_failure_counter(self) -> None:
        """Reset daily API failure counter (call at market open)."""
        with self._lock:
            self.state.api_failures_day = 0
            self.state.api_failures_hour = 0
            self._api_failures.clear()
            logger.info("API failure counters reset")
    
    # =========================================================================
    # EMERGENCY CONTROLS
    # =========================================================================
    
    def activate_emergency_shutdown(
        self,
        trigger: TriggerType,
        reason: str
    ) -> None:
        """
        Activate emergency shutdown.
        
        This immediately halts all trading and triggers liquidation
        if configured.
        """
        with self._lock:
            self.state.level = ProtectionLevel.FROZEN
            self.state.frozen_at = datetime.now()
            self.state.frozen_reason = reason
            self.state.active_triggers.append(trigger)
            
            logger.critical(
                f"EMERGENCY SHUTDOWN ACTIVATED: {trigger.value} - {reason}"
            )
            
            # Save state immediately
            self._save_state()
            
            # Trigger all callbacks
            self._trigger_callbacks(
                ProtectionLevel.FROZEN,
                [trigger],
                [reason]
            )
    
    def activate_manual_kill_switch(self, reason: str = "Manual activation") -> None:
        """Manually activate kill switch."""
        self.activate_emergency_shutdown(TriggerType.MANUAL, reason)
    
    def deactivate_kill_switch(self, force: bool = False) -> bool:
        """
        Deactivate kill switch.
        
        Args:
            force: Force deactivation even if manual reset required
            
        Returns:
            True if successfully deactivated
        """
        with self._lock:
            if self.config.manual_reset_required and not force:
                logger.warning(
                    "Cannot deactivate - manual reset required. Use force=True."
                )
                return False
            
            if TriggerType.MANUAL in self.state.active_triggers:
                self.state.active_triggers.remove(TriggerType.MANUAL)
            
            # Check if other triggers still active
            if not self.state.active_triggers:
                self.state.level = ProtectionLevel.NORMAL
                self.state.frozen_at = None
                self.state.frozen_reason = ""
                logger.info("Kill switch deactivated - system operational")
            else:
                self.state.level = ProtectionLevel.RESTRICTED
                logger.info(
                    f"Kill switch deactivated - system in restricted mode "
                    f"due to: {self.state.active_triggers}"
                )
            
            self._save_state()
            return True
    
    def request_trading_permission(self) -> Tuple[bool, str]:
        """
        Request permission to execute a trade.
        
        Returns:
            Tuple of (allowed, reason)
        """
        with self._lock:
            if self.state.level == ProtectionLevel.FROZEN:
                return False, f"Trading frozen: {self.state.frozen_reason}"
            
            if self.state.level == ProtectionLevel.LIQUIDATION:
                return False, "System in liquidation mode"
            
            if self.state.level == ProtectionLevel.RESTRICTED:
                # Check specific triggers
                if TriggerType.DAILY_LOSS in self.state.active_triggers:
                    return False, "Daily loss limit reached"
                
                if TriggerType.DRAWDOWN in self.state.active_triggers:
                    return False, "Drawdown limit reached"
                
                # Allow reducing positions but not adding
                return True, "Restricted mode - position reduction only"
            
            return True, "Trading allowed"
    
    # =========================================================================
    # MONITORING
    # =========================================================================
    
    def start_monitoring(self, interval_seconds: int = 10) -> None:
        """Start background monitoring thread."""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            logger.warning("Monitoring already running")
            return
        
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self._monitoring_thread.start()
        logger.info(f"Capital protection monitoring started (interval: {interval_seconds}s)")
    
    def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self._stop_monitoring.set()
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
        logger.info("Capital protection monitoring stopped")
    
    def _monitoring_loop(self, interval: int) -> None:
        """Background monitoring loop."""
        while not self._stop_monitoring.wait(interval):
            try:
                # Check for auto-recovery
                if self.state.level == ProtectionLevel.FROZEN:
                    self._check_recovery()
                
                # Check for daily reset
                now = datetime.now()
                if now.date() > self.state.last_reset_time.date():
                    self._daily_reset()
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
    
    def _check_recovery(self) -> None:
        """Check if system can recover from frozen state."""
        if not self.config.auto_recovery_enabled:
            return
        
        if not self.state.frozen_at:
            return
        
        # Check cooldown
        elapsed = (datetime.now() - self.state.frozen_at).total_seconds() / 60
        if elapsed < self.config.recovery_cooldown_minutes:
            return
        
        # Check if triggers are resolved
        if TriggerType.MANUAL in self.state.active_triggers:
            return  # Manual trigger requires manual reset
        
        if TriggerType.API_FAILURE in self.state.active_triggers:
            # Check if API failures have stopped
            if self.state.api_failures_hour > 0:
                return  # Still having failures
        
        # Attempt recovery
        logger.info("Attempting automatic recovery from frozen state")
        self.state.level = ProtectionLevel.RESTRICTED
        self.state.frozen_at = None
        self.state.active_triggers = [
            t for t in self.state.active_triggers
            if t in {TriggerType.DAILY_LOSS, TriggerType.DRAWDOWN}
        ]
        self._save_state()
    
    def _daily_reset(self) -> None:
        """Reset daily counters."""
        with self._lock:
            self.state.daily_pnl = 0.0
            self.state.daily_pnl_pct = 0.0
            self.state.api_failures_day = 0
            self.state.last_reset_time = datetime.now()
            
            # Reset daily loss trigger if active
            if TriggerType.DAILY_LOSS in self.state.active_triggers:
                self.state.active_triggers.remove(TriggerType.DAILY_LOSS)
            
            # Check if we can return to normal
            if self.state.level == ProtectionLevel.FROZEN:
                if TriggerType.DAILY_LOSS in self.state.active_triggers:
                    self.state.level = ProtectionLevel.RESTRICTED
            
            logger.info("Daily protection counters reset")
            self._save_state()
    
    # =========================================================================
    # CALLBACKS AND ALERTS
    # =========================================================================
    
    def register_protection_callback(self, callback: Callable) -> None:
        """Register a callback for protection events."""
        self._protection_callbacks.append(callback)
    
    def register_alert_callback(self, callback: Callable) -> None:
        """Register a callback for alerts."""
        self._alert_callbacks.append(callback)
    
    def _trigger_callbacks(
        self,
        level: ProtectionLevel,
        triggers: List[TriggerType],
        warnings: List[str]
    ) -> None:
        """Trigger all registered callbacks."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "level": level.value,
            "triggers": [t.value for t in triggers],
            "warnings": warnings,
            "state": {
                "daily_pnl_pct": self.state.daily_pnl_pct,
                "drawdown": self.state.current_drawdown,
                "api_failures_hour": self.state.api_failures_hour,
                "capital": self.state.current_capital
            }
        }
        
        for callback in self._protection_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in protection callback: {e}")
        
        # Send alerts
        for callback in self._alert_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    # =========================================================================
    # STATE PERSISTENCE
    # =========================================================================
    
    def _save_state(self) -> None:
        """Save state to file."""
        try:
            state_dict = {
                "level": self.state.level.value,
                "active_triggers": [t.value for t in self.state.active_triggers],
                "daily_pnl": self.state.daily_pnl,
                "daily_pnl_pct": self.state.daily_pnl_pct,
                "current_drawdown": self.state.current_drawdown,
                "peak_capital": self.state.peak_capital,
                "current_capital": self.state.current_capital,
                "api_failures_hour": self.state.api_failures_hour,
                "api_failures_day": self.state.api_failures_day,
                "last_reset_time": self.state.last_reset_time.isoformat(),
                "frozen_at": self.state.frozen_at.isoformat() if self.state.frozen_at else None,
                "frozen_reason": self.state.frozen_reason,
                "warnings": self.state.warnings
            }
            
            Path(self.state_file).parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, 'w') as f:
                json.dump(state_dict, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving state: {e}")
    
    def _load_state(self) -> None:
        """Load state from file."""
        try:
            if Path(self.state_file).exists():
                with open(self.state_file, 'r') as f:
                    state_dict = json.load(f)
                
                self.state.level = ProtectionLevel(state_dict.get("level", "normal"))
                self.state.active_triggers = [
                    TriggerType(t) for t in state_dict.get("active_triggers", [])
                ]
                self.state.daily_pnl = state_dict.get("daily_pnl", 0.0)
                self.state.daily_pnl_pct = state_dict.get("daily_pnl_pct", 0.0)
                self.state.current_drawdown = state_dict.get("current_drawdown", 0.0)
                self.state.peak_capital = state_dict.get("peak_capital", self.initial_capital)
                self.state.current_capital = state_dict.get("current_capital", self.initial_capital)
                self.state.api_failures_hour = state_dict.get("api_failures_hour", 0)
                self.state.api_failures_day = state_dict.get("api_failures_day", 0)
                
                if state_dict.get("last_reset_time"):
                    self.state.last_reset_time = datetime.fromisoformat(
                        state_dict["last_reset_time"]
                    )
                
                if state_dict.get("frozen_at"):
                    self.state.frozen_at = datetime.fromisoformat(
                        state_dict["frozen_at"]
                    )
                
                self.state.frozen_reason = state_dict.get("frozen_reason", "")
                self.state.warnings = state_dict.get("warnings", [])
                
                logger.info(f"Loaded protection state: level={self.state.level.value}")
                
        except Exception as e:
            logger.warning(f"Could not load state file: {e}")
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def get_status(self) -> Dict[str, Any]:
        """Get current protection status."""
        with self._lock:
            return {
                "level": self.state.level.value,
                "active_triggers": [t.value for t in self.state.active_triggers],
                "daily_pnl": self.state.daily_pnl,
                "daily_pnl_pct": self.state.daily_pnl_pct,
                "current_drawdown": self.state.current_drawdown,
                "peak_capital": self.state.peak_capital,
                "current_capital": self.state.current_capital,
                "api_failures_hour": self.state.api_failures_hour,
                "api_failures_day": self.state.api_failures_day,
                "frozen": self.state.level == ProtectionLevel.FROZEN,
                "frozen_reason": self.state.frozen_reason,
                "warnings": self.state.warnings,
                "config": {
                    "max_daily_loss_pct": self.config.max_daily_loss_pct,
                    "max_drawdown_pct": self.config.max_drawdown_pct,
                    "max_single_position_pct": self.config.max_single_position_pct,
                    "max_correlated_assets_pct": self.config.max_correlated_assets_pct,
                    "max_api_failures_per_hour": self.config.max_api_failures_per_hour
                }
            }
    
    def update_correlation_matrix(
        self,
        returns: np.ndarray,
        symbols: List[str]
    ) -> None:
        """Update correlation matrix for exposure monitoring."""
        with self._lock:
            self.symbols = symbols
            self.correlation_matrix = np.corrcoef(returns)
    
    def get_correlation_clusters(self) -> Dict[int, List[str]]:
        """Get assets grouped by correlation cluster."""
        if len(self.correlation_matrix) == 0:
            return {}
        
        clusters: Dict[int, List[str]] = {}
        assigned = set()
        cluster_id = 0
        
        for i, symbol in enumerate(self.symbols):
            if symbol in assigned:
                continue
            
            cluster_id += 1
            clusters[cluster_id] = [symbol]
            assigned.add(symbol)
            
            for j, other in enumerate(self.symbols):
                if other in assigned:
                    continue
                if self.correlation_matrix[i, j] > self.config.correlation_threshold:
                    clusters[cluster_id].append(other)
                    assigned.add(other)
        
        return clusters


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_default_protection_engine(
    initial_capital: float = 100000.0,
    max_daily_loss_pct: float = 0.03,
    max_drawdown_pct: float = 0.10
) -> CapitalProtectionEngine:
    """Create a protection engine with default settings."""
    config = ProtectionConfig(
        max_daily_loss_pct=max_daily_loss_pct,
        max_drawdown_pct=max_drawdown_pct
    )
    return CapitalProtectionEngine(
        config=config,
        initial_capital=initial_capital
    )
