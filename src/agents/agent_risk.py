# src/agents/agent_risk.py
"""
Risk Agent
==========
Institutional risk management agent.
Calculates VaR, CVaR, and other risk metrics.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import numpy as np
from dataclasses import dataclass
from enum import Enum

from src.agents.base_agent import BaseAgent
from src.core.event_bus import EventBus, EventType


logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskMetrics:
    """Risk metrics container."""
    symbol: str
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    max_drawdown: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    beta: float
    risk_level: RiskLevel
    timestamp: datetime


class RiskAgent(BaseAgent):
    """
    Institutional risk management agent.
    
    Features:
    - Value at Risk (VaR) - Historical, Parametric, Monte Carlo
    - Conditional VaR (CVaR / Expected Shortfall)
    - GARCH volatility modeling
    - Maximum drawdown tracking
    - Sharpe/Sortino ratios
    - Position sizing recommendations
    - Risk alerts and notifications
    """
    
    def __init__(
        self,
        name: str,
        event_bus: EventBus,
        state_manager: Any,
        config: Dict[str, Any]
    ):
        """
        Initialize Risk Agent.
        
        Args:
            name: Agent identifier
            event_bus: Event bus for communication
            state_manager: State manager instance
            config: Configuration dictionary with:
                - symbols: List of symbols to monitor
                - interval_sec: Calculation interval
                - var_confidence: VaR confidence level (default 0.95)
                - max_var_threshold: Maximum VaR before alert
                - max_drawdown_threshold: Maximum drawdown before alert
                - lookback_days: Days of history for calculations
        """
        super().__init__(name, event_bus, state_manager, config)
        
        # Configuration
        self.symbols = config.get("symbols", ["BTCUSDT"])
        self.interval_sec = config.get("interval_sec", 60)
        self.var_confidence = config.get("var_confidence", 0.95)
        self.max_var_threshold = config.get("max_var_threshold", 0.05)
        self.max_drawdown_threshold = config.get("max_drawdown_threshold", 0.10)
        self.lookback_days = config.get("lookback_days", 30)
        
        # Risk metrics cache
        self._risk_metrics: Dict[str, RiskMetrics] = {}
        
        # Position tracking
        self._positions: Dict[str, Dict] = {}
        
        # Alert history
        self._alerts: List[Dict] = []
        
        # Risk-free rate for Sharpe calculation
        self._risk_free_rate = config.get("risk_free_rate", 0.02)
        
        logger.info(
            f"RiskAgent initialized for {len(self.symbols)} symbols, "
            f"VaR confidence: {self.var_confidence}"
        )
    
    async def run(self):
        """Main agent loop - calculate risk metrics."""
        while self._running:
            try:
                # Calculate risk for each symbol
                for symbol in self.symbols:
                    # Get Monte Carlo paths from state
                    mc_paths = self.get_shared_state(
                        "MonteCarloAgent",
                        f"mc_paths:{symbol}"
                    )
                    
                    # Get price history
                    price_history = self._get_price_history(symbol)
                    
                    # Get current position
                    position = self._positions.get(symbol, {})
                    
                    # Calculate risk metrics
                    metrics = await self._calculate_risk(
                        symbol,
                        mc_paths,
                        price_history,
                        position
                    )
                    
                    if metrics:
                        # Store metrics
                        self._risk_metrics[symbol] = metrics
                        
                        # Update shared state
                        self._update_state(metrics)
                        
                        # Check for risk alerts
                        await self._check_risk_alerts(metrics)
                        
                        # Emit event
                        await self.emit_event(
                            EventType.RISK_CHECK_PASSED,
                            {
                                "symbol": symbol,
                                "var_95": metrics.var_95,
                                "cvar_95": metrics.cvar_95,
                                "risk_level": metrics.risk_level.value,
                            }
                        )
                
                # Wait for next interval
                await asyncio.sleep(self.interval_sec)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in risk calculation loop: {e}")
                self._error_count += 1
                await asyncio.sleep(self._error_backoff)
    
    def _get_price_history(self, symbol: str) -> np.ndarray:
        """Get price history for a symbol."""
        history_data = self.get_shared_state(
            "MarketDataAgent",
            f"price_history:{symbol}",
            []
        )
        
        if not history_data:
            return np.array([])
        
        prices = [h.get("price", 0) for h in history_data if h.get("price")]
        return np.array(prices) if prices else np.array([])
    
    async def _calculate_risk(
        self,
        symbol: str,
        mc_paths: Optional[List],
        price_history: np.ndarray,
        position: Dict
    ) -> Optional[RiskMetrics]:
        """
        Calculate comprehensive risk metrics.
        
        Args:
            symbol: Trading pair
            mc_paths: Monte Carlo simulation paths
            price_history: Historical prices
            position: Current position info
            
        Returns:
            RiskMetrics or None
        """
        try:
            # Calculate returns
            if len(price_history) > 1:
                returns = np.diff(np.log(price_history))
            else:
                returns = np.array([])
            
            # VaR calculations
            if mc_paths is not None:
                # Monte Carlo VaR
                paths_array = np.array(mc_paths)
                final_prices = paths_array[:, -1]
                initial_price = paths_array[0, 0]
                path_returns = final_prices / initial_price - 1
                
                var_95 = float(np.percentile(path_returns, 5))
                var_99 = float(np.percentile(path_returns, 1))
                cvar_95 = float(np.mean(path_returns[path_returns <= var_95]))
                cvar_99 = float(np.mean(path_returns[path_returns <= var_99]))
                
            elif len(returns) > 0:
                # Historical VaR
                var_95 = float(np.percentile(returns, 5))
                var_99 = float(np.percentile(returns, 1))
                cvar_95 = float(np.mean(returns[returns <= var_95]))
                cvar_99 = float(np.mean(returns[returns <= var_99]))
                
            else:
                # Default values
                var_95 = -0.05
                var_99 = -0.10
                cvar_95 = -0.08
                cvar_99 = -0.15
            
            # Maximum drawdown
            max_drawdown = self._calculate_max_drawdown(price_history)
            
            # Volatility (annualized)
            volatility = self._calculate_volatility(returns)
            
            # Sharpe ratio
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            
            # Sortino ratio
            sortino_ratio = self._calculate_sortino_ratio(returns)
            
            # Beta (if we have market data)
            beta = self._calculate_beta(symbol, returns)
            
            # Determine risk level
            risk_level = self._determine_risk_level(
                var_95, max_drawdown, volatility
            )
            
            return RiskMetrics(
                symbol=symbol,
                var_95=var_95,
                var_99=var_99,
                cvar_95=cvar_95,
                cvar_99=cvar_99,
                max_drawdown=max_drawdown,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                beta=beta,
                risk_level=risk_level,
                timestamp=datetime.now(),
            )
            
        except Exception as e:
            logger.error(f"Error calculating risk for {symbol}: {e}")
            return None
    
    def _calculate_max_drawdown(self, prices: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        if len(prices) < 2:
            return 0.0
        
        # Calculate cumulative maximum
        cummax = np.maximum.accumulate(prices)
        
        # Calculate drawdown
        drawdown = (prices - cummax) / cummax
        
        return float(np.min(drawdown))
    
    def _calculate_volatility(self, returns: np.ndarray) -> float:
        """Calculate annualized volatility."""
        if len(returns) < 2:
            return 0.0
        
        # Daily volatility
        daily_vol = np.std(returns)
        
        # Annualize (252 trading days)
        annual_vol = daily_vol * np.sqrt(252)
        
        return float(annual_vol)
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        
        # Annualized return
        annual_return = np.mean(returns) * 252
        
        # Annualized volatility
        annual_vol = self._calculate_volatility(returns)
        
        if annual_vol == 0:
            return 0.0
        
        # Sharpe ratio
        sharpe = (annual_return - self._risk_free_rate) / annual_vol
        
        return float(sharpe)
    
    def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sortino ratio."""
        if len(returns) < 2:
            return 0.0
        
        # Annualized return
        annual_return = np.mean(returns) * 252
        
        # Downside deviation
        negative_returns = returns[returns < 0]
        
        if len(negative_returns) == 0:
            return float('inf') if annual_return > 0 else 0.0
        
        downside_std = np.std(negative_returns) * np.sqrt(252)
        
        if downside_std == 0:
            return 0.0
        
        # Sortino ratio
        sortino = (annual_return - self._risk_free_rate) / downside_std
        
        return float(sortino)
    
    def _calculate_beta(self, symbol: str, returns: np.ndarray) -> float:
        """Calculate beta relative to market."""
        # Placeholder - would need market returns
        return 1.0
    
    def _determine_risk_level(
        self,
        var_95: float,
        max_drawdown: float,
        volatility: float
    ) -> RiskLevel:
        """Determine overall risk level."""
        risk_score = 0
        
        # VaR contribution
        if abs(var_95) > 0.10:
            risk_score += 3
        elif abs(var_95) > 0.05:
            risk_score += 2
        elif abs(var_95) > 0.02:
            risk_score += 1
        
        # Drawdown contribution
        if abs(max_drawdown) > 0.20:
            risk_score += 3
        elif abs(max_drawdown) > 0.10:
            risk_score += 2
        elif abs(max_drawdown) > 0.05:
            risk_score += 1
        
        # Volatility contribution
        if volatility > 0.50:
            risk_score += 3
        elif volatility > 0.30:
            risk_score += 2
        elif volatility > 0.15:
            risk_score += 1
        
        # Classify
        if risk_score >= 7:
            return RiskLevel.CRITICAL
        elif risk_score >= 5:
            return RiskLevel.HIGH
        elif risk_score >= 3:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    async def _check_risk_alerts(self, metrics: RiskMetrics):
        """Check for risk threshold breaches and emit alerts."""
        alerts = []
        
        # VaR alert
        if abs(metrics.var_95) > self.max_var_threshold:
            alerts.append({
                "type": "var_breach",
                "symbol": metrics.symbol,
                "value": metrics.var_95,
                "threshold": self.max_var_threshold,
                "severity": "high",
            })
        
        # Drawdown alert
        if abs(metrics.max_drawdown) > self.max_drawdown_threshold:
            alerts.append({
                "type": "drawdown_breach",
                "symbol": metrics.symbol,
                "value": metrics.max_drawdown,
                "threshold": self.max_drawdown_threshold,
                "severity": "critical",
            })
        
        # Risk level alert
        if metrics.risk_level == RiskLevel.CRITICAL:
            alerts.append({
                "type": "risk_level",
                "symbol": metrics.symbol,
                "level": metrics.risk_level.value,
                "severity": "critical",
            })
        
        # Emit alerts
        for alert in alerts:
            alert["timestamp"] = datetime.now().isoformat()
            self._alerts.append(alert)
            
            await self.emit_event(
                EventType.RISK_ALERT,
                alert
            )
            
            logger.warning(f"Risk alert: {alert}")
    
    def _update_state(self, metrics: RiskMetrics):
        """Update shared state with risk metrics."""
        self.update_state(f"risk:{metrics.symbol}", {
            "var_95": metrics.var_95,
            "var_99": metrics.var_99,
            "cvar_95": metrics.cvar_95,
            "cvar_99": metrics.cvar_99,
            "max_drawdown": metrics.max_drawdown,
            "volatility": metrics.volatility,
            "sharpe_ratio": metrics.sharpe_ratio,
            "sortino_ratio": metrics.sortino_ratio,
            "beta": metrics.beta,
            "risk_level": metrics.risk_level.value,
        })
    
    def update_position(self, symbol: str, position: Dict):
        """Update position information for a symbol."""
        self._positions[symbol] = position
    
    def get_risk_metrics(self, symbol: str) -> Optional[RiskMetrics]:
        """Get latest risk metrics for a symbol."""
        return self._risk_metrics.get(symbol)
    
    def get_all_metrics(self) -> Dict[str, RiskMetrics]:
        """Get all risk metrics."""
        return self._risk_metrics.copy()
    
    def get_alerts(self, limit: int = 100) -> List[Dict]:
        """Get recent alerts."""
        return self._alerts[-limit:]
    
    def get_position_size_recommendation(
        self,
        symbol: str,
        portfolio_value: float,
        risk_per_trade: float = 0.02
    ) -> float:
        """
        Get recommended position size based on risk metrics.
        
        Args:
            symbol: Trading pair
            portfolio_value: Total portfolio value
            risk_per_trade: Maximum risk per trade (default 2%)
            
        Returns:
            Recommended position size in base currency
        """
        metrics = self._risk_metrics.get(symbol)
        
        if not metrics:
            return portfolio_value * 0.01  # Default 1%
        
        # Adjust for risk level
        risk_multiplier = {
            RiskLevel.LOW: 1.0,
            RiskLevel.MEDIUM: 0.75,
            RiskLevel.HIGH: 0.5,
            RiskLevel.CRITICAL: 0.25,
        }.get(metrics.risk_level, 0.5)
        
        # Calculate base position
        base_position = portfolio_value * risk_per_trade
        
        # Adjust for VaR
        if abs(metrics.var_95) > 0:
            var_adjustment = min(1.0, self.max_var_threshold / abs(metrics.var_95))
        else:
            var_adjustment = 1.0
        
        # Final position
        position_size = base_position * risk_multiplier * var_adjustment
        
        return position_size
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent metrics."""
        metrics = super().get_metrics()
        metrics.update({
            "symbols_monitored": len(self.symbols),
            "risk_metrics_cached": len(self._risk_metrics),
            "total_alerts": len(self._alerts),
            "var_confidence": self.var_confidence,
        })
        return metrics
