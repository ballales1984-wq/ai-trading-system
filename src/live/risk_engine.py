"""
Advanced Risk Management Engine for Live Trading
================================================

This module provides professional-grade risk management features:
- Dynamic Stop Loss based on ATR/volatility
- Dynamic Take Profit
- Intelligent Trailing Stop
- Max Drawdown Protection (kill-switch)
- Advanced Position Monitoring
- Risk Event Logging

Author: Quantum AI Trading System
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class RiskEngine:
    """
    Professional Risk Management Engine for Live Trading
    
    Features:
    - ATR-based dynamic stop loss
    - ATR-based take profit
    - Intelligent trailing stop
    - Max drawdown kill-switch
    - Position risk monitoring
    """
    
    def __init__(
        self,
        max_drawdown: float = 0.20,
        sl_multiplier: float = 2.0,
        tp_multiplier: float = 3.0,
        trailing_multiplier: float = 1.5,
        enable_kill_switch: bool = True,
        trailing_start_threshold: float = 0.01  # 1% profit before trailing activates
    ):
        """
        Initialize Risk Engine.
        
        Args:
            max_drawdown: Maximum allowed drawdown (0.20 = 20%)
            sl_multiplier: ATR multiplier for stop loss
            tp_multiplier: ATR multiplier for take profit
            trailing_multiplier: ATR multiplier for trailing stop
            enable_kill_switch: Enable max drawdown protection
            trailing_start_threshold: Profit % before trailing stop activates
        """
        self.max_drawdown = max_drawdown
        self.sl_multiplier = sl_multiplier
        self.tp_multiplier = tp_multiplier
        self.trailing_multiplier = trailing_multiplier
        self.enable_kill_switch = enable_kill_switch
        self.trailing_start_threshold = trailing_start_threshold
        
        # State tracking
        self.highest_equity = None
        self.trailing_stops: Dict[str, float] = {}
        self.entry_prices: Dict[str, float] = {}
        self.position_directions: Dict[str, str] = {}  # 'long' or 'short'
        
        # Risk event log
        self.risk_events: List[Dict] = []
        
        logger.info(f"RiskEngine initialized: max_dd={max_drawdown}, "
                   f"sl={sl_multiplier}, tp={tp_multiplier}, ts={trailing_multiplier}")
    
    # =========================================================================
    # MAX DRAWDOWN PROTECTION
    # =========================================================================
    
    def check_max_drawdown(self, current_equity: float) -> Tuple[bool, float]:
        """
        Check if max drawdown has been breached.
        
        Args:
            current_equity: Current account equity
            
        Returns:
            Tuple of (should_kill, current_drawdown)
        """
        if not self.enable_kill_switch:
            return False, 0.0
        
        # Initialize highest equity on first call
        if self.highest_equity is None:
            self.highest_equity = current_equity
            return False, 0.0
        
        # Update highest equity
        self.highest_equity = max(self.highest_equity, current_equity)
        
        # Calculate drawdown
        if self.highest_equity > 0:
            drawdown = 1 - (current_equity / self.highest_equity)
        else:
            drawdown = 1.0
        
        # Check if kill-switch should trigger
        should_kill = drawdown >= self.max_drawdown
        
        if should_kill:
            self._log_risk_event(
                event_type="KILL_SWITCH",
                details=f"Max drawdown {drawdown:.2%} >= {self.max_drawdown:.2%}",
                equity=current_equity,
                drawdown=drawdown
            )
            logger.critical(f"KILL SWITCH TRIGGERED: Drawdown {drawdown:.2%}")
        
        return should_kill, drawdown
    
    def reset_drawdown_tracking(self, new_equity: float):
        """Reset drawdown tracking (e.g., after a deposit or reset)."""
        self.highest_equity = new_equity
        logger.info(f"Drawdown tracking reset to {new_equity}")
    
    # =========================================================================
    # STOP LOSS CALCULATION
    # =========================================================================
    
    def compute_stop_loss(
        self,
        entry_price: float,
        atr: float,
        direction: str = "long"
    ) -> float:
        """
        Compute dynamic stop loss based on ATR.
        
        Args:
            entry_price: Position entry price
            atr: Average True Range value
            direction: 'long' or 'short'
            
        Returns:
            Stop loss price
        """
        sl_distance = atr * self.sl_multiplier
        
        if direction.lower() == "long":
            return entry_price - sl_distance
        else:  # short
            return entry_price + sl_distance
    
    def compute_stop_loss_from_dataframe(
        self,
        df: pd.DataFrame,
        entry_price: float,
        direction: str = "long"
    ) -> float:
        """
        Compute stop loss from OHLCV dataframe.
        
        Args:
            df: OHLCV DataFrame with 'close', 'high', 'low'
            entry_price: Position entry price
            direction: 'long' or 'short'
            
        Returns:
            Stop loss price
        """
        # Calculate ATR if not present
        if 'atr_14' not in df.columns:
            high = df['high']
            low = df['low']
            close = df['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(14).mean().iloc[-1]
        else:
            atr = df['atr_14'].iloc[-1]
        
        # If ATR is NaN or 0, use percentage
        if pd.isna(atr) or atr == 0:
            atr = entry_price * 0.02  # 2% fallback
        
        return self.compute_stop_loss(entry_price, atr, direction)
    
    # =========================================================================
    # TAKE PROFIT CALCULATION
    # =========================================================================
    
    def compute_take_profit(
        self,
        entry_price: float,
        atr: float,
        direction: str = "long"
    ) -> float:
        """
        Compute dynamic take profit based on ATR.
        
        Args:
            entry_price: Position entry price
            atr: Average True Range value
            direction: 'long' or 'short'
            
        Returns:
            Take profit price
        """
        tp_distance = atr * self.tp_multiplier
        
        if direction.lower() == "long":
            return entry_price + tp_distance
        else:  # short
            return entry_price - tp_distance
    
    def compute_take_profit_from_dataframe(
        self,
        df: pd.DataFrame,
        entry_price: float,
        direction: str = "long"
    ) -> float:
        """
        Compute take profit from OHLCV dataframe.
        
        Args:
            df: OHLCV DataFrame
            entry_price: Position entry price
            direction: 'long' or 'short'
            
        Returns:
            Take profit price
        """
        # Calculate ATR if not present
        if 'atr_14' not in df.columns:
            high = df['high']
            low = df['low']
            close = df['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(14).mean().iloc[-1]
        else:
            atr = df['atr_14'].iloc[-1]
        
        # If ATR is NaN or 0, use percentage
        if pd.isna(atr) or atr == 0:
            atr = entry_price * 0.03  # 3% fallback
        
        return self.compute_take_profit(entry_price, atr, direction)
    
    # =========================================================================
    # TRAILING STOP
    # =========================================================================
    
    def init_trailing_stop(
        self,
        asset: str,
        entry_price: float,
        atr: float,
        direction: str = "long"
    ):
        """
        Initialize trailing stop for a new position.
        
        Args:
            asset: Trading symbol
            entry_price: Entry price
            atr: ATR value
            direction: 'long' or 'short'
        """
        initial_stop = self.compute_stop_loss(entry_price, atr, direction)
        self.trailing_stops[asset] = initial_stop
        self.entry_prices[asset] = entry_price
        self.position_directions[asset] = direction.lower()
        
        logger.info(f"Trailing stop initialized for {asset}: {initial_stop:.2f}")
    
    def update_trailing_stop(
        self,
        asset: str,
        current_price: float,
        atr: float
    ) -> float:
        """
        Update trailing stop based on current price.
        
        The trailing stop only moves in the favorable direction:
        - For long positions: moves up as price rises
        - For short positions: moves down as price falls
        
        Args:
            asset: Trading symbol
            current_price: Current market price
            atr: ATR value
            
        Returns:
            Updated trailing stop price
        """
        if asset not in self.trailing_stops:
            return current_price  # No trailing stop set
        
        direction = self.position_directions.get(asset, "long")
        entry_price = self.entry_prices.get(asset, current_price)
        
        # Calculate profit percentage
        if direction == "long":
            profit_pct = (current_price - entry_price) / entry_price
        else:
            profit_pct = (entry_price - current_price) / entry_price
        
        # Only activate trailing stop after threshold profit
        if profit_pct < self.trailing_start_threshold:
            return self.trailing_stops[asset]
        
        # Calculate new stop level
        ts_distance = atr * self.trailing_multiplier
        
        if direction == "long":
            # For longs, trailing stop moves UP
            new_stop = current_price - ts_distance
            # Only move up, never down
            self.trailing_stops[asset] = max(self.trailing_stops[asset], new_stop)
        else:
            # For shorts, trailing stop moves DOWN
            new_stop = current_price + ts_distance
            # Only move down, never up
            self.trailing_stops[asset] = min(self.trailing_stops[asset], new_stop)
        
        return self.trailing_stops[asset]
    
    def get_trailing_stop(self, asset: str) -> Optional[float]:
        """Get current trailing stop for an asset."""
        return self.trailing_stops.get(asset)
    
    def clear_trailing_stop(self, asset: str):
        """Clear trailing stop when position is closed."""
        if asset in self.trailing_stops:
            del self.trailing_stops[asset]
        if asset in self.entry_prices:
            del self.entry_prices[asset]
        if asset in self.position_directions:
            del self.position_directions[asset]
    
    # =========================================================================
    # EXIT SIGNAL CHECK
    # =========================================================================
    
    def check_exit_signal(
        self,
        asset: str,
        current_price: float,
        df: Optional[pd.DataFrame] = None,
        atr: Optional[float] = None
    ) -> Optional[str]:
        """
        Check if any exit condition is met.
        
        Args:
            asset: Trading symbol
            current_price: Current market price
            df: OHLCV DataFrame (optional)
            atr: ATR value (optional, calculated from df if not provided)
            
        Returns:
            Exit signal: 'STOP_LOSS', 'TAKE_PROFIT', 'TRAILING_STOP', or None
        """
        if asset not in self.entry_prices:
            return None
        
        entry_price = self.entry_prices[asset]
        direction = self.position_directions.get(asset, "long")
        
        # Get ATR
        if atr is None and df is not None:
            if 'atr_14' in df.columns:
                atr = df['atr_14'].iloc[-1]
            else:
                high = df['high']
                low = df['low']
                close = df['close']
                tr1 = high - low
                tr2 = abs(high - close.shift(1))
                tr3 = abs(low - close.shift(1))
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr = tr.rolling(14).mean().iloc[-1]
        
        if atr is None or pd.isna(atr):
            atr = entry_price * 0.02  # Fallback
        
        # Calculate exit levels
        sl = self.compute_stop_loss(entry_price, atr, direction)
        tp = self.compute_take_profit(entry_price, atr, direction)
        ts = self.update_trailing_stop(asset, current_price, atr)
        
        # Check exit conditions
        if direction == "long":
            if current_price <= sl:
                self._log_risk_event("STOP_LOSS", f"{asset} stop loss triggered", current_price)
                return "STOP_LOSS"
            if current_price >= tp:
                self._log_risk_event("TAKE_PROFIT", f"{asset} take profit triggered", current_price)
                return "TAKE_PROFIT"
            if current_price <= ts:
                self._log_risk_event("TRAILING_STOP", f"{asset} trailing stop triggered", current_price)
                return "TRAILING_STOP"
        else:  # short
            if current_price >= sl:
                self._log_risk_event("STOP_LOSS", f"{asset} stop loss triggered", current_price)
                return "STOP_LOSS"
            if current_price <= tp:
                self._log_risk_event("TAKE_PROFIT", f"{asset} take profit triggered", current_price)
                return "TAKE_PROFIT"
            if current_price >= ts:
                self._log_risk_event("TRAILING_STOP", f"{asset} trailing stop triggered", current_price)
                return "TRAILING_STOP"
        
        return None
    
    # =========================================================================
    # POSITION RISK MONITORING
    # =========================================================================
    
    def get_position_risk(
        self,
        asset: str,
        entry_price: float,
        current_price: float,
        quantity: float,
        direction: str = "long"
    ) -> Dict:
        """
        Get comprehensive risk metrics for a position.
        
        Args:
            asset: Trading symbol
            entry_price: Entry price
            current_price: Current market price
            quantity: Position size
            direction: 'long' or 'short'
            
        Returns:
            Dictionary with risk metrics
        """
        # Calculate P&L
        if direction.lower() == "long":
            pnl = (current_price - entry_price) * quantity
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl = (entry_price - current_price) * quantity
            pnl_pct = (entry_price - current_price) / entry_price
        
        # Get stop levels
        atr = abs(current_price * 0.02)  # Estimate ATR
        sl = self.compute_stop_loss(entry_price, atr, direction)
        tp = self.compute_take_profit(entry_price, atr, direction)
        
        # Distance to stops
        if direction.lower() == "long":
            dist_to_sl = (entry_price - sl) / entry_price
            dist_to_tp = (tp - entry_price) / entry_price
        else:
            dist_to_sl = (sl - entry_price) / entry_price
            dist_to_tp = (entry_price - tp) / entry_price
        
        # Risk-reward ratio
        rr_ratio = dist_to_tp / dist_to_sl if dist_to_sl > 0 else 0
        
        return {
            "asset": asset,
            "entry_price": entry_price,
            "current_price": current_price,
            "quantity": quantity,
            "direction": direction,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "stop_loss": sl,
            "take_profit": tp,
            "trailing_stop": self.get_trailing_stop(asset),
            "distance_to_sl": dist_to_sl,
            "distance_to_tp": dist_to_tp,
            "risk_reward_ratio": rr_ratio
        }
    
    # =========================================================================
    # RISK EVENT LOGGING
    # =========================================================================
    
    def _log_risk_event(
        self,
        event_type: str,
        details: str,
        price: float = None,
        equity: float = None,
        drawdown: float = None
    ):
        """Log a risk event."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "details": details,
            "price": price,
            "equity": equity,
            "drawdown": drawdown
        }
        self.risk_events.append(event)
        logger.warning(f"RISK EVENT: {event_type} - {details}")
    
    def get_risk_events(self, limit: int = 50) -> List[Dict]:
        """Get recent risk events."""
        return self.risk_events[-limit:]
    
    def clear_risk_events(self):
        """Clear risk event log."""
        self.risk_events = []
    
    # =========================================================================
    # PORTFOLIO-LEVEL RISK
    # =========================================================================
    
    def check_portfolio_risk(
        self,
        positions: Dict[str, Dict],
        current_prices: Dict[str, float],
        total_equity: float
    ) -> Dict:
        """
        Check overall portfolio risk.
        
        Args:
            positions: Dictionary of open positions
            current_prices: Current prices for all assets
            total_equity: Total account equity
            
        Returns:
            Portfolio risk metrics
        """
        total_exposure = 0
        total_pnl = 0
        max_risk = 0
        
        for asset, pos in positions.items():
            if asset not in current_prices:
                continue
            
            current_price = current_prices[asset]
            entry_price = pos.get('entry_price', current_price)
            quantity = pos.get('quantity', 0)
            direction = pos.get('direction', 'long')
            
            # Calculate position value
            position_value = current_price * quantity
            total_exposure += position_value
            
            # Calculate P&L
            if direction == "long":
                pnl = (current_price - entry_price) * quantity
            else:
                pnl = (entry_price - current_price) * quantity
            
            total_pnl += pnl
            
            # Track max risk per position
            risk_pct = abs(pnl / total_equity) if total_equity > 0 else 0
            max_risk = max(max_risk, risk_pct)
        
        # Check drawdown
        should_kill, drawdown = self.check_max_drawdown(total_equity)
        
        return {
            "total_equity": total_equity,
            "total_exposure": total_exposure,
            "total_pnl": total_pnl,
            "max_position_risk": max_risk,
            "current_drawdown": drawdown,
            "kill_switch_triggered": should_kill,
            "num_positions": len(positions)
        }


class RiskManager:
    """
    High-level Risk Manager that combines RiskEngine with position management.
    """
    
    def __init__(self, config: Dict = None):
        """Initialize Risk Manager with configuration."""
        config = config or {}
        
        self.engine = RiskEngine(
            max_drawdown=config.get('max_drawdown', 0.20),
            sl_multiplier=config.get('sl_multiplier', 2.0),
            tp_multiplier=config.get('tp_multiplier', 3.0),
            trailing_multiplier=config.get('trailing_multiplier', 1.5),
            trailing_start_threshold=config.get('trailing_threshold', 0.01)
        )
        
        self.positions = {}
        self.prices = {}
        
        logger.info("RiskManager initialized")
    
    def open_position(
        self,
        asset: str,
        direction: str,
        entry_price: float,
        quantity: float,
        atr: float = None
    ):
        """Open a new position with risk management."""
        self.positions[asset] = {
            'direction': direction,
            'entry_price': entry_price,
            'quantity': quantity,
            'atr': atr or (entry_price * 0.02)
        }
        
        # Initialize trailing stop
        self.engine.init_trailing_stop(asset, entry_price, self.positions[asset]['atr'], direction)
        
        logger.info(f"Position opened: {asset} {direction} {quantity} @ {entry_price}")
    
    def close_position(self, asset: str):
        """Close an existing position."""
        if asset in self.positions:
            del self.positions[asset]
            self.engine.clear_trailing_stop(asset)
            logger.info(f"Position closed: {asset}")
    
    def update_price(self, asset: str, price: float):
        """Update current price for an asset."""
        self.prices[asset] = price
    
    def check_exits(self) -> List[Tuple[str, str]]:
        """
        Check for exit signals on all positions.
        
        Returns:
            List of (asset, exit_reason) tuples
        """
        exits = []
        
        for asset, pos in list(self.positions.items()):
            if asset not in self.prices:
                continue
            
            current_price = self.prices[asset]
            exit_reason = self.engine.check_exit_signal(
                asset, 
                current_price,
                atr=pos.get('atr')
            )
            
            if exit_reason:
                exits.append((asset, exit_reason))
        
        return exits
    
    def get_portfolio_status(self) -> Dict:
        """Get overall portfolio risk status."""
        return self.engine.check_portfolio_risk(
            self.positions,
            self.prices,
            sum(p['quantity'] * self.prices.get(a, 0) for a, p in self.positions.items())
        )
