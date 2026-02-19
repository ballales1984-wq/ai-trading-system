# src/core/engine.py
"""
Trading Engine - Orchestrator
============================
Central orchestrator that coordinates all trading components.
Event-driven architecture for professional trading.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
from pathlib import Path

from src.core.event_bus import (
    EventBus, EventType, Event, EventHandler,
    create_event, SignalEventHandler, RiskEventHandler
)
from src.core.state_manager import StateManager


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EngineState(Enum):
    """Engine states."""
    STOPPED = "stopped"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"


class TradingMode(Enum):
    """Trading modes."""
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"


@dataclass
class EngineConfig:
    """Engine configuration."""
    mode: TradingMode = TradingMode.PAPER
    initial_balance: float = 100000
    
    # Risk limits
    max_position_pct: float = 0.3  # 30% max position
    max_order_pct: float = 0.1  # 10% max per order
    max_daily_loss_pct: float = 0.05  # 5% max daily loss
    max_drawdown_pct: float = 0.15  # 15% max drawdown
    
    # Execution
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # State
    state_save_interval: int = 60  # seconds
    snapshot_interval: int = 300  # seconds
    
    # Monitoring
    log_level: str = "INFO"


class TradingEngine:
    """
    Central trading engine orchestrator.
    Coordinates all components via event bus.
    """
    
    def __init__(self, config: EngineConfig):
        """
        Initialize trading engine.
        
        Args:
            config: Engine configuration
        """
        self.config = config
        self.state = EngineState.STOPPED
        
        # Core components
        self.event_bus = EventBus()
        self.state_manager = StateManager()
        
        # Component references (set externally)
        self.broker = None
        self.risk_manager = None
        self.signal_generator = None
        self.portfolio_manager = None
        
        # Background tasks
        self._tasks: List[asyncio.Task] = []
        
        # Statistics
        self.stats = {
            'started_at': None,
            'signals_processed': 0,
            'orders_placed': 0,
            'orders_filled': 0,
            'orders_rejected': 0,
            'total_pnl': 0.0,
            'emergency_stops': 0
        }
        
        # Callbacks
        self.on_signal: Optional[Callable] = None
        self.on_order: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
        self.on_risk_alert: Optional[Callable] = None
        
        # Setup logging
        self._setup_logging()
        
        # Subscribe to events
        self._subscribe_events()
        
        logger.info("Trading engine initialized")
    
    def _setup_logging(self):
        """Setup file logging."""
        log_dir = Path("logs/engine")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # File handler
        file_handler = logging.FileHandler(
            log_dir / f"engine_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(file_handler)
    
    def _subscribe_events(self):
        """Subscribe to system events."""
        # Subscribe to signal events
        self.event_bus.subscribe(
            EventType.SIGNAL_GENERATED,
            SignalEventHandler(self._handle_signal)
        )
        
        # Subscribe to risk events
        self.event_bus.subscribe(
            EventType.RISK_ALERT,
            RiskEventHandler(
                on_alert=self._handle_risk_alert,
                on_emergency=self._handle_emergency
            )
        )
        
        logger.debug("Event subscriptions registered")
    
    async def start(self) -> bool:
        """
        Start the trading engine.
        
        Returns:
            True if started successfully
        """
        if self.state != EngineState.STOPPED:
            logger.warning(f"Cannot start from state: {self.state}")
            return False
        
        self.state = EngineState.INITIALIZING
        logger.info("Starting trading engine...")
        
        try:
            # Restore previous state if exists
            await self._restore_state()
            
            # Connect to broker
            if self.broker:
                if not await self.broker.connect():
                    raise ConnectionError("Failed to connect to broker")
            
            # Start background tasks
            await self._start_tasks()
            
            # Publish engine started event
            await self.event_bus.publish(create_event(
                EventType.ENGINE_STARTED,
                {'mode': self.config.mode.value},
                'engine'
            ))
            
            self.stats['started_at'] = datetime.now()
            self.state = EngineState.RUNNING
            
            logger.info(f"Trading engine started in {self.config.mode.value} mode")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start engine: {e}")
            self.state = EngineState.ERROR
            
            await self.event_bus.publish(create_event(
                EventType.ENGINE_ERROR,
                {'error': str(e)},
                'engine'
            ))
            
            if self.on_error:
                await self.on_error({'type': 'start_failed', 'error': str(e)})
            
            return False
    
    async def stop(self, close_positions: bool = True) -> bool:
        """
        Stop the trading engine.
        
        Args:
            close_positions: Whether to close open positions
            
        Returns:
            True if stopped successfully
        """
        if self.state != EngineState.RUNNING:
            return False
        
        self.state = EngineState.STOPPING
        logger.info("Stopping trading engine...")
        
        try:
            # Cancel background tasks
            for task in self._tasks:
                task.cancel()
            self._tasks.clear()
            
            # Close positions if requested
            if close_positions and self.broker:
                logger.info("Closing all positions...")
                # Close all positions
                positions = await self.broker.get_positions()
                for pos in positions:
                    if pos.quantity != 0:
                        side = 'SELL' if pos.quantity > 0 else 'BUY'
                        # Create closing order
                        pass
            
            # Save final state
            await self._save_state()
            
            # Disconnect broker
            if self.broker:
                await self.broker.disconnect()
            
            # Publish stopped event
            await self.event_bus.publish(create_event(
                EventType.ENGINE_STOPPED,
                {'stats': self.stats},
                'engine'
            ))
            
            self.state = EngineState.STOPPED
            logger.info("Trading engine stopped")
            
            return True
            
        except Exception as e:
            logger.error(f"Error stopping engine: {e}")
            self.state = EngineState.ERROR
            return False
    
    async def _start_tasks(self):
        """Start background tasks."""
        # Market data processor
        task = asyncio.create_task(self._market_data_loop())
        self._tasks.append(task)
        
        # State saver
        task = asyncio.create_task(self._state_save_loop())
        self._tasks.append(task)
        
        # Health checker
        task = asyncio.create_task(self._health_check_loop())
        self._tasks.append(task)
        
        logger.info("Background tasks started")
    
    async def _market_data_loop(self):
        """Process market data and generate signals."""
        while self.state == EngineState.RUNNING:
            try:
                if self.signal_generator:
                    # Get market data
                    # Generate signals
                    # Publish signal events
                    pass
                
                await asyncio.sleep(1)  # Process every second
                
            except Exception as e:
                logger.error(f"Market data loop error: {e}")
    
    async def _state_save_loop(self):
        """Periodic state saving."""
        while self.state == EngineState.RUNNING:
            try:
                await asyncio.sleep(self.config.state_save_interval)
                await self._save_state()
                
            except Exception as e:
                logger.error(f"State save error: {e}")
    
    async def _health_check_loop(self):
        """Periodic health checks."""
        while self.state == EngineState.RUNNING:
            try:
                # Check broker connection
                if self.broker and not self.broker.is_connected:
                    logger.error("Broker disconnected!")
                    await self.stop(close_positions=True)
                    break
                
                # Check risk limits
                if self.risk_manager:
                    should_exit, reason = await self._check_risk_limits()
                    
                    if should_exit:
                        logger.critical(f"Risk limit exceeded: {reason}")
                        await self.stop(close_positions=True)
                        break
                
                # Log status periodically
                if self.broker:
                    balance = await self.broker.get_balance()
                    logger.info(
                        f"Health - Equity: {balance.total_equity:.2f}, "
                        f"PnL: {self.stats['total_pnl']:.2f}"
                    )
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Health check error: {e}")
    
    async def _check_risk_limits(self) -> tuple[bool, str]:
        """Check risk limits."""
        if not self.broker:
            return False, ""
        
        try:
            balance = await self.broker.get_balance()
            initial = self.config.initial_balance
            
            # Check daily loss
            daily_pnl = balance.realized_pnl + balance.unrealized_pnl
            daily_loss_pct = (initial - balance.total_equity) / initial
            
            if daily_loss_pct >= self.config.max_daily_loss_pct:
                return True, f"Daily loss {daily_loss_pct:.2%} exceeded"
            
            # Check drawdown
            drawdown_pct = (self.stats.get('equity_high', initial) - balance.total_equity) / initial
            
            if drawdown_pct >= self.config.max_drawdown_pct:
                return True, f"Drawdown {drawdown_pct:.2%} exceeded"
            
            return False, ""
            
        except Exception as e:
            logger.error(f"Risk check error: {e}")
            return False, ""
    
    async def _save_state(self):
        """Save current state."""
        if not self.broker:
            return
        
        try:
            # Get current state
            balance = await self.broker.get_balance()
            positions = await self.broker.get_positions()
            
            # Save portfolio state
            from src.core.state_manager import PortfolioState, PositionState
            self.state_manager.save_portfolio_state(PortfolioState(
                total_equity=balance.total_equity,
                available_balance=balance.available_balance,
                unrealized_pnl=balance.unrealized_pnl,
                realized_pnl=balance.realized_pnl
            ))
            
            # Save positions
            for pos in positions:
                self.state_manager.save_position(PositionState(
                    symbol=pos.symbol,
                    quantity=pos.quantity,
                    entry_price=pos.entry_price,
                    current_price=pos.current_price,
                    unrealized_pnl=pos.unrealized_pnl,
                    realized_pnl=pos.realized_pnl,
                    commission=pos.commission,
                    leverage=pos.leverage
                ))
            
            # Log event
            self.state_manager.log_event('state_saved', {
                'equity': balance.total_equity,
                'positions': len(positions)
            })
            
            # Publish event
            await self.event_bus.publish(create_event(
                EventType.STATE_SAVED,
                {'equity': balance.total_equity},
                'engine'
            ))
            
            logger.debug("State saved")
            
        except Exception as e:
            logger.error(f"Save state error: {e}")
    
    async def _restore_state(self):
        """Restore state from database."""
        try:
            state = self.state_manager.restore_from_snapshot()
            
            if state['positions']:
                logger.info(f"Restored {len(state['positions'])} positions")
            
            # Subscribe to state saved event
            self.event_bus.subscribe(
                EventType.STATE_SAVED,
                lambda e: logger.debug(f"Auto-saved: {e.data}")
            )
            
        except Exception as e:
            logger.warning(f"Could not restore state: {e}")
    
    async def _handle_signal(self, data: Dict):
        """Handle incoming trading signal."""
        try:
            symbol = data.get('symbol')
            action = data.get('action')
            quantity = data.get('quantity')
            confidence = data.get('confidence', 0.5)
            
            self.stats['signals_processed'] += 1
            
            logger.info(f"Processing signal: {symbol} {action} conf={confidence:.2f}")
            
            # Check with risk manager
            if self.risk_manager:
                is_valid, reason = await self.risk_manager.check_signal(data)
                
                if not is_valid:
                    logger.warning(f"Signal rejected by risk: {reason}")
                    
                    await self.event_bus.publish(create_event(
                        EventType.SIGNAL_REJECTED,
                        {'symbol': symbol, 'reason': reason},
                        'engine'
                    ))
                    
                    self.stats['orders_rejected'] += 1
                    return
            
            # Execute order
            if self.broker and action in ['BUY', 'SELL']:
                # Place order through broker
                # This would call self.broker.place_order()
                pass
            
            # Publish executed event
            await self.event_bus.publish(create_event(
                EventType.SIGNAL_EXECUTED,
                data,
                'engine'
            ))
            
        except Exception as e:
            logger.error(f"Error handling signal: {e}")
    
    async def _handle_risk_alert(self, data: Dict):
        """Handle risk alert."""
        logger.warning(f"Risk alert: {data}")
        
        if self.on_risk_alert:
            await self.on_risk_alert(data)
    
    async def _handle_emergency(self, data: Dict):
        """Handle emergency situation."""
        logger.critical(f"EMERGENCY: {data}")
        
        self.stats['emergency_stops'] += 1
        
        # Stop engine with position closing
        await self.stop(close_positions=True)
    
    async def process_signal(self, signal: Dict) -> Dict:
        """
        Process a trading signal through the engine.
        
        Args:
            signal: Signal dictionary
            
        Returns:
            Result dictionary
        """
        if self.state != EngineState.RUNNING:
            return {'success': False, 'error': 'Engine not running'}
        
        # Publish signal event
        await self.event_bus.publish(create_event(
            EventType.SIGNAL_GENERATED,
            signal,
            'external'
        ))
        
        return {'success': True, 'status': 'signal_processed'}
    
    async def get_status(self) -> Dict:
        """Get engine status."""
        balance = None
        positions = []
        
        if self.broker:
            try:
                balance = await self.broker.get_balance()
                positions = await self.broker.get_positions()
            except:
                pass
        
        return {
            'state': self.state.value,
            'mode': self.config.mode.value,
            'balance': balance.to_dict() if balance else None,
            'positions': len(positions),
            'stats': self.stats,
            'event_stats': self.event_bus.get_event_stats()
        }
    
    def get_event_history(self, limit: int = 100) -> List[Dict]:
        """Get event history."""
        events = self.event_bus.get_event_history(limit=limit)
        return [e.to_dict() for e in events]


async def create_engine(
    mode: str = 'paper',
    initial_balance: float = 100000,
    **kwargs
) -> TradingEngine:
    """
    Factory function to create trading engine.
    
    Args:
        mode: Trading mode ('paper', 'live')
        initial_balance: Initial balance
        **kwargs: Additional config parameters
        
    Returns:
        TradingEngine instance
    """
    config = EngineConfig(
        mode=TradingMode[mode.upper()],
        initial_balance=initial_balance,
        **kwargs
    )
    
    return TradingEngine(config)
