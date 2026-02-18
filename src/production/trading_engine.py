# src/production/trading_engine.py
"""
Production Trading Engine
=======================
Complete trading engine for real money with:
- Event-driven architecture
- Position tracking
- Portfolio management
- Safety features
- Comprehensive logging
"""

import asyncio
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path

from src.production.broker_interface import (
    BrokerInterface, create_broker, Order, OrderSide, OrderType, Position, AccountBalance
)
from src.production.order_manager import OrderManager, RiskManager, RetryConfig, OrderRequest


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TradingMode(Enum):
    """Trading mode enumeration."""
    PAPER = "paper"
    TESTNET = "testnet"
    LIVE = "live"


class EngineState(Enum):
    """Engine state enumeration."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class TradingConfig:
    """Trading engine configuration."""
    mode: TradingMode = TradingMode.PAPER
    initial_balance: float = 100000
    
    # Risk settings
    max_position_size: float = 0.3  # 30% max position
    max_order_size: float = 0.1  # 10% max per order
    max_daily_loss: float = 0.05  # 5% max daily loss
    max_leverage: int = 3
    
    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Safety
    emergency_close_on_error: bool = True
    max_daily_orders: int = 100
    
    # Monitoring
    health_check_interval: int = 60  # seconds
    position_check_interval: int = 10  # seconds


@dataclass
class Trade:
    """Represents a completed trade."""
    order_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    commission: float
    pnl: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side,
            'quantity': self.quantity,
            'price': self.price,
            'commission': self.commission,
            'pnl': self.pnl,
            'timestamp': self.timestamp.isoformat()
        }


class ProductionTradingEngine:
    """
    Production-ready trading engine.
    """
    
    def __init__(
        self,
        config: TradingConfig,
        signal_callback: Optional[Callable] = None
    ):
        """
        Initialize trading engine.
        
        Args:
            config: Trading configuration
            signal_callback: Callback for trading signals
        """
        self.config = config
        self.signal_callback = signal_callback
        
        # State
        self.state = EngineState.STOPPED
        self.broker: Optional[BrokerInterface] = None
        self.order_manager: Optional[OrderManager] = None
        
        # Position tracking
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        
        # Statistics
        self.stats = {
            'started_at': None,
            'total_orders': 0,
            'filled_orders': 0,
            'cancelled_orders': 0,
            'rejected_orders': 0,
            'total_pnl': 0.0,
            'daily_pnl': 0.0,
            'equity_high': config.initial_balance,
            'drawdown': 0.0
        }
        
        # Callbacks
        self.on_trade: Optional[Callable] = None
        self.on_position_change: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
        self.on_risk_alert: Optional[Callable] = None
        
        # Background tasks
        self._tasks: List[asyncio.Task] = []
        
        # Logging
        self._setup_logging()
        
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
    
    async def start(self) -> bool:
        """
        Start the trading engine.
        
        Returns:
            True if started successfully
        """
        if self.state != EngineState.STOPPED:
            logger.warning(f"Cannot start from state: {self.state}")
            return False
        
        self.state = EngineState.STARTING
        logger.info("Starting trading engine...")
        
        try:
            # Create broker
            broker_config = {
                'testnet': self.config.mode != TradingMode.LIVE
            }
            
            if self.config.mode == TradingMode.PAPER:
                broker_config['initial_balance'] = self.config.initial_balance
            
            self.broker = create_broker(
                'paper' if self.config.mode == TradingMode.PAPER else 'binance',
                broker_config
            )
            
            # Connect
            if not await self.broker.connect():
                raise ConnectionError("Failed to connect to broker")
            
            # Create risk manager
            risk_config = {
                'max_position_size': self.config.max_position_size,
                'max_order_size': self.config.max_order_size,
                'max_daily_loss': self.config.max_daily_loss,
                'max_leverage': self.config.max_leverage
            }
            risk_manager = RiskManager(risk_config)
            
            # Create order manager
            retry_config = RetryConfig(
                max_retries=self.config.max_retries,
                base_delay=self.config.retry_delay
            )
            
            self.order_manager = OrderManager(
                broker=self.broker,
                risk_manager=risk_manager,
                retry_config=retry_config,
                on_order_filled=self._on_order_filled,
                on_order_cancelled=self._on_order_cancelled,
                on_error=self._on_order_error
            )
            
            # Load existing positions
            await self._load_positions()
            
            # Update stats
            balance = await self.broker.get_balance()
            self.stats['equity_high'] = balance.total_equity
            self.stats['started_at'] = datetime.now()
            
            # Start background tasks
            await self._start_background_tasks()
            
            self.state = EngineState.RUNNING
            logger.info(f"Trading engine started in {self.config.mode.value} mode")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start engine: {e}")
            self.state = EngineState.ERROR
            if self.on_error:
                await self.on_error({
                    'type': 'engine_start_failed',
                    'error': str(e)
                })
            return False
    
    async def stop(self, close_positions: bool = False) -> bool:
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
            if close_positions and self.order_manager:
                logger.info("Closing all positions...")
                await self.order_manager.close_all_positions()
            
            # Disconnect broker
            if self.broker:
                await self.broker.disconnect()
            
            self.state = EngineState.STOPPED
            logger.info("Trading engine stopped")
            
            return True
            
        except Exception as e:
            logger.error(f"Error stopping engine: {e}")
            self.state = EngineState.ERROR
            return False
    
    async def _start_background_tasks(self):
        """Start background monitoring tasks."""
        # Position monitoring
        task = asyncio.create_task(self._position_monitor())
        self._tasks.append(task)
        
        # Health check
        task = asyncio.create_task(self._health_check())
        self._tasks.append(task)
        
        logger.info("Background tasks started")
    
    async def _position_monitor(self):
        """Monitor positions and update PnL."""
        while self.state == EngineState.RUNNING:
            try:
                # Get current positions
                positions = await self.broker.get_positions()
                
                # Update positions dict
                for pos in positions:
                    old_pnl = self.positions.get(pos.symbol, Position(pos.symbol)).unrealized_pnl
                    self.positions[pos.symbol] = pos
                    
                    # Check for significant changes
                    if abs(pos.unrealized_pnl - old_pnl) > 100:
                        logger.info(f"Position PnL change: {pos.symbol} {pos.unrealized_pnl:.2f}")
                        
                        if self.on_position_change:
                            await self.on_position_change(pos.to_dict())
                
                # Update daily PnL
                balance = await self.broker.get_balance()
                self.stats['daily_pnl'] = balance.unrealized_pnl + balance.realized_pnl
                
                # Check drawdown
                if balance.total_equity > self.stats['equity_high']:
                    self.stats['equity_high'] = balance.total_equity
                
                drawdown = (self.stats['equity_high'] - balance.total_equity) / self.stats['equity_high']
                self.stats['drawdown'] = drawdown
                
                # Risk check
                if drawdown > 0.15:  # 15% drawdown
                    logger.critical(f"High drawdown: {drawdown:.2%}")
                    if self.on_risk_alert:
                        await self.on_risk_alert({
                            'type': 'high_drawdown',
                            'drawdown': drawdown
                        })
                
            except Exception as e:
                logger.error(f"Position monitor error: {e}")
            
            await asyncio.sleep(self.config.position_check_interval)
    
    async def _health_check(self):
        """Perform periodic health checks."""
        while self.state == EngineState.RUNNING:
            try:
                # Check broker connection
                if not self.broker.is_connected:
                    logger.error("Broker disconnected!")
                    if self.config.emergency_close_on_error:
                        await self.stop(close_positions=True)
                    break
                
                # Check order manager
                if self.order_manager:
                    should_exit, reason = await self.order_manager.check_emergency_conditions()
                    
                    if should_exit:
                        logger.critical(f"Emergency exit: {reason}")
                        
                        if self.config.emergency_close_on_error:
                            await self.stop(close_positions=True)
                        
                        if self.on_risk_alert:
                            await self.on_risk_alert({
                                'type': 'emergency_exit',
                                'reason': reason
                            })
                        break
                
                # Log stats periodically
                balance = await self.broker.get_balance()
                logger.info(
                    f"Health check - Equity: {balance.total_equity:.2f}, "
                    f"Daily PnL: {self.stats['daily_pnl']:.2f}, "
                    f"Positions: {len(self.positions)}"
                )
                
            except Exception as e:
                logger.error(f"Health check error: {e}")
            
            await asyncio.sleep(self.config.health_check_interval)
    
    async def process_signal(self, signal: Dict) -> Dict:
        """
        Process a trading signal.
        
        Args:
            signal: Signal dictionary with symbol, action, confidence, etc.
            
        Returns:
            Result dictionary
        """
        if self.state != EngineState.RUNNING:
            return {'success': False, 'error': 'Engine not running'}
        
        symbol = signal.get('symbol')
        action = signal.get('action')  # BUY, SELL, CLOSE
        quantity = signal.get('quantity')
        confidence = signal.get('confidence', 0.5)
        
        if not symbol or not action:
            return {'success': False, 'error': 'Invalid signal'}
        
        # Filter by confidence
        if confidence < 0.6:
            return {'success': False, 'error': 'Low confidence'}
        
        try:
            # Get current position
            position = self.positions.get(symbol)
            current_qty = position.quantity if position else 0
            
            # Determine order
            if action == 'CLOSE':
                if current_qty > 0:
                    order = OrderRequest(
                        symbol=symbol,
                        side='SELL',
                        quantity=abs(current_qty),
                        order_type='MARKET',
                        metadata={'reason': 'signal_close', 'signal': signal}
                    )
                else:
                    return {'success': True, 'message': 'No position to close'}
                    
            elif action == 'BUY':
                if not quantity:
                    # Calculate quantity based on position size
                    balance = await self.broker.get_balance()
                    quantity = (balance.total_equity * self.config.max_order_size) / 50000  # rough estimate
                
                order = OrderRequest(
                    symbol=symbol,
                    side='BUY',
                    quantity=quantity,
                    order_type='MARKET',
                    metadata={'reason': 'signal_buy', 'signal': signal}
                )
                
            elif action == 'SELL':
                if current_qty <= 0:
                    return {'success': False, 'error': 'No position to sell'}
                
                order = OrderRequest(
                    symbol=symbol,
                    side='SELL',
                    quantity=quantity or abs(current_qty),
                    order_type='MARKET',
                    metadata={'reason': 'signal_sell', 'signal': signal}
                )
                
            else:
                return {'success': False, 'error': f'Unknown action: {action}'}
            
            # Place order
            result = await self.order_manager.place_order(order)
            
            self.stats['total_orders'] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing signal: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _on_order_filled(self, result: Dict):
        """Handle order filled callback."""
        logger.info(f"Order filled: {result}")
        
        self.stats['filled_orders'] += 1
        
        if result.get('pnl'):
            self.stats['total_pnl'] += result['pnl']
        
        if self.on_trade:
            await self.on_trade(result)
    
    async def _on_order_cancelled(self, result: Dict):
        """Handle order cancelled callback."""
        logger.info(f"Order cancelled: {result}")
        self.stats['cancelled_orders'] += 1
    
    async def _on_order_error(self, error: Dict):
        """Handle order error callback."""
        logger.error(f"Order error: {error}")
        
        if error.get('type') == 'order_rejected':
            self.stats['rejected_orders'] += 1
        
        if self.on_error:
            await self.on_error(error)
    
    async def _load_positions(self):
        """Load existing positions from broker."""
        try:
            positions = await self.broker.get_positions()
            self.positions = {p.symbol: p for p in positions}
            logger.info(f"Loaded {len(self.positions)} positions")
        except Exception as e:
            logger.error(f"Error loading positions: {e}")
    
    async def get_status(self) -> Dict:
        """Get engine status."""
        balance = await self.broker.get_balance()
        
        return {
            'state': self.state.value,
            'mode': self.config.mode.value,
            'balance': balance.to_dict(),
            'positions': {s: p.to_dict() for s, p in self.positions.items()},
            'stats': self.stats,
            'open_orders': len(self.order_manager.pending_orders) if self.order_manager else 0
        }
    
    async def get_positions_summary(self) -> List[Dict]:
        """Get positions summary."""
        return [
            {
                'symbol': symbol,
                'quantity': pos.quantity,
                'entry_price': pos.entry_price,
                'current_price': pos.current_price,
                'unrealized_pnl': pos.unrealized_pnl,
                'pnl_percent': pos.pnl_percent
            }
            for symbol, pos in self.positions.items()
            if pos.quantity != 0
        ]


async def create_production_engine(
    mode: str = 'paper',
    initial_balance: float = 100000,
    signal_callback: Optional[Callable] = None
) -> ProductionTradingEngine:
    """
    Factory function to create production trading engine.
    
    Args:
        mode: Trading mode ('paper', 'testnet', 'live')
        initial_balance: Initial balance
        signal_callback: Signal callback
        
    Returns:
        ProductionTradingEngine instance
    """
    config = TradingConfig(
        mode=TradingMode[mode.upper()],
        initial_balance=initial_balance
    )
    
    engine = ProductionTradingEngine(
        config=config,
        signal_callback=signal_callback
    )
    
    return engine
