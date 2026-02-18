# src/core/state_manager.py
"""
State Manager
============
Persistent state management for the trading engine.
Handles portfolio state, positions, orders, and model snapshots.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
from contextlib import contextmanager


logger = logging.getLogger(__name__)


@dataclass
class PortfolioState:
    """Portfolio state snapshot."""
    total_equity: float = 0.0
    available_balance: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class PositionState:
    """Position state."""
    symbol: str
    quantity: float = 0.0
    entry_price: float = 0.0
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    commission: float = 0.0
    leverage: float = 1.0
    opened_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        data = asdict(self)
        data['opened_at'] = self.opened_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data


@dataclass
class OrderState:
    """Order state."""
    order_id: str
    symbol: str
    side: str
    quantity: float
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data


@dataclass
class ModelState:
    """Model state snapshot."""
    model_id: str
    model_type: str
    version: str
    file_path: str
    metrics: Dict[str, float] = field(default_factory=dict)
    trained_at: datetime = field(default_factory=datetime.now)
    loaded_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        data = asdict(self)
        data['trained_at'] = self.trained_at.isoformat()
        data['loaded_at'] = self.loaded_at.isoformat()
        return data


class StateManager:
    """
    Manages persistent state for the trading engine.
    Uses SQLite for persistence.
    """
    
    def __init__(self, db_path: str = "data/trading_state.db"):
        """
        Initialize state manager.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self._ensure_db_dir()
        self._init_database()
        
        logger.info(f"State manager initialized with database: {db_path}")
    
    def _ensure_db_dir(self):
        """Ensure database directory exists."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
    
    @contextmanager
    def _get_connection(self):
        """Get database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def _init_database(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Portfolio table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS portfolio (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    total_equity REAL,
                    available_balance REAL,
                    unrealized_pnl REAL,
                    realized_pnl REAL,
                    timestamp TEXT
                )
            """)
            
            # Positions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    symbol TEXT PRIMARY KEY,
                    quantity REAL,
                    entry_price REAL,
                    current_price REAL,
                    unrealized_pnl REAL,
                    realized_pnl REAL,
                    commission REAL,
                    leverage REAL,
                    opened_at TEXT,
                    updated_at TEXT
                )
            """)
            
            # Orders table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    order_id TEXT PRIMARY KEY,
                    symbol TEXT,
                    side TEXT,
                    quantity REAL,
                    filled_quantity REAL,
                    avg_fill_price REAL,
                    status TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
            """)
            
            # Models table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    model_id TEXT PRIMARY KEY,
                    model_type TEXT,
                    version TEXT,
                    file_path TEXT,
                    metrics TEXT,
                    trained_at TEXT,
                    loaded_at TEXT
                )
            """)
            
            # Trades table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id TEXT,
                    symbol TEXT,
                    side TEXT,
                    quantity REAL,
                    price REAL,
                    commission REAL,
                    pnl REAL,
                    timestamp TEXT
                )
            """)
            
            # Event log table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS event_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT,
                    data TEXT,
                    timestamp TEXT
                )
            """)
            
            conn.commit()
    
    # Portfolio methods
    def save_portfolio_state(self, state: PortfolioState):
        """Save portfolio state."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO portfolio (total_equity, available_balance, unrealized_pnl, realized_pnl, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (
                state.total_equity,
                state.available_balance,
                state.unrealized_pnl,
                state.realized_pnl,
                state.timestamp.isoformat()
            ))
            conn.commit()
        
        logger.debug(f"Portfolio state saved: equity={state.total_equity}")
    
    def get_latest_portfolio_state(self) -> Optional[PortfolioState]:
        """Get latest portfolio state."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM portfolio ORDER BY timestamp DESC LIMIT 1
            """)
            row = cursor.fetchone()
            
            if row:
                return PortfolioState(
                    total_equity=row['total_equity'],
                    available_balance=row['available_balance'],
                    unrealized_pnl=row['unrealized_pnl'],
                    realized_pnl=row['realized_pnl'],
                    timestamp=datetime.fromisoformat(row['timestamp'])
                )
        
        return None
    
    def get_portfolio_history(self, hours: int = 24) -> List[PortfolioState]:
        """Get portfolio history."""
        since = datetime.now() - timedelta(hours=hours)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM portfolio WHERE timestamp > ? ORDER BY timestamp DESC
            """, (since.isoformat(),))
            
            return [
                PortfolioState(
                    total_equity=row['total_equity'],
                    available_balance=row['available_balance'],
                    unrealized_pnl=row['unrealized_pnl'],
                    realized_pnl=row['realized_pnl'],
                    timestamp=datetime.fromisoformat(row['timestamp'])
                )
                for row in cursor.fetchall()
            ]
    
    # Position methods
    def save_position(self, position: PositionState):
        """Save or update position."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO positions
                (symbol, quantity, entry_price, current_price, unrealized_pnl, realized_pnl, commission, leverage, opened_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                position.symbol,
                position.quantity,
                position.entry_price,
                position.current_price,
                position.unrealized_pnl,
                position.realized_pnl,
                position.commission,
                position.leverage,
                position.opened_at.isoformat(),
                position.updated_at.isoformat()
            ))
            conn.commit()
        
        logger.debug(f"Position saved: {position.symbol} qty={position.quantity}")
    
    def get_position(self, symbol: str) -> Optional[PositionState]:
        """Get position for symbol."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM positions WHERE symbol = ?", (symbol,))
            row = cursor.fetchone()
            
            if row:
                return PositionState(
                    symbol=row['symbol'],
                    quantity=row['quantity'],
                    entry_price=row['entry_price'],
                    current_price=row['current_price'],
                    unrealized_pnl=row['unrealized_pnl'],
                    realized_pnl=row['realized_pnl'],
                    commission=row['commission'],
                    leverage=row['leverage'],
                    opened_at=datetime.fromisoformat(row['opened_at']),
                    updated_at=datetime.fromisoformat(row['updated_at'])
                )
        
        return None
    
    def get_all_positions(self) -> List[PositionState]:
        """Get all positions."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM positions WHERE quantity != 0")
            
            return [
                PositionState(
                    symbol=row['symbol'],
                    quantity=row['quantity'],
                    entry_price=row['entry_price'],
                    current_price=row['current_price'],
                    unrealized_pnl=row['unrealized_pnl'],
                    realized_pnl=row['realized_pnl'],
                    commission=row['commission'],
                    leverage=row['leverage'],
                    opened_at=datetime.fromisoformat(row['opened_at']),
                    updated_at=datetime.fromisoformat(row['updated_at'])
                )
                for row in cursor.fetchall()
            ]
    
    def close_position(self, symbol: str):
        """Close position (set to zero)."""
        position = self.get_position(symbol)
        if position:
            position.quantity = 0
            position.updated_at = datetime.now()
            self.save_position(position)
    
    # Order methods
    def save_order(self, order: OrderState):
        """Save order."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO orders
                (order_id, symbol, side, quantity, filled_quantity, avg_fill_price, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                order.order_id,
                order.symbol,
                order.side,
                order.quantity,
                order.filled_quantity,
                order.avg_fill_price,
                order.status,
                order.created_at.isoformat(),
                order.updated_at.isoformat()
            ))
            conn.commit()
    
    def get_order(self, order_id: str) -> Optional[OrderState]:
        """Get order by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,))
            row = cursor.fetchone()
            
            if row:
                return OrderState(
                    order_id=row['order_id'],
                    symbol=row['symbol'],
                    side=row['side'],
                    quantity=row['quantity'],
                    filled_quantity=row['filled_quantity'],
                    avg_fill_price=row['avg_fill_price'],
                    status=row['status'],
                    created_at=datetime.fromisoformat(row['created_at']),
                    updated_at=datetime.fromisoformat(row['updated_at'])
                )
        
        return None
    
    def get_open_orders(self) -> List[OrderState]:
        """Get all open orders."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM orders WHERE status IN ('pending', 'open', 'partially_filled')")
            
            return [
                OrderState(
                    order_id=row['order_id'],
                    symbol=row['symbol'],
                    side=row['side'],
                    quantity=row['quantity'],
                    filled_quantity=row['filled_quantity'],
                    avg_fill_price=row['avg_fill_price'],
                    status=row['status'],
                    created_at=datetime.fromisoformat(row['created_at']),
                    updated_at=datetime.fromisoformat(row['updated_at'])
                )
                for row in cursor.fetchall()
            ]
    
    # Model methods
    def save_model_state(self, model: ModelState):
        """Save model state."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO models
                (model_id, model_type, version, file_path, metrics, trained_at, loaded_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                model.model_id,
                model.model_type,
                model.version,
                model.file_path,
                json.dumps(model.metrics),
                model.trained_at.isoformat(),
                model.loaded_at.isoformat()
            ))
            conn.commit()
    
    def get_model_state(self, model_id: str) -> Optional[ModelState]:
        """Get model state."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM models WHERE model_id = ?", (model_id,))
            row = cursor.fetchone()
            
            if row:
                return ModelState(
                    model_id=row['model_id'],
                    model_type=row['model_type'],
                    version=row['version'],
                    file_path=row['file_path'],
                    metrics=json.loads(row['metrics']),
                    trained_at=datetime.fromisoformat(row['trained_at']),
                    loaded_at=datetime.fromisoformat(row['loaded_at'])
                )
        
        return None
    
    # Trade methods
    def save_trade(self, trade: Dict):
        """Save trade."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO trades (order_id, symbol, side, quantity, price, commission, pnl, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade.get('order_id'),
                trade.get('symbol'),
                trade.get('side'),
                trade.get('quantity'),
                trade.get('price'),
                trade.get('commission', 0),
                trade.get('pnl', 0),
                datetime.now().isoformat()
            ))
            conn.commit()
    
    def get_trades(self, symbol: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """Get trade history."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            if symbol:
                cursor.execute("""
                    SELECT * FROM trades WHERE symbol = ? ORDER BY timestamp DESC LIMIT ?
                """, (symbol, limit))
            else:
                cursor.execute("SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?", (limit,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    # Event log methods
    def log_event(self, event_type: str, data: Dict):
        """Log event."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO event_log (event_type, data, timestamp)
                VALUES (?, ?, ?)
            """, (
                event_type,
                json.dumps(data),
                datetime.now().isoformat()
            ))
            conn.commit()
    
    # Snapshot methods
    def create_snapshot(self, portfolio: Dict, positions: List[Dict], orders: List[Dict]):
        """Create full state snapshot."""
        # Save portfolio
        self.save_portfolio_state(PortfolioState(**portfolio))
        
        # Save positions
        for pos in positions:
            self.save_position(PositionState(**pos))
        
        # Save orders
        for order in orders:
            self.save_order(OrderState(**order))
        
        logger.info("State snapshot created")
    
    def restore_from_snapshot(self) -> Dict:
        """Restore state from latest snapshot."""
        portfolio = self.get_latest_portfolio_state()
        positions = self.get_all_positions()
        orders = self.get_open_orders()
        
        return {
            'portfolio': portfolio.to_dict() if portfolio else None,
            'positions': [p.to_dict() for p in positions],
            'orders': [o.to_dict() for o in orders]
        }
