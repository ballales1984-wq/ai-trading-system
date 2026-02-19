"""
Data Adapter - Bridge between API and Real Trading System
=========================================================
Connects FastAPI endpoints to the actual trading system data sources.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import sqlite3
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.core.state_manager import StateManager
    from src.core.portfolio.portfolio_manager import PortfolioManager
    STATE_MANAGER_AVAILABLE = True
except ImportError:
    STATE_MANAGER_AVAILABLE = False
    print("Warning: Real trading system modules not available, using mock data")

try:
    from trading_simulator import TradingSimulator
    SIMULATOR_AVAILABLE = True
except ImportError:
    SIMULATOR_AVAILABLE = False


class DataAdapter:
    """Adapter to connect API to real trading system data."""
    
    def __init__(self):
        """Initialize data adapter."""
        self.state_manager = None
        self.simulator = None
        
        # Try to initialize real data sources
        if STATE_MANAGER_AVAILABLE:
            try:
                db_path = project_root / "data" / "trading_state.db"
                self.state_manager = StateManager(str(db_path))
                print(f"Connected to trading state database: {db_path}")
            except Exception as e:
                print(f"Could not connect to state manager: {e}")
        
        if SIMULATOR_AVAILABLE:
            try:
                # Try to load existing simulator state
                self.simulator = TradingSimulator(initial_balance=1000000.0)
                print("Connected to trading simulator")
            except Exception as e:
                print(f"Could not connect to simulator: {e}")
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary from real system."""
        # Try simulator first (most likely to have data)
        if self.simulator:
            try:
                status = self.simulator.check_portfolio()
                return {
                    "total_value": status.get('total_value', 0),
                    "cash_balance": status.get('balance', 0),
                    "market_value": status.get('total_value', 0) - status.get('balance', 0),
                    "total_pnl": status.get('total_pnl', 0),
                    "unrealized_pnl": status.get('total_pnl', 0),
                    "realized_pnl": 0.0,
                    "daily_pnl": 0.0,
                    "daily_return_pct": 0.0,
                    "total_return_pct": status.get('pnl_percent', 0),
                    "leverage": 1.0,
                    "buying_power": status.get('balance', 0),
                    "num_positions": status.get('open_positions', 0),
                }
            except Exception as e:
                print(f"Error getting portfolio from simulator: {e}")
        
        # Try state manager
        if self.state_manager:
            try:
                # Get latest portfolio state
                with self.state_manager._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT * FROM portfolio 
                        ORDER BY timestamp DESC LIMIT 1
                    """)
                    row = cursor.fetchone()
                    if row:
                        return {
                            "total_value": row['total_equity'] or 0,
                            "cash_balance": row['available_balance'] or 0,
                            "market_value": (row['total_equity'] or 0) - (row['available_balance'] or 0),
                            "total_pnl": (row['unrealized_pnl'] or 0) + (row['realized_pnl'] or 0),
                            "unrealized_pnl": row['unrealized_pnl'] or 0,
                            "realized_pnl": row['realized_pnl'] or 0,
                            "daily_pnl": 0.0,
                            "daily_return_pct": 0.0,
                            "total_return_pct": 0.0,
                            "leverage": 1.0,
                            "buying_power": row['available_balance'] or 0,
                            "num_positions": 0,  # Will be calculated from positions
                        }
            except Exception as e:
                print(f"Error getting portfolio from state manager: {e}")
        
        # Fallback to mock data
        return {
            "total_value": 1005000.0,
            "cash_balance": 500000.0,
            "market_value": 505000.0,
            "total_pnl": 5000.0,
            "unrealized_pnl": 4500.0,
            "realized_pnl": 500.0,
            "daily_pnl": 250.0,
            "daily_return_pct": 0.025,
            "total_return_pct": 0.5,
            "leverage": 1.0,
            "buying_power": 500000.0,
            "num_positions": 2,
        }
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """Get positions from real system."""
        positions = []
        
        # Try simulator
        if self.simulator:
            try:
                status = self.simulator.check_portfolio()
                positions_detail = status.get('positions_detail', {})
                for symbol, pos in positions_detail.items():
                    positions.append({
                        "position_id": f"pos_{symbol}",
                        "symbol": symbol,
                        "side": "LONG",
                        "quantity": pos.get('quantity', 0),
                        "entry_price": pos.get('entry_price', 0),
                        "current_price": pos.get('current_price', 0),
                        "market_value": pos.get('current_price', 0) * pos.get('quantity', 0),
                        "unrealized_pnl": (pos.get('current_price', 0) - pos.get('entry_price', 0)) * pos.get('quantity', 0),
                        "realized_pnl": 0.0,
                        "leverage": 1.0,
                        "margin_used": pos.get('entry_price', 0) * pos.get('quantity', 0),
                        "stop_loss": None,
                        "take_profit": None,
                        "opened_at": datetime.now().isoformat(),
                        "updated_at": datetime.now().isoformat(),
                    })
            except Exception as e:
                print(f"Error getting positions from simulator: {e}")
        
        # Try state manager
        if self.state_manager and not positions:
            try:
                with self.state_manager._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT * FROM positions")
                    rows = cursor.fetchall()
                    for row in rows:
                        positions.append({
                            "position_id": f"pos_{row['symbol']}",
                            "symbol": row['symbol'],
                            "side": "LONG",
                            "quantity": row['quantity'] or 0,
                            "entry_price": row['entry_price'] or 0,
                            "current_price": row['current_price'] or 0,
                            "market_value": (row['current_price'] or 0) * (row['quantity'] or 0),
                            "unrealized_pnl": row['unrealized_pnl'] or 0,
                            "realized_pnl": row['realized_pnl'] or 0,
                            "leverage": row['leverage'] or 1.0,
                            "margin_used": (row['entry_price'] or 0) * (row['quantity'] or 0),
                            "stop_loss": None,
                            "take_profit": None,
                            "opened_at": row['opened_at'] or datetime.now().isoformat(),
                            "updated_at": row['updated_at'] or datetime.now().isoformat(),
                        })
            except Exception as e:
                print(f"Error getting positions from state manager: {e}")
        
        return positions
    
    def get_orders(self) -> List[Dict[str, Any]]:
        """Get orders from real system."""
        orders = []
        
        # Try state manager
        if self.state_manager:
            try:
                with self.state_manager._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT * FROM orders ORDER BY created_at DESC LIMIT 50")
                    rows = cursor.fetchall()
                    for row in rows:
                        orders.append({
                            "order_id": row['order_id'],
                            "symbol": row['symbol'],
                            "side": row['side'],
                            "order_type": "MARKET",
                            "quantity": row['quantity'] or 0,
                            "price": row['avg_fill_price'],
                            "stop_price": None,
                            "status": row['status'] or "PENDING",
                            "filled_quantity": row['filled_quantity'] or 0,
                            "average_price": row['avg_fill_price'],
                            "commission": 0.0,
                            "created_at": row['created_at'] or datetime.now().isoformat(),
                            "updated_at": row['updated_at'] or datetime.now().isoformat(),
                            "strategy_id": None,
                            "broker": "binance",
                            "error_message": None,
                        })
            except Exception as e:
                print(f"Error getting orders from state manager: {e}")
        
        return orders
    
    def get_portfolio_history(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get portfolio history from real system."""
        history = []
        
        # Try state manager
        if self.state_manager:
            try:
                with self.state_manager._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT * FROM portfolio 
                        ORDER BY timestamp DESC LIMIT ?
                    """, (days,))
                    rows = cursor.fetchall()
                    for row in rows:
                        history.append({
                            "date": row['timestamp'][:10] if row['timestamp'] else datetime.now().strftime('%Y-%m-%d'),
                            "value": row['total_equity'] or 0,
                            "daily_return": 0.0,
                        })
            except Exception as e:
                print(f"Error getting history from state manager: {e}")
        
        return history


# Global adapter instance
_adapter_instance = None

def get_data_adapter() -> DataAdapter:
    """Get or create data adapter instance."""
    global _adapter_instance
    if _adapter_instance is None:
        _adapter_instance = DataAdapter()
    return _adapter_instance
