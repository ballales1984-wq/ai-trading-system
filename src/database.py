"""
ML Database Module for AI Trading System
=========================================
SQLite database for storing ML predictions, price history,
model performance, and backtest results.

Author: AI Trading System
Version: 1.0.0
"""

import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from contextlib import contextmanager
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class MLDatabase:
    """
    SQLite database manager for ML components.
    Provides persistent storage for:
    - ML predictions
    - Price history
    - Model performance metrics
    - Backtest results
    """
    
    def __init__(self, db_path: str = "data/ml_trading.db"):
        """
        Initialize ML database.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._ensure_db_dir()
        self._init_database()
        logger.info(f"ML Database initialized: {db_path}")
    
    def _ensure_db_dir(self):
        """Ensure database directory exists."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    @contextmanager
    def _connection(self):
        """Context manager for database connections."""
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()
    
    def _init_database(self):
        """Initialize database schema with all ML tables."""
        with self._connection() as conn:
            cursor = conn.cursor()
            
            # Table 1: ML Predictions
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ml_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    prediction TEXT NOT NULL,
                    probability REAL,
                    features TEXT,
                    actual_outcome TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Table 2: Price History
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS price_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    source TEXT DEFAULT 'binance',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(timestamp, symbol)
                )
            """)
            
            # Table 3: Model Performance
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    accuracy REAL,
                    precision_score REAL,
                    recall_score REAL,
                    f1_score REAL,
                    auc_roc REAL,
                    confusion_matrix TEXT,
                    training_samples INTEGER,
                    test_samples INTEGER,
                    training_time_seconds REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Table 4: Backtest Results
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS backtest_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    strategy_name TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    initial_balance REAL NOT NULL,
                    final_balance REAL NOT NULL,
                    total_return REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    win_rate REAL,
                    total_trades INTEGER,
                    profitable_trades INTEGER,
                    losing_trades INTEGER,
                    avg_profit REAL,
                    avg_loss REAL,
                    params TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Table 5: Model Metadata
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL UNIQUE,
                    model_type TEXT,
                    version TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    last_trained TEXT,
                    accuracy REAL,
                    is_active INTEGER DEFAULT 1,
                    filepath TEXT
                )
            """)
            
            # Table 6: Training Data
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    features TEXT NOT NULL,
                    target INTEGER NOT NULL,
                    timeframe TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for better performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_timestamp ON ml_predictions(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_symbol ON ml_predictions(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_price_timestamp ON price_history(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_price_symbol ON price_history(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_perf_model ON model_performance(model_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_backtest_strategy ON backtest_results(strategy_name)")
            
            conn.commit()
            logger.info("ML Database tables initialized successfully")
    
    # ==================== ML PREDICTIONS ====================
    
    def save_prediction(
        self,
        timestamp: datetime,
        symbol: str,
        model_name: str,
        prediction: str,
        probability: float,
        features: Optional[Dict] = None,
        actual_outcome: Optional[str] = None
    ) -> int:
        """
        Save ML prediction to database.
        
        Args:
            timestamp: Prediction timestamp
            symbol: Trading symbol
            model_name: Name of the ML model
            prediction: Prediction (BUY/SELL/HOLD)
            probability: Prediction probability
            features: Feature dictionary
            actual_outcome: Actual outcome (for verification)
            
        Returns:
            Prediction ID
        """
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO ml_predictions 
                (timestamp, symbol, model_name, prediction, probability, features, actual_outcome)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp,
                symbol,
                model_name,
                prediction,
                probability,
                json.dumps(features) if features else None,
                actual_outcome
            ))
            return cursor.lastrowid
    
    def get_predictions(
        self,
        symbol: Optional[str] = None,
        model_name: Optional[str] = None,
        limit: int = 100
    ) -> pd.DataFrame:
        """Get ML predictions from database."""
        query = "SELECT * FROM ml_predictions WHERE 1=1"
        params = []
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if model_name:
            query += " AND model_name = ?"
            params.append(model_name)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        with self._connection() as conn:
            return pd.read_sql_query(query, conn, params=params)
    
    def update_actual_outcome(self, prediction_id: int, actual_outcome: str):
        """Update actual outcome for a prediction."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE ml_predictions SET actual_outcome = ? WHERE id = ?
            """, (actual_outcome, prediction_id))
    
    # ==================== PRICE HISTORY ====================
    
    def save_price(
        self,
        timestamp: datetime,
        symbol: str,
        open_price: float,
        high: float,
        low: float,
        close: float,
        volume: float,
        source: str = "binance"
    ) -> int:
        """
        Save price data to database.
        
        Args:
            timestamp: Price timestamp
            symbol: Trading symbol
            open_price: Open price
            high: High price
            low: Low price
            close: Close price
            volume: Volume
            source: Data source
            
        Returns:
            Price ID
        """
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO price_history
                (timestamp, symbol, open, high, low, close, volume, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp,
                symbol, open_price, high, low, close, volume, source
            ))
            return cursor.lastrowid
    
    def save_prices_bulk(self, prices: List[Dict]):
        """Save multiple price records."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.executemany("""
                INSERT OR REPLACE INTO price_history
                (timestamp, symbol, open, high, low, close, volume, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                (
                    p['timestamp'].isoformat() if isinstance(p.get('timestamp'), datetime) else p.get('timestamp'),
                    p['symbol'],
                    p.get('open'),
                    p.get('high'),
                    p.get('low'),
                    p.get('close'),
                    p.get('volume'),
                    p.get('source', 'binance')
                ) for p in prices
            ])
    
    def get_price_history(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """Get price history from database."""
        query = "SELECT * FROM price_history WHERE symbol = ?"
        params = [symbol]
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat() if isinstance(start_time, datetime) else start_time)
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat() if isinstance(end_time, datetime) else end_time)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        with self._connection() as conn:
            return pd.read_sql_query(query, conn, params=params)
    
    # ==================== MODEL PERFORMANCE ====================
    
    def save_model_performance(
        self,
        timestamp: datetime,
        model_name: str,
        accuracy: float,
        precision_score: float,
        recall_score: float,
        f1_score: float,
        auc_roc: Optional[float] = None,
        confusion_matrix: Optional[Dict] = None,
        training_samples: int = 0,
        test_samples: int = 0,
        training_time_seconds: float = 0.0
    ) -> int:
        """
        Save model performance metrics.
        
        Args:
            timestamp: Training timestamp
            model_name: Model name
            accuracy: Accuracy score
            precision_score: Precision score
            recall_score: Recall score
            f1_score: F1 score
            auc_roc: AUC-ROC score
            confusion_matrix: Confusion matrix dict
            training_samples: Number of training samples
            test_samples: Number of test samples
            training_time_seconds: Training time
            
        Returns:
            Performance ID
        """
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO model_performance
                (timestamp, model_name, accuracy, precision_score, recall_score, 
                 f1_score, auc_roc, confusion_matrix, training_samples, 
                 test_samples, training_time_seconds)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp,
                model_name, accuracy, precision_score, recall_score, f1_score,
                auc_roc, json.dumps(confusion_matrix) if confusion_matrix else None,
                training_samples, test_samples, training_time_seconds
            ))
            return cursor.lastrowid
    
    def get_model_performance(
        self,
        model_name: Optional[str] = None,
        limit: int = 50
    ) -> pd.DataFrame:
        """Get model performance records."""
        query = "SELECT * FROM model_performance"
        params = []
        
        if model_name:
            query += " WHERE model_name = ?"
            params.append(model_name)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        with self._connection() as conn:
            return pd.read_sql_query(query, conn, params=params)
    
    def get_best_model(self, metric: str = "accuracy") -> Optional[Dict]:
        """Get best performing model based on metric."""
        allowed_metrics = ["accuracy", "precision_score", "recall_score", "f1_score", "auc_roc"]
        if metric not in allowed_metrics:
            metric = "accuracy"
        
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                SELECT * FROM model_performance 
                ORDER BY {metric} DESC LIMIT 1
            """)
            row = cursor.fetchone()
            return dict(row) if row else None
    
    # ==================== BACKTEST RESULTS ====================
    
    def save_backtest_result(
        self,
        timestamp: datetime,
        strategy_name: str,
        symbol: str,
        initial_balance: float,
        final_balance: float,
        total_return: float,
        sharpe_ratio: float,
        max_drawdown: float,
        win_rate: float,
        total_trades: int,
        profitable_trades: int,
        losing_trades: int,
        avg_profit: float,
        avg_loss: float,
        params: Optional[Dict] = None
    ) -> int:
        """
        Save backtest results.
        
        Args:
            timestamp: Backtest timestamp
            strategy_name: Strategy name
            symbol: Trading symbol
            initial_balance: Initial balance
            final_balance: Final balance
            total_return: Total return percentage
            sharpe_ratio: Sharpe ratio
            max_drawdown: Maximum drawdown
            win_rate: Win rate percentage
            total_trades: Total number of trades
            profitable_trades: Number of profitable trades
            losing_trades: Number of losing trades
            avg_profit: Average profit
            avg_loss: Average loss
            params: Strategy parameters
            
        Returns:
            Backtest ID
        """
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO backtest_results
                (timestamp, strategy_name, symbol, initial_balance, final_balance,
                 total_return, sharpe_ratio, max_drawdown, win_rate, total_trades,
                 profitable_trades, losing_trades, avg_profit, avg_loss, params)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp,
                strategy_name, symbol, initial_balance, final_balance,
                total_return, sharpe_ratio, max_drawdown, win_rate, total_trades,
                profitable_trades, losing_trades, avg_profit, avg_loss,
                json.dumps(params) if params else None
            ))
            return cursor.lastrowid
    
    def get_backtest_results(
        self,
        strategy_name: Optional[str] = None,
        symbol: Optional[str] = None,
        limit: int = 50
    ) -> pd.DataFrame:
        """Get backtest results."""
        query = "SELECT * FROM backtest_results WHERE 1=1"
        params = []
        
        if strategy_name:
            query += " AND strategy_name = ?"
            params.append(strategy_name)
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        with self._connection() as conn:
            return pd.read_sql_query(query, conn, params=params)
    
    def get_best_backtest(
        self,
        metric: str = "sharpe_ratio"
    ) -> Optional[Dict]:
        """Get best backtest result."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                SELECT * FROM backtest_results 
                ORDER BY {metric} DESC LIMIT 1
            """)
            row = cursor.fetchone()
            return dict(row) if row else None
    
    # ==================== MODEL METADATA ====================
    
    def save_model_metadata(
        self,
        model_name: str,
        model_type: str,
        version: str,
        accuracy: float,
        filepath: str
    ) -> int:
        """Save model metadata."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO model_metadata
                (model_name, model_type, version, last_trained, accuracy, filepath)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (model_name, model_type, version, datetime.now().isoformat(), accuracy, filepath))
            return cursor.lastrowid
    
    def get_model_metadata(self, model_name: str) -> Optional[Dict]:
        """Get model metadata."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM model_metadata WHERE model_name = ?", (model_name,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def get_active_models(self) -> List[Dict]:
        """Get all active models."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM model_metadata WHERE is_active = 1")
            return [dict(row) for row in cursor.fetchall()]
    
    # ==================== TRAINING DATA ====================
    
    def save_training_data(
        self,
        timestamp: datetime,
        symbol: str,
        features: Dict,
        target: int,
        timeframe: str = "1h"
    ) -> int:
        """Save training data sample."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO training_data (timestamp, symbol, features, target, timeframe)
                VALUES (?, ?, ?, ?, ?)
            """, (
                timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp,
                symbol, json.dumps(features), target, timeframe
            ))
            return cursor.lastrowid
    
    def get_training_data(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        limit: int = 10000
    ) -> pd.DataFrame:
        """Get training data."""
        query = "SELECT * FROM training_data WHERE 1=1"
        params = []
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if timeframe:
            query += " AND timeframe = ?"
            params.append(timeframe)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        with self._connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)
            # Parse features JSON
            if 'features' in df.columns:
                df['features'] = df['features'].apply(json.loads)
            return df
    
    # ==================== UTILITY METHODS ====================
    
    def get_statistics(self) -> Dict:
        """Get database statistics."""
        with self._connection() as conn:
            cursor = conn.cursor()
            stats = {}
            
            tables = ['ml_predictions', 'price_history', 'model_performance', 
                     'backtest_results', 'model_metadata', 'training_data']
            
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                stats[table] = cursor.fetchone()[0]
            
            return stats
    
    def cleanup_old_data(self, days: int = 90):
        """Remove old data from database."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cutoff = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            cutoff = cutoff.replace(day=cutoff.day - days)
            
            tables = ['ml_predictions', 'price_history', 'model_performance', 
                     'backtest_results', 'training_data']
            
            for table in tables:
                cursor.execute(f"DELETE FROM {table} WHERE timestamp < ?", (cutoff.isoformat(),))
            
            logger.info(f"Cleaned up data older than {days} days")


# ==================== STANDALONE FUNCTIONS ====================

def get_ml_database(db_path: str = "data/ml_trading.db") -> MLDatabase:
    """Get ML database instance."""
    return MLDatabase(db_path)


if __name__ == "__main__":
    # Test the database
    logging.basicConfig(level=logging.INFO)
    
    db = MLDatabase("data/test_ml.db")
    print("\n=== ML Database Test ===")
    print(f"Database stats: {db.get_statistics()}")
    
    # Test saving a prediction
    pred_id = db.save_prediction(
        timestamp=datetime.now(),
        symbol="BTCUSDT",
        model_name="xgboost_v1",
        prediction="BUY",
        probability=0.75,
        features={"rsi": 30, "macd": 1.5, "volume": 1000}
    )
    print(f"Saved prediction ID: {pred_id}")
    
    # Test saving price
    price_id = db.save_price(
        timestamp=datetime.now(),
        symbol="BTCUSDT",
        open_price=50000,
        high=51000,
        low=49000,
        close=50500,
        volume=1000
    )
    print(f"Saved price ID: {price_id}")
    
    # Test saving model performance
    perf_id = db.save_model_performance(
        timestamp=datetime.now(),
        model_name="xgboost_v1",
        accuracy=0.75,
        precision_score=0.72,
        recall_score=0.78,
        f1_score=0.75,
        auc_roc=0.80,
        training_samples=1000,
        test_samples=200
    )
    print(f"Saved performance ID: {perf_id}")
    
    # Test saving backtest
    bt_id = db.save_backtest_result(
        timestamp=datetime.now(),
        strategy_name="ml_strategy",
        symbol="BTCUSDT",
        initial_balance=10000,
        final_balance=15000,
        total_return=50.0,
        sharpe_ratio=1.5,
        max_drawdown=10.0,
        win_rate=60.0,
        total_trades=100,
        profitable_trades=60,
        losing_trades=40,
        avg_profit=100,
        avg_loss=50
    )
    print(f"Saved backtest ID: {bt_id}")
    
    print("\n=== Final Stats ===")
    print(f"Database stats: {db.get_statistics()}")
    print("\n✅ ML Database test completed!")

