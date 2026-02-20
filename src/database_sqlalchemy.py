"""
SQLAlchemy Database Module for AI Trading System
=================================================
Modern database layer using SQLAlchemy ORM with support for:
- SQLite (development)
- PostgreSQL (production)
- Alembic migrations

Author: AI Trading System
Version: 2.0.0
"""

import logging
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from contextlib import contextmanager

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Text, Boolean,
    DateTime, ForeignKey, Index, UniqueConstraint, event
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.pool import StaticPool
from sqlalchemy.dialects.postgresql import JSONB

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

Base = declarative_base()


# ==================== MODELS ====================

class MLPrediction(Base):
    """ML Predictions table"""
    __tablename__ = 'ml_predictions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    model_name = Column(String(100), nullable=False)
    prediction = Column(String(20), nullable=False)
    probability = Column(Float)
    features = Column(Text)  # JSON string for SQLite, JSONB for PostgreSQL
    actual_outcome = Column(String(20))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_pred_timestamp_symbol', 'timestamp', 'symbol'),
    )


class PriceHistory(Base):
    """Price History table"""
    __tablename__ = 'price_history'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    source = Column(String(50), default='binance')
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint('timestamp', 'symbol', name='uix_price_timestamp_symbol'),
        Index('idx_price_timestamp_symbol', 'timestamp', 'symbol'),
    )


class ModelPerformance(Base):
    """Model Performance Metrics table"""
    __tablename__ = 'model_performance'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    model_name = Column(String(100), nullable=False, index=True)
    accuracy = Column(Float)
    precision_score = Column(Float)
    recall_score = Column(Float)
    f1_score = Column(Float)
    auc_roc = Column(Float)
    confusion_matrix = Column(Text)  # JSON string
    training_samples = Column(Integer)
    test_samples = Column(Integer)
    training_time_seconds = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)


class BacktestResult(Base):
    """Backtest Results table"""
    __tablename__ = 'backtest_results'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    strategy_name = Column(String(100), nullable=False, index=True)
    symbol = Column(String(20), nullable=False)
    initial_balance = Column(Float, nullable=False)
    final_balance = Column(Float, nullable=False)
    total_return = Column(Float)
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    win_rate = Column(Float)
    total_trades = Column(Integer)
    profitable_trades = Column(Integer)
    losing_trades = Column(Integer)
    avg_profit = Column(Float)
    avg_loss = Column(Float)
    params = Column(Text)  # JSON string
    created_at = Column(DateTime, default=datetime.utcnow)


class ModelMetadata(Base):
    """Model Metadata table"""
    __tablename__ = 'model_metadata'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(100), nullable=False, unique=True)
    model_type = Column(String(50))
    version = Column(String(20))
    created_at = Column(DateTime, default=datetime.utcnow)
    last_trained = Column(DateTime)
    accuracy = Column(Float)
    is_active = Column(Boolean, default=True)
    filepath = Column(String(500))


class TrainingData(Base):
    """Training Data table"""
    __tablename__ = 'training_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    features = Column(Text, nullable=False)  # JSON string
    target = Column(Integer, nullable=False)
    timeframe = Column(String(10))
    created_at = Column(DateTime, default=datetime.utcnow)


class TradingSignalRecord(Base):
    """Trading Signals Records table"""
    __tablename__ = 'trading_signals'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    action = Column(String(10), nullable=False)
    confidence = Column(Float)
    entry_price = Column(Float)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    position_size = Column(Float)
    technical_score = Column(Float)
    sentiment_score = Column(Float)
    ml_score = Column(Float)
    reason = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


# ==================== DATABASE MANAGER ====================

class SQLAlchemyDatabase:
    """
    SQLAlchemy Database Manager for AI Trading System.
    
    Supports:
    - SQLite (default for development)
    - PostgreSQL (for production)
    - Connection pooling
    - Session management
    """
    
    def __init__(
        self,
        db_url: Optional[str] = None,
        db_path: str = "data/ml_trading.db",
        echo: bool = False
    ):
        """
        Initialize SQLAlchemy database.
        
        Args:
            db_url: Full database URL (e.g., postgresql://user:pass@host/db)
            db_path: Path for SQLite database (used if db_url not provided)
            echo: Echo SQL statements for debugging
        """
        if db_url:
            self.db_url = db_url
            self.engine = create_engine(
                db_url,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,
                echo=echo
            )
        else:
            # SQLite for development
            self.db_url = f"sqlite:///{db_path}"
            self.engine = create_engine(
                self.db_url,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
                echo=echo
            )
        
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        self._ensure_db_dir(db_path)
        self._init_database()
        
        logger.info(f"SQLAlchemy Database initialized: {self.db_url}")
    
    def _ensure_db_dir(self, db_path: str):
        """Ensure database directory exists for SQLite."""
        if 'sqlite' in self.db_url:
            from pathlib import Path
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    def _init_database(self):
        """Initialize database schema."""
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database tables created successfully")
    
    @contextmanager
    def get_session(self) -> Session:
        """Context manager for database sessions."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
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
        """Save ML prediction to database."""
        with self.get_session() as session:
            pred = MLPrediction(
                timestamp=timestamp,
                symbol=symbol,
                model_name=model_name,
                prediction=prediction,
                probability=probability,
                features=json.dumps(features) if features else None,
                actual_outcome=actual_outcome
            )
            session.add(pred)
            session.flush()
            return pred.id
    
    def get_predictions(
        self,
        symbol: Optional[str] = None,
        model_name: Optional[str] = None,
        limit: int = 100
    ) -> pd.DataFrame:
        """Get ML predictions from database."""
        with self.get_session() as session:
            query = session.query(MLPrediction)
            
            if symbol:
                query = query.filter(MLPrediction.symbol == symbol)
            if model_name:
                query = query.filter(MLPrediction.model_name == model_name)
            
            query = query.order_by(MLPrediction.timestamp.desc()).limit(limit)
            
            results = query.all()
            
            if not results:
                return pd.DataFrame()
            
            data = [{
                'id': r.id,
                'timestamp': r.timestamp,
                'symbol': r.symbol,
                'model_name': r.model_name,
                'prediction': r.prediction,
                'probability': r.probability,
                'features': json.loads(r.features) if r.features else None,
                'actual_outcome': r.actual_outcome,
                'created_at': r.created_at
            } for r in results]
            
            return pd.DataFrame(data)
    
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
        """Save price data to database."""
        with self.get_session() as session:
            price = PriceHistory(
                timestamp=timestamp,
                symbol=symbol,
                open=open_price,
                high=high,
                low=low,
                close=close,
                volume=volume,
                source=source
            )
            session.merge(price)  # Use merge for upsert
            session.flush()
            return price.id
    
    def save_prices_bulk(self, prices: List[Dict]):
        """Save multiple price records."""
        with self.get_session() as session:
            for p in prices:
                price = PriceHistory(
                    timestamp=p.get('timestamp'),
                    symbol=p['symbol'],
                    open=p.get('open'),
                    high=p.get('high'),
                    low=p.get('low'),
                    close=p.get('close'),
                    volume=p.get('volume'),
                    source=p.get('source', 'binance')
                )
                session.merge(price)
    
    def get_price_history(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """Get price history from database."""
        with self.get_session() as session:
            query = session.query(PriceHistory).filter(PriceHistory.symbol == symbol)
            
            if start_time:
                query = query.filter(PriceHistory.timestamp >= start_time)
            if end_time:
                query = query.filter(PriceHistory.timestamp <= end_time)
            
            query = query.order_by(PriceHistory.timestamp.desc()).limit(limit)
            
            results = query.all()
            
            if not results:
                return pd.DataFrame()
            
            data = [{
                'timestamp': r.timestamp,
                'symbol': r.symbol,
                'open': r.open,
                'high': r.high,
                'low': r.low,
                'close': r.close,
                'volume': r.volume,
                'source': r.source
            } for r in results]
            
            return pd.DataFrame(data)
    
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
        """Save model performance metrics."""
        with self.get_session() as session:
            perf = ModelPerformance(
                timestamp=timestamp,
                model_name=model_name,
                accuracy=accuracy,
                precision_score=precision_score,
                recall_score=recall_score,
                f1_score=f1_score,
                auc_roc=auc_roc,
                confusion_matrix=json.dumps(confusion_matrix) if confusion_matrix else None,
                training_samples=training_samples,
                test_samples=test_samples,
                training_time_seconds=training_time_seconds
            )
            session.add(perf)
            session.flush()
            return perf.id
    
    def get_best_model(self, metric: str = "accuracy") -> Optional[Dict]:
        """Get best performing model based on metric."""
        allowed_metrics = ["accuracy", "precision_score", "recall_score", "f1_score", "auc_roc"]
        if metric not in allowed_metrics:
            metric = "accuracy"
        
        with self.get_session() as session:
            result = session.query(ModelPerformance).order_by(
                getattr(ModelPerformance, metric).desc()
            ).first()
            
            if result:
                return {
                    'model_name': result.model_name,
                    'accuracy': result.accuracy,
                    'f1_score': result.f1_score,
                    'timestamp': result.timestamp
                }
            return None
    
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
        """Save backtest results."""
        with self.get_session() as session:
            bt = BacktestResult(
                timestamp=timestamp,
                strategy_name=strategy_name,
                symbol=symbol,
                initial_balance=initial_balance,
                final_balance=final_balance,
                total_return=total_return,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                total_trades=total_trades,
                profitable_trades=profitable_trades,
                losing_trades=losing_trades,
                avg_profit=avg_profit,
                avg_loss=avg_loss,
                params=json.dumps(params) if params else None
            )
            session.add(bt)
            session.flush()
            return bt.id
    
    # ==================== TRADING SIGNALS ====================
    
    def save_trading_signal(
        self,
        timestamp: datetime,
        symbol: str,
        action: str,
        confidence: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        position_size: float,
        technical_score: float = 0.0,
        sentiment_score: float = 0.0,
        ml_score: float = 0.0,
        reason: str = ""
    ) -> int:
        """Save trading signal to database."""
        with self.get_session() as session:
            signal = TradingSignalRecord(
                timestamp=timestamp,
                symbol=symbol,
                action=action,
                confidence=confidence,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                technical_score=technical_score,
                sentiment_score=sentiment_score,
                ml_score=ml_score,
                reason=reason
            )
            session.add(signal)
            session.flush()
            return signal.id
    
    def get_recent_signals(self, symbol: Optional[str] = None, limit: int = 50) -> pd.DataFrame:
        """Get recent trading signals."""
        with self.get_session() as session:
            query = session.query(TradingSignalRecord)
            
            if symbol:
                query = query.filter(TradingSignalRecord.symbol == symbol)
            
            query = query.order_by(TradingSignalRecord.timestamp.desc()).limit(limit)
            
            results = query.all()
            
            if not results:
                return pd.DataFrame()
            
            data = [{
                'timestamp': r.timestamp,
                'symbol': r.symbol,
                'action': r.action,
                'confidence': r.confidence,
                'entry_price': r.entry_price,
                'stop_loss': r.stop_loss,
                'take_profit': r.take_profit,
                'position_size': r.position_size,
                'technical_score': r.technical_score,
                'sentiment_score': r.sentiment_score,
                'ml_score': r.ml_score,
                'reason': r.reason
            } for r in results]
            
            return pd.DataFrame(data)


# ==================== FACTORY FUNCTION ====================

def get_database(
    db_url: Optional[str] = None,
    db_path: str = "data/ml_trading.db"
) -> SQLAlchemyDatabase:
    """
    Factory function to get database instance.
    
    Args:
        db_url: PostgreSQL URL for production (optional)
        db_path: SQLite path for development
        
    Returns:
        SQLAlchemyDatabase instance
    """
    # Check environment variables for production database
    if not db_url:
        db_url = os.environ.get('DATABASE_URL')
    
    return SQLAlchemyDatabase(db_url=db_url, db_path=db_path)


# ==================== COMPATIBILITY LAYER ====================

class MLDatabase(SQLAlchemyDatabase):
    """
    Compatibility layer for existing code using MLDatabase.
    Extends SQLAlchemyDatabase with legacy method names.
    """
    
    def __init__(self, db_path: str = "data/ml_trading.db"):
        super().__init__(db_path=db_path)
    
    def update_actual_outcome(self, prediction_id: int, actual_outcome: str):
        """Update actual outcome for a prediction."""
        with self.get_session() as session:
            pred = session.query(MLPrediction).filter(MLPrediction.id == prediction_id).first()
            if pred:
                pred.actual_outcome = actual_outcome
    
    def get_model_performance(
        self,
        model_name: Optional[str] = None,
        limit: int = 50
    ) -> pd.DataFrame:
        """Get model performance records."""
        with self.get_session() as session:
            query = session.query(ModelPerformance)
            
            if model_name:
                query = query.filter(ModelPerformance.model_name == model_name)
            
            query = query.order_by(ModelPerformance.timestamp.desc()).limit(limit)
            
            results = query.all()
            
            if not results:
                return pd.DataFrame()
            
            data = [{
                'id': r.id,
                'timestamp': r.timestamp,
                'model_name': r.model_name,
                'accuracy': r.accuracy,
                'precision_score': r.precision_score,
                'recall_score': r.recall_score,
                'f1_score': r.f1_score,
                'auc_roc': r.auc_roc,
                'training_samples': r.training_samples,
                'test_samples': r.test_samples,
                'training_time_seconds': r.training_time_seconds
            } for r in results]
            
            return pd.DataFrame(data)
    
    def get_backtest_results(
        self,
        strategy_name: Optional[str] = None,
        symbol: Optional[str] = None,
        limit: int = 50
    ) -> pd.DataFrame:
        """Get backtest results."""
        with self.get_session() as session:
            query = session.query(BacktestResult)
            
            if strategy_name:
                query = query.filter(BacktestResult.strategy_name == strategy_name)
            if symbol:
                query = query.filter(BacktestResult.symbol == symbol)
            
            query = query.order_by(BacktestResult.timestamp.desc()).limit(limit)
            
            results = query.all()
            
            if not results:
                return pd.DataFrame()
            
            data = [{
                'id': r.id,
                'timestamp': r.timestamp,
                'strategy_name': r.strategy_name,
                'symbol': r.symbol,
                'initial_balance': r.initial_balance,
                'final_balance': r.final_balance,
                'total_return': r.total_return,
                'sharpe_ratio': r.sharpe_ratio,
                'max_drawdown': r.max_drawdown,
                'win_rate': r.win_rate,
                'total_trades': r.total_trades,
                'profitable_trades': r.profitable_trades,
                'losing_trades': r.losing_trades,
                'avg_profit': r.avg_profit,
                'avg_loss': r.avg_loss,
                'params': json.loads(r.params) if r.params else None
            } for r in results]
            
            return pd.DataFrame(data)
