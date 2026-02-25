"""
TimescaleDB Models for Time-Series Data
=======================================
High-performance time-series storage using TimescaleDB hypertables.
Optimized for financial market data with automatic partitioning.
"""

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from decimal import Decimal
import json

from sqlalchemy import (
    Column, Integer, Float, String, DateTime, Boolean,
    Text, JSON, ForeignKey, Index, UniqueConstraint,
    create_engine, text, select, func
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.dialects.postgresql import insert

Base = declarative_base()


# ============================================================================
# TIMESCALEDB HYPERTABLE MODELS
# ============================================================================

class OHLCVBar(Base):
    """
    OHLCV (Open-High-Low-Close-Volume) price data.
    Stored as TimescaleDB hypertable for efficient time-series queries.
    """
    __tablename__ = "ohlcv_bars"
    
    time = Column(DateTime, primary_key=True, nullable=False)
    symbol = Column(String(20), primary_key=True, nullable=False)
    interval = Column(String(10), primary_key=True, nullable=False)  # 1m, 5m, 1h, 1d
    
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, default=0.0)
    quote_volume = Column(Float, default=0.0)
    trades_count = Column(Integer, default=0)
    vwap = Column(Float, nullable=True)  # Volume Weighted Average Price
    
    # Additional metrics
    twap = Column(Float, nullable=True)  # Time Weighted Average Price
    volatility = Column(Float, nullable=True)  # Rolling volatility
    
    __table_args__ = (
        Index('ix_ohlcv_symbol_time', 'symbol', 'time'),
        Index('ix_ohlcv_time_desc', text('time DESC')),
        UniqueConstraint('time', 'symbol', 'interval', name='uq_ohlcv_bar'),
    )
    
    @classmethod
    def create_hypertable(cls, engine):
        """Create TimescaleDB hypertable for efficient time-series queries."""
        try:
            with engine.connect() as conn:
                # Create hypertable with 1-day chunks
                conn.execute(text("""
                    SELECT create_hypertable(
                        'ohlcv_bars', 
                        'time',
                        chunk_time_interval => INTERVAL '1 day',
                        if_not_exists => TRUE
                    );
                """))
                # Create compression policy
                conn.execute(text("""
                    ALTER TABLE ohlcv_bars SET (
                        timescaledb.compress,
                        timescaledb.compress_segmentby = 'symbol,interval'
                    );
                """))
                # Compress data older than 7 days
                conn.execute(text("""
                    SELECT add_compression_policy(
                        'ohlcv_bars',
                        INTERVAL '7 days',
                        if_not_exists => TRUE
                    );
                """))
                conn.commit()
        except Exception as e:
            print(f"TimescaleDB hypertable creation skipped (may not be available): {e}")


class TradeTick(Base):
    """
    Individual trade ticks for detailed analysis.
    High-frequency trade data storage.
    """
    __tablename__ = "trade_ticks"
    
    time = Column(DateTime, primary_key=True, nullable=False)
    symbol = Column(String(20), primary_key=True, nullable=False)
    trade_id = Column(String(50), nullable=False)
    
    price = Column(Float, nullable=False)
    quantity = Column(Float, nullable=False)
    quote_quantity = Column(Float, nullable=True)
    
    is_buyer_maker = Column(Boolean, default=False)
    is_best_match = Column(Boolean, default=True)
    
    # Market context
    bid_price = Column(Float, nullable=True)
    ask_price = Column(Float, nullable=True)
    spread = Column(Float, nullable=True)
    
    __table_args__ = (
        Index('ix_trades_symbol_time', 'symbol', 'time'),
        UniqueConstraint('time', 'symbol', 'trade_id', name='uq_trade_tick'),
    )
    
    @classmethod
    def create_hypertable(cls, engine):
        """Create TimescaleDB hypertable for trade ticks."""
        try:
            with engine.connect() as conn:
                conn.execute(text("""
                    SELECT create_hypertable(
                        'trade_ticks',
                        'time',
                        chunk_time_interval => INTERVAL '4 hours',
                        if_not_exists => TRUE
                    );
                """))
                conn.commit()
        except Exception as e:
            print(f"TimescaleDB hypertable creation skipped: {e}")


class OrderBookSnapshot(Base):
    """
    Order book snapshots for market depth analysis.
    Periodic snapshots of bid/ask levels.
    """
    __tablename__ = "orderbook_snapshots"
    
    time = Column(DateTime, primary_key=True, nullable=False)
    symbol = Column(String(20), primary_key=True, nullable=False)
    
    # Top of book
    best_bid = Column(Float, nullable=False)
    best_ask = Column(Float, nullable=False)
    spread = Column(Float, nullable=False)
    mid_price = Column(Float, nullable=False)
    
    # Depth levels (JSON arrays)
    bids = Column(JSON, nullable=True)  # [[price, qty], ...]
    asks = Column(JSON, nullable=True)
    
    # Aggregated depth
    bid_volume_1pct = Column(Float, nullable=True)  # Volume within 1% of mid
    ask_volume_1pct = Column(Float, nullable=True)
    bid_volume_5pct = Column(Float, nullable=True)
    ask_volume_5pct = Column(Float, nullable=True)
    
    # Imbalance metrics
    imbalance = Column(Float, nullable=True)  # (bid_vol - ask_vol) / total_vol
    pressure = Column(Float, nullable=True)  # Directional pressure indicator
    
    __table_args__ = (
        Index('ix_orderbook_symbol_time', 'symbol', 'time'),
    )
    
    @classmethod
    def create_hypertable(cls, engine):
        """Create TimescaleDB hypertable for orderbook snapshots."""
        try:
            with engine.connect() as conn:
                conn.execute(text("""
                    SELECT create_hypertable(
                        'orderbook_snapshots',
                        'time',
                        chunk_time_interval => INTERVAL '1 day',
                        if_not_exists => TRUE
                    );
                """))
                conn.commit()
        except Exception as e:
            print(f"TimescaleDB hypertable creation skipped: {e}")


class FundingRate(Base):
    """
    Perpetual futures funding rates.
    Important for crypto derivatives trading.
    """
    __tablename__ = "funding_rates"
    
    time = Column(DateTime, primary_key=True, nullable=False)
    symbol = Column(String(20), primary_key=True, nullable=False)
    exchange = Column(String(20), primary_key=True, nullable=False)
    
    funding_rate = Column(Float, nullable=False)
    funding_time = Column(DateTime, nullable=False)  # Next funding time
    estimated_rate = Column(Float, nullable=True)
    
    # Open interest
    open_interest = Column(Float, nullable=True)
    open_interest_value = Column(Float, nullable=True)
    
    __table_args__ = (
        Index('ix_funding_symbol_time', 'symbol', 'time'),
    )
    
    @classmethod
    def create_hypertable(cls, engine):
        """Create TimescaleDB hypertable for funding rates."""
        try:
            with engine.connect() as conn:
                conn.execute(text("""
                    SELECT create_hypertable(
                        'funding_rates',
                        'time',
                        chunk_time_interval => INTERVAL '1 day',
                        if_not_exists => TRUE
                    );
                """))
                conn.commit()
        except Exception as e:
            print(f"TimescaleDB hypertable creation skipped: {e}")


class LiquidationEvent(Base):
    """
    Liquidation events from exchanges.
    Important for market sentiment and cascade risk.
    """
    __tablename__ = "liquidation_events"
    
    time = Column(DateTime, primary_key=True, nullable=False)
    symbol = Column(String(20), primary_key=True, nullable=False)
    exchange = Column(String(20), primary_key=True, nullable=False)
    
    side = Column(String(10), nullable=False)  # BUY, SELL
    price = Column(Float, nullable=False)
    quantity = Column(Float, nullable=False)
    value = Column(Float, nullable=False)
    
    order_type = Column(String(20), nullable=True)
    time_in_force = Column(String(10), nullable=True)
    
    __table_args__ = (
        Index('ix_liquidation_symbol_time', 'symbol', 'time'),
    )
    
    @classmethod
    def create_hypertable(cls, engine):
        """Create TimescaleDB hypertable for liquidation events."""
        try:
            with engine.connect() as conn:
                conn.execute(text("""
                    SELECT create_hypertable(
                        'liquidation_events',
                        'time',
                        chunk_time_interval => INTERVAL '1 day',
                        if_not_exists => TRUE
                    );
                """))
                conn.commit()
        except Exception as e:
            print(f"TimescaleDB hypertable creation skipped: {e}")


class PortfolioHistory(Base):
    """
    Portfolio value and metrics history.
    Time-series of portfolio performance.
    """
    __tablename__ = "portfolio_history"
    
    time = Column(DateTime, primary_key=True, nullable=False)
    portfolio_id = Column(String(50), primary_key=True, nullable=False)
    
    # Value metrics
    total_value = Column(Float, nullable=False)
    cash = Column(Float, nullable=False)
    equity = Column(Float, nullable=False)
    
    # PnL metrics
    unrealized_pnl = Column(Float, default=0.0)
    realized_pnl = Column(Float, default=0.0)
    daily_pnl = Column(Float, default=0.0)
    
    # Risk metrics
    var_95 = Column(Float, nullable=True)
    var_99 = Column(Float, nullable=True)
    drawdown = Column(Float, default=0.0)
    leverage = Column(Float, default=1.0)
    
    # Position count
    num_positions = Column(Integer, default=0)
    num_long = Column(Integer, default=0)
    num_short = Column(Integer, default=0)
    
    # Performance metrics
    sharpe = Column(Float, nullable=True)
    sortino = Column(Float, nullable=True)
    win_rate = Column(Float, nullable=True)
    
    __table_args__ = (
        Index('ix_portfolio_time', 'time'),
    )
    
    @classmethod
    def create_hypertable(cls, engine):
        """Create TimescaleDB hypertable for portfolio history."""
        try:
            with engine.connect() as conn:
                conn.execute(text("""
                    SELECT create_hypertable(
                        'portfolio_history',
                        'time',
                        chunk_time_interval => INTERVAL '1 day',
                        if_not_exists => TRUE
                    );
                """))
                conn.commit()
        except Exception as e:
            print(f"TimescaleDB hypertable creation skipped: {e}")


class RiskMetricsHistory(Base):
    """
    Historical risk metrics for monitoring and analysis.
    """
    __tablename__ = "risk_metrics_history"
    
    time = Column(DateTime, primary_key=True, nullable=False)
    portfolio_id = Column(String(50), primary_key=True, nullable=False)
    
    # VaR metrics
    var_1d_95 = Column(Float, nullable=True)
    var_1d_99 = Column(Float, nullable=True)
    var_5d_95 = Column(Float, nullable=True)
    cvar_1d_95 = Column(Float, nullable=True)
    cvar_1d_99 = Column(Float, nullable=True)
    
    # Volatility
    volatility_daily = Column(Float, nullable=True)
    volatility_annualized = Column(Float, nullable=True)
    
    # Drawdown
    current_drawdown = Column(Float, default=0.0)
    max_drawdown = Column(Float, default=0.0)
    drawdown_duration = Column(Integer, default=0)  # days
    
    # Correlation
    avg_correlation = Column(Float, nullable=True)
    diversification_ratio = Column(Float, nullable=True)
    
    # Beta
    beta = Column(Float, nullable=True)
    tracking_error = Column(Float, nullable=True)
    
    # Liquidity
    liquidity_score = Column(Float, nullable=True)
    concentration_score = Column(Float, nullable=True)
    
    __table_args__ = (
        Index('ix_risk_metrics_time', 'time'),
    )
    
    @classmethod
    def create_hypertable(cls, engine):
        """Create TimescaleDB hypertable for risk metrics."""
        try:
            with engine.connect() as conn:
                conn.execute(text("""
                    SELECT create_hypertable(
                        'risk_metrics_history',
                        'time',
                        chunk_time_interval => INTERVAL '1 day',
                        if_not_exists => TRUE
                    );
                """))
                conn.commit()
        except Exception as e:
            print(f"TimescaleDB hypertable creation skipped: {e}")


class SignalTimeseries(Base):
    """
    Trading signals time-series data.
    TimescaleDB hypertable for signal analysis.
    """
    __tablename__ = "signals_ts"
    
    timestamp = Column(DateTime, primary_key=True, nullable=False)
    symbol = Column(String(20), primary_key=True, nullable=False)
    
    action = Column(String(10), nullable=False)  # BUY, SELL, HOLD
    confidence = Column(Float, default=0.0)
    price_at_signal = Column(Float, default=0.0)
    strategy = Column(String(50), nullable=True)
    
    # Execution tracking
    executed = Column(Boolean, default=False)
    execution_price = Column(Float, nullable=True)
    execution_time = Column(DateTime, nullable=True)
    result_pnl = Column(Float, nullable=True)
    
    # Analysis factors
    technical_score = Column(Float, nullable=True)
    sentiment_score = Column(Float, nullable=True)
    ml_score = Column(Float, nullable=True)
    monte_carlo_level = Column(Integer, default=1)
    
    __table_args__ = (
        Index('ix_signals_ts_symbol_time', 'symbol', 'timestamp'),
    )
    
    @classmethod
    def create_hypertable(cls, engine):
        """Create TimescaleDB hypertable for signals."""
        try:
            with engine.connect() as conn:
                conn.execute(text("""
                    SELECT create_hypertable(
                        'signals_ts',
                        'timestamp',
                        chunk_time_interval => INTERVAL '1 day',
                        if_not_exists => TRUE
                    );
                """))
                conn.commit()
        except Exception as e:
            print(f"TimescaleDB hypertable creation skipped: {e}")


# ============================================================================
# CONTINUOUS AGGREGATES
# ============================================================================

def create_continuous_aggregates(engine):
    """
    Create TimescaleDB continuous aggregates for efficient queries.
    Pre-aggregated views for common time-based queries.
    """
    aggregates = [
        # 5-minute OHLCV aggregate from 1-minute bars
        """
        CREATE MATERIALIZED VIEW IF NOT EXISTS ohlcv_5m
        WITH (timescaledb.continuous) AS
        SELECT
            time_bucket('5 minutes', time) AS bucket,
            symbol,
            FIRST(open, time) AS open,
            MAX(high) AS high,
            MIN(low) AS low,
            LAST(close, time) AS close,
            SUM(volume) AS volume,
            SUM(quote_volume) AS quote_volume,
            SUM(trades_count) AS trades_count
        FROM ohlcv_bars
        WHERE interval = '1m'
        GROUP BY bucket, symbol
        WITH DATA;
        """,
        # 1-hour OHLCV aggregate
        """
        CREATE MATERIALIZED VIEW IF NOT EXISTS ohlcv_1h
        WITH (timescaledb.continuous) AS
        SELECT
            time_bucket('1 hour', time) AS bucket,
            symbol,
            FIRST(open, time) AS open,
            MAX(high) AS high,
            MIN(low) AS low,
            LAST(close, time) AS close,
            SUM(volume) AS volume,
            SUM(quote_volume) AS quote_volume,
            SUM(trades_count) AS trades_count
        FROM ohlcv_bars
        WHERE interval = '1m'
        GROUP BY bucket, symbol
        WITH DATA;
        """,
        # Daily OHLCV aggregate
        """
        CREATE MATERIALIZED VIEW IF NOT EXISTS ohlcv_1d
        WITH (timescaledb.continuous) AS
        SELECT
            time_bucket('1 day', time) AS bucket,
            symbol,
            FIRST(open, time) AS open,
            MAX(high) AS high,
            MIN(low) AS low,
            LAST(close, time) AS close,
            SUM(volume) AS volume,
            SUM(quote_volume) AS quote_volume,
            SUM(trades_count) AS trades_count
        FROM ohlcv_bars
        WHERE interval = '1m'
        GROUP BY bucket, symbol
        WITH DATA;
        """,
        # Hourly trade volume aggregate
        """
        CREATE MATERIALIZED VIEW IF NOT EXISTS trade_volume_1h
        WITH (timescaledb.continuous) AS
        SELECT
            time_bucket('1 hour', time) AS bucket,
            symbol,
            COUNT(*) AS trade_count,
            SUM(quantity) AS total_volume,
            SUM(quantity * price) AS total_value,
            AVG(price) AS avg_price,
            STDDEV(price) AS price_std
        FROM trade_ticks
        GROUP BY bucket, symbol
        WITH DATA;
        """,
        # Daily signals aggregate (for trading signals analysis)
        # Uses signals_ts hypertable for time-series optimized queries
        """
        CREATE MATERIALIZED VIEW IF NOT EXISTS daily_signals
        WITH (timescaledb.continuous) AS
        SELECT
            time_bucket('1 day', timestamp) AS bucket,
            symbol,
            COUNT(*) AS signal_count,
            AVG(CASE WHEN action = 'BUY' THEN 1 ELSE 0 END) AS buy_ratio,
            AVG(confidence) AS avg_confidence,
            AVG(CASE WHEN executed = TRUE THEN 1 ELSE 0 END) AS execution_rate,
            SUM(CASE WHEN executed = TRUE AND result_pnl IS NOT NULL THEN result_pnl ELSE 0 END) AS total_result_pnl,
            AVG(technical_score) AS avg_technical_score,
            AVG(sentiment_score) AS avg_sentiment_score,
            AVG(ml_score) AS avg_ml_score
        FROM signals_ts
        WHERE timestamp > NOW() - INTERVAL '365 days'
        GROUP BY bucket, symbol
        WITH DATA;
        """,
        # Weekly portfolio performance aggregate
        """
        CREATE MATERIALIZED VIEW IF NOT EXISTS weekly_performance
        WITH (timescaledb.continuous) AS
        SELECT
            time_bucket('7 days', time) AS bucket,
            portfolio_id,
            FIRST(total_value, time) AS start_value,
            LAST(total_value, time) AS end_value,
            (LAST(total_value, time) - FIRST(total_value, time)) AS value_change,
            ((LAST(total_value, time) - FIRST(total_value, time)) / FIRST(total_value, time) * 100) AS return_pct,
            MAX(drawdown) AS max_drawdown,
            AVG(sharpe) AS avg_sharpe,
            AVG(win_rate) AS avg_win_rate,
            SUM(daily_pnl) AS total_pnl,
            COUNT(*) AS observations
        FROM portfolio_history
        WHERE time > NOW() - INTERVAL '2 years'
        GROUP BY bucket, portfolio_id
        WITH DATA;
        """,
        # Hourly risk metrics aggregate
        """
        CREATE MATERIALIZED VIEW IF NOT EXISTS hourly_risk_metrics
        WITH (timescaledb.continuous) AS
        SELECT
            time_bucket('1 hour', time) AS bucket,
            portfolio_id,
            AVG(var_1d_95) AS avg_var_95,
            AVG(var_1d_99) AS avg_var_99,
            MAX(current_drawdown) AS max_drawdown,
            AVG(volatility_annualized) AS avg_volatility,
            AVG(beta) AS avg_beta,
            AVG(liquidity_score) AS avg_liquidity
        FROM risk_metrics_history
        WHERE time > NOW() - INTERVAL '90 days'
        GROUP BY bucket, portfolio_id
        WITH DATA;
        """,
    ]
    
    refresh_policies = [
        # Refresh policies for continuous aggregates
        "SELECT add_continuous_aggregate_policy('ohlcv_5m', start_offset => INTERVAL '3 hours', end_offset => INTERVAL '5 minutes', schedule_interval => INTERVAL '5 minutes', if_not_exists => TRUE);",
        "SELECT add_continuous_aggregate_policy('ohlcv_1h', start_offset => INTERVAL '1 day', end_offset => INTERVAL '1 hour', schedule_interval => INTERVAL '1 hour', if_not_exists => TRUE);",
        "SELECT add_continuous_aggregate_policy('ohlcv_1d', start_offset => INTERVAL '7 days', end_offset => INTERVAL '1 day', schedule_interval => INTERVAL '1 day', if_not_exists => TRUE);",
        "SELECT add_continuous_aggregate_policy('trade_volume_1h', start_offset => INTERVAL '1 day', end_offset => INTERVAL '1 hour', schedule_interval => INTERVAL '1 hour', if_not_exists => TRUE);",
        # New aggregates refresh policies
        "SELECT add_continuous_aggregate_policy('daily_signals', start_offset => INTERVAL '7 days', end_offset => INTERVAL '1 day', schedule_interval => INTERVAL '1 day', if_not_exists => TRUE);",
        "SELECT add_continuous_aggregate_policy('weekly_performance', start_offset => INTERVAL '30 days', end_offset => INTERVAL '7 days', schedule_interval => INTERVAL '1 day', if_not_exists => TRUE);",
        "SELECT add_continuous_aggregate_policy('hourly_risk_metrics', start_offset => INTERVAL '7 days', end_offset => INTERVAL '1 hour', schedule_interval => INTERVAL '1 hour', if_not_exists => TRUE);",
    ]
    
    retention_policies = [
        # Retention policies for raw data (keep for different periods based on importance)
        # Raw 1-minute OHLCV: keep for 90 days
        "SELECT add_retention_policy('ohlcv_bars', INTERVAL '90 days', if_not_exists => TRUE);",
        # Trade ticks: keep for 30 days (high volume)
        "SELECT add_retention_policy('trade_ticks', INTERVAL '30 days', if_not_exists => TRUE);",
        # Orderbook snapshots: keep for 7 days (very high volume)
        "SELECT add_retention_policy('orderbook_snapshots', INTERVAL '7 days', if_not_exists => TRUE);",
        # Funding rates: keep for 1 year
        "SELECT add_retention_policy('funding_rates', INTERVAL '1 year', if_not_exists => TRUE);",
        # Liquidation events: keep for 1 year
        "SELECT add_retention_policy('liquidation_events', INTERVAL '1 year', if_not_exists => TRUE);",
        # Portfolio history: keep for 5 years
        "SELECT add_retention_policy('portfolio_history', INTERVAL '5 years', if_not_exists => TRUE);",
        # Risk metrics: keep for 2 years
        "SELECT add_retention_policy('risk_metrics_history', INTERVAL '2 years', if_not_exists => TRUE);",
        # Signals time-series: keep for 1 year
        "SELECT add_retention_policy('signals_ts', INTERVAL '1 year', if_not_exists => TRUE);",
    ]
    
    try:
        with engine.connect() as conn:
            for agg_sql in aggregates:
                try:
                    conn.execute(text(agg_sql))
                except Exception as e:
                    print(f"Aggregate creation skipped: {e}")
            
            for policy_sql in refresh_policies:
                try:
                    conn.execute(text(policy_sql))
                except Exception as e:
                    print(f"Policy creation skipped: {e}")
            
            for retention_sql in retention_policies:
                try:
                    conn.execute(text(retention_sql))
                except Exception as e:
                    print(f"Retention policy creation skipped: {e}")
            
            conn.commit()
    except Exception as e:
        print(f"Continuous aggregates creation skipped: {e}")


# ============================================================================
# DATABASE INITIALIZATION
# ============================================================================

def init_timescaledb(database_url: str):
    """
    Initialize TimescaleDB with all hypertables and aggregates.
    
    Args:
        database_url: PostgreSQL connection string with TimescaleDB extension
    """
    engine = create_engine(
        database_url,
        pool_size=20,
        max_overflow=40,
        pool_pre_ping=True,
        pool_recycle=3600,
        echo=False
    )
    
    # Create extension
    try:
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;"))
            conn.commit()
    except Exception as e:
        print(f"TimescaleDB extension not available: {e}")
    
    # Create tables
    Base.metadata.create_all(engine)
    
    # Create hypertables
    OHLCVBar.create_hypertable(engine)
    TradeTick.create_hypertable(engine)
    OrderBookSnapshot.create_hypertable(engine)
    FundingRate.create_hypertable(engine)
    LiquidationEvent.create_hypertable(engine)
    PortfolioHistory.create_hypertable(engine)
    RiskMetricsHistory.create_hypertable(engine)
    SignalTimeseries.create_hypertable(engine)
    
    # Create continuous aggregates
    create_continuous_aggregates(engine)
    
    return engine


# ============================================================================
# QUERY HELPERS
# ============================================================================

class TimeSeriesQueries:
    """Helper class for common time-series queries."""
    
    @staticmethod
    def get_ohlcv_range(session, symbol: str, interval: str, 
                         start: datetime, end: datetime) -> List[Dict]:
        """Get OHLCV bars for a time range."""
        query = select(OHLCVBar).where(
            OHLCVBar.symbol == symbol,
            OHLCVBar.interval == interval,
            OHLCVBar.time >= start,
            OHLCVBar.time <= end
        ).order_by(OHLCVBar.time)
        
        results = session.execute(query).scalars().all()
        return [
            {
                'time': r.time,
                'open': r.open,
                'high': r.high,
                'low': r.low,
                'close': r.close,
                'volume': r.volume,
                'vwap': r.vwap
            }
            for r in results
        ]
    
    @staticmethod
    def get_recent_trades(session, symbol: str, limit: int = 100) -> List[Dict]:
        """Get most recent trades for a symbol."""
        query = select(TradeTick).where(
            TradeTick.symbol == symbol
        ).order_by(TradeTick.time.desc()).limit(limit)
        
        results = session.execute(query).scalars().all()
        return [
            {
                'time': r.time,
                'price': r.price,
                'quantity': r.quantity,
                'is_buyer_maker': r.is_buyer_maker
            }
            for r in reversed(results)
        ]
    
    @staticmethod
    def get_portfolio_history(session, portfolio_id: str,
                               days: int = 30) -> List[Dict]:
        """Get portfolio history for the last N days."""
        start = datetime.utcnow() - timedelta(days=days)
        
        query = select(PortfolioHistory).where(
            PortfolioHistory.portfolio_id == portfolio_id,
            PortfolioHistory.time >= start
        ).order_by(PortfolioHistory.time)
        
        results = session.execute(query).scalars().all()
        return [
            {
                'time': r.time,
                'total_value': r.total_value,
                'unrealized_pnl': r.unrealized_pnl,
                'realized_pnl': r.realized_pnl,
                'drawdown': r.drawdown,
                'leverage': r.leverage
            }
            for r in results
        ]
    
    @staticmethod
    def get_risk_metrics_trend(session, portfolio_id: str,
                                days: int = 30) -> List[Dict]:
        """Get risk metrics trend for the last N days."""
        start = datetime.utcnow() - timedelta(days=days)
        
        query = select(RiskMetricsHistory).where(
            RiskMetricsHistory.portfolio_id == portfolio_id,
            RiskMetricsHistory.time >= start
        ).order_by(RiskMetricsHistory.time)
        
        results = session.execute(query).scalars().all()
        return [
            {
                'time': r.time,
                'var_1d_95': r.var_1d_95,
                'cvar_1d_95': r.cvar_1d_95,
                'volatility_annualized': r.volatility_annualized,
                'max_drawdown': r.max_drawdown,
                'beta': r.beta
            }
            for r in results
        ]
