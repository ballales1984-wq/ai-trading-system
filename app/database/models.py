"""
Database Models
==============
SQLAlchemy models for the trading system.
Supports PostgreSQL (production) and SQLite (development).
"""

from datetime import datetime
from typing import Optional
from enum import Enum as PyEnum

from sqlalchemy import (
    Column, Integer, Float, String, DateTime, Boolean,
    Text, JSON, ForeignKey, Index, Enum, UniqueConstraint,
    create_engine
)
from sqlalchemy.orm import (
    declarative_base, relationship, sessionmaker, Session
)

Base = declarative_base()


# ============================================================================
# ENUMS
# ============================================================================

class OrderSideEnum(str, PyEnum):
    BUY = "BUY"
    SELL = "SELL"


class OrderTypeEnum(str, PyEnum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"


class OrderStatusEnum(str, PyEnum):
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


# ============================================================================
# MODELS
# ============================================================================

class PriceRecord(Base):
    """Historical and real-time price data."""
    __tablename__ = "prices"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, default=0.0)
    quote_volume = Column(Float, default=0.0)
    interval = Column(String(10), default="1m")  # 1m, 5m, 1h, 1d
    source = Column(String(30), default="binance")

    __table_args__ = (
        UniqueConstraint("symbol", "timestamp", "interval", name="uq_price_record"),
        Index("ix_prices_symbol_ts", "symbol", "timestamp"),
    )


class MacroEvent(Base):
    """Economic calendar / macro events."""
    __tablename__ = "macro_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_date = Column(DateTime, nullable=False, index=True)
    event_type = Column(String(100), nullable=False)
    country = Column(String(5), default="US")
    impact = Column(String(10), default="medium")  # low, medium, high
    actual = Column(Float, nullable=True)
    forecast = Column(Float, nullable=True)
    previous = Column(Float, nullable=True)
    description = Column(Text, nullable=True)
    source = Column(String(30), default="trading_economics")

    __table_args__ = (
        Index("ix_macro_date_impact", "event_date", "impact"),
    )


class NaturalEvent(Base):
    """Natural / climate events."""
    __tablename__ = "natural_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_date = Column(DateTime, nullable=False, index=True)
    event_type = Column(String(100), nullable=False)  # drought, hurricane, flood
    region = Column(String(100), nullable=True)
    intensity = Column(Float, default=0.0)  # 0-1 scale
    description = Column(Text, nullable=True)
    affected_commodities = Column(JSON, nullable=True)  # ["oil", "gas", "wheat"]
    source = Column(String(30), default="open_meteo")


class NewsRecord(Base):
    """News articles with sentiment scores."""
    __tablename__ = "news"

    id = Column(Integer, primary_key=True, autoincrement=True)
    published_at = Column(DateTime, nullable=False, index=True)
    title = Column(String(500), nullable=False)
    summary = Column(Text, nullable=True)
    sentiment_score = Column(Float, default=0.0)  # -1 to +1
    sentiment_label = Column(String(20), default="neutral")
    relevance_score = Column(Float, default=0.0)  # 0-1
    source = Column(String(50), nullable=True)
    source_url = Column(String(500), nullable=True)
    symbols = Column(JSON, nullable=True)  # ["BTCUSDT", "ETHUSDT"]
    api_source = Column(String(30), default="newsapi")

    __table_args__ = (
        Index("ix_news_published", "published_at"),
    )


class InnovationRecord(Base):
    """Technology / innovation events."""
    __tablename__ = "innovations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_date = Column(DateTime, nullable=False, index=True)
    innovation_type = Column(String(100), nullable=False)
    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=True)
    potential_impact = Column(Float, default=0.0)  # 0-1
    affected_sectors = Column(JSON, nullable=True)
    source = Column(String(30), default="google_patents")


class OrderRecord(Base):
    """Trading orders."""
    __tablename__ = "orders"

    id = Column(Integer, primary_key=True, autoincrement=True)
    order_id = Column(String(50), unique=True, nullable=False, index=True)
    broker_order_id = Column(String(50), nullable=True)
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(String(10), nullable=False)
    order_type = Column(String(20), nullable=False)
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=True)
    stop_price = Column(Float, nullable=True)
    filled_quantity = Column(Float, default=0.0)
    avg_fill_price = Column(Float, default=0.0)
    commission = Column(Float, default=0.0)
    status = Column(String(20), default="NEW")
    broker = Column(String(20), default="binance")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    strategy = Column(String(50), nullable=True)
    notes = Column(Text, nullable=True)

    __table_args__ = (
        Index("ix_orders_symbol_status", "symbol", "status"),
    )


class TradeRecord(Base):
    """Executed trades (fills)."""
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(String(50), unique=True, nullable=False)
    order_id = Column(String(50), ForeignKey("orders.order_id"), nullable=False)
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(String(10), nullable=False)
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    commission = Column(Float, default=0.0)
    pnl = Column(Float, default=0.0)
    timestamp = Column(DateTime, default=datetime.utcnow)
    broker = Column(String(20), default="binance")

    order = relationship("OrderRecord", backref="trades")


class PositionRecord(Base):
    """Current positions snapshot."""
    __tablename__ = "positions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(String(10), default="LONG")
    quantity = Column(Float, default=0.0)
    entry_price = Column(Float, default=0.0)
    current_price = Column(Float, default=0.0)
    unrealized_pnl = Column(Float, default=0.0)
    realized_pnl = Column(Float, default=0.0)
    leverage = Column(Float, default=1.0)
    opened_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    strategy = Column(String(50), nullable=True)


class PortfolioSnapshot(Base):
    """Portfolio state snapshots for performance tracking."""
    __tablename__ = "portfolio_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    total_equity = Column(Float, default=0.0)
    available_balance = Column(Float, default=0.0)
    unrealized_pnl = Column(Float, default=0.0)
    realized_pnl = Column(Float, default=0.0)
    num_positions = Column(Integer, default=0)
    drawdown = Column(Float, default=0.0)
    sharpe_ratio = Column(Float, nullable=True)
    metadata_json = Column(JSON, nullable=True)


class SignalRecord(Base):
    """Trading signals generated by the engine."""
    __tablename__ = "signals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    action = Column(String(10), nullable=False)  # BUY, SELL, HOLD
    confidence = Column(Float, default=0.0)
    price_at_signal = Column(Float, default=0.0)
    strategy = Column(String(50), nullable=True)
    monte_carlo_level = Column(Integer, default=1)
    factors = Column(JSON, nullable=True)  # {"technical": 0.7, "sentiment": 0.3}
    executed = Column(Boolean, default=False)
    result_pnl = Column(Float, nullable=True)


class EnergyRecord(Base):
    """Energy commodity data from EIA (oil, gas, etc.)."""
    __tablename__ = "energy_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    energy_type = Column(String(50), nullable=False)  # crude_oil, natural_gas, etc.
    product_name = Column(String(100), nullable=True)  # WTI, Brent, Henry Hub, etc.
    value = Column(Float, nullable=False)
    unit = Column(String(20), nullable=True)  # USD/barrel, USD/MMBtu, etc.
    area = Column(String(50), nullable=True)  # US, OPEC, Europe, etc.
    source = Column(String(30), default="eia")
    metadata_json = Column(JSON, nullable=True)

    __table_args__ = (
        Index("ix_energy_type_ts", "energy_type", "timestamp"),
    )


class SourceWeight(Base):
    """API source reliability weights (reinforcement learning)."""
    __tablename__ = "source_weights"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_name = Column(String(50), unique=True, nullable=False)
    weight = Column(Float, default=1.0)
    accuracy = Column(Float, default=0.5)
    total_predictions = Column(Integer, default=0)
    correct_predictions = Column(Integer, default=0)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# ============================================================================
# DATABASE SETUP
# ============================================================================

def get_engine(database_url: str = "sqlite:///data/trading.db"):
    """Create SQLAlchemy engine."""
    return create_engine(
        database_url,
        echo=False,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20,
    )


def create_tables(engine):
    """Create all tables."""
    Base.metadata.create_all(engine)


def get_session(engine) -> Session:
    """Create a new database session."""
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()


def init_database(database_url: str = "sqlite:///data/trading.db") -> Session:
    """Initialize database and return session."""
    engine = get_engine(database_url)
    create_tables(engine)
    return get_session(engine)
