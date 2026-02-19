"""
Database Package
===============
SQLAlchemy models and repository pattern for the trading system.
"""

from app.database.models import (
    Base,
    PriceRecord,
    MacroEvent,
    NaturalEvent,
    NewsRecord,
    InnovationRecord,
    OrderRecord,
    TradeRecord,
    PositionRecord,
    PortfolioSnapshot,
    SignalRecord,
    SourceWeight,
    get_engine,
    create_tables,
    get_session,
    init_database,
)

from app.database.repository import (
    TradingRepository,
    PriceRepository,
    MacroEventRepository,
    NewsRepository,
    OrderRepository,
    TradeRepository,
    PositionRepository,
    PortfolioRepository,
    SignalRepository,
    SourceWeightRepository,
)

__all__ = [
    "Base",
    "PriceRecord",
    "MacroEvent",
    "NaturalEvent",
    "NewsRecord",
    "InnovationRecord",
    "OrderRecord",
    "TradeRecord",
    "PositionRecord",
    "PortfolioSnapshot",
    "SignalRecord",
    "SourceWeight",
    "get_engine",
    "create_tables",
    "get_session",
    "init_database",
    "TradingRepository",
    "PriceRepository",
    "MacroEventRepository",
    "NewsRepository",
    "OrderRepository",
    "TradeRepository",
    "PositionRepository",
    "PortfolioRepository",
    "SignalRepository",
    "SourceWeightRepository",
]
