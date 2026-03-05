"""
Tests for app/database/models.py - Database Models
"""

import pytest
from datetime import datetime


class TestDatabaseEnums:
    """Test suite for database enums."""

    def test_order_side_enum(self):
        """Test OrderSideEnum values."""
        from app.database.models import OrderSideEnum
        
        assert OrderSideEnum.BUY.value == "BUY"
        assert OrderSideEnum.SELL.value == "SELL"

    def test_order_type_enum(self):
        """Test OrderTypeEnum values."""
        from app.database.models import OrderTypeEnum
        
        assert OrderTypeEnum.MARKET.value == "MARKET"
        assert OrderTypeEnum.LIMIT.value == "LIMIT"
        assert OrderTypeEnum.STOP_LOSS.value == "STOP_LOSS"
        assert OrderTypeEnum.TAKE_PROFIT.value == "TAKE_PROFIT"

    def test_order_status_enum(self):
        """Test OrderStatusEnum values."""
        from app.database.models import OrderStatusEnum
        
        assert OrderStatusEnum.NEW.value == "NEW"
        assert OrderStatusEnum.PARTIALLY_FILLED.value == "PARTIALLY_FILLED"
        assert OrderStatusEnum.FILLED.value == "FILLED"
        assert OrderStatusEnum.CANCELLED.value == "CANCELLED"
        assert OrderStatusEnum.REJECTED.value == "REJECTED"
        assert OrderStatusEnum.EXPIRED.value == "EXPIRED"


class TestDatabaseModels:
    """Test suite for database models."""

    def test_price_record_creation(self):
        """Test PriceRecord model creation."""
        from app.database.models import PriceRecord, Base
        from sqlalchemy import inspect
        
        # Check table name
        assert PriceRecord.__tablename__ == "prices"
        
        # Check columns exist
        mapper = inspect(PriceRecord)
        column_names = [col.key for col in mapper.columns]
        
        assert "id" in column_names
        assert "symbol" in column_names
        assert "timestamp" in column_names
        assert "open" in column_names
        assert "high" in column_names
        assert "low" in column_names
        assert "close" in column_names
        assert "volume" in column_names

    def test_macro_event_creation(self):
        """Test MacroEvent model creation."""
        from app.database.models import MacroEvent
        
        assert MacroEvent.__tablename__ == "macro_events"

    def test_natural_event_creation(self):
        """Test NaturalEvent model creation."""
        from app.database.models import NaturalEvent
        
        assert NaturalEvent.__tablename__ == "natural_events"

    def test_news_record_creation(self):
        """Test NewsRecord model creation."""
        from app.database.models import NewsRecord
        
        assert NewsRecord.__tablename__ == "news"

    def test_innovation_record_creation(self):
        """Test InnovationRecord model creation."""
        from app.database.models import InnovationRecord
        
        assert InnovationRecord.__tablename__ == "innovations"

    def test_order_record_creation(self):
        """Test OrderRecord model creation."""
        from app.database.models import OrderRecord
        
        assert OrderRecord.__tablename__ == "orders"

    def test_trade_record_creation(self):
        """Test TradeRecord model creation."""
        from app.database.models import TradeRecord
        
        assert TradeRecord.__tablename__ == "trades"

    def test_position_record_creation(self):
        """Test PositionRecord model creation."""
        from app.database.models import PositionRecord
        
        assert PositionRecord.__tablename__ == "positions"

    def test_portfolio_snapshot_creation(self):
        """Test PortfolioSnapshot model creation."""
        from app.database.models import PortfolioSnapshot
        
        assert PortfolioSnapshot.__tablename__ == "portfolio_snapshots"

    def test_signal_record_creation(self):
        """Test SignalRecord model creation."""
        from app.database.models import SignalRecord
        
        assert SignalRecord.__tablename__ == "signals"

    def test_energy_record_creation(self):
        """Test EnergyRecord model creation."""
        from app.database.models import EnergyRecord
        
        assert EnergyRecord.__tablename__ == "energy_records"


class TestBaseClass:
    """Test suite for Base class."""

    def test_base_declarative(self):
        """Test Base is declarative base."""
        from app.database.models import Base
        from sqlalchemy.orm import declarative_base
        
        # Base should be a declarative base
        assert hasattr(Base, 'registry')
        assert hasattr(Base, 'metadata')


class TestModelRelationships:
    """Test suite for model relationships."""

    def test_trade_order_relationship(self):
        """Test TradeRecord has relationship to OrderRecord."""
        from app.database.models import TradeRecord, OrderRecord
        
        # Check relationship exists
        assert hasattr(TradeRecord, 'order')


class TestDatabaseInit:
    """Test suite for database initialization."""

    def test_database_imports(self):
        """Test database module can be imported."""
        from app.database import models
        
        assert models is not None

    def test_database_enums_import(self):
        """Test database enums can be imported."""
        from app.database.models import OrderSideEnum, OrderTypeEnum, OrderStatusEnum
        
        assert OrderSideEnum is not None
        assert OrderTypeEnum is not None
        assert OrderStatusEnum is not None

    def test_main_models_import(self):
        """Test main models can be imported."""
        from app.database.models import (
            PriceRecord, MacroEvent, NaturalEvent, NewsRecord,
            InnovationRecord, OrderRecord, TradeRecord, PositionRecord,
            PortfolioSnapshot, SignalRecord, EnergyRecord
        )
        
        assert PriceRecord is not None
        assert MacroEvent is not None
        assert NaturalEvent is not None
        assert NewsRecord is not None
        assert InnovationRecord is not None
        assert OrderRecord is not None
        assert TradeRecord is not None
        assert PositionRecord is not None
        assert PortfolioSnapshot is not None
        assert SignalRecord is not None
        assert EnergyRecord is not None
