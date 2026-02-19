"""
Data Repository
==============
Repository pattern for database operations.
Provides CRUD operations for all trading entities.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from sqlalchemy.orm import Session
from sqlalchemy import desc, func, and_

from app.database.models import (
    PriceRecord, MacroEvent, NaturalEvent, NewsRecord,
    InnovationRecord, OrderRecord, TradeRecord, PositionRecord,
    PortfolioSnapshot, SignalRecord, SourceWeight
)

logger = logging.getLogger(__name__)


class BaseRepository:
    """Base repository with common CRUD operations."""

    def __init__(self, session: Session):
        self.session = session

    def commit(self):
        """Commit current transaction."""
        try:
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            logger.error(f"Commit failed: {e}")
            raise

    def close(self):
        """Close session."""
        self.session.close()


class PriceRepository(BaseRepository):
    """Repository for price data."""

    def save_price(self, record: PriceRecord) -> PriceRecord:
        """Save a price record (upsert)."""
        existing = (
            self.session.query(PriceRecord)
            .filter_by(
                symbol=record.symbol,
                timestamp=record.timestamp,
                interval=record.interval,
            )
            .first()
        )
        if existing:
            existing.open = record.open
            existing.high = record.high
            existing.low = record.low
            existing.close = record.close
            existing.volume = record.volume
            existing.quote_volume = record.quote_volume
            self.commit()
            return existing
        self.session.add(record)
        self.commit()
        return record

    def save_prices_bulk(self, records: List[PriceRecord]):
        """Bulk insert price records."""
        self.session.bulk_save_objects(records)
        self.commit()

    def get_prices(
        self,
        symbol: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        interval: str = "1m",
        limit: int = 1000,
    ) -> List[PriceRecord]:
        """Get price records for a symbol."""
        query = (
            self.session.query(PriceRecord)
            .filter_by(symbol=symbol, interval=interval)
        )
        if start:
            query = query.filter(PriceRecord.timestamp >= start)
        if end:
            query = query.filter(PriceRecord.timestamp <= end)
        return query.order_by(PriceRecord.timestamp).limit(limit).all()

    def get_latest_price(self, symbol: str) -> Optional[PriceRecord]:
        """Get the most recent price for a symbol."""
        return (
            self.session.query(PriceRecord)
            .filter_by(symbol=symbol)
            .order_by(desc(PriceRecord.timestamp))
            .first()
        )


class MacroEventRepository(BaseRepository):
    """Repository for macro economic events."""

    def save_event(self, event: MacroEvent) -> MacroEvent:
        self.session.add(event)
        self.commit()
        return event

    def get_events(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        impact: Optional[str] = None,
        country: Optional[str] = None,
    ) -> List[MacroEvent]:
        query = self.session.query(MacroEvent)
        if start:
            query = query.filter(MacroEvent.event_date >= start)
        if end:
            query = query.filter(MacroEvent.event_date <= end)
        if impact:
            query = query.filter_by(impact=impact)
        if country:
            query = query.filter_by(country=country)
        return query.order_by(MacroEvent.event_date).all()

    def get_upcoming_events(self, days: int = 7) -> List[MacroEvent]:
        now = datetime.utcnow()
        return self.get_events(start=now, end=now + timedelta(days=days))


class NewsRepository(BaseRepository):
    """Repository for news and sentiment data."""

    def save_news(self, record: NewsRecord) -> NewsRecord:
        self.session.add(record)
        self.commit()
        return record

    def save_news_bulk(self, records: List[NewsRecord]):
        self.session.bulk_save_objects(records)
        self.commit()

    def get_news(
        self,
        symbol: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        min_sentiment: Optional[float] = None,
        limit: int = 100,
    ) -> List[NewsRecord]:
        query = self.session.query(NewsRecord)
        if start:
            query = query.filter(NewsRecord.published_at >= start)
        if end:
            query = query.filter(NewsRecord.published_at <= end)
        if min_sentiment is not None:
            query = query.filter(NewsRecord.sentiment_score >= min_sentiment)
        if symbol:
            query = query.filter(NewsRecord.symbols.contains([symbol]))
        return query.order_by(desc(NewsRecord.published_at)).limit(limit).all()

    def get_average_sentiment(
        self, symbol: str, hours: int = 24
    ) -> float:
        since = datetime.utcnow() - timedelta(hours=hours)
        result = (
            self.session.query(func.avg(NewsRecord.sentiment_score))
            .filter(
                NewsRecord.published_at >= since,
                NewsRecord.symbols.contains([symbol]),
            )
            .scalar()
        )
        return result or 0.0


class OrderRepository(BaseRepository):
    """Repository for orders."""

    def save_order(self, order: OrderRecord) -> OrderRecord:
        existing = (
            self.session.query(OrderRecord)
            .filter_by(order_id=order.order_id)
            .first()
        )
        if existing:
            existing.status = order.status
            existing.filled_quantity = order.filled_quantity
            existing.avg_fill_price = order.avg_fill_price
            existing.commission = order.commission
            existing.updated_at = datetime.utcnow()
            self.commit()
            return existing
        self.session.add(order)
        self.commit()
        return order

    def get_order(self, order_id: str) -> Optional[OrderRecord]:
        return (
            self.session.query(OrderRecord)
            .filter_by(order_id=order_id)
            .first()
        )

    def get_open_orders(self, symbol: Optional[str] = None) -> List[OrderRecord]:
        query = self.session.query(OrderRecord).filter(
            OrderRecord.status.in_(["NEW", "PARTIALLY_FILLED"])
        )
        if symbol:
            query = query.filter_by(symbol=symbol)
        return query.order_by(desc(OrderRecord.created_at)).all()

    def get_order_history(
        self,
        symbol: Optional[str] = None,
        limit: int = 100,
    ) -> List[OrderRecord]:
        query = self.session.query(OrderRecord)
        if symbol:
            query = query.filter_by(symbol=symbol)
        return query.order_by(desc(OrderRecord.created_at)).limit(limit).all()


class TradeRepository(BaseRepository):
    """Repository for trades."""

    def save_trade(self, trade: TradeRecord) -> TradeRecord:
        self.session.add(trade)
        self.commit()
        return trade

    def get_trades(
        self,
        symbol: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[TradeRecord]:
        query = self.session.query(TradeRecord)
        if symbol:
            query = query.filter_by(symbol=symbol)
        if start:
            query = query.filter(TradeRecord.timestamp >= start)
        if end:
            query = query.filter(TradeRecord.timestamp <= end)
        return query.order_by(desc(TradeRecord.timestamp)).limit(limit).all()

    def get_total_pnl(self, symbol: Optional[str] = None) -> float:
        query = self.session.query(func.sum(TradeRecord.pnl))
        if symbol:
            query = query.filter_by(symbol=symbol)
        return query.scalar() or 0.0


class PositionRepository(BaseRepository):
    """Repository for positions."""

    def save_position(self, position: PositionRecord) -> PositionRecord:
        existing = (
            self.session.query(PositionRecord)
            .filter_by(symbol=position.symbol)
            .first()
        )
        if existing:
            existing.quantity = position.quantity
            existing.current_price = position.current_price
            existing.unrealized_pnl = position.unrealized_pnl
            existing.updated_at = datetime.utcnow()
            self.commit()
            return existing
        self.session.add(position)
        self.commit()
        return position

    def get_positions(self) -> List[PositionRecord]:
        return (
            self.session.query(PositionRecord)
            .filter(PositionRecord.quantity != 0)
            .all()
        )

    def get_position(self, symbol: str) -> Optional[PositionRecord]:
        return (
            self.session.query(PositionRecord)
            .filter_by(symbol=symbol)
            .first()
        )

    def close_position(self, symbol: str, realized_pnl: float):
        position = self.get_position(symbol)
        if position:
            position.quantity = 0
            position.realized_pnl += realized_pnl
            position.updated_at = datetime.utcnow()
            self.commit()


class PortfolioRepository(BaseRepository):
    """Repository for portfolio snapshots."""

    def save_snapshot(self, snapshot: PortfolioSnapshot) -> PortfolioSnapshot:
        self.session.add(snapshot)
        self.commit()
        return snapshot

    def get_snapshots(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[PortfolioSnapshot]:
        query = self.session.query(PortfolioSnapshot)
        if start:
            query = query.filter(PortfolioSnapshot.timestamp >= start)
        if end:
            query = query.filter(PortfolioSnapshot.timestamp <= end)
        return query.order_by(PortfolioSnapshot.timestamp).limit(limit).all()

    def get_latest_snapshot(self) -> Optional[PortfolioSnapshot]:
        return (
            self.session.query(PortfolioSnapshot)
            .order_by(desc(PortfolioSnapshot.timestamp))
            .first()
        )


class SignalRepository(BaseRepository):
    """Repository for trading signals."""

    def save_signal(self, signal: SignalRecord) -> SignalRecord:
        self.session.add(signal)
        self.commit()
        return signal

    def get_signals(
        self,
        symbol: Optional[str] = None,
        action: Optional[str] = None,
        limit: int = 100,
    ) -> List[SignalRecord]:
        query = self.session.query(SignalRecord)
        if symbol:
            query = query.filter_by(symbol=symbol)
        if action:
            query = query.filter_by(action=action)
        return query.order_by(desc(SignalRecord.timestamp)).limit(limit).all()

    def get_signal_accuracy(self, strategy: Optional[str] = None) -> float:
        query = self.session.query(SignalRecord).filter(
            SignalRecord.executed == True,
            SignalRecord.result_pnl.isnot(None),
        )
        if strategy:
            query = query.filter_by(strategy=strategy)
        total = query.count()
        if total == 0:
            return 0.0
        wins = query.filter(SignalRecord.result_pnl > 0).count()
        return wins / total


class SourceWeightRepository(BaseRepository):
    """Repository for API source weights (reinforcement learning)."""

    def get_weight(self, source_name: str) -> float:
        record = (
            self.session.query(SourceWeight)
            .filter_by(source_name=source_name)
            .first()
        )
        return record.weight if record else 1.0

    def update_weight(
        self,
        source_name: str,
        correct: bool,
        learning_rate: float = 0.01,
    ):
        record = (
            self.session.query(SourceWeight)
            .filter_by(source_name=source_name)
            .first()
        )
        if not record:
            record = SourceWeight(source_name=source_name)
            self.session.add(record)

        record.total_predictions += 1
        if correct:
            record.correct_predictions += 1

        record.accuracy = (
            record.correct_predictions / record.total_predictions
            if record.total_predictions > 0
            else 0.5
        )

        # Reinforcement learning weight update
        if correct:
            record.weight = min(2.0, record.weight + learning_rate)
        else:
            record.weight = max(0.1, record.weight - learning_rate)

        record.last_updated = datetime.utcnow()
        self.commit()

    def get_all_weights(self) -> Dict[str, float]:
        records = self.session.query(SourceWeight).all()
        return {r.source_name: r.weight for r in records}


# ============================================================================
# UNIFIED REPOSITORY
# ============================================================================

class TradingRepository:
    """
    Unified repository that provides access to all sub-repositories.
    
    Usage:
        repo = TradingRepository(session)
        prices = repo.prices.get_prices("BTCUSDT")
        repo.orders.save_order(order)
    """

    def __init__(self, session: Session):
        self.session = session
        self.prices = PriceRepository(session)
        self.macro_events = MacroEventRepository(session)
        self.news = NewsRepository(session)
        self.orders = OrderRepository(session)
        self.trades = TradeRepository(session)
        self.positions = PositionRepository(session)
        self.portfolio = PortfolioRepository(session)
        self.signals = SignalRepository(session)
        self.source_weights = SourceWeightRepository(session)

    def close(self):
        """Close the session."""
        self.session.close()
