"""add_app_models

Revision ID: a1b2c3d4e5f6
Revises: f5b534ab38b6
Create Date: 2026-02-20 23:20:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, None] = 'f5b534ab38b6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # === PRICES TABLE ===
    op.create_table('prices',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('open', sa.Float(), nullable=False),
        sa.Column('high', sa.Float(), nullable=False),
        sa.Column('low', sa.Float(), nullable=False),
        sa.Column('close', sa.Float(), nullable=False),
        sa.Column('volume', sa.Float(), server_default='0.0'),
        sa.Column('quote_volume', sa.Float(), server_default='0.0'),
        sa.Column('interval', sa.String(10), server_default='1m'),
        sa.Column('source', sa.String(30), server_default='binance'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('symbol', 'timestamp', 'interval', name='uq_price_record')
    )
    op.create_index('ix_prices_symbol', 'prices', ['symbol'])
    op.create_index('ix_prices_timestamp', 'prices', ['timestamp'])
    op.create_index('ix_prices_symbol_ts', 'prices', ['symbol', 'timestamp'])

    # === MACRO_EVENTS TABLE ===
    op.create_table('macro_events',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('event_date', sa.DateTime(), nullable=False),
        sa.Column('event_type', sa.String(100), nullable=False),
        sa.Column('country', sa.String(5), server_default='US'),
        sa.Column('impact', sa.String(10), server_default='medium'),
        sa.Column('actual', sa.Float(), nullable=True),
        sa.Column('forecast', sa.Float(), nullable=True),
        sa.Column('previous', sa.Float(), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('source', sa.String(30), server_default='trading_economics'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_macro_events_event_date', 'macro_events', ['event_date'])
    op.create_index('ix_macro_date_impact', 'macro_events', ['event_date', 'impact'])

    # === NATURAL_EVENTS TABLE ===
    op.create_table('natural_events',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('event_date', sa.DateTime(), nullable=False),
        sa.Column('event_type', sa.String(100), nullable=False),
        sa.Column('region', sa.String(100), nullable=True),
        sa.Column('intensity', sa.Float(), server_default='0.0'),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('affected_commodities', sa.JSON(), nullable=True),
        sa.Column('source', sa.String(30), server_default='open_meteo'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_natural_events_event_date', 'natural_events', ['event_date'])

    # === NEWS TABLE ===
    op.create_table('news',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('published_at', sa.DateTime(), nullable=False),
        sa.Column('title', sa.String(500), nullable=False),
        sa.Column('summary', sa.Text(), nullable=True),
        sa.Column('sentiment_score', sa.Float(), server_default='0.0'),
        sa.Column('sentiment_label', sa.String(20), server_default='neutral'),
        sa.Column('relevance_score', sa.Float(), server_default='0.0'),
        sa.Column('source', sa.String(50), nullable=True),
        sa.Column('source_url', sa.String(500), nullable=True),
        sa.Column('symbols', sa.JSON(), nullable=True),
        sa.Column('api_source', sa.String(30), server_default='newsapi'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_news_published', 'news', ['published_at'])

    # === INNOVATIONS TABLE ===
    op.create_table('innovations',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('event_date', sa.DateTime(), nullable=False),
        sa.Column('innovation_type', sa.String(100), nullable=False),
        sa.Column('title', sa.String(500), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('potential_impact', sa.Float(), server_default='0.0'),
        sa.Column('affected_sectors', sa.JSON(), nullable=True),
        sa.Column('source', sa.String(30), server_default='google_patents'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_innovations_event_date', 'innovations', ['event_date'])

    # === ORDERS TABLE ===
    op.create_table('orders',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('order_id', sa.String(50), nullable=False),
        sa.Column('broker_order_id', sa.String(50), nullable=True),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('side', sa.String(10), nullable=False),
        sa.Column('order_type', sa.String(20), nullable=False),
        sa.Column('quantity', sa.Float(), nullable=False),
        sa.Column('price', sa.Float(), nullable=True),
        sa.Column('stop_price', sa.Float(), nullable=True),
        sa.Column('filled_quantity', sa.Float(), server_default='0.0'),
        sa.Column('avg_fill_price', sa.Float(), server_default='0.0'),
        sa.Column('commission', sa.Float(), server_default='0.0'),
        sa.Column('status', sa.String(20), server_default='NEW'),
        sa.Column('broker', sa.String(20), server_default='binance'),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now()),
        sa.Column('strategy', sa.String(50), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('order_id', name='uq_order_id')
    )
    op.create_index('ix_orders_order_id', 'orders', ['order_id'])
    op.create_index('ix_orders_symbol', 'orders', ['symbol'])
    op.create_index('ix_orders_symbol_status', 'orders', ['symbol', 'status'])

    # === TRADES TABLE ===
    op.create_table('trades',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('trade_id', sa.String(50), nullable=False),
        sa.Column('order_id', sa.String(50), nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('side', sa.String(10), nullable=False),
        sa.Column('quantity', sa.Float(), nullable=False),
        sa.Column('price', sa.Float(), nullable=False),
        sa.Column('commission', sa.Float(), server_default='0.0'),
        sa.Column('pnl', sa.Float(), server_default='0.0'),
        sa.Column('timestamp', sa.DateTime(), server_default=sa.func.now()),
        sa.Column('broker', sa.String(20), server_default='binance'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('trade_id', name='uq_trade_id'),
        sa.ForeignKeyConstraint(['order_id'], ['orders.order_id'], name='fk_trades_order_id')
    )
    op.create_index('ix_trades_trade_id', 'trades', ['trade_id'])
    op.create_index('ix_trades_order_id', 'trades', ['order_id'])
    op.create_index('ix_trades_symbol', 'trades', ['symbol'])

    # === POSITIONS TABLE ===
    op.create_table('positions',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('side', sa.String(10), server_default='LONG'),
        sa.Column('quantity', sa.Float(), server_default='0.0'),
        sa.Column('entry_price', sa.Float(), server_default='0.0'),
        sa.Column('current_price', sa.Float(), server_default='0.0'),
        sa.Column('unrealized_pnl', sa.Float(), server_default='0.0'),
        sa.Column('realized_pnl', sa.Float(), server_default='0.0'),
        sa.Column('leverage', sa.Float(), server_default='1.0'),
        sa.Column('opened_at', sa.DateTime(), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now()),
        sa.Column('strategy', sa.String(50), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_positions_symbol', 'positions', ['symbol'])

    # === PORTFOLIO_SNAPSHOTS TABLE ===
    op.create_table('portfolio_snapshots',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('total_equity', sa.Float(), server_default='0.0'),
        sa.Column('available_balance', sa.Float(), server_default='0.0'),
        sa.Column('unrealized_pnl', sa.Float(), server_default='0.0'),
        sa.Column('realized_pnl', sa.Float(), server_default='0.0'),
        sa.Column('num_positions', sa.Integer(), server_default='0'),
        sa.Column('drawdown', sa.Float(), server_default='0.0'),
        sa.Column('sharpe_ratio', sa.Float(), nullable=True),
        sa.Column('metadata_json', sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_portfolio_snapshots_timestamp', 'portfolio_snapshots', ['timestamp'])

    # === SIGNALS TABLE ===
    op.create_table('signals',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('action', sa.String(10), nullable=False),
        sa.Column('confidence', sa.Float(), server_default='0.0'),
        sa.Column('price_at_signal', sa.Float(), server_default='0.0'),
        sa.Column('strategy', sa.String(50), nullable=True),
        sa.Column('monte_carlo_level', sa.Integer(), server_default='1'),
        sa.Column('factors', sa.JSON(), nullable=True),
        sa.Column('executed', sa.Boolean(), server_default='0'),
        sa.Column('result_pnl', sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_signals_timestamp', 'signals', ['timestamp'])
    op.create_index('ix_signals_symbol', 'signals', ['symbol'])

    # === ENERGY_RECORDS TABLE ===
    op.create_table('energy_records',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('energy_type', sa.String(50), nullable=False),
        sa.Column('product_name', sa.String(100), nullable=True),
        sa.Column('value', sa.Float(), nullable=False),
        sa.Column('unit', sa.String(20), nullable=True),
        sa.Column('area', sa.String(50), nullable=True),
        sa.Column('source', sa.String(30), server_default='eia'),
        sa.Column('metadata_json', sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_energy_records_timestamp', 'energy_records', ['timestamp'])
    op.create_index('ix_energy_type_ts', 'energy_records', ['energy_type', 'timestamp'])

    # === SOURCE_WEIGHTS TABLE ===
    op.create_table('source_weights',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('source_name', sa.String(50), nullable=False),
        sa.Column('weight', sa.Float(), server_default='1.0'),
        sa.Column('accuracy', sa.Float(), server_default='0.5'),
        sa.Column('total_predictions', sa.Integer(), server_default='0'),
        sa.Column('correct_predictions', sa.Integer(), server_default='0'),
        sa.Column('last_updated', sa.DateTime(), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('source_name', name='uq_source_name')
    )


def downgrade() -> None:
    # Drop tables in reverse order
    op.drop_table('source_weights')
    op.drop_index('ix_energy_type_ts', table_name='energy_records')
    op.drop_index('ix_energy_records_timestamp', table_name='energy_records')
    op.drop_table('energy_records')
    op.drop_index('ix_signals_symbol', table_name='signals')
    op.drop_index('ix_signals_timestamp', table_name='signals')
    op.drop_table('signals')
    op.drop_index('ix_portfolio_snapshots_timestamp', table_name='portfolio_snapshots')
    op.drop_table('portfolio_snapshots')
    op.drop_index('ix_positions_symbol', table_name='positions')
    op.drop_table('positions')
    op.drop_index('ix_trades_symbol', table_name='trades')
    op.drop_index('ix_trades_order_id', table_name='trades')
    op.drop_index('ix_trades_trade_id', table_name='trades')
    op.drop_table('trades')
    op.drop_index('ix_orders_symbol_status', table_name='orders')
    op.drop_index('ix_orders_symbol', table_name='orders')
    op.drop_index('ix_orders_order_id', table_name='orders')
    op.drop_table('orders')
    op.drop_index('ix_innovations_event_date', table_name='innovations')
    op.drop_table('innovations')
    op.drop_index('ix_news_published', table_name='news')
    op.drop_table('news')
    op.drop_index('ix_natural_events_event_date', table_name='natural_events')
    op.drop_table('natural_events')
    op.drop_index('ix_macro_date_impact', table_name='macro_events')
    op.drop_index('ix_macro_events_event_date', table_name='macro_events')
    op.drop_table('macro_events')
    op.drop_index('ix_prices_symbol_ts', table_name='prices')
    op.drop_index('ix_prices_timestamp', table_name='prices')
    op.drop_index('ix_prices_symbol', table_name='prices')
    op.drop_table('prices')
