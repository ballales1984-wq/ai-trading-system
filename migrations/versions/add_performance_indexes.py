"""
Alembic migration: add performance indexes
==========================================
Aggiunge indici su tabelle critiche per migliorare
le query più frequenti del sistema AI Trading.

Usa _safe_index() per essere resiliente a schemi parziali
(es: SQLite locale, tabelle non ancora create, colonne mancanti).
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = 'add_performance_indexes'
down_revision = 'a1b2c3d4e5f6'
branch_labels = None
depends_on = None


def _safe_index(table: str, name: str, cols: list, unique: bool = False) -> None:
    """Create index, silently skipping if table/column doesn't exist yet."""
    try:
        op.create_index(name, table, cols, unique=unique, if_not_exists=True)
    except Exception as exc:
        print(f"  [skip] {name} on {table}: {exc}")


def upgrade() -> None:
    # ── orders ────────────────────────────────────────────────────────────
    _safe_index('orders', 'ix_orders_symbol', ['symbol'])
    _safe_index('orders', 'ix_orders_status', ['status'])
    _safe_index('orders', 'ix_orders_created_at', ['created_at'])
    _safe_index('orders', 'ix_orders_user_symbol_status', ['user_id', 'symbol', 'status'])

    # ── positions ──────────────────────────────────────────────────────────
    _safe_index('positions', 'ix_positions_symbol', ['symbol'])
    _safe_index('positions', 'ix_positions_account_symbol', ['account_id', 'symbol'])
    _safe_index('positions', 'ix_positions_side', ['side'])

    # ── portfolio_history ──────────────────────────────────────────────────
    _safe_index('portfolio_history', 'ix_portfolio_history_account_date', ['account_id', 'date'])

    # ── ohlcv_bars (TimescaleDB) ──────────────────────────────────────────
    _safe_index('ohlcv_bars', 'ix_ohlcv_bars_symbol_interval_ts',
                ['symbol', 'interval', 'timestamp'], unique=True)

    # ── trade_ticks ────────────────────────────────────────────────────────
    _safe_index('trade_ticks', 'ix_trade_ticks_symbol_ts', ['symbol', 'timestamp'])


def downgrade() -> None:
    for table, index in [
        ('orders',            'ix_orders_symbol'),
        ('orders',            'ix_orders_status'),
        ('orders',            'ix_orders_created_at'),
        ('orders',            'ix_orders_user_symbol_status'),
        ('positions',         'ix_positions_symbol'),
        ('positions',         'ix_positions_account_symbol'),
        ('positions',         'ix_positions_side'),
        ('portfolio_history', 'ix_portfolio_history_account_date'),
        ('ohlcv_bars',        'ix_ohlcv_bars_symbol_interval_ts'),
        ('trade_ticks',       'ix_trade_ticks_symbol_ts'),
    ]:
        try:
            op.drop_index(index, table_name=table, if_exists=True)
        except Exception:
            pass
