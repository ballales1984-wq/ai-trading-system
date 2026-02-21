-- ============================================================
-- AI Trading System - Database Initialization
-- ============================================================
-- Run this script to initialize the database with TimescaleDB

-- Create TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Create schemas
CREATE SCHEMA IF NOT EXISTS trading;
CREATE SCHEMA IF NOT EXISTS market_data;
CREATE SCHEMA IF NOT EXISTS analytics;

-- ============================================================
-- MARKET DATA TABLES
-- ============================================================

-- OHLCV Bars (TimescaleDB hypertable)
CREATE TABLE IF NOT EXISTS market_data.ohlcv_bars (
    time        TIMESTAMPTZ NOT NULL,
    symbol      VARCHAR(20) NOT NULL,
    interval    VARCHAR(10) NOT NULL,  -- 1m, 5m, 1h, 1d
    open        DOUBLE PRECISION NOT NULL,
    high        DOUBLE PRECISION NOT NULL,
    low         DOUBLE PRECISION NOT NULL,
    close       DOUBLE PRECISION NOT NULL,
    volume      DOUBLE PRECISION DEFAULT 0,
    quote_volume DOUBLE PRECISION DEFAULT 0,
    trades_count INTEGER DEFAULT 0,
    vwap        DOUBLE PRECISION,
    twap        DOUBLE PRECISION,
    volatility  DOUBLE PRECISION,
    PRIMARY KEY (time, symbol, interval)
);

-- Create hypertable
SELECT create_hypertable(
    'market_data.ohlcv_bars',
    'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Enable compression
ALTER TABLE market_data.ohlcv_bars SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol,interval'
);

SELECT add_compression_policy(
    'market_data.ohlcv_bars',
    INTERVAL '7 days',
    if_not_exists => TRUE
);

-- Trade ticks
CREATE TABLE IF NOT EXISTS market_data.trade_ticks (
    time            TIMESTAMPTZ NOT NULL,
    symbol          VARCHAR(20) NOT NULL,
    trade_id        VARCHAR(50) NOT NULL,
    price           DOUBLE PRECISION NOT NULL,
    quantity        DOUBLE PRECISION NOT NULL,
    quote_quantity  DOUBLE PRECISION,
    is_buyer_maker  BOOLEAN DEFAULT FALSE,
    is_best_match   BOOLEAN DEFAULT TRUE,
    bid_price       DOUBLE PRECISION,
    ask_price       DOUBLE PRECISION,
    spread          DOUBLE PRECISION,
    PRIMARY KEY (time, symbol, trade_id)
);

SELECT create_hypertable(
    'market_data.trade_ticks',
    'time',
    chunk_time_interval => INTERVAL '4 hours',
    if_not_exists => TRUE
);

-- Order book snapshots
CREATE TABLE IF NOT EXISTS market_data.orderbook_snapshots (
    time            TIMESTAMPTZ NOT NULL,
    symbol          VARCHAR(20) NOT NULL,
    best_bid        DOUBLE PRECISION NOT NULL,
    best_ask        DOUBLE PRECISION NOT NULL,
    spread          DOUBLE PRECISION NOT NULL,
    mid_price       DOUBLE PRECISION NOT NULL,
    bids            JSONB,
    asks            JSONB,
    bid_volume_1pct DOUBLE PRECISION,
    ask_volume_1pct DOUBLE PRECISION,
    bid_volume_5pct DOUBLE PRECISION,
    ask_volume_5pct DOUBLE PRECISION,
    imbalance       DOUBLE PRECISION,
    pressure        DOUBLE PRECISION,
    PRIMARY KEY (time, symbol)
);

SELECT create_hypertable(
    'market_data.orderbook_snapshots',
    'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Funding rates
CREATE TABLE IF NOT EXISTS market_data.funding_rates (
    time                TIMESTAMPTZ NOT NULL,
    symbol              VARCHAR(20) NOT NULL,
    exchange            VARCHAR(20) NOT NULL,
    funding_rate        DOUBLE PRECISION NOT NULL,
    funding_time        TIMESTAMPTZ NOT NULL,
    estimated_rate      DOUBLE PRECISION,
    open_interest       DOUBLE PRECISION,
    open_interest_value DOUBLE PRECISION,
    PRIMARY KEY (time, symbol, exchange)
);

SELECT create_hypertable(
    'market_data.funding_rates',
    'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- ============================================================
-- TRADING TABLES
-- ============================================================

-- Orders
CREATE TABLE IF NOT EXISTS trading.orders (
    id              SERIAL PRIMARY KEY,
    order_id        VARCHAR(50) UNIQUE NOT NULL,
    broker_order_id VARCHAR(50),
    symbol          VARCHAR(20) NOT NULL,
    side            VARCHAR(10) NOT NULL,  -- BUY, SELL
    order_type      VARCHAR(20) NOT NULL,  -- MARKET, LIMIT, STOP, STOP_LIMIT
    quantity        DOUBLE PRECISION NOT NULL,
    price           DOUBLE PRECISION,
    stop_price      DOUBLE PRECISION,
    time_in_force   VARCHAR(10) DEFAULT 'GTC',
    status          VARCHAR(20) DEFAULT 'NEW',
    filled_quantity DOUBLE PRECISION DEFAULT 0,
    average_price   DOUBLE PRECISION,
    broker          VARCHAR(20) NOT NULL,
    strategy        VARCHAR(50),
    error_message   TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    executed_at     TIMESTAMPTZ
);

CREATE INDEX idx_orders_symbol ON trading.orders(symbol);
CREATE INDEX idx_orders_status ON trading.orders(status);
CREATE INDEX idx_orders_created ON trading.orders(created_at DESC);

-- Trades
CREATE TABLE IF NOT EXISTS trading.trades (
    id              SERIAL PRIMARY KEY,
    trade_id        VARCHAR(50) UNIQUE NOT NULL,
    order_id        VARCHAR(50) NOT NULL,
    broker_trade_id VARCHAR(50),
    symbol          VARCHAR(20) NOT NULL,
    side            VARCHAR(10) NOT NULL,
    quantity        DOUBLE PRECISION NOT NULL,
    price           DOUBLE PRECISION NOT NULL,
    commission      DOUBLE PRECISION DEFAULT 0,
    commission_asset VARCHAR(10),
    broker          VARCHAR(20) NOT NULL,
    executed_at     TIMESTAMPTZ DEFAULT NOW(),
    FOREIGN KEY (order_id) REFERENCES trading.orders(order_id)
);

CREATE INDEX idx_trades_symbol ON trading.trades(symbol);
CREATE INDEX idx_trades_executed ON trading.trades(executed_at DESC);

-- Positions
CREATE TABLE IF NOT EXISTS trading.positions (
    id              SERIAL PRIMARY KEY,
    symbol          VARCHAR(20) NOT NULL,
    side            VARCHAR(10) NOT NULL,  -- LONG, SHORT
    quantity        DOUBLE PRECISION NOT NULL,
    entry_price     DOUBLE PRECISION NOT NULL,
    current_price   DOUBLE PRECISION NOT NULL,
    unrealized_pnl  DOUBLE PRECISION DEFAULT 0,
    realized_pnl    DOUBLE PRECISION DEFAULT 0,
    leverage        DOUBLE PRECISION DEFAULT 1.0,
    margin          DOUBLE PRECISION,
    status          VARCHAR(20) DEFAULT 'OPEN',
    opened_at       TIMESTAMPTZ DEFAULT NOW(),
    closed_at       TIMESTAMPTZ,
    UNIQUE(symbol, side, status) WHERE status = 'OPEN'
);

CREATE INDEX idx_positions_symbol ON trading.positions(symbol);
CREATE INDEX idx_positions_status ON trading.positions(status);

-- ============================================================
-- ANALYTICS TABLES
-- ============================================================

-- Portfolio history
CREATE TABLE IF NOT EXISTS analytics.portfolio_history (
    time            TIMESTAMPTZ NOT NULL,
    portfolio_id    VARCHAR(50) NOT NULL,
    total_value     DOUBLE PRECISION NOT NULL,
    cash            DOUBLE PRECISION NOT NULL,
    equity          DOUBLE PRECISION NOT NULL,
    unrealized_pnl  DOUBLE PRECISION DEFAULT 0,
    realized_pnl    DOUBLE PRECISION DEFAULT 0,
    daily_pnl       DOUBLE PRECISION DEFAULT 0,
    var_95          DOUBLE PRECISION,
    var_99          DOUBLE PRECISION,
    drawdown        DOUBLE PRECISION DEFAULT 0,
    leverage        DOUBLE PRECISION DEFAULT 1.0,
    num_positions   INTEGER DEFAULT 0,
    num_long        INTEGER DEFAULT 0,
    num_short       INTEGER DEFAULT 0,
    sharpe          DOUBLE PRECISION,
    sortino         DOUBLE PRECISION,
    win_rate        DOUBLE PRECISION,
    PRIMARY KEY (time, portfolio_id)
);

SELECT create_hypertable(
    'analytics.portfolio_history',
    'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Risk metrics history
CREATE TABLE IF NOT EXISTS analytics.risk_metrics (
    time                TIMESTAMPTZ NOT NULL,
    portfolio_id        VARCHAR(50) NOT NULL,
    var_1d_95           DOUBLE PRECISION,
    var_1d_99           DOUBLE PRECISION,
    var_5d_95           DOUBLE PRECISION,
    cvar_1d_95          DOUBLE PRECISION,
    cvar_1d_99          DOUBLE PRECISION,
    volatility_daily    DOUBLE PRECISION,
    volatility_annualized DOUBLE PRECISION,
    current_drawdown    DOUBLE PRECISION DEFAULT 0,
    max_drawdown        DOUBLE PRECISION DEFAULT 0,
    drawdown_duration   INTEGER DEFAULT 0,
    avg_correlation     DOUBLE PRECISION,
    diversification_ratio DOUBLE PRECISION,
    beta                DOUBLE PRECISION,
    tracking_error      DOUBLE PRECISION,
    liquidity_score     DOUBLE PRECISION,
    concentration_score DOUBLE PRECISION,
    PRIMARY KEY (time, portfolio_id)
);

SELECT create_hypertable(
    'analytics.risk_metrics',
    'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Signals
CREATE TABLE IF NOT EXISTS analytics.signals (
    id              SERIAL PRIMARY KEY,
    signal_id       VARCHAR(50) UNIQUE NOT NULL,
    symbol          VARCHAR(20) NOT NULL,
    signal_type     VARCHAR(50) NOT NULL,
    direction       VARCHAR(10) NOT NULL,  -- LONG, SHORT, NEUTRAL
    strength        DOUBLE PRECISION DEFAULT 0.5,
    price_target    DOUBLE PRECISION,
    stop_loss       DOUBLE PRECISION,
    strategy        VARCHAR(50),
    confidence      DOUBLE PRECISION,
    metadata        JSONB,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    expires_at      TIMESTAMPTZ,
    status          VARCHAR(20) DEFAULT 'ACTIVE'
);

CREATE INDEX idx_signals_symbol ON analytics.signals(symbol);
CREATE INDEX idx_signals_type ON analytics.signals(signal_type);
CREATE INDEX idx_signals_created ON analytics.signals(created_at DESC);

-- ============================================================
-- CONTINUOUS AGGREGATES
-- ============================================================

-- 5-minute OHLCV aggregate
CREATE MATERIALIZED VIEW IF NOT EXISTS market_data.ohlcv_5m
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
FROM market_data.ohlcv_bars
WHERE interval = '1m'
GROUP BY bucket, symbol
WITH DATA;

SELECT add_continuous_aggregate_policy(
    'market_data.ohlcv_5m',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '5 minutes',
    if_not_exists => TRUE
);

-- 1-hour OHLCV aggregate
CREATE MATERIALIZED VIEW IF NOT EXISTS market_data.ohlcv_1h
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
FROM market_data.ohlcv_bars
WHERE interval = '1m'
GROUP BY bucket, symbol
WITH DATA;

SELECT add_continuous_aggregate_policy(
    'market_data.ohlcv_1h',
    start_offset => INTERVAL '1 day',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

-- Daily OHLCV aggregate
CREATE MATERIALIZED VIEW IF NOT EXISTS market_data.ohlcv_1d
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
FROM market_data.ohlcv_bars
WHERE interval = '1m'
GROUP BY bucket, symbol
WITH DATA;

SELECT add_continuous_aggregate_policy(
    'market_data.ohlcv_1d',
    start_offset => INTERVAL '7 days',
    end_offset => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- ============================================================
-- GRANT PERMISSIONS
-- ============================================================

-- Grant permissions to trading user
GRANT ALL PRIVILEGES ON SCHEMA market_data TO trading;
GRANT ALL PRIVILEGES ON SCHEMA trading TO trading;
GRANT ALL PRIVILEGES ON SCHEMA analytics TO trading;

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA market_data TO trading;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA trading TO trading;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA analytics TO trading;

GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA trading TO trading;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA analytics TO trading;

-- Default privileges for future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA market_data GRANT ALL ON TABLES TO trading;
ALTER DEFAULT PRIVILEGES IN SCHEMA trading GRANT ALL ON TABLES TO trading;
ALTER DEFAULT PRIVILEGES IN SCHEMA analytics GRANT ALL ON TABLES TO trading;

-- ============================================================
-- COMPLETION MESSAGE
-- ============================================================

DO $$
BEGIN
    RAISE NOTICE 'Database initialization completed successfully!';
    RAISE NOTICE 'TimescaleDB extension installed';
    RAISE NOTICE 'Schemas created: market_data, trading, analytics';
    RAISE NOTICE 'Hypertables created for time-series data';
    RAISE NOTICE 'Continuous aggregates configured';
END $$;
