# 📊 AI Trading System - Database Schema (2026 Enhanced Edition)

> Complete database schema documentation with relationships and field definitions
> **Version: 2.0.0** - Incorporates 2026-era improvements

---

## 🗄️ Database Overview

| Database | Type | Purpose |
|----------|------|---------|
| `ai_trading` | PostgreSQL | Primary relational data store |
| `timeseries` | TimescaleDB | OHLCV market data, ticks, equity snapshots |
| `redis` | Redis | Caching & session management |

---

## 📋 Entity Relationship Diagram (Corrected)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USERS                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  id (PK)          UUID                                                      │
│  email            VARCHAR(255) UNIQUE                                       │
│  username         VARCHAR(100) UNIQUE                                       │
│  password_hash    VARCHAR(255)                                              │
│  role             ENUM('user', 'admin', 'manager')                         │
│  country_code     VARCHAR(2)            -- NEW: KYC/Regulatory              │
│  kyc_status       VARCHAR(20)          -- NEW: pending/verified/rejected   │
│  is_active        BOOLEAN DEFAULT TRUE                                     │
│  created_at       TIMESTAMP                                                 │
│  updated_at       TIMESTAMP                                                 │
│  last_login       TIMESTAMP                                                 │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 │ 1:N
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           STRATEGIES                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│  id (PK)          UUID                                                      │
│  user_id (FK)     UUID → users.id                                          │
│  name             VARCHAR(100)                                              │
│  description      TEXT                                                      │
│  strategy_type    ENUM('momentum', 'mean_reversion', 'multi', 'ai')        │
│  model_version    VARCHAR(50)           -- NEW: AI model versioning        │
│  ml_framework     VARCHAR(30)           -- NEW: pytorch/tensorflow/etc      │
│  inference_endpoint TEXT               -- NEW: external LLM endpoint       │
│  parameters       JSONB                                                     │
│  is_active        BOOLEAN DEFAULT FALSE                                     │
│  created_at       TIMESTAMP                                                 │
│  updated_at       TIMESTAMP                                                 │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 │ 1:N
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ORDERS                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│  id (PK)          UUID                                                      │
│  user_id (FK)     UUID → users.id                                          │
│  account_id (FK)  UUID → accounts.id                                        │
│  strategy_id (FK) UUID → strategies.id                                     │
│  broker_order_id  VARCHAR(100)                                              │
│  symbol           VARCHAR(20)                                               │
│  side             ENUM('buy', 'sell')                                       │
│  order_type       VARCHAR(20)                                               │
│  quantity         DECIMAL(20, 8)                                            │
│  price            DECIMAL(20, 8)                                            │
│  status           ENUM('new', 'pending', 'filled', 'cancelled', ...)       │
│  created_at       TIMESTAMP                                                 │
│  executed_at      TIMESTAMP                                                 │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 │ 1:N (order can have many fills)
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             FILLS (NEW)                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  id (PK)          UUID                                                      │
│  order_id (FK)    UUID → orders.id            -- NEW: explicit link       │
│  strategy_id (FK) UUID → strategies.id                                     │
│  user_id (FK)     UUID → users.id                                          │
│  broker_fill_id   VARCHAR(100)                                              │
│  symbol           VARCHAR(20)                                               │
│  side             ENUM('buy', 'sell')                                       │
│  quantity         DECIMAL(20, 8)                                            │
│  price            DECIMAL(20, 8)                                            │
│  liquidity_type   VARCHAR(20)           -- NEW: maker/taker                 │
│  commission       DECIMAL(20, 8)                                             │
│  realized_pnl     DECIMAL(20, 8)                                            │
│  executed_at      TIMESTAMP                                                 │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 │ N:M (via trade_positions)
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           POSITIONS                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  id (PK)          UUID                                                      │
│  user_id (FK)     UUID → users.id                                          │
│  symbol           VARCHAR(20)                                               │
│  side             ENUM('long', 'short')     -- REMOVED: 'flat'            │
│  quantity         DECIMAL(20, 8)                                            │
│  entry_price      DECIMAL(20, 8)                                            │
│  current_price    DECIMAL(20, 8)                                            │
│  unrealized_pnl   DECIMAL(20, 8)                                            │
│  opened_at        TIMESTAMP                                                 │
│  closed_at        TIMESTAMP                                                 │
│  status           ENUM('open', 'closed')                                    │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 │ 1:N
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ACCOUNTS (FIXED)                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  id (PK)          UUID                                                      │
│  user_id (FK)     UUID → users.id                                          │
│  broker           VARCHAR(50)                                                │
│  balance          DECIMAL(20, 8)                                            │
│  equity           DECIMAL(20, 8)                                            │
│  currency         VARCHAR(10) DEFAULT 'USD'                                 │
│  created_at       TIMESTAMP                                                 │
│  updated_at       TIMESTAMP                                                 │
└─────────────────────────────────────────────────────────────────────────────┘

                                 │
                                 │ 1:N
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          BACKTESTS (NEW)                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  id (PK)          UUID                                                      │
│  strategy_id (FK) UUID → strategies.id                                    │
│  parameters      JSONB                                                      │
│  start_date      DATE                                                       │
│  end_date        DATE                                                       │
│  sharpe          DECIMAL(12,4)                                              │
│  sortino         DECIMAL(12,4)                                              │
│  max_drawdown    DECIMAL(12,4)                                              │
│  cagr            DECIMAL(12,4)                                              │
│  report_json     JSONB                                                      │
│  created_at      TIMESTAMP                                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📝 Detailed Table Definitions (2026 Enhanced)

### 1. Users Table (Enhanced)

```sql
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) DEFAULT 'user' CHECK (role IN ('user', 'admin', 'manager')),
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    avatar_url TEXT,
    
    -- NEW: Compliance & Regulatory (2026)
    country_code VARCHAR(2),              -- ISO 3166-1 alpha-2
    kyc_status VARCHAR(20) DEFAULT 'pending' CHECK (kyc_status IN ('pending', 'verified', 'rejected', 'expired')),
    kyc_verified_at TIMESTAMP,
    tax_id VARCHAR(50),
    
    is_active BOOLEAN DEFAULT TRUE,
    is_verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,
    settings JSONB DEFAULT '{}'
);

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_role ON users(role);
CREATE INDEX idx_users_kyc_status ON users(kyc_status);
CREATE INDEX idx_users_country ON users(country_code);
```

### 2. Strategies Table (Enhanced with AI/ML)

```sql
CREATE TABLE strategies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    strategy_type VARCHAR(50) NOT NULL,
    
    -- NEW: AI/ML Specific Fields (2026)
    model_version VARCHAR(50),           -- e.g. "vLLM-3.1", "finetune-2026-02"
    ml_framework VARCHAR(30),            -- pytorch, tensorflow, xgboost, langchain, etc.
    inference_endpoint TEXT,              -- External LLM/API endpoint if used
    backtest_id UUID,                    -- Reference to successful backtest
    
    parameters JSONB DEFAULT '{}',
    risk_parameters JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT FALSE,
    is_public BOOLEAN DEFAULT FALSE,
    version INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_run_at TIMESTAMP,
    
    CONSTRAINT valid_strategy_type CHECK (
        strategy_type IN ('momentum', 'mean_reversion', 'multi', 'ai', 'custom', 'arbitrage', 'grid')
    )
);

CREATE INDEX idx_strategies_user_id ON strategies(user_id);
CREATE INDEX idx_strategies_type ON strategies(strategy_type);
CREATE INDEX idx_strategies_active ON strategies(is_active);
CREATE INDEX idx_strategies_model_version ON strategies(model_version);
```

### 3. NEW: Backtests Table

```sql
CREATE TABLE backtests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    strategy_id UUID REFERENCES strategies(id) ON DELETE CASCADE,
    name VARCHAR(100),
    parameters JSONB DEFAULT '{}',
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    
    -- Performance Metrics
    initial_capital DECIMAL(20, 8) DEFAULT 100000,
    final_capital DECIMAL(20, 8),
    sharpe DECIMAL(12, 4),
    sortino DECIMAL(12, 4),
    max_drawdown DECIMAL(12, 4),
    cagr DECIMAL(12, 4),
    calmar DECIMAL(12, 4),
    win_rate DECIMAL(10, 4),
    profit_factor DECIMAL(10, 4),
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    avg_trade_duration INTERVAL,
    
    -- Full Report
    report_json JSONB,                    -- Complete metrics, equity curve samples
    
    status VARCHAR(20) DEFAULT 'running' CHECK (status IN ('running', 'completed', 'failed', 'cancelled')),
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

CREATE INDEX idx_backtests_strategy_id ON backtests(strategy_id);
CREATE INDEX idx_backtests_status ON backtests(status);
CREATE INDEX idx_backtests_created_at ON backtests(created_at DESC);
```

### 4. Orders Table

```sql
CREATE TABLE orders (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    account_id UUID REFERENCES accounts(id) ON DELETE CASCADE,
    strategy_id UUID REFERENCES strategies(id) ON DELETE SET NULL,
    broker_order_id VARCHAR(100),
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL CHECK (side IN ('buy', 'sell')),
    order_type VARCHAR(20) NOT NULL,
    time_in_force VARCHAR(10) DEFAULT 'GTC' CHECK (time_in_force IN ('GTC', 'IOC', 'FOK', 'GTX')),
    quantity DECIMAL(20, 8) NOT NULL,
    price DECIMAL(20, 8),
    stop_price DECIMAL(20, 8),
    limit_price DECIMAL(20, 8),
    filled_quantity DECIMAL(20, 8) DEFAULT 0,
    filled_price DECIMAL(20, 8),
    average_price DECIMAL(20, 8),
    commission DECIMAL(20, 8) DEFAULT 0,
    status VARCHAR(20) DEFAULT 'new' CHECK (status IN ('new', 'pending', 'open', 'filled', 'partially_filled', 'cancelled', 'rejected', 'expired')),
    reject_reason TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    submitted_at TIMESTAMP,
    executed_at TIMESTAMP,
    expired_at TIMESTAMP,
    cancelled_at TIMESTAMP,
    client_order_id VARCHAR(100) UNIQUE,
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX idx_orders_user_id ON orders(user_id);
CREATE INDEX idx_orders_account_id ON orders(account_id);
CREATE INDEX idx_orders_strategy_id ON orders(strategy_id);
CREATE INDEX idx_orders_symbol ON orders(symbol);
CREATE INDEX idx_orders_status ON orders(status);
CREATE INDEX idx_orders_client_order_id ON orders(client_order_id);
CREATE INDEX idx_orders_created_at ON orders(created_at DESC);
```

### 5. NEW: Fills Table (Replaces/Amplifies Trades)

```sql
CREATE TABLE fills (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    order_id UUID REFERENCES orders(id) ON DELETE CASCADE,
    strategy_id UUID REFERENCES strategies(id) ON DELETE SET NULL,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    account_id UUID REFERENCES accounts(id) ON DELETE SET NULL,
    
    broker_fill_id VARCHAR(100),
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL CHECK (side IN ('buy', 'sell')),
    quantity DECIMAL(20, 8) NOT NULL,
    price DECIMAL(20, 8) NOT NULL,
    
    -- NEW: Liquidity Type
    liquidity_type VARCHAR(20),          -- 'maker' or 'taker'
    
    commission DECIMAL(20, 8) DEFAULT 0,
    realized_pnl DECIMAL(20, 8),         -- If closing position
    
    executed_at TIMESTAMPTZ NOT NULL,
    metadata JSONB DEFAULT '{}',
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_fills_order_id ON fills(order_id);
CREATE INDEX idx_fills_user_id ON fills(user_id);
CREATE INDEX idx_fills_account_id ON fills(account_id);
CREATE INDEX idx_fills_symbol ON fills(symbol);
CREATE INDEX idx_fills_executed_at ON fills(executed_at DESC);

-- Optional: Trades as a materialized view aggregating fills
CREATE MATERIALIZED VIEW trades_summary AS
SELECT 
    user_id,
    strategy_id,
    symbol,
    side,
    COUNT(*) as fill_count,
    SUM(quantity) as total_quantity,
    AVG(price) as avg_price,
    SUM(commission) as total_commission,
    SUM(realized_pnl) as total_realized_pnl,
    MIN(executed_at) as first_fill,
    MAX(executed_at) as last_fill
FROM fills
GROUP BY user_id, strategy_id, symbol, side;

CREATE UNIQUE INDEX idx_trades_summary_unique ON trades_summary(user_id, strategy_id, symbol, side);
```

### 6. Positions Table (Simplified)

```sql
CREATE TABLE positions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    strategy_id UUID REFERENCES strategies(id) ON DELETE SET NULL,
    account_id UUID REFERENCES accounts(id) ON DELETE SET NULL,
    symbol VARCHAR(20) NOT NULL,
    
    -- REMOVED: 'flat' - derived from quantity = 0
    side VARCHAR(10) NOT NULL CHECK (side IN ('long', 'short')),
    
    quantity DECIMAL(20, 8) NOT NULL,
    entry_price DECIMAL(20, 8) NOT NULL,
    current_price DECIMAL(20, 8),
    unrealized_pnl DECIMAL(20, 8) DEFAULT 0,
    realized_pnl DECIMAL(20, 8) DEFAULT 0,
    position_value DECIMAL(20, 8),
    leverage INTEGER DEFAULT 1,
    margin DECIMAL(20, 8) DEFAULT 0,
    liquidation_price DECIMAL(20, 8),
    status VARCHAR(20) DEFAULT 'open' CHECK (status IN ('open', 'closed', 'liquidated')),
    opened_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    closed_at TIMESTAMP,
    last_updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX idx_positions_user_id ON positions(user_id);
CREATE INDEX idx_positions_account_id ON positions(account_id);
CREATE INDEX idx_positions_symbol ON positions(symbol);
CREATE INDEX idx_positions_status ON positions(status);
CREATE INDEX idx_positions_opened_at ON positions(opened_at DESC);
```

### 7. Accounts Table

```sql
CREATE TABLE accounts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    broker VARCHAR(50) NOT NULL,
    account_id VARCHAR(100),
    account_name VARCHAR(100),
    account_type VARCHAR(20) DEFAULT 'live' CHECK (account_type IN ('live', 'paper', 'test')),
    balance DECIMAL(20, 8) DEFAULT 0,
    equity DECIMAL(20, 8) DEFAULT 0,
    available_balance DECIMAL(20, 8) DEFAULT 0,
    currency VARCHAR(10) DEFAULT 'USD',
    is_primary BOOLEAN DEFAULT FALSE,
    is_active BOOLEAN DEFAULT TRUE,
    last_sync_at TIMESTAMP,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_accounts_user_id ON accounts(user_id);
CREATE INDEX idx_accounts_broker ON accounts(broker);
CREATE INDEX idx_accounts_type ON accounts(account_type);
```

### 8. Market Data Tables (TimescaleDB Enhanced)

#### OHLCV Data

```sql
CREATE TABLE ohlcv (
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    open DECIMAL(20, 8) NOT NULL,
    high DECIMAL(20, 8) NOT NULL,
    low DECIMAL(20, 8) NOT NULL,
    close DECIMAL(20, 8) NOT NULL,
    volume DECIMAL(20, 8) NOT NULL,
    quote_volume DECIMAL(20, 8),
    n_trades INTEGER,
    taker_buy_base_volume DECIMAL(20, 8),
    taker_buy_quote_volume DECIMAL(20, 8),
    
    -- NEW: Exchange/Venue Tracking
    exchange VARCHAR(30) DEFAULT 'BINANCE',
    is_crypto BOOLEAN DEFAULT TRUE,
    asset_class VARCHAR(20) DEFAULT 'crypto',
    
    PRIMARY KEY (symbol, timeframe, timestamp)
);

SELECT create_hypertable('ohlcv', 'timestamp', 
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Compression (2026 Best Practice)
ALTER TABLE ohlcv SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol, timeframe',
    timescaledb.compress_orderby = 'timestamp DESC'
);

-- Add compression policy
SELECT add_compression_policy('ohlcv', INTERVAL '7 days');

CREATE INDEX idx_ohlcv_symbol_timestamp ON ohlcv(symbol, timestamp DESC);
CREATE INDEX idx_ohlcv_exchange ON ohlcv(exchange);
```

#### NEW: Continuous Aggregates

```sql
-- 1 Hour OHLCV Aggregate
CREATE MATERIALIZED VIEW ohlcv_1h
WITH (timescaledb.continuous) AS
SELECT 
    symbol,
    timeframe,
    time_bucket('1 hour', timestamp) AS bucket,
    FIRST(open, timestamp) AS open,
    MAX(high) AS high,
    MIN(low) AS low,
    LAST(close, timestamp) AS close,
    SUM(volume) AS volume,
    SUM(quote_volume) AS quote_volume,
    SUM(n_trades) AS n_trades,
    MAX(exchange) AS exchange,
    MAX(is_crypto) AS is_crypto
FROM ohlcv
WHERE timeframe IN ('1m', '5m', '15m')
GROUP BY symbol, bucket, timeframe
WITH NO DATA;

-- 1 Day OHLCV Aggregate
CREATE MATERIALIZED VIEW ohlcv_1d
WITH (timescaledb.continuous) AS
SELECT 
    symbol,
    time_bucket('1 day', timestamp) AS bucket,
    FIRST(open, timestamp) AS open,
    MAX(high) AS high,
    MIN(low) AS low,
    LAST(close, timestamp) AS close,
    SUM(volume) AS volume,
    SUM(quote_volume) AS quote_volume,
    SUM(n_trades) AS n_trades,
    AVG(close) AS avg_close,
    MAX(exchange) AS exchange
FROM ohlcv
WHERE timeframe = '1h'
GROUP BY symbol, bucket
WITH NO DATA;

-- Add refresh policies
SELECT add_continuous_aggregate_policy('ohlcv_1h',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour');

SELECT add_continuous_aggregate_policy('ohlcv_1d',
    start_offset => INTERVAL '32 days',
    end_offset => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 day');

-- Retention policy
SELECT add_retention_policy('ohlcv', INTERVAL '5 years');
```

#### Tick Data

```sql
CREATE TABLE ticks (
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    price DECIMAL(20, 8) NOT NULL,
    quantity DECIMAL(20, 8),
    is_buyer_maker BOOLEAN,
    exchange VARCHAR(30) DEFAULT 'BINANCE',
    PRIMARY KEY (symbol, timestamp, price)
);

SELECT create_hypertable('ticks', 'timestamp',
    chunk_time_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

-- Compression
ALTER TABLE ticks SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol',
    timescaledb.compress_orderby = 'timestamp DESC'
);

SELECT add_retention_policy('ticks', INTERVAL '90 days');
```

### 9. Performance & Analytics Tables

#### Equity Curve

```sql
CREATE TABLE equity_snapshots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    account_id UUID REFERENCES accounts(id) ON DELETE CASCADE,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    equity DECIMAL(20, 8) NOT NULL,
    balance DECIMAL(20, 8) NOT NULL,
    unrealized_pnl DECIMAL(20, 8) DEFAULT 0,
    realized_pnl DECIMAL(20, 8) DEFAULT 0,
    available_balance DECIMAL(20, 8),
    margin_used DECIMAL(20, 8),
    open_positions INTEGER DEFAULT 0,
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX idx_equity_user_id ON equity_snapshots(user_id, timestamp DESC);
SELECT create_hypertable('equity_snapshots', 'timestamp',
    chunk_time_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

-- Compression & Retention
ALTER TABLE equity_snapshots SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'user_id, account_id',
    timescaledb.compress_orderby = 'timestamp DESC'
);

SELECT add_retention_policy('equity_snapshots', INTERVAL '3 years');
```

#### Daily Performance

```sql
CREATE TABLE daily_performance (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    account_id UUID REFERENCES accounts(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    starting_equity DECIMAL(20, 8) NOT NULL,
    ending_equity DECIMAL(20, 8) NOT NULL,
    daily_pnl DECIMAL(20, 8) DEFAULT 0,
    daily_return_pct DECIMAL(10, 4) DEFAULT 0,
    realized_pnl DECIMAL(20, 8) DEFAULT 0,
    unrealized_pnl DECIMAL(20, 8) DEFAULT 0,
    commission DECIMAL(20, 8) DEFAULT 0,
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    open_positions INTEGER DEFAULT 0,
    max_drawdown DECIMAL(20, 8) DEFAULT 0,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, account_id, date)
);

CREATE INDEX idx_daily_perf_user_date ON daily_performance(user_id, date DESC);
```

### 10. Audit & Compliance Tables (Enhanced)

#### Audit Log with Hash Chain (2026)

```sql
CREATE TABLE audit_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50) NOT NULL,
    resource_id UUID,
    old_values JSONB,
    new_values JSONB,
    ip_address INET,
    user_agent TEXT,
    status VARCHAR(20) DEFAULT 'success',
    error_message TEXT,
    
    -- NEW: Immutable Hash Chain (2026)
    prev_hash TEXT,
    hash TEXT GENERATED ALWAYS AS (
        encode(digest(
            COALESCE(id::TEXT, '') || 
            COALESCE(user_id::TEXT, '') || 
            COALESCE(action, '') || 
            COALESCE(resource_type, '') || 
            COALESCE(created_at::TEXT, ''),
            'sha256'
        ), 'hex')
    ) STORED,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX idx_audit_logs_action ON audit_logs(action);
CREATE INDEX idx_audit_logs_resource ON audit_logs(resource_type, resource_id);
CREATE INDEX idx_audit_logs_created_at ON audit_logs(created_at DESC);
CREATE INDEX idx_audit_logs_hash ON audit_logs(hash);
CREATE INDEX idx_audit_logs_prev_hash ON audit_logs(prev_hash);

SELECT create_hypertable('audit_logs', 'created_at',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Compression
ALTER TABLE audit_logs SET (
    timescaledb.compress,
    timescaledb.compress_orderby = 'created_at DESC'
);

SELECT add_retention_policy('audit_logs', INTERVAL '7 years');
```

### 11. NEW: Risk & Margin Tables

```sql
CREATE TABLE position_limits (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    account_id UUID REFERENCES accounts(id) ON DELETE CASCADE,
    
    -- Limits
    max_position_size DECIMAL(20, 8),
    max_daily_loss DECIMAL(20, 8),
    max_drawdown_pct DECIMAL(10, 4),
    max_leverage DECIMAL(10, 2) DEFAULT 1,
    max_sector_exposure DECIMAL(10, 4),
    
    -- Symbol/Sector Specific
    symbol VARCHAR(20),
    sector VARCHAR(50),
    
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(user_id, account_id, symbol)
);

CREATE TABLE margin_requirements (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id UUID REFERENCES accounts(id) ON DELETE CASCADE,
    symbol VARCHAR(20) NOT NULL,
    initial_margin DECIMAL(10, 6),       -- As fraction (0.01 = 1%)
    maintenance_margin DECIMAL(10, 6),
    margin_currency VARCHAR(10) DEFAULT 'USD',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(account_id, symbol)
);

CREATE TABLE risk_snapshots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    account_id UUID REFERENCES accounts(id) ON DELETE CASCADE,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    -- VaR Metrics
    var_1d DECIMAL(20, 8),
    var_1w DECIMAL(20, 8),
    var_1m DECIMAL(20, 8),
    
    -- Exposure
    total_exposure DECIMAL(20, 8),
    long_exposure DECIMAL(20, 8),
    short_exposure DECIMAL(20, 8),
    net_exposure DECIMAL(20, 8),
    sector_exposures JSONB,              -- {sector: exposure, ...}
    
    -- Risk Metrics
    portfolio_beta DECIMAL(10, 4),
    correlation_to_benchmark DECIMAL(10, 4),
    
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX idx_risk_snapshots_user ON risk_snapshots(user_id, timestamp DESC);
CREATE INDEX idx_risk_snapshots_account ON risk_snapshots(account_id, timestamp DESC);

SELECT create_hypertable('risk_snapshots', 'timestamp',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);
```

### 12. NEW: Strategy Telemetry & Metrics

```sql
CREATE TABLE strategy_telemetry (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    strategy_id UUID REFERENCES strategies(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    -- Runtime Metrics
    execution_time_ms INTEGER,
    memory_usage_mb DECIMAL(10, 2),
    cpu_usage_pct DECIMAL(5, 2),
    
    -- Signal Metrics
    signals_generated INTEGER,
    signals_executed INTEGER,
    signal_confidence_avg DECIMAL(5, 4),
    
    -- Model Metrics (if AI)
    inference_time_ms INTEGER,
    model_load_time_ms INTEGER,
    prediction_confidence DECIMAL(5, 4),
    feature_count INTEGER,
    
    -- Errors
    errors_count INTEGER DEFAULT 0,
    last_error_message TEXT,
    
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX idx_telemetry_strategy ON strategy_telemetry(strategy_id, timestamp DESC);
CREATE INDEX idx_telemetry_user ON strategy_telemetry(user_id, timestamp DESC);

SELECT create_hypertable('strategy_telemetry', 'timestamp',
    chunk_time_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

-- Compression
ALTER TABLE strategy_telemetry SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'strategy_id',
    timescaledb.compress_orderby = 'timestamp DESC'
);

SELECT add_retention_policy('strategy_telemetry', INTERVAL '90 days');
```

### 13. NEW: Notifications & Alerts

```sql
CREATE TABLE notifications (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    type VARCHAR(50) NOT NULL,
    title VARCHAR(200) NOT NULL,
    message TEXT,
    severity VARCHAR(20) DEFAULT 'info' CHECK (severity IN ('info', 'warning', 'error', 'critical')),
    is_read BOOLEAN DEFAULT FALSE,
    read_at TIMESTAMP,
    action_url TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_notifications_user ON notifications(user_id, is_read, created_at DESC);
CREATE INDEX idx_notifications_unread ON notifications(user_id, is_read) WHERE is_read = FALSE;
```

---

## 🔗 Foreign Key Relationships (Updated)

| Parent Table | Child Table | Relationship |
|--------------|-------------|--------------|
| users | strategies | 1:N |
| users | orders | 1:N |
| users | fills | 1:N |
| users | positions | 1:N |
| users | accounts | 1:N |
| users | equity_snapshots | 1:N |
| users | audit_logs | 1:N |
| users | notifications | 1:N |
| users | backtests | 1:N |
| users | risk_snapshots | 1:N |
| users | strategy_telemetry | 1:N |
| strategies | orders | 1:N |
| strategies | fills | 1:N |
| strategies | positions | 1:N |
| strategies | backtests | 1:N |
| strategies | strategy_telemetry | 1:N |
| accounts | orders | 1:N |
| accounts | fills | 1:N |
| accounts | equity_snapshots | 1:N |
| accounts | daily_performance | 1:N |
| accounts | position_limits | 1:N |
| accounts | margin_requirements | 1:N |
| accounts | risk_snapshots | 1:N |
| orders | fills | 1:N |

---

## 📊 Common Queries

### Get User's Active Positions

```sql
SELECT * FROM positions 
WHERE user_id = 'uuid-here' 
AND status = 'open'
ORDER BY opened_at DESC;
```

### Get Order with All Fills

```sql
SELECT o.*, 
       json_agg(f) FILTER (WHERE f IS NOT NULL) AS fills
FROM orders o
LEFT JOIN fills f ON f.order_id = o.id
WHERE o.id = 'order-uuid'
GROUP BY o.id;
```

### Calculate Performance Metrics

```sql
SELECT 
    user_id,
    COUNT(*) as total_fills,
    SUM(quantity) FILTER (WHERE side = 'buy') as total_bought,
    SUM(quantity) FILTER (WHERE side = 'sell') as total_sold,
    SUM(commission) as total_commission,
    SUM(realized_pnl) as total_pnl
FROM fills
WHERE executed_at >= NOW() - INTERVAL '30 days'
GROUP BY user_id;
```

### Get Equity Curve

```sql
SELECT 
    date_trunc('hour', timestamp) as hour,
    AVG(equity) as avg_equity
FROM equity_snapshots
WHERE user_id = 'uuid-here'
AND timestamp >= NOW() - INTERVAL '7 days'
GROUP BY 1
ORDER BY 1;
```

### Get Strategy Performance from Backtest

```sql
SELECT 
    b.*,
    s.name as strategy_name
FROM backtests b
JOIN strategies s ON b.strategy_id = s.id
WHERE b.user_id = 'uuid-here'
AND b.status = 'completed'
ORDER BY b.sharpe DESC
LIMIT 10;
```

---

## 🔧 Migrations

Migrations are managed using Alembic. See `migrations/` directory for versioned schema changes.

```bash
# Generate migration
alembic revision --autogenerate -m "Add new table"

# Run migrations
alembic upgrade head

# Rollback
alembic downgrade -1
```

---

## 📝 Summary of Changes from v1.x to v2.0

| Category | Change |
|----------|--------|
| **Users** | Added `country_code`, `kyc_status`, `kyc_verified_at`, `tax_id` |
| **Strategies** | Added `model_version`, `ml_framework`, `inference_endpoint`, `backtest_id` |
| **Backtests** | NEW table for strategy backtesting & model versioning |
| **Orders** | Added `strategy_id` FK, improved status enum |
| **Trades→Fills** | NEW explicit fills table with order relationship |
| **Positions** | Removed `flat` side (derived from quantity) |
| **Accounts** | Fixed diagram inconsistency |
| **OHLCV** | Added `exchange`, `is_crypto`, `asset_class`, compression |
| **Continuous Aggs** | NEW `ohlcv_1h`, `ohlcv_1d` materialized views |
| **Audit Logs** | Added hash chain for immutability (`prev_hash`, `hash`) |
| **Risk** | NEW `position_limits`, `margin_requirements`, `risk_snapshots` |
| **Telemetry** | NEW `strategy_telemetry` for observability |
| **Notifications** | NEW `notifications` table |
| **Retention** | Added 5-year OHLCV, 3-year equity, 7-year audit policies |

---

*Last Updated: March 2026*
*Database Version: 2.0.0*
