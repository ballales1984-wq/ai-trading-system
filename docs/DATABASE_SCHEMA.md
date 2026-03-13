# 📊 AI Trading System - Database Schema

> Complete database schema documentation with relationships and field definitions

---

## 🗄️ Database Overview

| Database | Type | Purpose |
|----------|------|---------|
| `ai_trading` | PostgreSQL | Primary data store |
| `timeseries` | TimescaleDB | OHLCV market data |
| `redis` | Redis | Caching & session |

---

## 📋 Entity Relationship Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USERS                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  id (PK)          UUID                                                      │
│  email            VARCHAR(255) UNIQUE                                       │
│  username         VARCHAR(100) UNIQUE                                       │
│  password_hash    VARCHAR(255)                                              │
│  role             ENUM('user', 'admin', 'manager')                         │
│  is_active        BOOLEAN DEFAULT TRUE                                       │
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
│  strategy_type    ENUM('momentum', 'mean_reversion', 'multi', 'custom')   │
│  parameters       JSONB                                                     │
│  is_active        BOOLEAN DEFAULT FALSE                                     │
│  created_at       TIMESTAMP                                                 │
│  updated_at       TIMESTAMP                                                 │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 │ 1:N
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             TRADES                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  id (PK)          UUID                                                      │
│  strategy_id (FK) UUID → strategies.id                                      │
│  user_id (FK)     UUID → users.id                                          │
│  symbol           VARCHAR(20)                                               │
│  side             ENUM('buy', 'sell')                                       │
│  order_type       ENUM('market', 'limit', 'stop')                          │
│  quantity         DECIMAL(20, 8)                                            │
│  price            DECIMAL(20, 8)                                            │
│  status           ENUM('pending', 'filled', 'cancelled', 'rejected')       │
│  filled_price     DECIMAL(20, 8)                                            │
│  filled_at        TIMESTAMP                                                 │
│  created_at       TIMESTAMP                                                 │
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
│  side             ENUM('long', 'short', 'flat')                            │
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
│                        POSITIONS                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│  id (PK)          UUID                                                      │
│  user_id (FK)     UUID → users.id                                          │
│  balance          DECIMAL(20, 8)                                            │
│  equity           DECIMAL(20, 8)                                            │
│  currency         VARCHAR(10) DEFAULT 'USD'                                 │
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
│  broker_id        VARCHAR(50)                                               │
│  symbol           VARCHAR(20)                                               │
│  side             ENUM('buy', 'sell')                                       │
│  order_type       VARCHAR(20)                                               │
│  quantity         DECIMAL(20, 8)                                            │
│  price            DECIMAL(20, 8)                                            │
│  stop_price       DECIMAL(20, 8)                                            │
│  status           VARCHAR(20)                                               │
│  filled_quantity  DECIMAL(20, 8)                                            │
│  filled_price     DECIMAL(20, 8)                                            │
│  created_at       TIMESTAMP                                                 │
│  updated_at       TIMESTAMP                                                 │
│  executed_at      TIMESTAMP                                                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📝 Detailed Table Definitions

### 1. Users Table

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
```

### 2. Strategies Table

```sql
CREATE TABLE strategies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    strategy_type VARCHAR(50) NOT NULL,
    parameters JSONB DEFAULT '{}',
    risk_parameters JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT FALSE,
    is_public BOOLEAN DEFAULT FALSE,
    version INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_run_at TIMESTAMP,
    
    CONSTRAINT valid_strategy_type CHECK (
        strategy_type IN ('momentum', 'mean_reversion', 'multi', 'ai', 'custom')
    )
);

CREATE INDEX idx_strategies_user_id ON strategies(user_id);
CREATE INDEX idx_strategies_type ON strategies(strategy_type);
CREATE INDEX idx_strategies_active ON strategies(is_active);
```

### 3. Trades Table

```sql
CREATE TABLE trades (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    strategy_id UUID REFERENCES strategies(id) ON DELETE SET NULL,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL CHECK (side IN ('buy', 'sell')),
    order_type VARCHAR(20) NOT NULL,
    quantity DECIMAL(20, 8) NOT NULL,
    price DECIMAL(20, 8),
    commission DECIMAL(20, 8) DEFAULT 0,
    slippage DECIMAL(20, 8) DEFAULT 0,
    status VARCHAR(20) DEFAULT 'pending',
    filled_price DECIMAL(20, 8),
    filled_quantity DECIMAL(20, 8) DEFAULT 0,
    pnl DECIMAL(20, 8) DEFAULT 0,
    notes TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    filled_at TIMESTAMP,
    closed_at TIMESTAMP,
    
    CONSTRAINT valid_side CHECK (side IN ('buy', 'sell')),
    CONSTRAINT valid_status CHECK (status IN ('pending', 'open', 'filled', 'cancelled', 'rejected', 'closed'))
);

CREATE INDEX idx_trades_strategy_id ON trades(strategy_id);
CREATE INDEX idx_trades_user_id ON trades(user_id);
CREATE INDEX idx_trades_symbol ON trades(symbol);
CREATE INDEX idx_trades_status ON trades(status);
CREATE INDEX idx_trades_created_at ON trades(created_at DESC);
```

### 4. Positions Table

```sql
CREATE TABLE positions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    strategy_id UUID REFERENCES strategies(id) ON DELETE SET NULL,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL CHECK (side IN ('long', 'short', 'flat')),
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
CREATE INDEX idx_positions_symbol ON positions(symbol);
CREATE INDEX idx_positions_status ON positions(status);
CREATE INDEX idx_positions_opened_at ON positions(opened_at DESC);
```

### 5. Accounts Table

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

### 6. Orders Table

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
CREATE INDEX idx_orders_symbol ON orders(symbol);
CREATE INDEX idx_orders_status ON orders(status);
CREATE INDEX idx_orders_client_order_id ON orders(client_order_id);
CREATE INDEX idx_orders_created_at ON orders(created_at DESC);
```

### 7. Market Data Tables (TimescaleDB)

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
    PRIMARY KEY (symbol, timeframe, timestamp)
);

SELECT create_hypertable('ohlcv', 'timestamp', 
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

CREATE INDEX idx_ohlcv_symbol_timestamp ON ohlcv(symbol, timestamp DESC);
```

#### Tick Data

```sql
CREATE TABLE ticks (
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    price DECIMAL(20, 8) NOT NULL,
    quantity DECIMAL(20, 8),
    is_buyer_maker BOOLEAN,
    PRIMARY KEY (symbol, timestamp, price)
);

SELECT create_hypertable('ticks', 'timestamp',
    chunk_time_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);
```

### 8. Performance & Analytics Tables

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

### 9. Audit & Compliance Tables

#### Audit Log

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
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX idx_audit_logs_action ON audit_logs(action);
CREATE INDEX idx_audit_logs_created_at ON audit_logs(created_at DESC);
SELECT create_hypertable('audit_logs', 'created_at',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);
```

---

## 🔗 Foreign Key Relationships

| Parent Table | Child Table | Relationship |
|--------------|-------------|--------------|
| users | strategies | 1:N |
| users | trades | 1:N |
| users | positions | 1:N |
| users | accounts | 1:N |
| users | orders | 1:N |
| users | equity_snapshots | 1:N |
| strategies | trades | 1:N |
| strategies | positions | 1:N |
| strategies | orders | 1:N |
| accounts | orders | 1:N |
| accounts | equity_snapshots | 1:N |
| accounts | daily_performance | 1:N |

---

## 📊 Common Queries

### Get User's Active Positions

```sql
SELECT * FROM positions 
WHERE user_id = 'uuid-here' 
AND status = 'open'
ORDER BY opened_at DESC;
```

### Get User's Trade History

```sql
SELECT t.*, s.name as strategy_name
FROM trades t
LEFT JOIN strategies s ON t.strategy_id = s.id
WHERE t.user_id = 'uuid-here'
AND t.created_at >= NOW() - INTERVAL '30 days'
ORDER BY t.created_at DESC
LIMIT 100;
```

### Calculate Performance Metrics

```sql
SELECT 
    user_id,
    COUNT(*) as total_trades,
    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
    SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
    AVG(pnl) as avg_pnl,
    SUM(pnl) as total_pnl
FROM trades
WHERE created_at >= NOW() - INTERVAL '30 days'
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

*Last Updated: March 2026*
*Database Version: 1.2.0*
