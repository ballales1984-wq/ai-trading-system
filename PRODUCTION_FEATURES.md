# Production Features Implementation

This document describes the production-grade features implemented for the AI Trading System.

## Overview

The following production features have been implemented:

1. **Robust Database Persistence** - TimescaleDB for time-series data
2. **Production-Grade Structured Logging** - JSON logging with correlation IDs
3. **Containerized Deployment** - Multi-stage Docker builds with docker-compose
4. **Hardened Risk Engine** - Circuit breakers, kill switches, position limits
5. **CI/CD Pipeline** - Comprehensive GitHub Actions workflow

---

## 1. Database Persistence (TimescaleDB)

### Files Created
- [`app/database/timescale_models.py`](app/database/timescale_models.py) - Time-series models

### Features
- **Hypertables** for efficient time-series storage:
  - `OHLCVBar` - Price data with automatic partitioning
  - `TradeTick` - High-frequency trade data
  - `OrderBookSnapshot` - Market depth snapshots
  - `FundingRate` - Perpetual futures funding rates
  - `LiquidationEvent` - Exchange liquidation events
  - `PortfolioHistory` - Portfolio performance tracking
  - `RiskMetricsHistory` - Risk metrics over time

- **Continuous Aggregates** for pre-computed views:
  - 5-minute, 1-hour, and daily OHLCV aggregates
  - Hourly trade volume summaries

- **Compression Policies** for storage optimization:
  - Automatic compression after 7 days
  - Segment-by symbol and interval

### Usage

```python
from app.database.timescale_models import init_timescaledb, TimeSeriesQueries

# Initialize TimescaleDB
engine = init_timescaledb("postgresql://user:pass@localhost:5432/trading")

# Query OHLCV data
from datetime import datetime, timedelta
start = datetime.utcnow() - timedelta(days=30)
data = TimeSeriesQueries.get_ohlcv_range(session, "BTCUSDT", "1h", start, datetime.utcnow())
```

---

## 2. Production-Grade Structured Logging

### Files Created
- [`app/core/logging_production.py`](app/core/logging_production.py) - Production logging

### Features
- **JSON Formatting** with Elastic Common Schema (ECS) compatibility
- **Correlation IDs** for distributed tracing
- **Sensitive Data Masking** for API keys, passwords, tokens
- **Log Categories** for filtering (TRADING, RISK, SECURITY, AUDIT, etc.)
- **Multiple Output Handlers**:
  - Console (stdout)
  - Rotating file with compression
  - Elasticsearch direct ingestion

### Usage

```python
from app.core.logging_production import (
    setup_production_logging,
    get_trading_logger,
    new_correlation_id,
    LogCategory
)

# Setup logging
setup_production_logging(
    service_name="ai-trading-system",
    environment="production",
    log_level="INFO",
    enable_file_logging=True
)

# Get trading logger
logger = get_trading_logger(__name__, LogCategory.TRADING)

# Log with context
logger.log_order_created(
    order_id="ORD123",
    symbol="BTCUSDT",
    side="BUY",
    quantity=1.0,
    price=50000.0,
    order_type="LIMIT"
)

# Log risk violation
logger.log_risk_violation(
    limit_type="var_breach",
    current_value=0.025,
    limit_value=0.02,
    severity="critical"
)
```

### Sample JSON Output

```json
{
  "@timestamp": "2026-02-20T15:30:00.000Z",
  "message": "Order created: ORD123",
  "log.level": "info",
  "log.logger": "trading.engine",
  "service": {
    "name": "ai-trading-system",
    "environment": "production"
  },
  "correlation_id": "a1b2c3d4",
  "data": {
    "order_id": "ORD123",
    "symbol": "BTCUSDT",
    "side": "BUY",
    "quantity": 1.0,
    "price": 50000.0
  },
  "event": {
    "category": "trading",
    "type": "order_created"
  }
}
```

---

## 3. Containerized Deployment

### Files Created
- [`docker/Dockerfile.production`](docker/Dockerfile.production) - Multi-stage Dockerfile
- [`docker-compose.production.yml`](docker-compose.production.yml) - Full production stack
- [`docker/prometheus/prometheus.yml`](docker/prometheus/prometheus.yml) - Prometheus config
- [`docker/nginx/nginx.conf`](docker/nginx/nginx.conf) - Nginx reverse proxy

### Docker Multi-Stage Build

```dockerfile
# Build stages:
# 1. builder - Install dependencies
# 2. production - Main application
# 3. api - API server only
# 4. development - With dev tools
# 5. testing - Test runner
```

### Production Stack

```yaml
services:
  # Database
  postgres:      # TimescaleDB
  redis:         # Caching

  # Application
  trading-system: # Dashboard
  api:            # FastAPI backend

  # Monitoring
  prometheus:     # Metrics
  grafana:        # Dashboards

  # Infrastructure
  nginx:          # Reverse proxy
  backup:         # Database backups
```

### Commands

```bash
# Build production image
docker build -f docker/Dockerfile.production -t ai-trading-system:prod .

# Start full stack
docker-compose -f docker-compose.production.yml up -d

# Start with specific environment
docker-compose -f docker-compose.production.yml up -d --build

# View logs
docker-compose -f docker-compose.production.yml logs -f trading-system

# Scale API
docker-compose -f docker-compose.production.yml up -d --scale api=3
```

---

## 4. Hardened Risk Engine

### Files Created
- [`app/risk/hardened_risk_engine.py`](app/risk/hardened_risk_engine.py) - Hardened risk engine

### Features

#### Circuit Breakers
- **VaR Circuit** - Trips when VaR approaches limit
- **Drawdown Circuit** - Trips on drawdown threshold
- **Daily Loss Circuit** - Trips on daily loss limit
- **Leverage Circuit** - Trips on leverage breach
- **Concentration Circuit** - Trips on concentration risk

#### Kill Switches
- `MANUAL` - Manual activation
- `DRAWDOWN` - Automatic on max drawdown
- `VAR_BREACH` - Automatic on VaR breach
- `LEVERAGE_BREACH` - Automatic on leverage breach
- `LOSS_LIMIT` - Automatic on daily loss limit
- `VOLATILITY_SPIKE` - Automatic on volatility spike
- `SYSTEM_ERROR` - Automatic on system errors

#### Position Limits
- Single position size limit (default 10%)
- Sector concentration limit (default 25%)
- Asset class limit (default 50%)
- Gross exposure limit (default 200%)
- Maximum leverage (default 5x)

#### Risk Levels
- `GREEN` - Normal operations
- `YELLOW` - Caution - increased monitoring
- `ORANGE` - Warning - reduce exposure
- `RED` - Critical - halt new positions
- `BLACK` - Emergency - liquidate all

### Usage

```python
from app.risk.hardened_risk_engine import (
    HardenedRiskEngine,
    Position,
    Portfolio,
    KillSwitchType
)

# Initialize engine
engine = HardenedRiskEngine(
    initial_capital=100000.0,
    max_drawdown_pct=0.20,
    daily_loss_limit_pct=0.05,
    max_position_pct=0.10,
    max_leverage=5.0
)

# Check order risk
result = engine.check_order_risk(
    symbol="BTCUSDT",
    side="BUY",
    quantity=1.0,
    price=50000.0,
    portfolio=portfolio
)

if result.approved:
    print(f"Order approved, risk score: {result.risk_score}")
else:
    print(f"Order rejected: {result.reasons}")

# Activate kill switch
engine.activate_kill_switch(
    KillSwitchType.MANUAL,
    reason="Emergency stop",
    activated_by="admin"
)

# Emergency stop (halts all trading)
engine.emergency_stop("Critical system error")

# Get risk status
status = engine.get_risk_status(portfolio)
print(f"Risk level: {status['risk_level']}")
```

---

## 5. CI/CD Pipeline

### Files Created
- [`.github/workflows/ci-cd-production.yml`](.github/workflows/ci-cd-production.yml) - Production CI/CD

### Pipeline Stages

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Code Quality   │────▶│    Security     │────▶│      Test       │
│  - Black        │     │  - Bandit       │     │  - Unit tests   │
│  - Ruff         │     │  - pip-audit    │     │  - Coverage     │
│  - mypy         │     │  - Trivy        │     │  - Integration  │
└─────────────────┘     │  - Gitleaks     │     └─────────────────┘
                        └─────────────────┘              │
                                                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Deploy Prod    │◀────│  Deploy Staging │◀────│     Docker      │
│  - Kubernetes   │     │  - Kubernetes   │     │  - Build        │
│  - Verification │     │  - Smoke tests  │     │  - Push         │
│  - Notification │     │                 │     │  - Multi-arch   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Workflow Triggers

- **Push to main** → Deploy to production
- **Push to develop** → Deploy to staging
- **Pull request** → Run tests and security scans
- **Tag v*** → Create release and deploy
- **Manual dispatch** → Deploy to specified environment

### Required Secrets

```yaml
# Kubernetes
KUBE_CONFIG_STAGING
KUBE_CONFIG_PRODUCTION

# Container Registry
DOCKERHUB_USERNAME
DOCKERHUB_TOKEN

# Code Quality
CODECOV_TOKEN

# Notifications
SLACK_WEBHOOK
```

### Manual Deployment

```bash
# Trigger via GitHub CLI
gh workflow run ci-cd-production.yml \
  -f environment=production \
  -f skip_tests=false

# Or via GitHub UI
# Actions → AI Trading System - Production CI/CD → Run workflow
```

---

## Testing

### Test File
- [`tests/test_production_features.py`](tests/test_production_features.py)

### Run Tests

```bash
# Run all production feature tests
pytest tests/test_production_features.py -v

# Run with coverage
pytest tests/test_production_features.py --cov=app --cov-report=term-missing
```

---

## Quick Start

### 1. Start Infrastructure

```bash
# Start database and cache
docker-compose up -d postgres redis

# Wait for services
sleep 30
```

### 2. Initialize Database

```bash
# Run migrations
alembic upgrade head

# Initialize TimescaleDB
python -c "from app.database.timescale_models import init_timescaledb; init_timescaledb('postgresql://trading:trading_secret@localhost:5432/trading')"
```

### 3. Start Application

```bash
# Development
python main.py --mode dashboard

# Production
docker-compose -f docker-compose.production.yml up -d
```

### 4. Access Services

- Dashboard: http://localhost:8050
- API: http://localhost:8000
- Grafana: http://localhost:3000
- Prometheus: http://localhost:9090

---

## Monitoring

### Grafana Dashboards

Import the following dashboards:
- Node Exporter Full (ID: 1860)
- Redis Dashboard (ID: 11835)
- PostgreSQL Database (ID: 9628)
- Nginx (ID: 12708)

### Key Metrics

```yaml
# Trading metrics
trading_orders_total
trading_orders_rejected_total
trading_pnl_current
trading_positions_open

# Risk metrics
risk_var_current
risk_drawdown_current
risk_leverage_current
risk_circuit_breaker_state

# System metrics
http_requests_total
http_request_duration_seconds
database_connections_active
cache_hit_ratio
```

---

## Security Considerations

1. **Secrets Management**
   - Use Kubernetes secrets or external secret managers
   - Never commit secrets to git
   - Rotate API keys regularly

2. **Network Security**
   - Use TLS for all external communication
   - Restrict access to monitoring endpoints
   - Implement rate limiting

3. **Access Control**
   - Use non-root containers
   - Implement RBAC in Kubernetes
   - Enable authentication for Grafana

4. **Audit Logging**
   - All trading actions are logged
   - Risk events are tracked
   - Kill switch activations are recorded

---

## Troubleshooting

### Common Issues

1. **TimescaleDB not available**
   ```bash
   # Check if extension is installed
   docker exec -it trading-postgres psql -U trading -d trading -c "SELECT * FROM pg_extension WHERE extname = 'timescaledb';"
   ```

2. **Circuit breaker stuck open**
   ```python
   # Reset circuit breaker
   engine.reset_circuit_breaker("var")
   ```

3. **Kill switch won't deactivate**
   ```python
   # Force deactivate
   engine.deactivate_kill_switch(KillSwitchType.MANUAL)
   ```

4. **Logs not appearing in Elasticsearch**
   - Check network connectivity
   - Verify API key
   - Check fallback file: `logs/elasticsearch-fallback.json`

---

## Support

For issues or questions:
1. Check the logs: `docker-compose logs -f trading-system`
2. Review metrics in Grafana
3. Check circuit breaker status: `engine.get_risk_status(portfolio)`
