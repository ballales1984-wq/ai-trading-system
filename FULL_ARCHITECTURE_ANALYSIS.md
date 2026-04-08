# AI Trading System - Complete Architecture Analysis

## 1. Panoramica Sistema

```
AI Trading System v2.3.2 "Security Hardened"
├── Backend (Python/FastAPI): 71 files
├── Frontend (React/TypeScript): 41 files
├── Tests: 100+ test files
└── Total LOC: ~34,000
```

---

## 2. Backend Architecture (`app/`)

### 2.1 Entry Point
- **`app/main.py`**: FastAPI application
  - Title: "Hedge Fund Trading System"
  - Version: 2.1.0
  - Docs: `/docs`, `/redoc`, `/openapi.json`

### 2.2 API Routes (`app/api/routes/`)

| Route | File | Purpose |
|-------|------|---------|
| `/api/v1/auth` | `auth.py` | Login, register, JWT tokens, logout |
| `/api/v1/orders` | `orders.py` | Order CRUD, emergency stop |
| `/api/v1/portfolio` | `portfolio.py` | Portfolio summary, positions, optimization |
| `/api/v1/market` | `market.py` | Prices, candles, order book |
| `/api/v1/risk` | `risk.py` | VaR, CVaR, risk limits |
| `/api/v1/strategy` | `strategy.py` | Strategy management |
| `/api/v1/news` | `news.py` | News sentiment analysis |
| `/api/v1/health` | `health.py` | Health checks |
| `/api/v1/cache` | `cache.py` | Redis/in-memory cache |
| `/api/v1/waitlist` | `waitlist.py` | Waitlist management |
| `/api/v1/agents` | `agents.py` | AI agent execution |
| `/api/v1/payments` | `payments.py` | Stripe integration |
| `/ws/prices` | `ws.py` | WebSocket for real-time data |

### 2.3 Core Services (`app/core/`)

| Module | Purpose |
|--------|---------|
| `config.py` | Pydantic settings, env vars |
| `security.py` | JWT, password hashing, token blacklist |
| `security_middleware.py` | CSP, headers, monitoring |
| `rate_limiter.py` | 60 req/min, 1000/hr, 10000/day |
| `database.py` | PostgreSQL connection pooling |
| `cache.py` | Redis + in-memory cache |
| `rbac.py` | Role-based access control |
| `multi_tenant.py` | Multi-tenant user management |
| `logging.py` | Structured logging |
| `performance.py` | Performance monitoring |

### 2.4 Risk Module (`app/risk/`)

| File | Purpose |
|------|---------|
| `risk_engine.py` | VaR, CVaR, stress testing |
| `hardened_risk_engine.py` | Enhanced risk checks |
| `risk_book.py` | Position risk tracking |

### 2.5 Execution Module (`app/execution/`)

| File | Purpose |
|------|---------|
| `execution_engine.py` | Order execution orchestration |
| `order_manager.py` | Order lifecycle management |
| `broker_connector.py` | Generic broker interface |
| `connectors/binance_connector.py` | Binance API |
| `connectors/bybit_connector.py` | Bybit API |
| `connectors/coinbase_connector.py` | Coinbase API |
| `connectors/ib_connector.py` | Interactive Brokers |
| `connectors/paper_connector.py` | Paper trading |

### 2.6 Strategies (`app/strategies/`)

| File | Purpose |
|------|---------|
| `base_strategy.py` | Abstract strategy class |
| `momentum.py` | Momentum trading strategy |
| `mean_reversion.py` | Mean reversion strategy |
| `multi_strategy.py` | Multi-strategy portfolio |

### 2.7 Database (`app/database/`)

| File | Purpose |
|------|---------|
| `models.py` | SQLAlchemy models (13 tables) |
| `timescale_models.py` | TimescaleDB time-series models |
| `repository.py` | Data access layer |
| `async_repository.py` | Async data access |

### 2.8 Compliance (`app/compliance/`)

| File | Purpose |
|------|---------|
| `audit.py` | Audit logging |
| `alerts.py` | Alert system |
| `reporting.py` | Compliance reports |

---

## 3. Frontend Architecture (`frontend/src/`)

### 3.1 Pages (`frontend/src/pages/`)

| Page | Route | Purpose |
|------|-------|---------|
| `Marketing.tsx` | `/`, `/marketing` | Landing page |
| `Login.tsx` | `/login` | Authentication |
| `Dashboard.tsx` | `/dashboard` | Main dashboard |
| `Portfolio.tsx` | `/portfolio` | Portfolio view |
| `Market.tsx` | `/market` | Market prices |
| `Orders.tsx` | `/orders` | Order management |
| `News.tsx` | `/news` | News feed |
| `Strategy.tsx` | `/strategy` | Strategy config |
| `Risk.tsx` | `/risk` | Risk metrics |
| `Settings.tsx` | `/settings` | User settings |
| `MLMonitoring.tsx` | `/ml-monitoring` | ML model monitoring |
| `InvestorPortal.tsx` | `/investor-portal` | Investor dashboard |
| `AIAssistant.tsx` | `/ai-assistant` | AI chat |
| `Privacy.tsx` | `/privacy` | Privacy policy |
| `Terms.tsx` | `/terms` | Terms of service |
| `Cookies.tsx` | `/cookies` | Cookie policy |

### 3.2 Components

| Component | Purpose |
|-----------|---------|
| `Layout.tsx` | Main layout with sidebar |
| `ui/StatusBadge.tsx` | Status indicators |
| `ui/Skeleton.tsx` | Loading skeletons |
| `trading/OrderBook.tsx` | Order book display |
| `trading/CandlestickChart.tsx` | Trading charts |

### 3.3 Services

- **`services/api.ts`**: Axios API client for all endpoints

### 3.4 Hooks

| Hook | Purpose |
|------|---------|
| `useWebSocket.ts` | WebSocket connection with auto-reconnect |
| `useMarketData.ts` | Real-time market data |

---

## 4. Security Analysis

### 4.1 Authentication
- **JWT tokens** with `jti` for blacklist tracking
- Token expiration: 30 minutes (configurable)
- Refresh tokens: 7 days

### 4.2 Authorization
- **RBAC**: Admin, Trader, Viewer, API_User
- Role-based route protection

### 4.3 Rate Limiting
- 60 requests/minute
- 1000 requests/hour
- 10000 requests/day

### 4.4 Security Headers
```
Strict-Transport-Security: max-age=31536000
X-Frame-Options: DENY
X-Content-Type-Options: nosniff
Content-Security-Policy: (configured for Google Analytics, WebSocket)
```

### 4.5 Input Validation
- All Pydantic models have validation (patterns, ranges, lengths)
- Field-level validation on all API requests

---

## 5. Integration Points

### 5.1 Frontend → Backend
- **REST API**: `/api/v1/*` endpoints
- **WebSocket**: `/ws/prices` for real-time data
- **Authentication**: JWT in Authorization header

### 5.2 External APIs
- **Binance**: Market data, trading
- **Bybit**: Trading
- **Coinbase**: Trading
- **CoinMarketCap**: Market data
- **Stripe**: Payments

---

## 6. Database Schema

### 6.1 Main Tables (`app/database/models.py`)

| Table | Purpose |
|-------|---------|
| `prices` | Historical price data |
| `orders` | Trading orders |
| `trades` | Executed trades |
| `positions` | Current positions |
| `portfolio_snapshots` | Portfolio history |
| `signals` | Trading signals |
| `news` | News with sentiment |
| `macro_events` | Economic events |
| `energy_records` | Energy commodities |

### 6.2 Indexes
- All foreign keys indexed
- Composite indexes on frequently queried fields

---

## 7. Known Issues & TODOs

### 7.1 Minor Issues
- 2 TODO comments (Coinbase/Bybit WebSocket enhancement)
- Default users in dev mode only (production needs env vars)

### 7.2 Production Considerations
- Database: Switch from SQLite to PostgreSQL
- Redis: Enable for production caching
- API Keys: Set via environment variables

---

## 8. Summary

| Aspect | Status |
|--------|--------|
| API Routes | 13 main routes + WebSocket |
| Security | JWT, RBAC, Rate Limiting, CSP |
| Database | 13+ tables with indexes |
| Frontend | 16 pages, real-time WebSocket |
| Testing | 100+ test files |
| Documentation | Release notes, API docs |

**Architecture**: Clean, modular, production-ready with security hardening v2.3.2