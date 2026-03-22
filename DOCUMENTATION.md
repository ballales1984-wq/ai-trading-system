# AI Trading System - Technical Documentation

<div align="center">

![Version](https://img.shields.io/badge/version-2.3.0-blue)
![Python](https://img.shields.io/badge/python-3.11+-green)
![React](https://img.shields.io/badge/react-18+-blue)

</div>

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Backend Architecture](#backend-architecture)
3. [Frontend Architecture](#frontend-architecture)
4. [Concept Engine](#concept-engine)
5. [API Reference](#api-reference)
6. [Database Schema](#database-schema)
7. [Security](#security)
8. [Deployment](#deployment)
9. [Development](#development)

---

## 1. Architecture Overview

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                     AI Trading System                        │
├─────────────────────────────────────────────────────────────┤
│  Frontend (React + TypeScript + Vite)                     │
│  http://localhost:5173                                     │
├─────────────────────────────────────────────────────────────┤
│  Backend API (FastAPI)                    Python Dashboards │
│  http://localhost:8000                      Port 8050        │
├─────────────────────────────────────────────────────────────┤
│  AI Assistant (Streamlit)                                  │
│  Port 8501                                                 │
├─────────────────────────────────────────────────────────────┤
│  Core Modules:                                              │
│  - Decision Engine (Trading Logic)                         │
│  - Concept Engine (NLP/Semantic)                           │
│  - Risk Engine (VaR, CVaR, etc.)                          │
│  - Sentiment Analysis (News)                               │
│  - Technical Analysis (Indicators)                         │
└─────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Layer | Technology |
|-------|------------|
| API | FastAPI + Uvicorn |
| Database | PostgreSQL + Redis |
| Frontend | React 18 + TypeScript + Vite |
| Styling | Tailwind CSS |
| Charts | Recharts + Plotly |
| ML/AI | scikit-learn, sentence-transformers, FAISS |
| Analytics | Pandas, NumPy |

---

## 2. Backend Architecture

### Directory Structure

```
app/
├── main.py              # FastAPI application entry point
├── api/
│   ├── routes/        # API endpoints
│   │   ├── orders.py      # Order management
│   │   ├── portfolio.py   # Portfolio management
│   │   ├── market.py      # Market data
│   │   ├── risk.py        # Risk metrics
│   │   ├── news.py        # News & sentiment
│   │   └── auth.py        # Authentication
│   └── mock_data.py    # Demo data
├── core/
│   ├── config.py      # Configuration
│   ├── security.py    # JWT & auth
│   ├── database.py    # DB connection
│   └── cache.py       # Redis caching
├── database/
│   ├── models.py      # SQLAlchemy models
│   └── repository.py  # Data access
├── execution/
│   ├── broker_connector.py
│   ├── execution_engine.py
│   └── order_manager.py
├── portfolio/
│   ├── optimization.py
│   └── performance.py
└── risk/
    ├── risk_engine.py
    └── hardened_risk_engine.py
```

### Key Modules

#### Decision Engine (`decision_engine.py`)
- Trading signal generation
- Strategy execution
- Multi-asset support

#### Risk Engine (`app/risk/risk_engine.py`)
- VaR calculation
- CVaR calculation
- Drawdown monitoring
- Position sizing

#### Concept Engine (`concept_engine.py`)
- Semantic search with FAISS
- Embedding generation
- Financial concept extraction

#### Technical Analysis (`technical_analysis.py`)
- RSI, MACD, Bollinger Bands
- Moving averages
- Support/Resistance levels

---

## 3. Frontend Architecture

### Directory Structure

```
frontend/
├── src/
│   ├── pages/
│   │   ├── Dashboard.tsx    # Main dashboard
│   │   ├── Portfolio.tsx    # Portfolio view
│   │   ├── Market.tsx       # Market data
│   │   ├── Orders.tsx       # Order management
│   │   ├── Login.tsx        # Authentication
│   │   └── ...
│   ├── components/
│   │   ├── layout/         # Layout components
│   │   ├── trading/        # Trading components
│   │   └── ui/            # Reusable UI
│   ├── services/
│   │   └── api.ts         # API client
│   └── hooks/              # Custom React hooks
├── dist/                   # Production build
└── package.json
```

### State Management
- React Query for server state
- Context API for auth state
- Local state for UI

---

## 4. Concept Engine

### Overview

The Concept Engine is a semantic knowledge layer for financial concepts. It uses:

- **FAISS** for vector similarity search
- **sentence-transformers** for embeddings
- **Hybrid search** (semantic + keyword)

### Usage

```python
from concept_engine import ConceptEngine

# Initialize
engine = ConceptEngine()

# Semantic search
results = engine.search("staking crypto", k=5)

# Extract concepts from news
concepts = engine.extract_from_text("Bitcoin surge bullish sentiment")

# Analyze sentiment
sentiment = analyze_sentiment("market drop bearish")
```

### Concepts Categories

| Category | Examples |
|----------|----------|
| Trading | Long/Short, Leverage, Stop Loss |
| Risk | VaR, CVaR, Drawdown, Sharpe |
| Market | Volatility, Liquidity, Volume |
| Technical | RSI, MACD, Bollinger Bands |
| DeFi | Staking, Liquidity Pool, Yield Farming |
| Economics | Inflation, Interest Rate, GDP |

---

## 5. API Reference

### Portfolio Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/portfolio/summary` | Portfolio overview |
| GET | `/api/portfolio/positions` | Open positions |
| GET | `/api/portfolio/performance` | Performance metrics |
| GET | `/api/portfolio/allocation` | Asset allocation |
| GET | `/api/portfolio/history` | Equity curve |

### Order Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/orders` | List orders |
| POST | `/api/orders` | Create order |
| GET | `/api/orders/{id}` | Get order |
| PATCH | `/api/orders/{id}` | Update order |
| DELETE | `/api/orders/{id}` | Cancel order |
| POST | `/api/orders/emergency-stop` | Emergency stop |

### Market Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/market/prices` | All prices |
| GET | `/api/market/prices/{symbol}` | Specific price |
| GET | `/api/market/candles/{symbol}` | OHLCV data |

### Risk Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/risk/metrics` | Risk metrics |
| GET | `/api/risk/var` | Value at Risk |
| GET | `/api/risk/drawdown` | Drawdown data |

---

## 6. Database Schema

### Core Tables

```sql
-- Users
users (
    id UUID PRIMARY KEY,
    email VARCHAR(255) UNIQUE,
    password_hash VARCHAR(255),
    created_at TIMESTAMP,
    is_active BOOLEAN
)

-- Portfolios
portfolios (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users,
    name VARCHAR(255),
    balance DECIMAL,
    created_at TIMESTAMP
)

-- Positions
positions (
    id UUID PRIMARY KEY,
    portfolio_id UUID REFERENCES portfolios,
    symbol VARCHAR(20),
    side VARCHAR(10),
    quantity DECIMAL,
    entry_price DECIMAL,
    created_at TIMESTAMP
)

-- Orders
orders (
    id UUID PRIMARY KEY,
    portfolio_id UUID REFERENCES portfolios,
    symbol VARCHAR(20),
    side VARCHAR(10),
    order_type VARCHAR(20),
    quantity DECIMAL,
    price DECIMAL,
    status VARCHAR(20),
    created_at TIMESTAMP
)
```

---

## 7. Security

### Authentication
- JWT tokens with HS256
- Access tokens (15 min expiry)
- Refresh tokens (7 days)

### Rate Limiting
- 100 requests/minute per IP
- 1000 requests/minute authenticated

### Security Headers
- CORS configured
- X-Frame-Options: DENY
- X-Content-Type-Options: nosniff

---

## 8. Deployment

### Docker (Recommended)

```bash
# Build
docker-compose build

# Run
docker-compose up -d
```

### Manual

```bash
# Backend
python -m uvicorn app.main:app --reload --port 8000

# Frontend
cd frontend && npm run build
```

### Environment Variables

```env
DATABASE_URL=postgresql://user:pass@localhost:5432/db
REDIS_URL=redis://localhost:6379
SECRET_KEY=your-secret-key
DEMO_MODE=true
```

---

## 9. Development

### Setup

```bash
# Clone
git clone https://github.com/ballales1984-wq/ai-trading-system.git
cd ai-trading-system

# Install
pip install -r requirements.txt
cd frontend && npm install

# Run
python -m uvicorn app.main:app --reload
cd frontend && npm run dev
```

### Testing

```bash
pytest tests/
```

### Code Style

- Black for Python
- ESLint for JavaScript
- Prettier for formatting

---

## License

MIT License - See [LICENSE](LICENSE)

---

<div align="center">

*Documentation generated for AI Trading System v2.3.0*

</div>
