# 🤖 AI Trading System — Mini Hedge Fund

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](Dockerfile)
[![Tests](https://img.shields.io/badge/Tests-235+-green.svg)](tests/)
[![Production](https://img.shields.io/badge/Production-Ready-brightgreen.svg)](docker-compose.production.yml)
[![Coverage](https://img.shields.io/badge/Coverage-95%25-green.svg)](tests/)

A **professional-grade algorithmic trading system** that replicates hedge fund capabilities: multi-source data ingestion, ML-powered predictions, 5-level Monte Carlo simulations, institutional risk management, and automated execution.

> **🎉 Version 2.0 — Production Ready (95% Complete)**

---

## 🏗️ Architecture Overview

```
External APIs (18+)  →  API Registry  →  Central Database
                                              ↓
                                      Analysis Engine
                                    (Technical + Sentiment + Events)
                                              ↓
                                      Monte Carlo Engine (5 Levels)
                                              ↓
                                      Decision Engine
                                      (BUY/SELL/HOLD + Confidence)
                                              ↓
                                      Execution Engine → Exchanges
                                              ↓
                                      Dashboard + Alerts + Logs
```

> See [API_FLOW_DIAGRAM.md](API_FLOW_DIAGRAM.md) for the complete visual block diagram.

---

## ✨ Key Features

### 📊 Multi-Source Data Ingestion (18 APIs)
| Category | APIs | Purpose |
|---|---|---|
| **Market Data** | Binance, CoinGecko, Alpha Vantage, Quandl, CoinMarketCap | OHLCV prices, historical series, crypto rankings |
| **Sentiment** | NewsAPI, Benzinga, Twitter/X, GDELT | News sentiment, social mood |
| **Macro Events** | Trading Economics, EconPulse, Investing.com | Economic calendar, GDP, CPI |
| **Natural Events** | Open-Meteo, Climate TRACE, USGS | Weather, climate, hydrology |
| **Innovation** | EIA, Google Patents, Lens.org | Energy prices, tech patents |

### 🎲 Monte Carlo Simulation (5 Levels)
1. **Base** — Geometric Brownian Motion random walks
2. **Conditional** — Event-conditioned paths (macro + sentiment)
3. **Adaptive** — Reinforcement learning from past accuracy
4. **Multi-Factor** — Natural events, cross-correlations, regime switching
5. **Semantic History** — Pattern matching, black swan detection, fat tails

### 🧠 Decision Engine
- Weighted ensemble: Technical (30%) + Momentum (25%) + Correlation (20%) + Sentiment (15%) + ML (10%) + Monte Carlo (10%)
- ML Predictor (XGBoost/LightGBM/Random Forest)
- External sentiment blending with source reliability weighting
- Confidence scoring with strength labels (STRONG/MODERATE/WEAK)

### 🛡️ Institutional Risk Management
- Value at Risk (VaR) — Historical, Parametric, Monte Carlo
- Conditional VaR (CVaR / Expected Shortfall)
- GARCH/EGARCH/GJR-GARCH volatility models
- Fat-tail risk analysis
- Position limits, drawdown controls, correlation checks

### 📈 Execution Engine
- Best execution routing with slippage control
- Order book simulation
- Transaction Cost Analysis (TCA)
- Paper trading + Binance Testnet + Live execution
- Bybit and OKX exchange connectors

### 🖥️ Dashboard (22 Live Callbacks)
- Real-time portfolio, P&L, positions
- Rolling volatility, Sharpe ratio, drawdown charts
- Binance trading panel (Execute Order, Save Settings)
- Strategy allocation selector
- Order book, trade history, signal history

---

## 🆕 Recent Updates (v2.0 — Production Ready)

### 🎉 Production Features Completed

| Feature | Description | Status |
|---------|-------------|--------|
| **TimescaleDB** | Time-series database with hypertables, continuous aggregates, compression | ✅ |
| **Hardened Risk Engine** | Circuit breakers, kill switches, VaR/CVaR limits, position controls | ✅ |
| **Production Logging** | JSON structured logging, correlation IDs, sensitive data masking | ✅ |
| **CI/CD Pipeline** | GitHub Actions with code quality, security scans, Docker build, K8s deploy | ✅ |
| **Docker Production** | Multi-stage builds, Nginx reverse proxy, Prometheus metrics | ✅ |
| **Broker Connectors** | Binance, Bybit, Interactive Brokers, Paper trading | ✅ |

### New Features
- **HMM Regime Detection** — Hidden Markov Models for market regime identification
- **SQLAlchemy Database Layer** — Modern ORM with Alembic migrations
- **Enhanced ML Stack** — XGBoost, LightGBM, SHAP explainability
- **Deep Learning Ready** — PyTorch + Transformers for NLP sentiment
- **Redis Cache** — Hot data caching for improved performance
- **Twitter/X Integration** — Real-time social sentiment via Tweepy
- **Live News Feed** — Real-time crypto news from CoinGecko with dynamic fallback
- **Multi-Agent System** — Market makers, arbitrageurs, retail agents simulation
- **AutoML Engine** — Evolutionary strategy optimization
- **HFT Simulator** — Tick-by-tick simulation with order book

### Bug Fixes (Feb 2026)
- **Fixed Dashboard News Feed** — News was stuck showing static content. Now fetches live news from CoinGecko API with proper `?page=1` parameter and includes dynamic time-based fallback when API is unavailable.

### New Dependencies
| Category | Libraries |
|----------|-----------|
| ML/DL | xgboost, lightgbm, shap, hmmlearn, torch, transformers |
| Database | SQLAlchemy, psycopg, psycopg2-binary, alembic, redis, timescaledb |
| APIs | tweepy, ccxt |
| Broker | ib_insync (Interactive Brokers) |
| Monitoring | prometheus-client, grafana |

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/ballales1984-wq/ai-trading-system.git
cd ai-trading-system
pip install -r requirements.txt
```

### 2. Configure API Keys
Edit `.env` with your API keys:
```env
# Required
BINANCE_API_KEY=your_key
BINANCE_SECRET_KEY=your_secret
USE_BINANCE_TESTNET=true

# Recommended
NEWSAPI_KEY=your_newsapi_key
ALPHA_VANTAGE_API_KEY=your_av_key

# Optional (enhances Monte Carlo levels 2-5)
TRADING_ECONOMICS_API_KEY=your_te_key
EIA_API_KEY=your_eia_key
TWITTER_BEARER_TOKEN=your_twitter_token
```

### 3. Run Dashboard
```bash
python dashboard.py
# Open http://127.0.0.1:8050
```

### 4. Run Trading Engine
```bash
python main.py
```

### 5. Run with Docker
```bash
docker-compose up -d
```

---

## 📁 Project Structure

```
ai-trading-system/
├── main.py                     # Main entry point
├── dashboard.py                # Dash dashboard (22 callbacks)
├── decision_engine.py          # Signal generation + Monte Carlo 5 levels
├── data_collector.py           # Market data ingestion (ccxt)
├── technical_analysis.py       # RSI, MACD, Bollinger, patterns
├── sentiment_news.py           # Sentiment analysis (NLP)
├── ml_predictor.py             # ML price prediction
├── config.py                   # Configuration & settings
├── .env                        # API keys (15+ services)
│
├── src/
│   ├── external/               # 🌐 External API clients
│   │   ├── api_registry.py     # Central API factory & dispatcher
│   │   ├── market_data_apis.py # Binance, CoinGecko, Alpha Vantage, Quandl
│   │   ├── sentiment_apis.py   # NewsAPI, Benzinga, Twitter, GDELT
│   │   ├── macro_event_apis.py # Trading Economics, EconPulse
│   │   ├── natural_event_apis.py # Open-Meteo, Climate TRACE, USGS
│   │   ├── innovation_apis.py  # EIA, Google Patents, Lens.org
│   │   ├── bybit_client.py     # Bybit exchange connector
│   │   └── okx_client.py       # OKX exchange connector
│   │
│   ├── core/
│   │   ├── engine.py           # Core trading engine
│   │   ├── event_bus.py        # Event-driven architecture
│   │   ├── state_manager.py    # State persistence (SQLite)
│   │   ├── execution/          # Order management, best execution, TCA
│   │   ├── portfolio/          # Portfolio manager
│   │   └── risk/               # VaR, CVaR, GARCH, fat-tail risk
│   │
│   ├── automl/                 # AutoML engine
│   ├── strategy/               # Trading strategies
│   ├── hedgefund_ml.py         # Hedge fund ML strategies
│   ├── ml_enhanced.py          # Enhanced ML models
│   ├── portfolio_optimizer.py  # Mean-variance, risk parity
│   ├── risk_engine.py          # Risk management
│   ├── hmm_regime.py           # HMM regime detection (NEW)
│   ├── database_sqlalchemy.py  # SQLAlchemy ORM layer (NEW)
│   └── ...                     # 40+ modules
│
├── app/                        # FastAPI REST API
│   ├── api/routes/             # Market, orders, portfolio, risk, strategy
│   ├── execution/              # Broker connectors
│   └── risk/                   # Risk engine
│
├── migrations/                 # Alembic database migrations (NEW)
│
├── java-frontend/              # Spring Boot web dashboard
│
├── docker/                     # Docker configurations
├── .github/workflows/          # CI/CD pipeline
│
├── API_FLOW_DIAGRAM.md         # Visual block diagram (APIs → Engine → Output)
├── API_INTEGRATION_ARCHITECTURE.md  # Mermaid flow diagrams
├── ARCHITECTURE.md             # Technical architecture
├── ECOSYSTEM_MAP.md            # Complete ecosystem map
├── ROADMAP.md                  # Development roadmap
└── TODO_HEDGE_FUND.md          # Hedge fund implementation plan
```

---

## 🔄 How APIs Feed the Engine

```
Step 1: APIRegistry dispatches to all configured APIs
Step 2: Data normalized into unified schema (NormalizedRecord)
Step 3: Stored in database (6 tables: ohlcv, sentiment, events, natural, innovation, geopolitical)
Step 4: Analysis Engine computes indicators + sentiment + event impact
Step 5: Feature Engineering creates multi-factor vectors
Step 6: Monte Carlo runs 5-level simulation
Step 7: Decision Engine generates BUY/SELL/HOLD with probability + confidence
Step 8: Execution Router sends orders
Step 9: Dashboard displays results
Step 10: Feedback Loop updates source weights + model parameters
```

---

## 🧪 Testing

```bash
# Run all tests (235+ tests)
pytest

# Run with coverage
pytest --cov=src --cov=app

# Run production tests
pytest tests/test_production_features.py -v

# Test specific modules
python test_core.py
python test_execution.py
python test_dashboard_integration.py
python test_binance_testnet.py
python test_hmm_regime.py
python test_paper_trading.py
python test_hft_engine.py
```

---

## 📊 API Key Setup Guide

| API | Free Tier | Sign Up |
|---|---|---|
| **Binance** | ✅ Testnet free | [binance.com/api](https://www.binance.com/en/my/settings/api-management) |
| **NewsAPI** | ✅ 100 req/day | [newsapi.org](https://newsapi.org/register) |
| **Alpha Vantage** | ✅ 5 req/min | [alphavantage.co](https://www.alphavantage.co/support/#api-key) |
| **CoinGecko** | ✅ 30 req/min | [coingecko.com](https://www.coingecko.com/en/api) |
| **Quandl** | ✅ 300 req/min | [data.nasdaq.com](https://data.nasdaq.com/sign-up) |
| **Open-Meteo** | ✅ No key needed | [open-meteo.com](https://open-meteo.com/) |
| **GDELT** | ✅ No key needed | [gdeltproject.org](https://www.gdeltproject.org/) |
| **USGS Water** | ✅ No key needed | [waterservices.usgs.gov](https://waterservices.usgs.gov/) |
| **Climate TRACE** | ✅ No key needed | [climatetrace.org](https://climatetrace.org/) |
| **Trading Economics** | 💰 Paid | [tradingeconomics.com](https://tradingeconomics.com/api) |
| **EIA** | ✅ Free | [eia.gov](https://www.eia.gov/opendata/register.php) |
| **Twitter/X** | 💰 Basic plan | [developer.twitter.com](https://developer.twitter.com/) |
| **Benzinga** | 💰 Partner | [docs.benzinga.io](https://docs.benzinga.io/) |

---

## 🐳 Docker

### Development
```bash
# Full system
docker-compose up -d

# Hedge fund mode
docker-compose -f docker-compose.hedgefund.yml up -d
```

### Production Stack
```bash
# Start infrastructure
docker-compose -f docker-compose.production.yml up -d postgres redis

# Wait for services
sleep 30

# Start all services
docker-compose -f docker-compose.production.yml up -d

# Access services
# Dashboard:    http://localhost:8050
# API:          http://localhost:8000
# Grafana:      http://localhost:3000
# Prometheus:   http://localhost:9090
```

### Production Services
| Service | Port | Description |
|---------|------|-------------|
| trading-system | 8050 | Dashboard principale |
| api | 8000 | FastAPI backend |
| postgres | 5432 | TimescaleDB |
| redis | 6379 | Cache |
| prometheus | 9090 | Metriche |
| grafana | 3000 | Dashboard monitoring |
| nginx | 80/443 | Reverse proxy |

---

## 📈 Performance

The system is designed for:
- **Latency**: < 100ms signal generation
- **Throughput**: 1000+ Monte Carlo simulations per signal
- **Uptime**: 24/7 with auto-recovery
- **Scalability**: Add assets/strategies without code changes

---

## 📚 Documentation

| Document | Description |
|---|---|
| [API_FLOW_DIAGRAM.md](API_FLOW_DIAGRAM.md) | Visual block diagram: APIs → Database → Engine → Output |
| [API_INTEGRATION_ARCHITECTURE.md](API_INTEGRATION_ARCHITECTURE.md) | Mermaid diagrams of all data flows |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Technical architecture details |
| [ECOSYSTEM_MAP.md](ECOSYSTEM_MAP.md) | Complete ecosystem map |
| [COMPONENT_DIAGRAM.md](COMPONENT_DIAGRAM.md) | Component interaction diagram |
| [ROADMAP.md](ROADMAP.md) | Development roadmap |
| [DASHBOARD_README.md](DASHBOARD_README.md) | Dashboard usage guide |
| [STATO_PROGETTO.md](STATO_PROGETTO.md) | Project status (Italian) |
| [PRODUCTION_FEATURES.md](PRODUCTION_FEATURES.md) | Production features documentation |

---

## 📊 Project Status

```
COMPLETED:    ████████████████████████████████████████████████████████░░ 95%
REMAINING:    ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 5%
```

| Component | Status |
|-----------|--------|
| Core Architecture v2.0 | ✅ Complete |
| Event Bus System | ✅ Complete |
| State Manager (SQLite) | ✅ Complete |
| Trading Engine | ✅ Complete |
| Portfolio Manager | ✅ Complete |
| Risk Engine (Hardened) | ✅ Complete |
| Broker Interface | ✅ Complete |
| Dashboard v2.0 | ✅ Complete |
| ML Models | ✅ Complete |
| Production Stack | ✅ Complete |
| CI/CD Pipeline | ✅ Complete |
| Test Suite | ✅ Complete |

---

## 📄 License

MIT License — see [LICENSE](LICENSE)

---

*Built with Python 3.11+ | FastAPI | Dash | NumPy | Pandas | scikit-learn | XGBoost | LightGBM | PyTorch*

*Last Updated: 2026-02-20*
