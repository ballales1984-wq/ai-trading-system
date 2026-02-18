# 🚀 Quantum AI Trading System

**Advanced Quantitative Trading Framework for Crypto & Commodities**

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![API](https://img.shields.io/badge/FastAPI-Included-green)

---

## 🎯 Overview

A professional-grade quantitative trading system with institutional risk management, machine learning signals, and multi-asset portfolio optimization. Supports live trading on Binance Testnet, paper trading simulation, and comprehensive backtesting.

### Key Features

- **Event-Driven Architecture** - Modern, scalable design with async support
- **ML Signal Engine** - RandomForest + XGBoost ensemble with AutoML
- **Institutional Risk Engine** - VaR, CVaR, Monte Carlo simulations
- **Advanced Volatility Models** - GARCH, EGARCH, GJR-GARCH
- **Portfolio Optimization** - CVaR, Risk Parity, Mean-Variance
- **Live Trading** - Binance Testnet integration with WebSocket streaming
- **REST API** - FastAPI server for external integrations
- **Interactive Dashboard** - Real-time monitoring with Dash/Plotly
- **Docker Support** - Containerized deployment
- **Multi-Asset Support** - Crypto, Forex, Commodities

---

## 🏗️ Architecture

### High-Level System Design

```
┌─────────────────────────────────────────────────────────────────────┐
│                     QUANTUM AI TRADING SYSTEM                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                      DATA LAYER                                 │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │ │
│  │  │ Market Data  │  │  News/Sent   │  │   On-Chain Data     │  │ │
│  │  │  Collector   │  │   Analyzer   │  │                     │  │ │
│  │  └──────────────┘  └──────────────┘  └──────────────────────┘  │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                 ↓                                    │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                    SIGNAL LAYER                                 │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │ │
│  │  │ Technical    │  │     ML       │  │   Decision          │  │ │
│  │  │ Indicators   │  │   Models     │  │   Engine            │  │ │
│  │  └──────────────┘  └──────────────┘  └──────────────────────┘  │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                 ↓                                    │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                     RISK LAYER                                  │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │ │
│  │  │    VaR       │  │   GARCH      │  │   Risk              │  │ │
│  │  │   CVaR       │  │  Volatility  │  │   Parity            │  │ │
│  │  └──────────────┘  └──────────────┘  └──────────────────────┘  │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                 ↓                                    │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                   EXECUTION LAYER                               │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │ │
│  │  │   Order      │  │   Portfolio  │  │   Broker            │  │ │
│  │  │   Manager    │  │   Manager    │  │   Connector         │  │ │
│  │  └──────────────┘  └──────────────┘  └──────────────────────┘  │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                      API & UI                                   │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │ │
│  │  │   FastAPI    │  │    Dash      │  │   Java Frontend     │  │ │
│  │  │   Server     │  │  Dashboard   │  │   (Thymeleaf)       │  │ │
│  │  └──────────────┘  └──────────────┘  └──────────────────────┘  │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### Modular Application Structure (`app/`)

```
app/
├── main.py                    # FastAPI application entry
├── api/
│   ├── routes/
│   │   ├── health.py         # Health check endpoints
│   │   ├── market.py         # Market data endpoints
│   │   ├── orders.py         # Order management
│   │   ├── portfolio.py      # Portfolio operations
│   │   ├── risk.py           # Risk metrics
│   │   └── strategy.py       # Strategy management
│   └── __init__.py
├── core/
│   ├── config.py             # Configuration management
│   ├── logging.py            # Logging setup
│   └── security.py           # API security
├── database/                  # Database models (SQLAlchemy)
├── execution/
│   ├── broker_connector.py   # Broker abstraction
│   ├── execution_engine.py  # Order execution logic
│   ├── order_manager.py     # Order lifecycle
│   └── connectors/
│       ├── binance_connector.py
│       └── paper_connector.py
├── market_data/
│   ├── data_feed.py          # Market data feed
│   └── websocket_stream.py   # WebSocket streaming
├── portfolio/                # Portfolio management
├── risk/
│   └── risk_engine.py        # Risk calculations
└── strategies/
    ├── base_strategy.py      # Base strategy class
    ├── mean_reversion.py     # Mean reversion strategy
    ├── momentum.py           # Momentum strategy
    └── multi_strategy.py     # Multi-strategy ensemble
```

---

## 📁 Project Structure

```
ai-trading-system/
│
├── app/                       # Modular FastAPI application
│   ├── api/                  # REST API routes
│   ├── core/                 # Core utilities
│   ├── execution/           # Order execution
│   ├── market_data/         # Market data feeds
│   ├── portfolio/           # Portfolio management
│   ├── risk/                # Risk engine
│   └── strategies/          # Trading strategies
│
├── src/                      # Core trading system
│   ├── core/
│   │   ├── engine.py         # Trading engine orchestrator
│   │   ├── event_bus.py      # Event-driven messaging
│   │   ├── state_manager.py  # SQLite persistence
│   │   ├── portfolio/        # Portfolio management
│   │   ├── execution/        # Order execution
│   │   └── risk/
│   │       ├── institutional_risk_engine.py
│   │       ├── volatility_models.py
│   │       ├── fat_tail_risk.py
│   │       └── multiasset_cvar.py
│   │
│   ├── live/                 # Live trading modules
│   ├── automl/              # AutoML engine
│   ├── ml_model/            # ML models
│   └── hft/                 # High-frequency trading
│
├── dashboard/               # Dash dashboard (Python)
├── java-frontend/          # Java Spring Boot frontend
├── docker/                  # Docker configurations
├── tests/                   # Test suite
├── config.py                # Configuration
├── main.py                  # Entry point
├── api_server.py           # FastAPI server
├── requirements.txt         # Dependencies
└── docker-compose.yml       # Docker Compose
```

---

## 🧠 Machine Learning

### Signal Generation

- **Random Forest** - Ensemble tree-based signals
- **XGBoost** - Gradient boosting signals
- **LightGBM** - Fast gradient boosting
- **Feature Engineering** - Technical indicators, sentiment, on-chain
- **Walk-Forward Validation** - Out-of-sample testing
- **AutoML** - Automated model selection and hyperparameter tuning

### Training

```python
from src.ml_model import EnsembleModel

model = EnsembleModel(n_estimators=100)
model.fit(X_train, y_train)
signals = model.predict(X_test)
```

### AutoML Usage

```python
from src.automl.automl_engine import AutoMLEngine

automl = AutoMLEngine(objective='classification')
best_model = automl.optimize(X_train, y_train, time_limit=300)
```

---

## ⚠️ Risk Management

### Institutional-Grade Features

| Module | Description |
|--------|-------------|
| **VaR** | Value at Risk (Historical, Parametric, Monte Carlo) |
| **CVaR** | Conditional Expected Shortfall |
| **GARCH** | Conditional volatility modeling |
| **Fat-Tail** | Student-t distribution, Extreme Value Theory |
| **Stress Test** | Custom market crash scenarios |
| **Risk Parity** | Equal risk contribution allocation |
| **Multi-Asset CVaR** | Cross-asset portfolio risk |

### Usage

```python
from src.core.risk import InstitutionalRiskEngine

risk = InstitutionalRiskEngine(confidence=0.95)
report = risk.full_risk_report(returns)

print(f"VaR 95%: {report['historical_var']:.2%}")
print(f"CVaR: {report['expected_shortfall']:.2%}")
print(f"Monte Carlo 5%: {report['monte_carlo']['p5']:.2%}")
```

---

## 📊 Dashboard & UI

### Python Dash Dashboard

Real-time monitoring with Dash/Plotly:

- Portfolio positions
- P&L tracking
- Risk metrics visualization
- Signal visualization
- Monte Carlo distributions
- Technical charts

Run dashboard:
```bash
python main.py --mode dashboard
```

Access at: `http://localhost:8050`

### Java Frontend (Spring Boot)

Modern web interface with Thymeleaf templates:

```bash
cd java-frontend
mvn spring-boot:run
```

Access at: `http://localhost:8080`

---

## 🌐 REST API

FastAPI-based REST API for external integrations:

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/market/{symbol}` | GET | Market data |
| `/api/orders` | GET/POST | Order management |
| `/api/portfolio` | GET | Portfolio positions |
| `/api/risk/metrics` | GET | Risk metrics |
| `/api/strategy/signals` | GET | Trading signals |

### Run API Server

```bash
python api_server.py
```

Or with Docker:
```bash
docker-compose up api
```

Access API docs at: `http://localhost:8000/docs`

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure

Copy `.env.example` to `.env` and add your API keys:

```env
BINANCE_API_KEY=your_key
BINANCE_SECRET_KEY=your_secret
NEWS_API_KEY=your_key
COINMARKETCAP_API_KEY=your_key
```

### 3. Run Modes

```bash
# Dashboard
python main.py --mode dashboard

# Paper Trading
python main.py --mode simulate --assets BTCUSDT,ETHUSDT

# Live Trading (Testnet)
python main.py --mode live --assets BTCUSDT,ETHUSDT --simulation

# Backtest
python main.py --mode backtest --symbol BTCUSDT --days 365

# API Server
python api_server.py
```

### 4. Docker Deployment

```bash
# Start all services
docker-compose up -d

# Start with hedge fund mode
docker-compose -f docker-compose.hedgefund.yml up -d

# View logs
docker-compose logs -f
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_paper_trading.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run integration tests
pytest tests/test_phase2.py -v
```

---

## 📈 Risk Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| MAX_DRAWDOWN | 20% | Kill-switch threshold |
| STOPLOSS | 2x ATR | Stop loss multiplier |
| TAKEPROFIT | 3x ATR | Take profit multiplier |
| MAX_POSITION | 30% | Max position size |
| MAX_LEVERAGE | 3x | Maximum leverage |
| VAR_CONFIDENCE | 95% | VaR confidence level |

---

## 🔒 Safety Features

- ✅ Kill-switch at max drawdown
- ✅ Position size limits
- ✅ Order retry logic with exponential backoff
- ✅ SQLite state persistence
- ✅ Paper trading validation tests
- ✅ Risk limits enforcement
- ✅ Circuit breakers for extreme volatility

---

## 📝 License

MIT License - See [LICENSE](LICENSE) file.

---

## 👤 Author

**Quantum AI Trading System**

- GitHub: [ballales1984-wq](https://github.com/ballales1984-wq)
- Built with Python, Scikit-learn, XGBoost, FastAPI, Dash, SQLite

---

## ⚡ Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python 3.10+ |
| ML | Scikit-learn, XGBoost, LightGBM |
| Risk | SciPy, NumPy, Pandas |
| Trading | Binance API, WebSocket |
| API | FastAPI, Uvicorn |
| Dashboard | Dash, Plotly |
| Frontend | Java Spring Boot, Thymeleaf |
| Database | SQLite |
| Orchestration | Event Bus, AsyncIO |
| Container | Docker, Docker Compose |

---

## 📞 Support

For issues and feature requests, please open a GitHub issue.

---

*Built for professional quantitative trading with institutional-grade risk management.*
