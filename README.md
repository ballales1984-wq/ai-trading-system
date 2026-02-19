# 🤖 AI Trading System

> **Professional Quantitative Trading Platform** with institutional-grade risk management, machine learning signals, and multi-asset portfolio optimization.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![API](https://img.shields.io/badge/FastAPI-Included-green.svg)](https://fastapi.tiangolo.com/)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)]()

---

## ✨ Key Features

- **🏗️ Event-Driven Architecture** - Modern async design with modular components
- **🧠 ML Signal Engine** - RandomForest + XGBoost ensemble with AutoML
- **⚠️ Institutional Risk Engine** - VaR, CVaR, Monte Carlo, GARCH volatility
- **📊 Portfolio Optimization** - CVaR, Risk Parity, Mean-Variance
- **🚀 Live Trading** - Binance Testnet integration with WebSocket streaming
- **🌐 REST API** - FastAPI server for external integrations
- **📈 Interactive Dashboard** - Real-time monitoring with Dash/Plotly
- **🐳 Docker Support** - Containerized deployment
- **💰 Multi-Asset** - Crypto, Forex, Commodities

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACES                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐   │
│  │  Dashboard  │  │  REST API   │  │   Java Frontend     │   │
│  │  (Dash)     │  │  (FastAPI)  │  │   (Spring Boot)     │   │
│  └─────────────┘  └─────────────┘  └─────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      TRADING ENGINE                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐   │
│  │  Decision   │  │  Execution  │  │   Portfolio         │   │
│  │  Engine     │  │  Engine    │  │   Manager           │   │
│  └─────────────┘  └─────────────┘  └─────────────────────┘   │
│         ↓                ↓                  ↓                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐   │
│  │    Risk     │  │  Event Bus  │  │   State Manager     │   │
│  │  Engine     │  │   (Async)   │  │   (SQLite)          │   │
│  └─────────────┘  └─────────────┘  └─────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      ANALYTICS LAYER                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐   │
│  │ Technical   │  │ Sentiment   │  │   ML Models        │   │
│  │ Analysis    │  │ Analysis    │  │   (AutoML)         │   │
│  └─────────────┘  └─────────────┘  └─────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      DATA COLLECTION                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐   │
│  │   Market    │  │    News     │  │   On-Chain Data    │   │
│  │   Data      │  │   Feed      │  │                     │   │
│  └─────────────┘  └─────────────┘  └─────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/ballales1984-wq/ai-trading-system.git
cd ai-trading-system

# Install dependencies
pip install -r requirements.txt
```

### Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit with your API keys
nano .env
```

Required API keys:
- `BINANCE_API_KEY` / `BINANCE_SECRET_KEY` - For live trading
- `NEWS_API_KEY` - For sentiment analysis
- `COINMARKETCAP_API_KEY` - For market data

### Run Modes

```bash
# Start Dashboard
python main.py --mode dashboard

# Paper Trading Simulation
python main.py --mode simulate --assets BTCUSDT,ETHUSDT

# Live Trading (Testnet)
python main.py --mode live --assets BTCUSDT,ETHUSDT

# Backtest
python main.py --mode backtest --symbol BTCUSDT --days 365

# Start API Server
python api_server.py
```

### Docker

```bash
# Start all services
docker-compose up -d

# Start with hedge fund mode
docker-compose -f docker-compose.hedgefund.yml up -d
```

---

## 📁 Project Structure

```
ai-trading-system/
│
├── app/                      # Modular FastAPI application
│   ├── api/routes/          # REST API endpoints
│   ├── core/                # Configuration, security
│   ├── execution/           # Order execution & connectors
│   ├── market_data/         # Market data feeds
│   ├── portfolio/           # Portfolio management
│   ├── risk/                # Risk engine
│   └── strategies/          # Trading strategies
│
├── src/                      # Core trading system
│   ├── core/                # Engine, event bus, state
│   ├── live/                # Live trading modules
│   ├── automl/              # AutoML engine
│   ├── ml_model/            # ML models
│   └── hft/                 # High-frequency trading
│
├── dashboard/               # Dash dashboard (Python)
├── java-frontend/           # Java Spring Boot frontend
├── docker/                  # Docker configurations
├── tests/                   # Test suite
└── config.py                # Configuration
```

---

## 🧠 Machine Learning

### Signal Generation

The system uses ensemble ML models for signal generation:

- **Random Forest** - Tree-based ensemble signals
- **XGBoost** - Gradient boosting signals
- **LightGBM** - Fast gradient boosting
- **Feature Engineering** - Technical indicators, sentiment, on-chain data
- **Walk-Forward Validation** - Out-of-sample testing
- **AutoML** - Automated model selection

```python
from src.ml_model import EnsembleModel

model = EnsembleModel(n_estimators=100)
model.fit(X_train, y_train)
signals = model.predict(X_test)
```

---

## ⚠️ Risk Management

### Institutional-Grade Features

| Module | Description |
|--------|-------------|
| **VaR** | Value at Risk (Historical, Parametric, Monte Carlo) |
| **CVaR** | Conditional Expected Shortfall |
| **GARCH** | Conditional volatility modeling |
| **Fat-Tail** | Extreme Value Theory |
| **Stress Test** | Custom crash scenarios |
| **Risk Parity** | Equal risk contribution |

```python
from src.core.risk import InstitutionalRiskEngine

risk = InstitutionalRiskEngine(confidence=0.95)
report = risk.full_risk_report(returns)

print(f"VaR 95%: {report['historical_var']:.2%}")
print(f"CVaR: {report['expected_shortfall']:.2%}")
```

---

## 🌐 REST API

FastAPI server running on `http://localhost:8000`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/market/{symbol}` | GET | Market data |
| `/api/orders` | GET/POST | Order management |
| `/api/portfolio` | GET | Portfolio positions |
| `/api/risk/metrics` | GET | Risk metrics |
| `/api/strategy/signals` | GET | Trading signals |

API Documentation: `http://localhost:8000/docs`

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_paper_trading.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

---

## 📊 Dashboard

Access at `http://localhost:8050`

Features:
- Portfolio positions & P&L
- Risk metrics visualization
- Signal visualization
- Technical charts
- Monte Carlo distributions

---

## 🔒 Safety Features

- ✅ Kill-switch at max drawdown (20%)
- ✅ Position size limits (30% max)
- ✅ Stop-loss & take-profit automation
- ✅ Order retry with exponential backoff
- ✅ SQLite state persistence
- ✅ Circuit breakers for extreme volatility

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
| Frontend | Java Spring Boot |
| Database | SQLite |
| Container | Docker, Docker Compose |

---

## 📝 License

MIT License - See [LICENSE](LICENSE) file.

---

## 👤 Author

**Alessio Ballarè** - [ballales1984-wq](https://github.com/ballales1984-wq)

---

## 🙏 Acknowledgments

Built with Python, Scikit-learn, XGBoost, FastAPI, Dash, SQLite

---

*🤖 Professional quantitative trading with institutional-grade risk management*

