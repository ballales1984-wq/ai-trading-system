# 🤖 AI Trading System — Mini Hedge Fund

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](Dockerfile)
[![Tests](https://img.shields.io/badge/Tests-235+-green.svg)](tests/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Ready-brightgreen.svg)](app/)
[![Dash](https://img.shields.io/badge/Dash-Dashboard-orange.svg)](dashboard/)

Un **sistema di trading algoritmico di livello professionale** che replica le capacità di un hedge fund: ingestione dati multi-sorgente, predizioni ML, simulazioni Monte Carlo a 5 livelli, gestione del rischio istituzionale ed esecuzione automatizzata.

> **🎉 Versione 2.0 — Production Ready**

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/ballales1984-wq/ai-trading-system.git
cd ai-trading-system
pip install -r requirements.txt
```

### 2. Run Dashboard (Dash)
```bash
python dashboard.py
# Open http://127.0.0.1:8050
```

### 3. Run FastAPI (Swagger Docs)
```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
# Open http://127.0.0.1:8000/docs
```

### 4. Run with Docker
```bash
docker-compose up -d
```

---

## 📱 Interfacce Disponibili

| Servizio | Porta | URL | Descrizione |
|----------|-------|-----|-------------|
| **Dashboard** | 8050 | http://localhost:8050 | Dash trading interface |
| **FastAPI** | 8000 | http://localhost:8000/docs | REST API con Swagger |
| **PostgreSQL** | 5432 | localhost:5432 | Database TimescaleDB |
| **Redis** | 6379 | localhost:6379 | Cache |

---

## 🏗️ Architettura

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

---

## ✨ Caratteristiche Principali

### 📊 Ingestione Dati Multi-Sorgente (18+ API)
| Categoria | API | Scopo |
|-----------|-----|-------|
| **Market Data** | Binance, CoinGecko, Alpha Vantage | OHLCV prezzi, serie storiche |
| **Sentiment** | NewsAPI, Twitter/X, GDELT | Sentiment notizie, social mood |
| **Macro Events** | Trading Economics, EIA | Calendario economico, GDP, CPI |
| **Natural Events** | Open-Meteo, Climate TRACE | Meteo, clima, idrologia |

### 🎲 Simulazione Monte Carlo (5 Livelli)
1. **Base** — Geometric Brownian Motion random walks
2. **Conditional** — Percorsi condizionati agli eventi
3. **Adaptive** — Reinforcement learning
4. **Multi-Factor** — Eventi naturali, correlazioni, regime switching
5. **Semantic History** — Pattern matching, black swan detection

### 🧠 Decision Engine
- Ensemble pesato: Technical (30%) + Momentum (25%) + Correlation (20%) + Sentiment (15%) + ML (10%)
- ML Predictor (XGBoost/LightGBM/Random Forest)
- Scoring di confidenza (STRONG/MODERATE/WEAK)

### 🛡️ Gestione Rischio Istituzionale
- Value at Risk (VaR) — Historical, Parametric, Monte Carlo
- Conditional VaR (CVaR / Expected Shortfall)
- Modelli GARCH/EGARCH/GJR-GARCH
- Position limits, drawdown controls

### 📈 Execution Engine
- Best execution routing con slippage control
- Order book simulation
- Transaction Cost Analysis (TCA)
- Paper trading + Binance Testnet + Live
- Connettori: Binance, Bybit, OKX, Interactive Brokers

### 🖥️ Dashboard (22 Callbacks)
- Portfolio real-time, P&L, posizioni
- Volatilità, Sharpe ratio, drawdown charts
- Pannello trading Binance
- Order book, trade history, signal history

---

## 📁 Struttura Progetto

```
ai-trading-system/
├── main.py                     # Main entry point
├── dashboard.py                # Dash dashboard
├── app/                        # FastAPI REST API
│   ├── main.py                 # FastAPI app
│   ├── api/routes/             # API routes
│   ├── core/                   # Core modules
│   ├── execution/              # Broker connectors
│   └── database/                # Database layer
├── src/
│   ├── external/               # API clients (18+)
│   ├── core/                   # Trading engine
│   ├── decision/               # Decision engine
│   ├── strategy/               # Trading strategies
│   ├── agents/                 # AI agents
│   └── ml_enhanced.py          # ML models
├── docker/                     # Docker configs
├── tests/                      # Test suite (235+ tests)
└── docs/                       # Documentation
```

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov=app

# Test specific modules
python test_core.py
python test_execution.py
python test_dashboard_integration.py
python test_binance_testnet.py
```

---

## 🐳 Docker

### Development
```bash
docker-compose up -d
```

### Production
```bash
docker-compose -f docker-compose.production.yml up -d
```

---

## ⚙️ Configurazione API Keys

Crea un file `.env`:
```env
# Required
BINANCE_API_KEY=your_key
BINANCE_SECRET_KEY=your_secret
USE_BINANCE_TESTNET=true

# Optional
NEWSAPI_KEY=your_newsapi_key
ALPHA_VANTAGE_API_KEY=your_av_key
```

---

## 📊 Project Status

```
COMPLETED:    ████████████████████████████████████████████████████████░░ 95%
REMAINING:    ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 5%
```

| Componente | Status |
|------------|--------|
| Core Architecture v2.0 | ✅ |
| Event Bus System | ✅ |
| Trading Engine | ✅ |
| Portfolio Manager | ✅ |
| Risk Engine | ✅ |
| Broker Interface | ✅ |
| Dashboard v2.0 | ✅ |
| ML Models | ✅ |
| FastAPI | ✅ |
| Docker | ✅ |
| CI/CD | ✅ |

---

## 📄 Licenza

MIT License — vedi [LICENSE](LICENSE)

---

*Built with Python 3.11+ | FastAPI | Dash | NumPy | Pandas | scikit-learn | XGBoost | LightGBM*

