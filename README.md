# 🤖 AI Trading System - Professional Trading Platform

<div align="center">

![Version](https://img.shields.io/badge/version-2.3.0-blue)
![Python](https://img.shields.io/badge/python-3.11+-green)
![React](https://img.shields.io/badge/react-18+-blue)
![License](https://img.shields.io/badge/license-MIT-yellow)
![Status](https://img.shields.io/badge/status-active-success)

**Advanced algorithmic trading platform with AI-powered market analysis**

[English](#english) | [Italiano](#italiano)

</div>

---

## 🇬🇧 English

### Overview

AI Trading System is a comprehensive algorithmic trading platform that combines institutional-grade infrastructure with cutting-edge AI/ML technology:

- ⚡ **FastAPI Backend** - High-performance REST API with 88+ endpoints
- 🎨 **React Frontend** - Modern dark-themed dashboard with 10+ pages
- 🧠 **AI Assistant** - Natural language trading assistant (Streamlit)
- 🧠 **Concept Engine** - Semantic knowledge layer with FAISS
- 📊 **ML Monitoring** - Real-time ML model performance tracking
- 💼 **Investor Portal** - Professional investor reporting dashboard

### Key Features

| Feature | Description |
|---------|-------------|
| **Multi-Asset Trading** | Support for Crypto (BTC, ETH, SOL), Stocks, Forex, Options |
| **Risk Management** | VaR, CVaR, Drawdown, Sharpe Ratio, Monte Carlo Simulation |
| **Technical Analysis** | RSI, MACD, Bollinger Bands, Moving Averages, HMM Regime Detection |
| **Machine Learning** | Price prediction, trend detection, ensemble models (XGBoost, LightGBM) |
| **Sentiment Analysis** | News analysis with NLP, semantic concept extraction |
| **Auto Trading** | Fully automated trading with stop-loss and take-profit |
| **Backtesting** | Historical strategy testing with walk-forward analysis |
| **Broker Integration** | Binance, Bybit, Interactive Brokers, Paper Trading |

### Quick Start

```bash
# Clone repository
git clone https://github.com/ballales1984-wq/ai-trading-system.git
cd ai-trading-system

# Install dependencies
pip install -r requirements.txt
cd frontend && npm install

# Start backend (Terminal 1)
python -m uvicorn app.main:app --reload --port 8000

# Start frontend (Terminal 2)
cd frontend && npm run dev

# Access the application
open http://localhost:5173
```

### Services & Pages

| Service | URL | Port |
|---------|-----|------|
| Backend API | http://localhost:8000 | 8000 |
| API Docs | http://localhost:8000/docs | 8000 |
| React Frontend | http://localhost:5173 | 5173 |

### Frontend Pages

| Page | Route | Description |
|------|-------|-------------|
| Dashboard | `/dashboard` | Main trading dashboard with real-time data |
| Portfolio | `/portfolio` | Portfolio management and positions |
| Market | `/market` | Market prices and charts |
| Orders | `/orders` | Order history and management |
| News | `/news` | AI-powered news feed with sentiment |
| Strategy | `/strategy` | Trading strategy configuration |
| Risk | `/risk` | Risk metrics and limits |
| ML Monitoring | `/ml-monitoring` | ML model performance tracking |
| Investor Portal | `/investor-portal` | Professional investor reporting |
| Settings | `/settings` | System configuration |

### Project Structure

```
ai-trading-system/
├── app/                    # FastAPI backend
│   ├── api/routes/        # API endpoints (orders, portfolio, market, risk, etc.)
│   ├── core/              # Core utilities (security, logging, config)
│   ├── database/          # Database models and repository
│   ├── execution/         # Order execution and broker connectors
│   ├── portfolio/         # Portfolio management and optimization
│   ├── risk/              # Risk engine (VaR, CVaR, hardened risk)
│   └── strategies/        # Trading strategies (momentum, mean reversion)
├── frontend/               # React frontend
│   ├── src/
│   │   ├── pages/        # Page components (Dashboard, Portfolio, etc.)
│   │   ├── components/   # Reusable UI components
│   │   ├── services/     # API client services
│   │   └── hooks/        # Custom React hooks
│   └── dist/             # Production build
├── src/                   # Trading core modules
│   ├── decision/         # Decision engine and filters
│   ├── execution/        # Auto executor with SL/TP
│   ├── core/             # State manager, data collector
│   └── hft/              # High-frequency trading module
├── decision_engine/       # Trading decision logic
├── technical_analysis.py  # Technical indicators
├── sentiment_news.py      # Sentiment analysis
├── concept_engine.py      # Semantic knowledge layer
├── ml_predictor.py       # ML price prediction
├── main_auto_trader.py   # Auto trading main loop
└── requirements.txt      # Python dependencies
```

### Technology Stack

#### Backend
- **FastAPI** - Modern async web framework
- **SQLAlchemy** - Database ORM
- **Pydantic** - Data validation
- **Redis** - Caching
- **TimescaleDB** - Time-series data (production)

#### Frontend
- **React 18** - UI library
- **TypeScript** - Type safety
- **Vite** - Build tool
- **Tailwind CSS** - Styling
- **Recharts** - Charts and visualizations
- **ECharts** - Advanced charting

#### AI/ML
- **sentence-transformers** - Semantic embeddings
- **FAISS** - Vector similarity search
- **scikit-learn** - ML algorithms
- **XGBoost/LightGBM** - Gradient boosting
- **pandas** - Data analysis
- **hmmlearn** - Hidden Markov Models for regime detection

### API Endpoints

#### Portfolio
- `GET /api/portfolio/summary` - Portfolio overview
- `GET /api/portfolio/positions` - Open positions
- `GET /api/portfolio/performance` - Performance metrics
- `GET /api/portfolio/allocation` - Asset allocation
- `GET /api/portfolio/history` - Equity curve history

#### Orders
- `GET /api/orders` - List orders
- `POST /api/orders` - Create order
- `POST /api/orders/emergency-stop` - Emergency stop

#### Market
- `GET /api/market/prices` - Current prices
- `GET /api/market/candles/{symbol}` - OHLCV data
- `GET /api/market/sentiment` - Market sentiment

#### Risk
- `GET /api/risk/metrics` - Risk metrics (VaR, CVaR, drawdown)
- `GET /api/risk/limits` - Risk limits configuration
- `GET /api/risk/correlation` - Asset correlation matrix

#### Strategy
- `GET /api/strategy/` - List strategies
- `POST /api/strategy/` - Create strategy

### Risk Metrics

The system calculates:
- **VaR (Value at Risk)** - Maximum expected loss at 95% confidence
- **CVaR (Conditional VaR)** - Expected shortfall
- **Sharpe Ratio** - Risk-adjusted returns
- **Sortino Ratio** - Downside risk-adjusted returns
- **Max Drawdown** - Peak-to-trough loss
- **Calmar Ratio** - Return/max drawdown

### Auto Trading Features

- **Opportunity Filter Pro** - Advanced signal filtering
- **Monte Carlo Simulation** - Scenario analysis
- **Stop-Loss** - 4% automatic position closing
- **Take-Profit** - 5% profit target
- **StateManager** - Persistent trading state in SQLite
- **Real-time Updates** - WebSocket price streaming

### Development

```bash
# Backend only
python -m uvicorn app.main:app --reload --port 8000

# Frontend only
cd frontend && npm run dev

# Auto trader
python main_auto_trader.py --mode live --dry-run

# Run tests
pytest tests/ --cov=app
```

### Environment Variables

Create `.env` file:

```env
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/aitrading

# API Keys
BINANCE_API_KEY=your_key
BINANCE_SECRET=your_secret

# Security
SECRET_KEY=your_secret_key
ALGORITHM=HS256

# Demo Mode
DEMO_MODE=true
```

### Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### License

MIT License - See [LICENSE](LICENSE) for details.

---

## 🇮🇹 Italiano

### Panoramica

AI Trading System è una piattaforma di trading algoritmica completa che combina infrastruttura di livello istituzionale con tecnologia AI/ML all'avanguardia:

- ⚡ **Backend FastAPI** - API REST ad alte prestazioni con 88+ endpoint
- 🎨 **Frontend React** - Dashboard moderna con tema scuro e 10+ pagine
- 🧠 **Assistente AI** - Assistente trading in linguaggio naturale (Streamlit)
- 🧠 **Concept Engine** - Layer semantico con FAISS
- 📊 **ML Monitoring** - Monitoraggio performance modelli ML in tempo reale
- 💼 **Investor Portal** - Dashboard reporting per investitori professionali

### Caratteristiche Principali

| Caratteristica | Descrizione |
|----------------|-------------|
| **Trading Multi-Asset** | Crypto (BTC, ETH, SOL), Azioni, Forex, Opzioni |
| **Risk Management** | VaR, CVaR, Drawdown, Sharpe Ratio, Monte Carlo |
| **Analisi Tecnica** | RSI, MACD, Bollinger Bands, Medie Mobili, HMM |
| **Machine Learning** | Previsione prezzi, rilevamento trend, ensemble models |
| **Analisi Sentiment** | Analisi news con NLP, estrazione concetti semantici |
| **Auto Trading** | Trading automatizzato con stop-loss e take-profit |
| **Backtesting** | Test storico strategie con walk-forward analysis |
| **Broker Integration** | Binance, Bybit, Interactive Brokers, Paper Trading |

### Avvio Rapido

```bash
# Clona repository
git clone https://github.com/ballales1984-wq/ai-trading-system.git
cd ai-trading-system

# Installa dipendenze
pip install -r requirements.txt
cd frontend && npm install

# Avvia backend (Terminale 1)
python -m uvicorn app.main:app --reload --port 8000

# Avvia frontend (Terminale 2)
cd frontend && npm run dev

# Accedi all'applicazione
apri http://localhost:5173
```

### Stack Tecnologico

#### Backend
- **FastAPI** - Framework web asincrono moderno
- **SQLAlchemy** - ORM per database
- **Pydantic** - Validazione dati
- **Redis** - Caching
- **TimescaleDB** - Dati time-series (produzione)

#### Frontend
- **React 18** - Libreria UI
- **TypeScript** - Tipo-sicurezza
- **Vite** - Tool di build
- **Tailwind CSS** - Stiling
- **Recharts** - Grafici e visualizzazioni
- **ECharts** - Grafici avanzati

#### AI/ML
- **sentence-transformers** - Embedding semantici
- **FAISS** - Ricerca similarità vettoriale
- **scikit-learn** - Algoritmi ML
- **XGBoost/LightGBM** - Gradient boosting
- **pandas** - Analisi dati
- **hmmlearn** - Hidden Markov Models per rilevamento regime

### Variabili di Ambiente

Crea file `.env`:

```env
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/aitrading

# Chiavi API
BINANCE_API_KEY=tua_chiave
BINANCE_SECRET=tuo_segreto

# Sicurezza
SECRET_KEY=tua_chiave_segreta
ALGORITHM=HS256

# Modalità Demo
DEMO_MODE=true
```

### Licenza

Licenza MIT - Vedi [LICENSE](LICENSE) per dettagli.

---

<div align="center">

**🚀 Built with ❤️ for algorithmic trading**

*Copyright © 2024-2026 AI Trading System*

</div>
