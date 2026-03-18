# 🤖 AI Trading System - Professional Trading Platform

<div align="center">

![Version](https://img.shields.io/badge/version-2.3.0-blue)
![Python](https://img.shields.io/badge/python-3.14+-green)
![React](https://img.shields.io/badge/react-18+-blue)
![License](https://img.shields.io/badge/license-MIT-yellow)
![Status](https://img.shields.io/badge/status-active-success)

**Advanced algorithmic trading platform with AI-powered market analysis**

[English](#english) | [Italiano](#italiano)

</div>

---

## 🇬🇧 English

### Overview

AI Trading System is a comprehensive algorithmic trading platform that combines:

- ⚡ **FastAPI Backend** - High-performance REST API
- 🎨 **React Frontend** - Modern dark-themed dashboard
- 📊 **Python Dashboards** - Advanced analytics (Dash + Streamlit)
- 🧠 **AI Assistant** - Natural language trading assistant
- 🧠 **Concept Engine** - Semantic knowledge layer for financial concepts

### Key Features

| Feature | Description |
|---------|-------------|
| **Multi-Asset Trading** | Support for Crypto, Stocks, Forex, Options |
| **Risk Management** | VaR, CVaR, Drawdown, Sharpe Ratio |
| **Technical Analysis** | RSI, MACD, Bollinger Bands, Moving Averages |
| **Machine Learning** | Price prediction, trend detection, HMM regime detection |
| **Sentiment Analysis** | News analysis with NLP |
| **Semantic Search** | FAISS-powered concept engine |

### Quick Start

```bash
# Clone repository
git clone https://github.com/ballales1984-wq/ai-trading-system.git
cd ai-trading-system

# Install dependencies
pip install -r requirements.txt
cd frontend && npm install

# Start all services
./start_all_services.bat
```

### Services

| Service | URL | Port |
|---------|-----|------|
| Backend API | http://localhost:8000 | 8000 |
| API Docs | http://localhost:8000/docs | 8000 |
| React Frontend | http://localhost:5173 | 5173 |
| Python Dashboard | http://localhost:8050 | 8050 |
| AI Assistant | http://localhost:8501 | 8501 |

### Project Structure

```
ai-trading-system/
├── app/                    # FastAPI backend
│   ├── api/              # API routes
│   ├── core/             # Core utilities
│   ├── database/         # Database models
│   ├── execution/        # Order execution
│   ├── portfolio/        # Portfolio management
│   ├── risk/             # Risk engine
│   └── strategies/       # Trading strategies
├── frontend/              # React frontend
│   ├── src/
│   │   ├── pages/       # Page components
│   │   ├── components/  # Reusable components
│   │   └── services/    # API services
│   └── dist/            # Production build
├── dashboard/            # Python Dash dashboards
├── concept_engine.py     # Semantic knowledge layer
├── sentiment_news.py     # News sentiment analysis
├── decision_engine.py    # Trading decision engine
├── technical_analysis.py # Technical indicators
└── requirements.txt      # Python dependencies
```

### Technology Stack

#### Backend
- **FastAPI** - Modern async web framework
- **SQLAlchemy** - Database ORM
- **Pydantic** - Data validation
- **Redis** - Caching

#### Frontend
- **React 18** - UI library
- **TypeScript** - Type safety
- **Vite** - Build tool
- **Tailwind CSS** - Styling
- **Recharts** - Charts

#### AI/ML
- **sentence-transformers** - Semantic embeddings
- **FAISS** - Vector similarity search
- **scikit-learn** - ML algorithms
- **pandas** - Data analysis

### API Endpoints

#### Portfolio
- `GET /api/portfolio/summary` - Portfolio overview
- `GET /api/portfolio/positions` - Open positions
- `GET /api/portfolio/performance` - Performance metrics
- `GET /api/portfolio/allocation` - Asset allocation

#### Orders
- `GET /api/orders` - List orders
- `POST /api/orders` - Create order
- `POST /api/orders/emergency-stop` - Emergency stop

#### Market
- `GET /api/market/prices` - Current prices
- `GET /api/market/candles/{symbol}` - OHLCV data

### Risk Metrics

The system calculates:
- **VaR (Value at Risk)** - Maximum expected loss
- **CVaR (Conditional VaR)** - Expected shortfall
- **Sharpe Ratio** - Risk-adjusted returns
- **Max Drawdown** - Peak-to-trough loss
- **Sortino Ratio** - Downside risk-adjusted returns

### Development

```bash
# Backend only
python -m uvicorn app.main:app --reload --port 8000

# Frontend only
cd frontend && npm run dev

# Dashboard
cd dashboard && python app.py

# AI Assistant
streamlit run ai_financial_dashboard.py --server.port 8501
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

AI Trading System è una piattaforma di trading algoritmica completa che combina:

- ⚡ **Backend FastAPI** - API REST ad alte prestazioni
- 🎨 **Frontend React** - Dashboard moderna con tema scuro
- 📊 **Dashboard Python** - Analisi avanzata (Dash + Streamlit)
- 🧠 **Assistente AI** - Assistente trading in linguaggio naturale
- 🧠 **Concept Engine** - Layer semantico per concetti finanziari

### Caratteristiche Principali

| Caratteristica | Descrizione |
|---------------|-------------|
| **Trading Multi-Asset** | Crypto, Azioni, Forex, Opzioni |
| **Risk Management** | VaR, CVaR, Drawdown, Sharpe Ratio |
| **Analisi Tecnica** | RSI, MACD, Bollinger Bands, Medie Mobili |
| **Machine Learning** | Previsione prezzi, rilevamento trend, HMM |
| **Analisi Sentiment** | Analisi news con NLP |
| **Ricerca Semantica** | Concept engine con FAISS |

### Avvio Rapido

```bash
# Clona repository
git clone https://github.com/ballales1984-wq/ai-trading-system.git
cd ai-trading-system

# Installa dipendenze
pip install -r requirements.txt
cd frontend && npm install

# Avvia tutti i servizi
./start_all_services.bat
```

### Servizi

| Servizio | URL | Porta |
|----------|-----|-------|
| Backend API | http://localhost:8000 | 8000 |
| Documentazione API | http://localhost:8000/docs | 8000 |
| Frontend React | http://localhost:5173 | 5173 |
| Dashboard Python | http://localhost:8050 | 8050 |
| Assistente AI | http://localhost:8501 | 8501 |

### Stack Tecnologico

#### Backend
- **FastAPI** - Framework web asincrono moderno
- **SQLAlchemy** - ORM per database
- **Pydantic** - Validazione dati
- **Redis** - Caching

#### Frontend
- **React 18** - Libreria UI
- **TypeScript** - Tipo-sicurezza
- **Vite** - Tool di build
- **Tailwind CSS** - Stiling
- **Recharts** - Grafici

#### AI/ML
- **sentence-transformers** - Embedding semantici
- **FAISS** - Ricerca similarità vettoriale
- **scikit-learn** - Algoritmi ML
- **pandas** - Analisi dati

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
