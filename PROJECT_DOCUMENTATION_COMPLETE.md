rederne# 📋 DOCUMENTAZIONE COMPLETA DEL PROGETTO AI TRADING SYSTEM

## 1. PANORAMICA GENERALE DEL PROGETTO

### 1.1 Descrizione
L'**AI Trading System** è una piattaforma di trading algoritmico professionale progettata per replicare le capacità di un hedge fund. Il sistema utilizza un'architettura multi-agente, guidata dagli eventi, con un sistema modulare di strategie di trading.

### 1.2 Caratteristiche Principali
- **Architettura Event-Driven**: Pipeline dati asincrone, esecuzione non-bloccante
- **Previsioni Probabilistiche**: Simulazione Monte Carlo a 5 livelli
- **Design Risk-First**: Limiti VaR/CVaR, modellazione volatilità GARCH
- **Rilevamento Regime Adattivo**: HMM per rilevamento condizioni di mercato
- **Multi-Source Intelligence**: 18+ integrazioni API

### 1.3 Stack Tecnologico
| Componente | Tecnologia |
|------------|-----------|
| Backend | Python 3.11+, FastAPI, asyncio |
| ML | XGBoost, LSTM, scikit-learn |
| Database | PostgreSQL, TimescaleDB, Redis |
| Frontend | React 18, TypeScript, Tailwind CSS |
| DevOps | Docker, Kubernetes |

---

## 2. STRUTTURA DELLE CARTELLE

```
ai-trading-system/
├── app/                    # FastAPI application
│   ├── main.py            # Entry point FastAPI
│   ├── api/              # API routes
│   │   └── routes/       # Endpoint REST
│   │       ├── auth.py
│   │       ├── cache.py
│   │       ├── health.py
│   │       ├── market.py
│   │       ├── news.py
│   │       ├── orders.py
│   │       ├── payments.py
│   │       ├── portfolio.py
│   │       ├── risk.py
│   │       ├── strategy.py
│   │       └── waitlist.py
│   ├── core/             # Core functionality
│   │   ├── cache.py
│   │   ├── config.py
│   │   ├── connections.py
│   │   ├── data_adapter.py
│   │   ├── database.py
│   │   ├── logging.py
│   │   ├── logging_production.py
│   │   ├── rate_limiter.py
│   │   ├── rbac.py
│   │   ├── security.py
│   │   ├── structured_logging.py
│   │   └── unified_config.py
│   ├── database/         # Database layer
│   │   ├── async_repository.py
│   │   ├── models.py
│   │   ├── repository.py
│   │   └── timescale_models.py
│   ├── execution/        # Broker connectors
│   │   ├── broker_connector.py
│   │   ├── execution_engine.py
│   │   ├── order_manager.py
│   │   └── connectors/
│   │       ├── binance_connector.py
│   │       ├── ib_connector.py
│   │       └── paper_connector.py
│   ├── market_data/      # Market data
│   │   ├── data_feed.py
│   │   └── websocket_stream.py
│   ├── portfolio/        # Portfolio management
│   │   ├── optimization.py
│   │   └── performance.py
│   ├── risk/            # Risk management
│   │   ├── hardened_risk_engine.py
│   │   └── risk_engine.py
│   └── strategies/       # Trading strategies
│       ├── base_strategy.py
│       ├── mean_reversion.py
│       ├── momentum.py
│       └── multi_strategy.py
│
├── src/                  # Core trading logic (80+ moduli)
│   ├── agents/          # AI Agents
│   │   ├── base_agent.py
│   │   ├── agent_marketdata.py
│   │   ├── agent_montecarlo.py
│   │   ├── agent_risk.py
│   │   └── agent_supervisor.py
│   ├── core/            # Core infrastructure
│   │   ├── event_bus.py
│   │   ├── state_manager.py
│   │   ├── engine.py
│   │   ├── api_rate_manager.py
│   │   ├── capital_protection.py
│   │   ├── dynamic_allocation.py
│   │   ├── resource_monitor.py
│   │   ├── execution/
│   │   ├── portfolio/
│   │   └── risk/
│   ├── automl/          # AutoML
│   │   ├── automl_engine.py
│   │   ├── evolution.py
│   │   └── strategy_evolution_manager.py
│   ├── decision/        # Decision engine
│   │   ├── decision_automatic.py
│   │   ├── decision_montecarlo.py
│   │   └── filtro_opportunita.py
│   ├── decision_engine/ # Decision engine standalone
│   │   ├── __init__.py
│   │   ├── core.py
│   │   ├── signals.py
│   │   ├── monte_carlo.py
│   │   ├── five_question.py
│   │   └── external.py
│   ├── external/        # API integrations (18+ sources)
│   │   ├── api_registry.py
│   │   ├── bybit_client.py
│   │   ├── coinmarketcap_client.py
│   │   ├── okx_client.py
│   │   ├── sentiment_apis.py
│   │   ├── market_data_apis.py
│   │   └── ...
│   ├── hft/            # High-Frequency Trading
│   ├── live/           # Live trading
│   │   ├── binance_multi_ws.py
│   │   ├── live_streaming_manager.py
│   │   ├── portfolio_live.py
│   │   ├── position_sizing.py
│   │   └── risk_engine.py
│   ├── meta/           # Meta-learning
│   ├── models/         # ML models
│   ├── production/     # Production trading
│   ├── research/       # Research modules
│   ├── rl/            # Reinforcement learning
│   ├── simulations/   # Market simulations
│   ├── strategy/      # Trading strategies
│   │   ├── base_strategy.py
│   │   ├── momentum.py
│   │   ├── mean_reversion.py
│   │   └── strategy_comparison.py
│   └── [core files]
│       ├── ml_model.py
│       ├── ml_enhanced.py
│       ├── ml_model_xgb.py
│       ├── ml_tuning.py
│       ├── hmm_regime.py
│       ├── risk_engine.py
│       ├── risk_guard.py
│       ├── risk_optimizer.py
│       ├── technical_analysis.py
│       ├── data_collector.py
│       ├── sentiment_news.py
│       ├── trading_completo.py
│       ├── trading_ledger.py
│       └── ...
│
├── frontend/            # React Frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── layout/
│   │   │   └── ui/
│   │   ├── pages/
│   │   │   ├── Dashboard.tsx
│   │   │   ├── Market.tsx
│   │   │   ├── Orders.tsx
│   │   │   ├── PaymentTest.tsx
│   │   │   └── Portfolio.tsx
│   │   ├── services/
│   │   │   └── api.ts
│   │   ├── types/
│   │   └── App.tsx
│   ├── public/
│   └── package.json
│
├── tests/              # Test suite (311 tests)
│   ├── test_agents.py
│   ├── test_all_modules.py
│   ├── test_app.py
│   ├── test_cache_routes.py
│   ├── test_decision_engine.py
│   ├── test_edge_cases.py
│   ├── test_event_bus.py
│   ├── test_evolution.py
│   ├── test_new_modules.py
│   ├── test_production_features.py
│   ├── test_security.py
│   ├── test_strategies.py
│   ├── test_strategy_evolution.py
│   ├── test_technical_analysis.py
│   └── test_timescale_aggregates.py
│
├── docker/             # Docker configs
├── infra/             # Kubernetes configs
│   └── k8s/
│       ├── deployment.yaml
│       ├── service.yaml
│       ├── secrets.yaml
│       ├── configmap.yaml
│       ├── hpa.yaml
│       ├── storage.yaml
│       └── ingress.yaml
│
├── docs/               # Documentation
│   ├── ARCHITECTURE_V2.md
│   ├── API_V2.md
│   ├── CODE_REVIEW_REPORT.md
│   └── ...
│
├── scripts/            # Utility scripts
├── dashboard/          # Dash dashboard
├── landing/            # Landing page
├── migrations/         # Alembic migrations
├── models/            # Saved ML models
├── plans/             # Planning documents
├── api/               # API server
├── agent_coordination/# Multi-agent system
├── desktop_app/       # Desktop app (Tkinter)
├── java-frontend/     # Java frontend (experimental)
├── data/              # Data storage
├── logs/              # Log files
├── cache/             # Cache storage
└── [config files]
    ├── config.py
    ├── requirements.txt
    ├── docker-compose.yml
    ├── pyproject.toml
    ├── pytest.ini
    └── ...
```

---

## 3. COMPONENTI PRINCIPALI

### 3.1 Decision Engine (`decision_engine/`)
Il cuore del sistema che genera segnali di trading combinando:
- **Analisi Tecnica**: RSI, MACD, Bollinger Bands, EMA
- **Sentiment Analysis**: News, Twitter, social media
- **Monte Carlo**: Simulazione probabilistica a 5 livelli
- **ML Prediction**: XGBoost, modelli ensemble
- **HMM Regime Detection**: Rilevamento regime mercato

**File principali:**
- `core.py`: Strutture dati (TradingSignal, PortfolioState) e classe DecisionEngine
- `signals.py`: SignalGenerator per combinazione fattori
- `monte_carlo.py`: MonteCarloEngine per simulazioni
- `five_question.py`: Framework 5-domande (What, Why, How Much, When, Risk)
- `external.py`: Integrazione API esterne

### 3.2 Agenti AI (`src/agents/`)
Sistema multi-agente per orchestrazione:
- **MarketDataAgent**: Streaming dati mercato
- **MonteCarloAgent**: Simulazioni probabilistiche
- **RiskAgent**: Calcolo VaR/CVaR
- **SupervisorAgent**: Orchestrazione agenti

### 3.3 API REST (`app/api/`)
Endpoints FastAPI per:
- `/api/v1/orders`: Gestione ordini
- `/api/v1/portfolio`: Portfolio management
- `/api/v1/market`: Dati mercato
- `/api/v1/risk`: Metriche rischio
- `/api/v1/strategy`: Strategie trading
- `/api/v1/news`: Notizie e sentiment
- `/api/v1/auth`: Autenticazione
- `/api/v1/payments`: Pagamenti
- `/api/v1/cache`: Cache management

### 3.4 Database Layer
- **PostgreSQL**: Database relazionale
- **TimescaleDB**: Time-series data (OHLCV, trades)
- **Redis**: Cache e message broker

---

## 4. LOGICA DI TRADING

### 4.1 Flusso di Generazione Segnali
```
1. Market Data → Technical Analysis
2. News → Sentiment Analysis
3. Historical Data → HMM Regime Detection
4. Price Data → ML Prediction
5. All Data → Monte Carlo Simulation
6. Combine All → Signal Generator
7. Risk Check → Execute/Reject
8. Order Manager → Broker Execution
```

### 4.2 Pesi del Decision Engine
| Componente | Peso |
|------------|------|
| Technical Analysis | 30% |
| Momentum Signals | 25% |
| Cross-Asset Correlation | 20% |
| Sentiment Score | 15% |
| ML Prediction | 10% |

### 4.3 Monte Carlo 5 Livelli
| Livello | Descrizione |
|---------|-------------|
| 1 | Geometric Brownian Motion |
| 2 | Conditional (event-conditioned) |
| 3 | Adaptive (GARCH volatility) |
| 4 | Multi-Factor (correlations) |
| 5 | Semantic (news-aware) |

---

## 5. RIFERIMENTI E CONFIGURAZIONI

### 5.1 Variabili d'Ambiente (`.env`)
```env
# API Keys
BINANCE_API_KEY=
BINANCE_SECRET_KEY=
NEWS_API_KEY=
COINMARKETCAP_API_KEY=
ALPHA_VANTAGE_API_KEY=
EIA_API_KEY=

# Database
DATABASE_URL=postgresql://...
REDIS_URL=redis://...

# Trading
TRADING_MODE=paper
USE_BINANCE_TESTNET=true
SIMULATION_MODE=true

# Risk
MAX_POSITION_SIZE=0.1
MAX_DAILY_DRAWDOWN=0.05
VAR_CONFIDENCE=0.95

# Telegram
TELEGRAM_ENABLED=false
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
```

### 5.2 Configurazione Simboli (`config.py`)
- **Crypto**: BTC, ETH, SOL, XRP, ADA, DOT, AVAX, MATIC, etc.
- **Commodity Tokens**: PAXG (Gold), XAUT (Gold), WTI (Oil)
- **Fiat Tokens**: PEUR, PGBP, PJPY

### 5.3 Indicatori Tecnici
- RSI (period: 14, overbought: 70, oversold: 30)
- EMA (short: 12, medium: 26, long: 50)
- Bollinger Bands (period: 20, std: 2)
- MACD (fast: 12, slow: 26, signal: 9)
- ATR (period: 14)

---

## 6. ANNOTAZIONI E DOCUMENTAZIONE

### 6.1 Documenti Tecnici
| File | Descrizione |
|------|-------------|
| `README.md` | Documentazione principale |
| `docs/ARCHITECTURE_V2.md` | Architettura sistema |
| `docs/API_V2.md` | Documentazione API |
| `docs/REFACTOR_PLAN.md` | Piano refactoring |
| `ROADMAP_SAAS.md` | Roadmap SaaS |
| `ROADMAP_VISIVA.md` | Roadmap visuale |
| `HARDENING_PLAN.md` | Piano hardening |

### 6.2 Piani di Implementazione
- `plans/PROJECT_STRUCTURE_EXPLAINED.md`
- `plans/FIX_NEWS_FEED_PLAN.md`
- `plans/NEWS_USAGE_ANALYSIS.md`

### 6.3 Note di Rilascio
- `STABLE_RELEASE.md`
- `DEMO_RELEASE_CHECKLIST.md`
- `DEPLOYMENT_SUMMARY.md`

---

## 7. TEST

### 7.1 Suite di Test
- **311 test totali** con stato ✅ PASSED
- Runtime: ~8 minuti
- Coverage: src/, app/, decision_engine/

### 7.2 Esecuzione Test
```bash
# Tutti i test
pytest tests/ -v

# Con coverage
pytest tests/ --cov=src --cov=app --cov-report=html

# Test specifico
pytest tests/test_decision_engine.py -v

# Test integrazione
pytest tests/test_integration.py -v --run-integration
```

---

## 8. DEPLOYMENT

### 8.1 Docker Compose
```bash
# Avvio tutti i servizi
docker-compose up -d

# Servizi inclusi:
# - postgres (port 5432)
# - redis (port 6379)
# - trading-system (port 8050)
# - api (port 8000)
# - frontend (port 3000)
```

### 8.2 Kubernetes
File in `infra/k8s/`:
- deployment.yaml
- service.yaml
- secrets.yaml
- configmap.yaml
- hpa.yaml
- storage.yaml
- ingress.yaml

### 8.3 Vercel (Frontend)
- `vercel.json` configurato
- Deploy automatico da GitHub

---

## 9. RISCHIO E PERFORMANCE

### 9.1 Metriche di Rischio
- **VaR (95%, 99%)**: Value at Risk
- **CVaR**: Conditional VaR / Expected Shortfall
- **Max Drawdown**: Massima riduzione
- **Sharpe Ratio**: Rendimento corretto per rischio
- **Sortino Ratio**: Rischio downside corretto

### 9.2 Parametri di Rischio
| Parametro | Valore |
|-----------|-------|
| Max Position Size | 10% |
| Max Daily Drawdown | 5% |
| Max Correlation Exposure | 30% |
| VaR Confidence | 95% |
| CVaR Limit | 8% |

### 9.3 Performance Backtest
| Metrica | Valore | Benchmark |
|---------|--------|-----------|
| CAGR | 23.5% | 18.2% |
| Max Drawdown | 7.2% | 45.8% |
| Sharpe Ratio | 1.95 | 0.82 |
| Sortino Ratio | 2.45 | 1.12 |
| Win Rate | 68% | - |

---

## 10. FLUSSO DATI

### 10.1 Architettura Event-Driven
```
Exchange APIs → API Registry → Event Bus
                              ↓
                    ┌─────────┼─────────┐
                    ↓         ↓         ↓
              Agents    Strategy   Risk Engine
                    ↓         ↓         ↓
                    └─────────┼─────────┘
                              ↓
                    Order Manager → Brokers
                              ↓
                    Portfolio Update
```

### 10.2 API Routes Flow
```
Client Request
      ↓
FastAPI (app/main.py)
      ↓
Middleware (CORS, Logging, Auth)
      ↓
Route Handler (app/api/routes/)
      ↓
Business Logic (src/, decision_engine/)
      ↓
Database/Cache (PostgreSQL, Redis)
      ↓
Response to Client
```

---

## 11. ESECUZIONE

### 11.1 Quick Start
```bash
# Installazione
pip install -r requirements.txt

# Configurazione
cp .env.example .env

# Avvio Dashboard
python dashboard.py  # http://127.0.0.1:8050

# Avvio API
python -m uvicorn app.main:app --reload  # http://127.0.0.1:8000

# Avvio Frontend
cd frontend && npm install && npm run dev
```

### 11.2 Modalità Trading
- **Simulation**: Prezzi simulati
- **Paper Trading**: Binance Testnet
- **Live Trading**: Binance Real (solo con capitale reale)

---

## 12. AUTORE

**Alessio Ballini**  
Quantitative Developer | Python Engineer | AI Trading Systems

---

## 13. LICENZA

MIT License - vedi file `LICENSE`

---

> *"The goal of a trading system is not to predict the future, but to manage uncertainty in a way that preserves capital and captures opportunities."*

