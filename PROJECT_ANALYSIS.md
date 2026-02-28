# 📋 AI Trading System - Documentazione Completa del Progetto

## Panoramica del Progetto

**AI Trading System** è una piattaforma di trading algoritmico professionale progettata per replicare le capacità di un hedge fund. Il sistema utilizza un'architettura multi-agente, guidata dagli eventi, con strategie di trading modulari e gestione avanzata del rischio.

### Caratteristiche Principali

- **Evento-Driven Architecture**: Pipeline dati asincroni, esecuzione non-bloccante
- **Forecasting Probabilistico**: Simulazione Monte Carlo a 5 livelli
- **Risk-First Design**: Limiti VaR/CVaR, modellazione volatilità GARCH
- **Regime Modeling Adattivo**: Rilevamento regime di mercato HMM
- **Multi-Source Intelligence**: 18+ integrazioni API

---

## 📁 Struttura delle Directory Principali

```
ai-trading-system/
├── app/                    # Applicazione FastAPI
│   ├── api/routes/         # Endpoint REST
│   ├── core/              # Sicurezza, cache, DB
│   ├── execution/         # Connettori broker
│   └── database/         # Modelli SQLAlchemy
│
├── src/                   # Logica core del trading
│   ├── agents/            # AI agents (MonteCarlo, Risk, MarketData)
│   ├── core/              # Event bus, state manager
│   ├── decision/          # Decision engine
│   ├── strategy/          # Strategie di trading
│   ├── research/          # Alpha Lab, Feature Store
│   └── external/          # Integrazioni API esterne
│
├── frontend/             # Frontend React/TypeScript
├── dashboard/            # Dashboard Dash
├── tests/                # Suite di test (311 test)
├── docker/               # Configurazioni Docker
├── infra/                # Configurazioni Kubernetes
├── docs/                 # Documentazione
├── decision_engine/      # Motore decisionale
├── api/                  # API Vercel
├── landing/              # Pagina di landing
├── desktop_app/          # App desktop Kivy
├── scripts/              # Script utilità
└── migrations/           # Migrazioni database
```

---

## 📂 Directory Principali - Dettaglio Completo

### 1. `/app` - Applicazione FastAPI

```
app/
├── main.py                          # Entry point FastAPI
├── api/
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── auth.py                  # Autenticazione JWT
│   │   ├── cache.py                 # Gestione cache
│   │   ├── health.py                # Health check
│   │   ├── market.py                 # Dati mercato
│   │   ├── news.py                   # Notizie e sentiment
│   │   ├── orders.py                # Gestione ordini
│   │   ├── payments.py              # Pagamenti Stripe
│   │   ├── portfolio.py             # Portfolio management
│   │   ├── risk.py                  # Metriche rischio
│   │   ├── strategy.py              # Strategie
│   │   └── waitlist.py              # Lista attesa
│   └── mock_data.py                 # Dati mock
│
├── core/
│   ├── __init__.py
│   ├── cache.py                     # Sistema cache Redis
│   ├── config.py                    # Configurazione
│   ├── connections.py               # Connessioni DB
│   ├── data_adapter.py              # Adattatore dati
│   ├── database.py                  # Database SQLAlchemy
│   ├── logging_production.py        # Logging produzione
│   ├── logging.py                   # Logging base
│   ├── rate_limiter.py              # Rate limiting
│   ├── rbac.py                      # Role-Based Access Control
│   ├── security.py                  # Sicurezza JWT
│   ├── structured_logging.py        # Logging strutturato
│   └── unified_config.py            # Configurazione unificata
│
├── database/
│   ├── __init__.py
│   ├── async_repository.py          # Repository asincrono
│   ├── models.py                    # Modelli database
│   ├── repository.py                # Repository
│   └── timescale_models.py          # Modelli TimescaleDB
│
├── execution/
│   ├── __init__.py
│   ├── broker_connector.py           # Connettore broker
│   ├── execution_engine.py          # Motore esecuzione
│   ├── order_manager.py              # Gestione ordini
│   └── connectors/
│       ├── __init__.py
│       ├── binance_connector.py     # Connettore Binance
│       ├── ib_connector.py          # Connettore Interactive Brokers
│       └── paper_connector.py        # Connettore Paper Trading
│
├── market_data/
│   ├── __init__.py
│   ├── data_feed.py                 # Feed dati mercato
│   └── websocket_stream.py          # Stream WebSocket
│
├── portfolio/
│   ├── __init__.py
│   ├── optimization.py              # Ottimizzazione portfolio
│   └── performance.py               # Performance portfolio
│
└── risk/
    ├── __init__.py
    ├── hardened_risk_engine.py      # Motore rischio hardened
    └── risk_engine.py               # Motore rischio base
```

### 2. `/src` - Logica Core Trading

```
src/
├── __init__.py
├── account_manager.py               # Gestione account
├── allocation.py                    # Allocazione capitale
├── async_utils.py                   # Utility asincrone
├── backtest.py                     # Backtesting
├── backtest_multi.py               # Backtesting multi-asset
├── dashboard_investor.py           # Dashboard investitore
├── dashboard_performance.py         # Dashboard performance
├── data_loader.py                  # Caricamento dati
├── database_config.py              # Configurazione database
├── database.py                     # Database base
├── database_sqlalchemy.py          # SQLAlchemy ORM
├── error_handling.py               # Gestione errori
├── execution.py                    # Esecuzione ordini
├── features.py                     # Feature engineering
├── fund_simulator.py               # Simulatore fondo
├── hedgefund_ml.py                 # ML per hedge fund
├── hmm_regime.py                   # Modello regime HMM
├── ib_wrapper.py                   # Wrapper Interactive Brokers
├── indicators.py                   # Indicatori tecnici
├── kpi.py                          # KPI trading
├── live_portfolio_manager.py       # Gestione portfolio live
├── live_trading.py                 # Trading live
├── ml_enhanced.py                  # ML avanzato
├── ml_model.py                     # Modello ML base
├── ml_model_xgb.py                 # Modello XGBoost
├── ml_tuning.py                    # Tuning modelli ML
├── multi_asset_stream.py           # Stream multi-asset
├── multi_strategy_engine.py        # Motore multi-strategia
├── performance.py                  # Performance tracking
├── performance_monitor.py          # Monitoraggio performance
├── portfolio_optimizer.py          # Ottimizzazione portfolio
├── risk.py                         # Gestione rischio base
├── risk_engine.py                  # Motore rischio
├── risk_guard.py                   # Guardie rischio
├── risk_optimizer.py               # Ottimizzazione rischio
├── risk_trailing.py                # Trailing stop rischio
├── signal_engine.py                # Motore segnali
├── trade_log.py                    # Log trading
├── trading_completo.py             # Trading completo
├── trading_ledger.py               # Libro mastro trading
├── utils.py                        # Utility generali
├── utils_cache.py                  # Utility cache
├── utils_retry.py                  # Utility retry
└── walkforward.py                  # Walk-forward analysis
```

#### `/src/agents` - Sistema Multi-Agente
```
agents/
├── __init__.py
├── agent_marketdata.py             # Agente dati mercato
├── agent_montecarlo.py             # Agente simulazione Monte Carlo
├── agent_risk.py                   # Agente calcolo rischio
├── agent_supervisor.py             # Agente supervisione
└── base_agent.py                   # Classe base agente
```

#### `/src/automl` - AutoML Engine
```
automl/
├── __init__.py
├── automl_engine.py                # Motore AutoML
├── evolution.py                    # Algoritmo genetico
└── strategy_evolution_manager.py   # Evoluzione strategie
```

#### `/src/core` - Componenti Core
```
core/
├── __init__.py
├── api_rate_manager.py             # Gestione rate API
├── capital_protecction.py          # Protezione capitale
├── dynamic_allocation.py           # Allocazione dinamica
├── dynamic_capital_allocation.py   # Allocazione capitale dinamica
├── engine.py                       # Motore trading
├── event_bus.py                    # Event bus Pub/Sub
├── resource_monitor.py             # Monitoraggio risorse
├── state_manager.py                # Gestione stato
│
├── execution/
│   ├── __init__.py
│   ├── best_execution.py          # Best execution
│   ├── broker_interface.py         # Interfaccia broker
│   ├── order_manager.py            # Gestione ordini
│   ├── orderbook_simulator.py     # Simulatore order book
│   └── tca.py                      # Transaction Cost Analysis
│
├── performance/
│   ├── __init__.py
│   ├── async_logging.py            # Logging asincrono
│   ├── db_batcher.py               # Batching database
│   ├── event_loop.py               # Event loop
│   ├── message_bus.py              # Message bus
│   ├── metrics.py                  # Metriche
│   ├── prometheus_metrics.py      # Metriche Prometheus
│   ├── ring_buffer.py              # Buffer circolare
│   └── ws_batcher.py               # Batching WebSocket
│
├── portfolio/
│   ├── __init__.py
│   └── portfolio_manager.py       # Gestione portfolio
│
└── risk/
    ├── __init__.py
    ├── fat_tail_risk.py            # Rischio fat tail
    ├── institutional_risk_engine.py # Rischio istituzionale
    ├── multiasset_cvar.py          # CVaR multi-asset
    ├── risk_engine.py               # Motore rischio
    └── volatility_models.py        # Modelli volatilità
```

#### `/src/external` - Integrazioni API Esterne
```
external/
├── __init__.py
├── api_registry.py                 # Registry API
├── bybit_client.py                 # Client Bybit
├── cloudflare_radar_client.py     # Client Cloudflare Radar
├── coinmarketcap_client.py         # Client CoinMarketCap
├── innovation_apis.py              # API innovazione
├── macro_event_apis.py             # API eventi macro
├── market_data_apis.py             # API dati mercato
├── natural_event_apis.py           # API eventi naturali
├── okx_client.py                   # Client OKX
├── sentiment_apis.py               # API sentiment
└── weather_api.py                  # API meteo
```

#### `/src/strategy` - Strategie di Trading
```
strategy/
├── __init__.py
├── base_strategy.py                # Strategia base
├── mean_reversion.py               # Mean Reversion
├── momentum.py                     # Momentum
├── montblanck.py                   # Strategia Montblanck
└── strategy_comparison.py          # Confronto strategie
```

### 3. `/frontend` - Frontend React/TypeScript

```
frontend/
├── .gitignore
├── Dockerfile
├── index.html
├── nginx.conf
├── package.json
├── package-lock.json
├── postcss.config.js
├── tailwind.config.js
├── tsconfig.json
├── vite.config.ts
│
├── public/
│   ├── cancel.html
│   ├── success.html
│   └── (static assets)
│
└── src/
    ├── App.tsx                     # App principale
    ├── main.tsx                    # Entry point
    ├── index.css                   # Stili globali
    ├── vite-env.d.ts
    │
    ├── components/
    │   ├── NewsFeed.tsx            # Feed notizie
    │   ├── layout/
    │   │   └── Layout.tsx         # Layout base
    │   └── ui/
    │       ├── DemoBadge.tsx
    │       ├── EmptyState.tsx
    │       ├── ErrorBoundary.tsx
    │       ├── LoadingSpinner.tsx
    │       ├── Skeleton.tsx
    │       └── Toast.tsx
    │
    ├── pages/
    │   ├── Dashboard.tsx           # Dashboard principale
    │   ├── Market.tsx             # Pagina mercato
    │   ├── Orders.tsx             # Gestione ordini
    │   ├── PaymentTest.tsx        # Test pagamenti
    │   └── Portfolio.tsx          # Portfolio utente
    │
    ├── services/
    │   └── api.ts                 # Servizi API
    │
    └── types/
        └── index.ts               # Tipi TypeScript
```

### 4. `/decision_engine` - Motore Decisionale

```
decision_engine/
├── __init__.py
├── core.py                        # Logica core decisionale
├── external.py                    # Integrazioni esterne
├── five_question.py               # Motore 5 domande
├── monte_carlo.py                 # Simulazione Monte Carlo
└── signals.py                     # Generazione segnali
```

### 5. `/tests` - Suite di Test

```
tests/
├── __init__.py
├── test_agents.py                 # Test agenti
├── test_all_modules.py           # Test tutti i moduli
├── test_app.py                   # Test app FastAPI
├── test_cache_routes.py           # Test route cache
├── test_decision_engine.py        # Test motore decisionale
├── test_edge_cases.py             # Test casi limite
├── test_event_bus.py              # Test event bus
├── test_evolution.py              # Test evoluzione
├── test_new_modules.py            # Test nuovi moduli
├── test_production_features.py    # Test funzionalità produzione
├── test_security.py               # Test sicurezza
├── test_strategies.py             # Test strategie
├── test_strategy_evolution.py      # Test evoluzione strategie
├── test_technical_analysis.py     # Test analisi tecnica
└── test_timescale_aggregates.py   # Test aggregazioni TimescaleDB
```

### 6. `/docs` - Documentazione

```
docs/
├── AGENTS_ARCHITECTURE.md          # Architettura agenti
├── API_DOCS.md                     # Documentazione API
├── API_REFERENCE.md               # Riferimento API
├── API_V2.md                       # API versione 2
├── APP_CONSOLIDATION_PLAN.md      # Piano consolidamento app
├── ARCHITECTURE_V2.md             # Architettura v2
├── CODE_REVIEW_REPORT.md          # Rapporto code review
├── CONSOLIDATION_STATUS_REPORT.md # Stato consolidamento
├── DATA_PYRAMID.md                # Piramide dati
├── GUIDA_ITALIANA.md              # Guida in italiano
├── GUIDA_ROUTING.md               # Guida routing
├── NEWS_FEED_IMPLEMENTATION.md   # Implementazione feed notizie
├── README.md                       # README principale
├── REFACTOR_PLAN.md               # Piano refactoring
├── SENTIMENT_ANALYSIS_IMPLEMENTATION.md
├── SYSTEM_ARCHITECTURE.md         # Architettura sistema
├── TECHNICAL_DOCUMENTATION.md     # Documentazione tecnica
└── TRADE_HISTORY_IMPLEMENTATION.md
```

### 7. `/infra` - Infrastruttura Kubernetes

```
infra/k8s/
├── deployment.yaml                 # Deployment Kubernetes
├── service.yaml                    # Service Kubernetes
├── secrets.yaml                   # Secret Kubernetes
├── configmap.yaml                 # ConfigMap Kubernetes
├── hpa.yaml                       # Horizontal Pod Autoscaler
├── storage.yaml                   # Storage persistente
└── ingress.yaml                   # Ingress controller
```

### 8. `/docker` - Configurazioni Docker

```
docker/
├── Dockerfile                      # Dockerfile principale
├── Dockerfile.stable              # Dockerfile versione stabile
├── Dockerfile.backup              # Dockerfile backup
├── Dockerfile.render              # Dockerfile Render
├── Dockerfile.render.optimized    # Dockerfile ottimizzato
├── docker-compose.yml             # Compose principale
├── docker-compose.stable.yml      # Compose versione stabile
├── docker-compose.production.yml  # Compose produzione
└── docker-compose.hedgefund.yml   # Compose hedge fund
```

---

## 🔗 Riferimenti e Dipendenze

### Dipendenze Python Principali

| Pacchetto | Versione | Scopo |
|-----------|----------|-------|
| fastapi | ^0.109.0 | Framework API |
| uvicorn | ^0.27.0 | Server ASGI |
| sqlalchemy | ^2.0.0 | ORM database |
| pydantic | ^2.5.0 | Validazione dati |
| python-jose | ^3.3.0 | JWT tokens |
| passlib | ^1.7.4 | Hashing password |
| python-multipart | ^0.0.6 | Form data |
| aioredis | ^2.0.1 | Cache Redis |
| asyncpg | ^0.29.0 | PostgreSQL async |
| psycopg2-binary | ^2.9.9 | PostgreSQL |
| pandas | ^2.1.0 | Analisi dati |
| numpy | ^1.26.0 | Calcoli numerici |
| scikit-learn | ^1.4.0 | Machine learning |
| xgboost | ^2.0.0 | Gradient boosting |
| lightgbm | ^4.1.0 | Gradient boosting |
| ccxt | ^4.0.0 | Exchange APIs |
| websockets | ^12.0 | WebSocket client |
| aiohttp | ^3.9.0 | HTTP async |
| pytest | ^7.4.0 | Testing |
| pytest-asyncio | ^0.23.0 | Testing async |
| pytest-cov | ^4.1.0 | Coverage |

### Dipendenze Frontend

| Pacchetto | Scopo |
|-----------|-------|
| react ^18.2.0 | UI Framework |
| react-dom ^18.2.0 | React DOM |
| react-router-dom ^6.x | Routing |
| axios | HTTP client |
| recharts | Grafici |
| tailwindcss | Styling |
| vite | Build tool |
| typescript | Type safety |

---

## 🧠 Logica del Sistema

### Flusso dei Dati

```
1. 📡 Data Layer
   Exchange APIs → API Registry → TimescaleDB → Redis Cache

2. 🔬 Analysis Layer
   Technical Analysis → Sentiment Engine → Correlation Matrix → ML Predictor

3. 🧠 Decision Layer
   Monte Carlo Engine → Decision Engine → Risk Check

4. ⚡ Execution Layer
   Order Manager → Smart Router → Exchange Connectors

5. 📊 Presentation Layer
   Real-time Dashboard → API Server → WebSocket Stream
```

### Motore Monte Carlo (5 Livelli)

| Livello | Nome | Descrizione |
|---------|------|-------------|
| 1 | Base | Geometric Brownian Motion |
| 2 | Conditional | Event-conditioned paths |
| 3 | Adaptive | RL from past accuracy |
| 4 | Multi-Factor | Cross-correlations, regime switching |
| 5 | Semantic | Pattern matching, black swans |

### Sistema di Gestione Rischio

- **VaR (95%, 99%)**: Value at Risk
- **CVaR**: Conditional VaR / Expected Shortfall
- **Max Drawdown**: Massimo calo dal picco
- **Sharpe Ratio**: Rendimento aggiustato per rischio
- **Sortino Ratio**: Rischio downside aggiustato
- **Volatilità**: Volatilità annualizzata

### Strategie di Trading

1. **MomentumStrategy**: Rilevamento momentum prezzi
2. **MeanReversionStrategy**: Segnali basati su Z-score
3. **MultiStrategy**: Combinazione multi-strategia

---

## 📝 Annotazioni e Documentazione

### File di Configurazione

| File | Descrizione |
|------|-------------|
| `.env` | Variabili ambiente |
| `.env.example` | Template variabili ambiente |
| `config.py` | Configurazione principale |
| `alembic.ini` | Configurazione migrazioni |
| `pytest.ini` | Configurazione pytest |
| `pyproject.toml` | Configurazione progetto Python |
| `vercel.json` | Configurazione Vercel |

### File di Build e Deployment

| File | Descrizione |
|------|-------------|
| `build_exe.py` | Build executable |
| `build_exe.bat` | Build Windows batch |
| `build_exe.ps1` | Build PowerShell |
| `Dockerfile` | Container Docker |
| `docker-compose.yml` | Compose Docker |

### File di Avvio

| File | Descrizione |
|------|-------------|
| `main.py` | Entry point principale |
| `dashboard.py` | Avvio dashboard |
| `start_ai_trading.bat` | Avvio Windows |
| `start_stable.sh` | Avvio Linux stabile |
| `start_stable.bat` | Avvio Windows stabile |

---

## 🧪 Test

### Esecuzione Test

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov=app --cov-report=html

# Run specific test file
pytest tests/test_decision_engine.py -v

# Run integration tests
pytest tests/ -v --run-integration
```

### Risultati Test

| Stato | Conteggio |
|-------|-----------|
| ✅ PASSED | 311 |
| ⏱️ Runtime | ~8 minuti |

---

## 🚀 Quick Start

### Installazione

```bash
# Clone repository
git clone https://github.com/ballales1984-wq/ai-trading-system.git
cd ai-trading-system

# Crea virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Installa dipendenze
pip install -r requirements.txt

# Configura ambiente
cp .env.example .env

# Avvio dashboard
python dashboard.py  # http://127.0.0.1:8050

# Avvio API
python -m uvicorn app.main:app --reload  # http://127.0.0.1:8000/docs
```

### Frontend

```bash
cd frontend
npm install
npm run dev  # http://127.0.0.1:5173
```

---

## 📊 Feature Matrix

### Data Ingestion

| Source | Type | Update Frequency |
|--------|------|-----------------|
| Binance | OHLCV, Order Book | Real-time WebSocket |
| CoinGecko | Prices, Market Data | 60s |
| Alpha Vantage | Technical Indicators | Daily |
| NewsAPI | Sentiment Headlines | 15min |
| Twitter/X | Social Sentiment | Real-time stream |
| GDELT | Global Events | Hourly |
| Trading Economics | Macro Indicators | Daily |

### Decision Engine Weights

- Technical Analysis: 30%
- Momentum Signals: 25%
- Cross-Asset Correlation: 20%
- Sentiment Score: 15%
- ML Prediction: 10%

---

## 📈 Performance Targets

| Metrica | Target |
|---------|--------|
| Signal Latency | < 100ms |
| Monte Carlo Paths | 1000+ per signal |
| System Uptime | 99.9% |
| API Response | < 50ms |

---

## 📅 Roadmap

### Q1 2025
- [x] TimescaleDB continuous aggregates
- [x] React frontend with Tailwind CSS
- [x] CSS variables theming system
- [ ] Live trading with real capital
- [ ] Additional exchange support (OKX, Bybit)
- [ ] Advanced order types (iceberg, TWAP, VWAP)

### Q2 2025
- [ ] Multi-strategy portfolio allocation
- [ ] Options pricing and Greeks calculation
- [ ] Cross-exchange arbitrage detection
- [ ] Dark/Light theme toggle

### Q3 2025
- [ ] Reinforcement learning agent
- [ ] Alternative data integration (satellite, credit cards)
- [ ] White paper publication

---

## 👨‍💻 Autore

**Alessio Ballini**  
Quantitative Developer | Python Engineer | AI Trading Systems

---

## 📄 Licenza

MIT License - vedi file LICENSE per dettagli.

