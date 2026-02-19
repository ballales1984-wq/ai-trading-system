# 🔄 AI Trading System - API Integration Architecture

## Complete Flow Diagram: From APIs to Trading Decisions

```mermaid
flowchart TB
    subgraph EXTERNAL_APIS["🌐 External Data Sources"]
        subgraph MARKET_DATA["📊 Market Data APIs"]
            BN[Binance API<br/>Crypto Prices]
            CG[CoinGecko API<br/>Crypto Market Data]
            AV[Alpha Vantage<br/>Stocks/Forex/Commodities]
            QD[Quandl/Nasdaq<br/>Quantitative Data]
        end
        
        subgraph SENTIMENT["📰 News & Sentiment APIs"]
            NA[NewsAPI.org<br/>General News]
            BZ[Benzinga API<br/>Financial News]
            TW[Twitter/X API<br/>Social Sentiment]
            GD[GDELT API<br/>Geopolitical Events]
        end
        
        subgraph MACRO["📅 Economic Calendar APIs"]
            TE[Trading Economics<br/>Macro Events]
            EP[EconPulse<br/>USA/Global Events]
            IC[Investing.com<br/>Global Events]
        end
        
        subgraph NATURAL["🌡️ Natural Events APIs"]
            OM[Open-Meteo<br/>Weather Data]
            CT[Climate TRACE<br/>Climate Emissions]
            WD[USGS Water Data<br/>Hydrological]
        end
        
        subgraph FUTURES["🔮 Future Indicators APIs"]
            EIA[EIA API<br/>Energy/Petrol/Gas]
            GP[Google Patents<br/>Innovation Data]
        end
    end
    
    subgraph DATA_LAYER["💾 Data Ingestion Layer"]
        NORM[Normalizer<br/>Data Standardization]
        VAL[Validator<br/>Data Quality Check]
        CACHE[(Redis Cache)]
        DB[(PostgreSQL<br/>TimeSeries DB)]
    end
    
    subgraph PROCESSING["⚙️ Processing Engine"]
        IND[Technical Indicators<br/>RSI/MACD/MA]
        SENT[Sentiment Analyzer<br/>NLP Processing]
        EVT[Event Processor<br/>Macro/Natural Link]
        FE[Feature Engineering<br/>Multi-factor]
    end
    
    subgraph MONTE_CARLO["🎲 Monte Carlo Simulation Engine"]
        MC1[MC Level 1<br/>Base Random Walk]
        MC2[MC Level 2<br/>Conditional Events]
        MC3[MC Level 3<br/>Adaptive Learning]
        MC4[MC Level 4<br/>Multi-factor]
        MC5[MC Level 5<br/>Semantic History]
    end
    
    subgraph DECISION["🧠 Decision Engine"]
        SIG[Signal Generator<br/>BUY/SELL/HOLD]
        ALLO[Portfolio Allocator<br/>Risk Parity/MeanVar]
        RISK[Risk Manager<br/>VaR/CVaR/Limits]
        EXEC[Execution Engine<br/>Order Routing]
    end
    
    subgraph OUTPUT["📤 Output Layer"]
        DASH[Dashboard<br/>Real-time Monitor]
        ALERT[Alert System<br/>Telegram/Email]
        LOG[(Trade Log<br/>Audit Trail)]
        API[REST API<br/>External Access]
    end
    
    %% Connections
    BN --> NORM
    CG --> NORM
    AV --> NORM
    QD --> NORM
    NA --> NORM
    BZ --> NORM
    TW --> NORM
    GD --> NORM
    TE --> NORM
    EP --> NORM
    IC --> NORM
    OM --> NORM
    CT --> NORM
    WD --> NORM
    EIA --> NORM
    GP --> NORM
    
    NORM --> VAL
    VAL --> CACHE
    CACHE -.->|Hot Data| DB
    VAL -->|Cold Data| DB
    
    DB --> IND
    DB --> SENT
    DB --> EVT
    DB --> FE
    
    IND --> MC1
    SENT --> MC2
    EVT --> MC2
    FE --> MC3
    
    MC1 --> MC2
    MC2 --> MC3
    MC3 --> MC4
    MC4 --> MC5
    
    MC5 --> SIG
    SIG --> ALLO
    ALLO --> RISK
    RISK --> EXEC
    
    EXEC --> DASH
    EXEC --> ALERT
    EXEC --> LOG
    EXEC --> API
```

---

## 🔄 API-to-Database Integration Schema

```mermaid
flowchart LR
    subgraph COLLECTORS["📥 Data Collectors"]
        MC[Market Collector<br/>src/data_collector.py]
        SC[Sentiment Collector<br/>sentiment_news.py]
        EC[Event Collector<br/>economic_calendar.py]
    end
    
    subgraph STORAGE["💾 Database Tables"]
        subgraph OHLCV["ohlcv_data"]
            TB1[(asset)]
            TB2[(timestamp)]
            TB3[(open)]
            TB4[(high)]
            TB5[(low)]
            TB6[(close)]
            TB7[(volume)]
        end
        
        subgraph NEWS["news_sentiment"]
            NB1[(id)]
            NB2[(timestamp)]
            NB3[(title)]
            NB4[(content)]
            NB5[(sentiment_score)]
            NB6[(source)]
        end
        
        subgraph EVENTS["market_events"]
            EC1[(id)]
            EC2[(timestamp)]
            EC3[(event_type)]
            EC4[(impact)]
            EC5[(region)]
        end
        
        subgraph FEATURES["engineered_features"]
            FB1[(id)]
            FB2[(timestamp)]
            FB3[(rsi)]
            FB4[(macd)]
            FB5[(bb_position)]
            FB6[(sentiment_weight)]
        end
    end
    
    MC --> OHLCV
    SC --> NEWS
    EC --> EVENTS
    NEWS --> FEATURES
    OHLCV --> FEATURES
    EVENTS --> FEATURES
```

---

## ⚙️ Processing Pipeline

```mermaid
flowchart TB
    subgraph INPUT["📥 Input Data"]
        PRICE[Price Data]
        NEWS[News Data]
        EVENTS[Event Data]
    end
    
    subgraph PROCESS["🔧 Processing Stages"]
        direction LR
        P1[Data Cleaning]
        P2[Feature Extraction]
        P3[Normalization]
        P4[PCA/Dimensionality Reduction]
    end
    
    subgraph ML["🤖 ML Models"]
        direction LR
        M1[Price Predictor<br/>LSTM/Transformer]
        M2[Sentiment Classifier<br/>BERT]
        M3[Risk Model<br/>XGBoost]
        M4[Portfolio Optimizer<br/>RL Agent]
    end
    
    subgraph OUTPUTS["📤 Outputs"]
        O1[Price Forecast]
        O2[Trade Signals]
        O3[Risk Metrics]
        O4[Portfolio Weights]
    end
    
    INPUT --> PROCESS
    PROCESS --> ML
    ML --> OUTPUTS
```

---

## 🎲 Monte Carlo Simulation Layers

```mermaid
flowchart LR
    subgraph LAYERS["Simulation Layers"]
        direction TB
        L1["🎲 Level 1: Base Monte Carlo<br/>- Random walk simulation<br/>- GBM model<br/>- Historical volatility"]
        L2["🎲 Level 2: Conditional MC<br/>- Event-conditioned paths<br/>- Macro event impact<br/>- Sentiment weighting"]
        L3["🎲 Level 3: Adaptive MC<br/>- Reinforcement learning<br/>- Path correction<br/>- Dynamic volatility"]
        L4["🎲 Level 4: Multi-Factor MC<br/>- Multiple factors<br/>- Cross-asset correlation<br/>- Regime switching"]
        L5["🎲 Level 5: Semantic History<br/>- Historical pattern matching<br/>- Cause-effect modeling<br/>- Black swan detection"]
    end
    
    L1 -->|Enhanced with| L2
    L2 -->|Learned from| L3
    L3 -->|Multiplied by| L4
    L4 -->|Contextualized by| L5
```

---

## 🧠 Decision Engine Architecture

```mermaid
flowchart TB
    subgraph INPUTS["📥 Inputs"]
        MC_RES[Monte Carlo Results]
        TECH[Technical Signals]
        SENT[Sentiment Scores]
        RISK[Risk Metrics]
    end
    
    subgraph DECISION["🧠 Decision Process"]
        AGG[Signal Aggregator<br/>Weighted Ensemble]
        FILT[Filter Layer<br/>Market Regime Check]
        VALID[Validator<br/>Risk Limits]
        EXEC[Execution Router<br/>Best Venue]
    end
    
    subgraph OUTPUT["📤 Decisions"]
        BUY[BUY Signal<br/>Confidence %]
        SELL[SELL Signal<br/>Confidence %]
        HOLD[HOLD Signal<br/>Reason]
    end
    
    INPUTS --> AGG
    AGG --> FILT
    FILT --> VALID
    VALID --> EXEC
    EXEC --> BUY
    EXEC --> SELL
    EXEC --> HOLD
```

---

## 📊 Complete System Architecture

```mermaid
flowchart TB
    subgraph FRONTEND["🎨 Frontend Layer"]
        DASH[Dashboard<br/>React/Vue]
        APP[Mobile App]
        API_DOC[API Documentation]
    end
    
    subgraph BACKEND["⚙️ Backend Layer"]
        subgraph API["REST API (FastAPI)"]
            ROUTE_MKT[Market Routes]
            ROUTE_ORD[Order Routes]
            ROUTE_PF[Portfolio Routes]
            ROUTE_RISK[Risk Routes]
        end
        
        subgraph CORE["Core Engine"]
            ENGINE[Trading Engine<br/>src/core/engine.py]
            EXEC[Execution Engine<br/>src/core/execution]
            RISK[Risk Engine<br/>src/core/risk]
            PORT[Portfolio Manager<br/>src/core/portfolio]
        end
        
        subgraph ML["ML Pipeline"]
            PRED[Price Predictor<br/>src/ml_predictor.py]
            OPT[Portfolio Optimizer<br/>src/portfolio_optimizer.py]
            AUTO[AutoML<br/>src/automl]
        end
    end
    
    subgraph DATA["💾 Data Layer"]
        DB[(PostgreSQL)]
        REDIS[(Redis)]
        TS[(TimescaleDB)]
        S3[(S3/Blob Storage)]
    end
    
    subgraph EXTERNAL["🌐 External Services"]
        EXCHANGES[Binance/Kraken<br/>Exchanges]
        BROKERS[Brokers<br/>Interactive Brokers]
        NOTIFY[Notifications<br/>Telegram/Slack]
    end
    
    FRONTEND --> API
    API --> CORE
    API --> ML
    CORE --> DATA
    ML --> DATA
    EXEC --> EXCHANGES
    EXEC --> BROKERS
    CORE --> NOTIFY
```

---

## 📋 API Mapping to System Components

| API Category | APIs Used | Source File | Database Table | Usage |
|--------------|-----------|-------------|----------------|-------|
| **Market Data** | Binance, CoinGecko, Alpha Vantage | [`data_collector.py`](data_collector.py) | `ohlcv_data` | Price feeds, historical data |
| **Sentiment** | NewsAPI, Benzinga, Twitter | [`sentiment_news.py`](sentiment_news.py) | `news_sentiment` | Market mood, signal weighting |
| **Economic** | Trading Economics, Investing.com | [`config.py`](config.py) | `market_events` | Event-conditioned MC |
| **Weather** | Open-Meteo, Climate TRACE | [`config.py`](config.py) | `natural_events` | Commodity correlation |
| **Energy** | EIA API | [`config.py`](config.py) | `energy_prices` | Stress testing |
| **Innovation** | Google Patents | [`config.py`](config.py) | `innovation_index` | Long-term scenarios |

---

## 🔄 Data Flow Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         EXTERNAL APIs                                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐               │
│  │ Binance  │  │ NewsAPI  │  │ Trading  │  │ Open-    │               │
│  │          │  │          │  │ Economics│  │ Meteo    │               │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘               │
│       │             │             │             │                       │
│       └─────────────┴──────┬──────┴─────────────┘                       │
│                            ▼                                            │
│                   ┌────────────────┐                                    │
│                   │  Normalizer &  │                                    │
│                   │  Validator     │                                    │
│                   └───────┬────────┘                                    │
│                           │                                              │
│                           ▼                                              │
│                   ┌────────────────┐                                    │
│                   │   PostgreSQL   │                                    │
│                   │   + Redis      │                                    │
│                   └───────┬────────┘                                    │
│                           │                                              │
│       ┌───────────────────┼───────────────────┐                       │
│       ▼                   ▼                   ▼                        │
│ ┌──────────┐       ┌──────────┐       ┌──────────┐                   │
│ │Technical │       │Sentiment │       │ Event    │                   │
│ │Indicators│       │Analyzer  │       │Processor │                   │
│ └────┬─────┘       └────┬─────┘       └────┬─────┘                   │
│       │                 │                 │                           │
│       └─────────────────┼─────────────────┘                           │
│                         ▼                                               │
│              ┌────────────────────┐                                     │
│              │ Monte Carlo Engine │                                     │
│              │  (5 Levels)        │                                     │
│              └─────────┬──────────┘                                     │
│                        ▼                                                │
│              ┌────────────────────┐                                     │
│              │ Decision Engine     │                                     │
│              │ Signals + Risk      │                                     │
│              └─────────┬──────────┘                                     │
│                        │                                                │
│                        ▼                                                │
│              ┌────────────────────┐                                     │
│              │ Execution Engine   │                                     │
│              │ + Dashboard        │                                     │
│              └────────────────────┘                                     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🔗 Key Files Reference

| Component | File Path | Description |
|-----------|-----------|-------------|
| Main Engine | [`src/core/engine.py`](src/core/engine.py) | Core trading engine |
| Data Collector | [`data_collector.py`](data_collector.py) | Market data ingestion |
| Sentiment | [`sentiment_news.py`](sentiment_news.py) | News sentiment analysis |
| ML Predictor | [`src/ml_predictor.py`](src/ml_predictor.py) | Price prediction |
| Risk Engine | [`src/risk_engine.py`](src/risk_engine.py) | Risk management |
| Execution | [`src/execution.py`](src/execution.py) | Order execution |
| Portfolio | [`src/portfolio_optimizer.py`](src/portfolio_optimizer.py) | Portfolio optimization |

---

## 🚀 Implementation Status

| Component | Status | Priority |
|-----------|--------|----------|
| ✅ Market Data APIs (Binance) | Implemented | P0 |
| ✅ Sentiment APIs | Implemented | P1 |
| ✅ Economic Calendar | Configured | P1 |
| ⚠️ Weather/Natural Events | Planned | P2 |
| ⚠️ Innovation Patents API | Planned | P3 |
| ✅ Monte Carlo (Level 1-3) | Implemented | P0 |
| 🔄 Monte Carlo (Level 4-5) | In Progress | P1 |
| ✅ Decision Engine | Implemented | P0 |

---

*Document generated for AI Trading System v2.0*
*Architecture version: 2026.02*
