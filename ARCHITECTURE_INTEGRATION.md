# AI Trading System - Architecture Diagram

```
+-----------------------------------------------------------------------------------------------------+
|                                    EXTERNAL DATA SOURCES                                      |
+-----------------------------------------------------------------------------------------------------+

  +---------------------+  +---------------------+  +---------------------+
  |   PRICE & MARKET    |  |    NEWS & SENTIMENT |  |   ECONOMIC CALENDAR |
  |        APIs         |  |        APIs          |  |        APIs         |
  +---------------------+  +---------------------+  +---------------------+
  | - Binance           |  | - NewsAPI.org       |  | - Trading Economics |
  | - Bybit (NEW)       |  | - Benzinga          |  | - EconPulse         |
  | - OKX (NEW)         |  | - Twitter/X         |  | - Investing.com     |
  | - CoinGecko         |  | - GDELT             |  | - ForexFactory      |
  | - Alpha Vantage     |  | - CryptoPanic       |  |                     |
  +----------+----------+  +----------+----------+  +----------+----------+
             |                          |                          |
  +---------------------+  +---------------------+  +---------------------+
  |  NATURAL EVENTS     |  |   GEOPOLITICS      |  |  SPECIAL INDICATORS |
  |        APIs         |  |        APIs         |  |        APIs         |
  +---------------------+  +---------------------+  +---------------------+
  | - Open-Meteo        |  | - EventKG           |  | - EIA API           |
  | - Climate TRACE     |  | - Wikidata Events   |  | - Google Patents    |
  | - USGS Water        |  | - GDELT             |  | - Lens.org          |
  | - Copernicus        |  |                     |  |                     |
  +----------+----------+  +----------+----------+  +----------+----------+
             |                          |                          |
-------------+--------------------------+--------------------------+----------------------------
             |                          |                          |
             +--------------------------+--------------------------+
                                         |
                                         V
+-----------------------------------------------------------------------------------------------------+
|                                    DATA NORMALIZATION LAYER                                  |
+-----------------------------------------------------------------------------------------------------+
   +-----------------------------------------------------------------------------+
   |                          DATA NORMALIZER                                    |
   |  +---------+  +---------+  +---------+  +---------+  +---------+           |
   |  | Prices  |  | Events  |  | News    |  | Climate |  | Market  |           |
   |  | Parser  |  | Parser  |  | Parser  |  | Parser  |  | Parser  |           |
   |  +----+----+  +----+----+  +----+----+  +----+----+  +----+----+         |
   |       |            |            |            |            |              |
   |       +------------+-------------+------------+------------+              |
   |                                |                                         |
   |                                V                                         |
   |                    +---------------------+                              |
   |                    |  Unified Data Model |                              |
   |                    +----------+----------+                              |
   +-----------------------------+--------------------------------------------+
                                 |
                                 V
+-----------------------------------------------------------------------------------------------------+
|                                    CENTRAL DATABASE                                          |
+-----------------------------------------------------------------------------------------------------+
   +-----------------------------------------------------------------------------+
   |                              TABLES                                         |
   |  +----------------+  +----------------+  +----------------+              |
   |  | prices         |  | events_macro   |  | news           |              |
   |  | - asset        |  | - date         |  | - date         |              |
   |  | - timestamp    |  | - type         |  | - title        |              |
   |  | - open/high/   |  | - impact       |  | - sentiment    |              |
   |  |   low/close    |  | - currency     |  | - source       |              |
   |  | - volume       |  |                |  | - assets       |              |
   |  +----------------+  +----------------+  +----------------+              |
   |                                                                            |
   |  +----------------+  +----------------+  +----------------+              |
   |  | events_natural |  | innovations    |  | semantic_vars  |              |
   |  | - date         |  | - date         |  | - date         |              |
   |  | - type         |  | - type         |  | - event_id     |              |
   |  | - intensity    |  | - impact       |  | - price_impact |              |
   |  | - region       |  | - sector       |  | - confidence   |              |
   |  +----------------+  +----------------+  +----------------+              |
   +-----------------------------------------------------------------------------+
                                         |
                                         V
+-----------------------------------------------------------------------------------------------------+
|                                    TRADING ENGINE CORE                                       |
+-----------------------------------------------------------------------------------------------------+
   +-----------------------------------------------------------------------------+
   |                          ANALYSIS ENGINE                                   |
   |                                                                            |
   |  +---------------------+  +---------------------+                        |
   |  |  TECHNICAL ANALYSIS  |  |   SENTIMENT ENGINE |                        |
   |  +---------------------+  +---------------------+                        |
   |  | - Moving Averages   |  | - News Scoring      |                        |
   |  | - RSI, MACD, ATR    |  | - Social Sentiment  |                        |
   |  | - Bollinger Bands   |  | - Macro Context     |                        |
   |  | - Volume Profile    |  | - Source Weights    |                        |
   |  +----------+----------+  +----------+----------+                        |
   |             |                        |                                   |
   |             +-------------+----------+                                   |
   |                         |                                                 |
   |                         V                                                 |
   |  +-------------------------------------------------------------------------+
   |  |                    SIGNAL GENERATOR                              |
   |  |  - Multi-factor combination                                     |
   |  |  - ML-based weighting (Reinforcement Learning)               |
   |  |  - Confidence scoring                                           |
   |  +-------------------------------+---------------------------------+
   |                                  |                                   |
   +----------------------------------+------------------------------------+
                                 |
                                 V
+-----------------------------------------------------------------------------------------------------+
|                               MONTE CARLO SIMULATION LAYER                                   |
+-----------------------------------------------------------------------------------------------------+

   LEVEL 1: Basic Monte Carlo
   +------------------------------------------------------------------------+
   |  - Random price paths (Geometric Brownian Motion)                     |
   |  - Historical volatility                                             |
   +------------------------------------------------------------------------+
                                      |
                                      V

   LEVEL 2: Conditional Monte Carlo
   +------------------------------------------------------------------------+
   |  - Incorporates MACRO EVENTS (rate changes, GDP, employment)          |
   |  - Historical event impact analysis                                  |
   +------------------------------------------------------------------------+
                                      |
                                      V

   LEVEL 3: Adaptive Monte Carlo
   +------------------------------------------------------------------------+
   |  - Real-time SENTIMENT adjustment                                     |
   |  - News-driven volatility modulation                                  |
   |  - Source reliability weighting                                       |
   +------------------------------------------------------------------------+
                                      |
                                      V

   LEVEL 4: Multi-Factor Monte Carlo
   +------------------------------------------------------------------------+
   |  - NATURAL EVENTS inclusion (weather, climate)                        |
   |  - Seasonal patterns                                                 |
   |  - Commodities correlation                                           |
   +------------------------------------------------------------------------+
                                      |
                                      V

   LEVEL 5: Semantic History Monte Carlo
   +------------------------------------------------------------------------+
   |  - GEOPOLITICAL events (wars, treaties, elections)                  |
   |  - INNOVATION cycles (patents, tech breakthroughs)                   |
   |  - Causal inference modeling                                         |
   |  - Stress testing with rare events                                   |
   +------------------------------------------------------------------------+

                                         |
                                         V
+-----------------------------------------------------------------------------------------------------+
|                                    DECISION & EXECUTION LAYER                               |
+-----------------------------------------------------------------------------------------------------+
   +-----------------------------------------------------------------------------+
   |                        DECISION ENGINE                                     |
   |                                                                            |
   |   Inputs:                              Outputs:                           |
   |   -------                              -------                            |
   |   - Signal Strength -----------------> BUY / SELL / HOLD                 |
   |   - Monte Carlo Probabilities --------> Position Size                    |
   |   - Risk Metrics ---------------------> Stop Loss Level                  |
   |   - Portfolio Constraints ------------> Take Profit Level               |
   |   - Execution Cost -------------------> Execution Strategy              |
   |                                                                            |
   +-----------------------------------------------------------------------------+
                                         |
                                         V
   +-----------------------------------------------------------------------------+
   |                      EXECUTION ENGINES                                   |
   |                                                                            |
   |  +-----------------+  +-----------------+  +-----------------+              |
   |  |   BINANCE       |  |    BYBIT        |  |     OKX         |              |
   |  |   Connector    |  |   Connector     |  |   Connector     |              |
   |  |   (existing)    |  |   (NEW)         |  |   (NEW)         |              |
   |  +-----------------+  +-----------------+  +-----------------+              |
   |                                                                            |
   |  +-----------------+  +-----------------+  +-----------------+              |
   |  |  Order Manager  |  |  Risk Engine    |  |  Portfolio      |              |
   |  |  - Order Types  |  |  - VaR          |  |  Manager        |              |
   |  |  - Execution    |  |  - Limits       |  |  - Allocation   |              |
   |  |  - TCA          |  |  - Drawdown     |  |  - Rebalance    |              |
   |  +-----------------+  +-----------------+  +-----------------+              |
   +-----------------------------------------------------------------------------+
                                         |
                                         V
+-----------------------------------------------------------------------------------------------------+
|                                    OUTPUT & MONITORING                                      |
+-----------------------------------------------------------------------------------------------------+
   +-----------------+  +-----------------+  +-----------------+  +-----------------+
   |    DASHBOARD    |  |   TELEGRAM     |  |    LOGS         |  |   BACKTEST     |
   |   (Real-time)   |  |   Alerts       |  |   (All trades) |  |   Reports      |
   +-----------------+  +-----------------+  +-----------------+  +-----------------+
   | - Portfolio     |  | - Signals       |  | - Decisions     |  | - Performance  |
   | - P&L           |  | - Risk Alerts   |  | - Executions    |  | - Sharpe       |
   | - Positions     |  | - Errors        |  | - Errors        |  | - Drawdown     |
   | - Charts        |  |                 |  |                 |  | - Win Rate     |
   +-----------------+  +-----------------+  +-----------------+  +-----------------+
```

## Data Flow Summary

1. EXTERNAL APIs -> Data Normalizer -> CENTRAL DATABASE
   
2. DATABASE -> Analysis Engine -> Signals + Weights

3. SIGNALS + Monte Carlo Simulations -> Decision Engine -> Trading Actions

4. TRADING ACTIONS -> Exchange Connectors -> Market Execution

5. EXECUTION RESULTS -> Dashboard + Alerts + Logs

## Key Components

| Component | Purpose | Files |
|-----------|---------|-------|
| External APIs | Data ingestion from exchanges and sources | src/external/*.py |
| Data Normalizer | Standardize data format | src/data_loader.py |
| Technical Analysis | Calculate indicators | technical_analysis.py |
| Sentiment Engine | News/social scoring | sentiment_news.py |
| Signal Generator | Combine factors | src/signal_engine.py |
| Monte Carlo | Price simulations | src/performance.py |
| Decision Engine | Trading decisions | decision_engine.py |
| Execution | Order placement | src/execution.py, src/external/ |
| Risk Engine | Risk management | src/risk_engine.py |
| Portfolio | Position management | src/portfolio_optimizer.py |
| Dashboard | Visualization | dashboard.py, dashboard_api.py |

## API Integration Status

| API Type | Exchange/Source | Status |
|----------|-----------------|--------|
| Prices | Binance | Existing |
| Prices | Bybit | NEW |
| Prices | OKX | NEW |
| Prices | CoinGecko | Planned |
| News | NewsAPI | Planned |
| News | Twitter/X | Planned |
| Macro | Trading Economics | Planned |
| Climate | Open-Meteo | Planned |
| Patents | Google Patents | Planned |

