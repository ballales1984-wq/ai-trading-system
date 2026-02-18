# 🤖 AI Trading System - Quantum Quant Framework

A professional-grade quantitative trading system with machine learning, live trading, risk management, and portfolio optimization for cryptocurrency and commodity-linked assets.

> **Status**: 🚀 Production Ready v2.0 | **Level**: Hedge Fund Ready

---

## 🎯 Features

### Core Trading
- **Data Collection**: Real-time crypto prices from Binance API (28+ trading pairs)
- **Technical Analysis**: RSI, EMA, SMA, Bollinger Bands, MACD, VWAP, Stochastic, ATR, ADX
- **Cross-Market Analysis**: Correlations between crypto and commodity assets
- **News/Sentiment**: Market sentiment analysis with Fear & Greed index
- **Decision Engine**: Probabilistic trading signals with risk management

### Machine Learning (Quant Level)
- **Random Forest Classifier**: Supervised ML for signal generation
- **XGBoost**: Advanced gradient boosting for prediction
- **Feature Engineering**: Returns, volatility, momentum, regime detection
- **Walk-Forward Validation**: Proper time-series cross-validation
- **Ensemble Models**: Combine RF + XGBoost for robust signals

### Live Trading
- **Real-time WebSocket**: Multi-asset streaming from Binance
- **Paper Trading**: Safe simulation mode
- **Binance Futures Testnet**: Real order execution (test money)
- **ML Ensemble Live**: Real-time prediction in production

### Risk Management (Professional)
- **Dynamic Stop Loss**: ATR-based, adapts to volatility
- **Dynamic Take Profit**: ATR-based risk/reward
- **Trailing Stop**: Intelligent follow-with profit
- **Max Drawdown Protection**: Kill-switch at configurable threshold
- **Portfolio Risk Monitoring**: Real-time exposure tracking

### Notifications
- **Telegram Bot**: Real-time alerts for:
  - Trading signals
  - Trade executions
  - Portfolio updates
  - Risk events
  - System errors

### Backtesting & Portfolio
- **Backtest Engine**: Long/short with transaction costs & slippage
- **Multi-Asset Portfolio**: Volatility parity, risk parity, momentum allocation
- **Risk Metrics**: Sharpe, Sortino, Calmar, VaR, Max Drawdown
- **Fund Simulation**: 2% management fee + 20% performance fee (HWM)
- **Performance Reports**: Professional hedge fund format

### Dashboard
- **Interactive Charts**: Candlestick with multiple indicators
- **ML Metrics**: Accuracy, confidence, feature importance
- **Equity Curve**: vs Benchmark comparison
- **Drawdown Chart**: Real-time risk visualization
- **Portfolio Analytics**: Multi-asset performance
- **Auto-Trading Panel**: Configure and run live trading
- **Commodities Panel**: Gold, silver, oil analysis

### Advanced Trading (Quant Level)
- **HFT Simulator**: Tick-by-tick simulation with orderbook, latency, slippage
- **Multi-Agent Market**: Market makers, takers, arbitrageurs simulation
- **RL Training Environment**: Gym-compatible environment for reinforcement learning
- **AutoML Engine**: Genetic algorithm for strategy evolution
- **Hyperparameter Optimization**: Bayesian-style parameter tuning
- **Strategy Genome**: Genetic representation of trading strategies

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    DASHBOARD (Plotly/Dash)                │
├─────────────────────────────────────────────────────────────┤
│ ML Models │ Signal Engine │ Risk Metrics │ Telegram       │
├─────────────────────────────────────────────────────────────┤
│ RandomForest │ XGBoost │ Ensemble │ Walk-Forward          │
├─────────────────────────────────────────────────────────────┤
│ Live Trading │ Risk Engine │ Portfolio │ Testnet          │
├─────────────────────────────────────────────────────────────┤
│ Backtest Engine │ Multi-Asset Portfolio │ Performance     │
├─────────────────────────────────────────────────────────────┤
│ Indicators │ Data Loader │ Binance API │ Sentiment       │
├─────────────────────────────────────────────────────────────┤
│ HFT Simulator │ Multi-Agent │ RL Env │ AutoML            │
└─────────────────────────────────────────────────────────────┘
```

### Core v2.0 Architecture (Event-Driven)

```
                           ┌──────────────┐
                           │  Dashboard   │
                           │ (Plotly/Dash)│
                           └─────┬────────┘
                                 │
                                 ▼
                    ┌───────────────────────────┐
                    │      State Manager        │
                    │ (SQLite persistence)     │
                    └─────┬───────────┬────────┘
                          │           │
          ┌───────────────┘           └───────────────┐
          ▼                                         ▼
 ┌───────────────────┐                       ┌───────────────────┐
 │   Risk Engine     │                       │    Event Bus      │
 │ - Max Drawdown    │                       │ - Pub/Sub events │
 │ - Position Limits │                       │ - Signal handling│
 │ - Emergency Stop  │                       └────────┬──────────┘
 └────────┬──────────┘                                │
          │                                           ▼
          ▼                                 ┌─────────────────────┐
 ┌───────────────────┐                     │ Order Manager       │
 │ Portfolio Manager │                     │ - Retry logic       │
 │ - Multi-asset     │                     │ - Risk validation   │
 │ - Position sizing │                     └────────┬──────────┘
 └────────┬──────────┘                              │
          │                                          ▼
          ▼                               ┌─────────────────────┐
 ┌───────────────────┐                  │ Broker Interface    │
 │   Trading Engine  │                  │ - Paper Trading    │
 │   Orchestrator    │                  │ - Live (Binance)   │
 └───────────────────┘                  └─────────────────────┘
```

---

## 📂 Project Structure

```
ai-trading-system/
│
├── src/                          # Quant modules
│   ├── __init__.py              # Package exports
│   ├── backtest.py              # Backtesting engine
│   ├── backtest_multi.py        # Multi-asset portfolio
│   ├── data_loader.py           # Data loading (CSV/API)
│   ├── indicators.py             # Technical indicators
│   ├── ml_model.py              # Random Forest signals
│   ├── ml_model_xgb.py          # XGBoost signals
│   ├── performance.py           # Hedge fund metrics
│   ├── risk.py                  # Risk analysis
│   ├── risk_engine.py           # Live risk management
│   ├── fund_simulator.py        # Fee structure simulation
│   ├── signal_engine.py         # Signal generation
│   ├── utils.py                 # Utilities
│   ├── walkforward.py           # Walk-forward optimization
│   │
│   ├── core/                   # NEW: Production Core v2.0
│   │   ├── __init__.py         # Core exports
│   │   ├── event_bus.py        # Event-driven pub/sub
│   │   ├── state_manager.py    # SQLite persistence
│   │   ├── engine.py           # Main orchestrator
│   │   ├── portfolio/
│   │   │   └── portfolio_manager.py  # Multi-asset portfolio
│   │   ├── risk/
│   │   │   └── risk_engine.py  # Professional risk
│   │   └── execution/
│   │       ├── broker_interface.py  # Paper/Live broker
│   │       └── order_manager.py     # Order execution + retry
│   │
│   ├── live/                   # Live trading modules
│   │   ├── binance_multi_ws.py
│   │   ├── portfolio_live.py
│   │   ├── position_sizing.py
│   │   ├── telegram_notifier.py
│   │   └── risk_engine.py
│   │
│   ├── models/                 # ML models
│   ├── hft/                   # HFT modules
│   ├── automl/                # AutoML
│   ├── meta/                  # Meta-evolution
│   ├── simulations/           # Market simulation
│   └── quant/                 # Quantitative strategies
│
├── dashboard/
│   └── app.py                 # Professional Dash dashboard
│
├── tests/
│   └── test_technical_analysis.py
│
├── config.py                   # Configuration
├── main.py                    # CLI entry point
├── dashboard.py               # Dashboard app
├── live_multi_asset.py        # Live trading system
├── auto_trader.py             # Auto trading
├── test_core.py              # Core module tests
├── test_paper_trading.py      # Paper trading validation (Phase 1)
├── test_phase2.py             # Testnet integration (Phase 2)
├── run_live.py                # Live trading entry point
├── ARCHITECTURE.md            # Architecture documentation
├── requirements.txt           # Dependencies
├── Dockerfile                 # Docker container
└── docker-compose.yml         # Docker orchestration
```

---

## 🚀 Quick Start

```bash
# Clone and install
git clone https://github.com/ballales1984-wq/ai-trading-system.git
cd ai-trading-system
pip install -r requirements.txt
pip install xgboost websocket-client

# Test core modules (new v2.0)
python test_core.py

# Start dashboard
python main.py --mode dashboard
```

---

## 💻 Usage Examples

### Core v2.0 - Paper Trading

```python
from src.core import (
    TradingEngine, PaperBroker, RiskEngine, 
    PortfolioManager, create_broker
)

# Create broker
broker = PaperBroker(initial_balance=100000)
await broker.connect()

# Create risk engine
risk = RiskEngine(
    initial_balance=100000,
    limits=RiskLimits(
        max_position_pct=0.3,
        max_daily_loss_pct=0.05,
        max_drawdown_pct=0.20
    )
)

# Create portfolio
portfolio = PortfolioManager(initial_balance=100000)

# Open position
position = portfolio.open_position("BTCUSDT", "long", 0.5, 45000)
print(f"Opened: {position.symbol} {position.quantity}")

# Update prices
portfolio.update_prices({"BTCUSDT": 46000})
print(f"PnL: ${portfolio.get_metrics().unrealized_pnl:.2f}")
```

### ML Signal Generation

```python
from src.ml_model import MLSignalModel
from src.indicators import calculate_all_indicators

# Prepare data with indicators
df = calculate_all_indicators(price_data)

# Train ML model
model = MLSignalModel('random_forest')
metrics = model.train(df)

# Generate signals
signals = model.predict_signals(df)
```

### XGBoost Model

```python
from src.ml_model_xgb import XGBSignalModel

model = XGBSignalModel(n_estimators=300, max_depth=6)
model.fit(df)
signals = model.predict_signals(df)
top_features = model.get_top_features(10)
```

### Live Trading with Telegram

```bash
# Start live trading with notifications
python main.py --mode live \
    --assets BTCUSDT,ETHUSDT,SOLUSDT \
    --telegram-token "YOUR_BOT_TOKEN" \
    --telegram-chat-id "YOUR_CHAT_ID"
```

---

## 📊 Supported Assets

### Cryptocurrencies
BTC, ETH, XRP, SOL, ADA, DOT, AVAX, MATIC, BNB, DOGE, LINK, ATOM, UNI, LTC, NEAR, APT, ARB, OP, INJ, SUI, SEI, TIA

### Commodity Tokens
PAXG (Gold), XAUT (Gold), STETH, FXS (Frax)

---

## 🧪 Testing

```bash
# Run core tests (v2.0)
python test_core.py

# Run Paper Trading validation tests (Phase 1)
python test_paper_trading.py

# Run Binance Testnet integration tests (Phase 2)
python test_phase2.py

# Run all tests
python -m pytest tests/ -v

# Quick check
python -m pytest tests/test_app.py -q
```

---

## 🐳 Docker

```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f
```

---

## ⚠️ Risk Warning

This is a research framework for educational purposes. Always use paper trading first, then small amounts on testnet. Do not use with real capital without proper backtesting and risk management.

---

## 📈 Performance Metrics Available

| Metric | Description |
|--------|-------------|
| Annual Return | Compound annual growth rate |
| Sharpe Ratio | Risk-adjusted return |
| Sortino Ratio | Downside risk-adjusted |
| Calmar Ratio | Return / Max Drawdown |
| Max Drawdown | Largest peak-to-trough |
| VaR 95% | Value at Risk (95% confidence) |
| Win Rate | Percentage of profitable trades |
| Profit Factor | Gross gains / Gross losses |

---

## 🔧 Configuration

### Environment Variables (.env)

```bash
# Binance API (optional for live trading)
BINANCE_API_KEY=your_api_key
BINANCE_SECRET=your_secret

# Telegram (optional for notifications)
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_CHAT_ID=your_chat_id
```

### Risk Parameters

```python
# In live_multi_asset.py or via CLI
risk_engine = RiskEngine(
    max_drawdown=0.20,       # Kill-switch at 20%
    sl_multiplier=2.0,       # Stop loss = 2x ATR
    tp_multiplier=3.0,       # Take profit = 3x ATR
    trailing_multiplier=1.5  # Trailing = 1.5x ATR
)
```

---

## 📝 License

MIT License

---

## 🎓 Architecture (Detailed)

See [ARCHITECTURE.md](ARCHITECTURE.md) for complete architecture documentation including:

- Complete Data Flow Diagram
- Core System Architecture (v2.0)
- Event-Driven Flow
- HFT & Multi-Agent Simulation
- Dashboard Architecture
- Complete System Overview

---

**Level**: Production Ready v2.0  
**Ready for**: Live Trading, Backtesting, Portfolio Management, SaaS  
**Safe Mode**: Paper Trading & Testnet Enabled
