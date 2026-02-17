# 🤖 AI Trading System - Quantum Quant Framework

A professional-grade quantitative trading system with machine learning, live trading, risk management, and portfolio optimization for cryptocurrency and commodity-linked assets.

> **Status**: 🚀 Production Ready | **Level**: Hedge Fund Ready

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
│   ├── performance.py            # Hedge fund metrics
│   ├── risk.py                  # Risk analysis
│   ├── risk_engine.py           # Live risk management
│   ├── fund_simulator.py        # Fee structure simulation
│   ├── signal_engine.py          # Signal generation
│   ├── utils.py                 # Utilities
│   ├── walkforward.py            # Walk-forward optimization
│   │
│   ├── live/                    # Live trading modules
│   │   ├── __init__.py
│   │   ├── binance_multi_ws.py  # WebSocket streaming
│   │   ├── portfolio_live.py    # Live portfolio
│   │   ├── position_sizing.py   # Dynamic sizing
│   │   ├── telegram_notifier.py # Telegram alerts
│   │   └── risk_engine.py       # Advanced risk management
│   │
│   ├── models/                  # ML models
│   │   ├── __init__.py
│   │   └── ensemble.py          # Ensemble model
│   │
│   │   ├── hft/                  # HFT modules
│   │   │   ├── hft_simulator.py  # Tick-by-tick simulator
│   │   │   └── hft_env.py        # RL training environment
│   │
│   │   ├── automl/               # AutoML
│   │   │   └── automl_engine.py  # Strategy evolution
│   │
│   │   ├── meta/                 # Meta-evolution
│   │   │   ├── meta_evolution_engine.py  # Hybrid agent evolution
│   │   │   ├── multi_market_evolution.py  # Multi-market migration
│   │   │   └── emergent_communication.py  # Agent communication
│   │
│   │   ├── simulations/          # Market simulation
│   │   │   └── multi_agent_market.py  # Multi-agent market
│   │   │
│   │   └── quant/│   └── quant/                  # Quantitative strategies
│
├── dashboard/
│   └── app.py                   # Professional Dash dashboard
│
├── tests/
│   ├── __init__.py
│   ├── test_technical_analysis.py
│   └── test_app.py              # Comprehensive tests
│
├── config.py                     # Configuration
├── main.py                       # CLI entry point
├── dashboard.py                   # Dashboard app
├── live_multi_asset.py           # Live trading system
├── auto_trader.py                # Auto trading
├── requirements.txt               # Dependencies
├── Dockerfile                     # Docker container
└── docker-compose.yml            # Docker orchestration
```

---

## 🚀 Quick Start

```bash
# Clone and install
git clone https://github.com/ballales1984-wq/ai-trading-system.git
cd ai-trading-system
pip install -r requirements.txt
pip install xgboost  # For advanced ML
pip install websocket-client  # For live trading

# Start dashboard
python main.py --mode dashboard
```

---

## 💻 Usage Examples

### Live Trading with Telegram
```bash
# Start live trading with notifications
python main.py --mode live \
    --assets BTCUSDT,ETHUSDT,SOLUSDT \
    --telegram-token "YOUR_BOT_TOKEN" \
    --telegram-chat-id "YOUR_CHAT_ID"
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

### Risk Engine (Live Trading)
```python
from src.live.risk_engine import RiskEngine

risk = RiskEngine(
    max_drawdown=0.20,      # 20% kill-switch
    sl_multiplier=2.0,      # ATR x 2 for SL
    tp_multiplier=3.0,      # ATR x 3 for TP
    trailing_multiplier=1.5  # ATR x 1.5 for trailing
)

# Check exits
exit_signal = risk.check_exit_signal(asset, current_price, atr)
if exit_signal:
    close_position(asset)
```

### Backtest
```python
from src.backtest import run_backtest
from src.performance import generate_performance_report

result = run_backtest(df, signals, initial_capital=10000)
print(generate_performance_report(result.strategy_returns, result.equity_curve))
```

### Multi-Asset Portfolio
```python
from src.backtest_multi import MultiAssetBacktest

backtest = MultiAssetBacktest(initial_capital=1_000_000)
backtest.add_asset('BTC', btc_prices, btc_signals)
backtest.add_asset('ETH', eth_prices, eth_signals)

returns, metrics = backtest.run_backtest('volatility_parity')
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
# Run all tests
python -m pytest tests/ -v

# Run specific test
python -m pytest tests/test_app.py -v

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
    max_drawdown=0.20,      # Kill-switch at 20%
    sl_multiplier=2.0,      # Stop loss = 2x ATR
    tp_multiplier=3.0,      # Take profit = 3x ATR
    trailing_multiplier=1.5  # Trailing = 1.5x ATR
)
```

---

## 📝 License

MIT License

---

## 🎓 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DASHBOARD (Plotly)                      │
├─────────────────────────────────────────────────────────────┤
│  ML Models  │  Signal Engine  │  Risk Metrics  │  Telegram │
├─────────────────────────────────────────────────────────────┤
│  RandomForest  │  XGBoost  │  Ensemble  │  Walk-Forward    │
├─────────────────────────────────────────────────────────────┤
│  Live Trading  │  Risk Engine  │  Portfolio  │  Testnet     │
├─────────────────────────────────────────────────────────────┤
│  Backtest Engine  │  Multi-Asset Portfolio  │  Performance │
├─────────────────────────────────────────────────────────────┤
│     Indicators      │    Data Loader    │   Binance API    │
└─────────────────────────────────────────────────────────────┘
```

---

**Level**: Production Ready  
**Ready for**: Live Trading, Backtesting, Portfolio Management  
**Safe Mode**: Paper Trading & Testnet Enabled
