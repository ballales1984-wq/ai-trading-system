# 🤖 AI Trading System - Quant Research Framework

A professional-grade quantitative trading system with machine learning, backtesting, and portfolio optimization for cryptocurrency and commodity-linked assets.

> **Status**: 🔬 Research Framework | **Level**: Hedge Fund Ready

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

### Backtesting & Risk
- **Backtest Engine**: Long/short with transaction costs & slippage
- **Multi-Asset Portfolio**: Volatility parity, risk parity allocation
- **Risk Metrics**: Sharpe, Sortino, Calmar, VaR, Max Drawdown
- **Fund Simulation**: 2% management fee + 20% performance fee (HWM)
- **Performance Reports**: Professional hedge fund format

### Dashboard
- **Interactive Charts**: Candlestick with multiple indicators
- **ML Metrics**: Accuracy, confidence, feature importance
- **Equity Curve**: vs Benchmark comparison
- **Drawdown Chart**: Real-time risk visualization
- **Portfolio Analytics**: Multi-asset performance

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
│   ├── fund_simulator.py        # Fee structure simulation
│   ├── signal_engine.py          # Signal generation
│   ├── utils.py                 # Utilities
│   └── walkforward.py            # Walk-forward optimization
│
├── dashboard/
│   └── app.py                   # Professional Dash dashboard
│
├── tests/
│   └── test_technical_analysis.py  # Unit tests
│
├── config.py                     # Configuration
├── main.py                      # CLI entry point
├── dashboard.py                 # Original dashboard
├── requirements.txt             # Dependencies
├── Dockerfile                   # Docker container
└── docker-compose.yml           # Docker orchestration
```

---

## 🚀 Quick Start

```bash
# Clone and install
git clone https://github.com/ballales1984-wq/ai-trading-system.git
cd ai-trading-system
pip install -r requirements.txt
pip install xgboost  # For advanced ML

# Start dashboard
python main.py --mode dashboard
```

---

## 💻 Usage Examples

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

### Fund Simulation
```python
from src.fund_simulator import FundSimulator

fund = FundSimulator(initial_capital=10_000_000)
adjusted, metrics = fund.apply_fees(equity_curve)
# Shows: gross/net return, fees, final AUM
```

---

## 📊 Supported Assets

### Cryptocurrencies
BTC, ETH, XRP, SOL, ADA, DOT, AVAX, MATIC, BNB, DOGE, LINK, ATOM, UNI, LTC, NEAR, APT, ARB, OP, INJ, SUI, SEI, TIA

### Commodity Tokens
PAXG (Gold), XAUT (Gold), STETH, FXS (Frax)

---

## 🎓 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DASHBOARD (Plotly)                      │
├─────────────────────────────────────────────────────────────┤
│  ML Models  │  Signal Engine  │  Risk Metrics  │  Fund   │
├─────────────────────────────────────────────────────────────┤
│  RandomForest  │  XGBoost  │  Ensemble  │  Walk-Forward    │
├─────────────────────────────────────────────────────────────┤
│  Backtest Engine  │  Multi-Asset Portfolio  │  Performance │
├─────────────────────────────────────────────────────────────┤
│     Indicators      │    Data Loader    │   Binance API    │
└─────────────────────────────────────────────────────────────┘
```

---

## 🧪 Testing

```bash
# Run tests
python -m pytest tests/ -v

# Run specific test
python -m pytest tests/test_technical_analysis.py -v
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

This is a research framework for educational purposes. Do not use with real capital without proper backtesting, live paper trading, and risk management.

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

## 📝 License

MIT License

---

**Level**: Quant Research Framework  
**Ready for**: Backtesting, Strategy Development, Portfolio Optimization  
**Next**: Live trading with paper money first!
