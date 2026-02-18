# 🚀 Quantum AI Trading System

**Advanced Quantitative Trading Framework for Crypto & Commodities**

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)

---

## 🎯 Overview

A professional-grade quantitative trading system with institutional risk management, machine learning signals, and multi-asset portfolio optimization.

### Key Features

- **Event-Driven Architecture** - Modern, scalable design
- **ML Signal Engine** - RandomForest + XGBoost ensemble
- **Institutional Risk Engine** - VaR, CVaR, Monte Carlo
- **Advanced Volatility Models** - GARCH, EGARCH, GJR-GARCH
- **Portfolio Optimization** - CVaR, Risk Parity, Mean-Variance
- **Live Trading** - Binance Testnet integration
- **Interactive Dashboard** - Real-time monitoring

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TRADING SYSTEM                           │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐   │
│  │   Signals   │→ │    Risk     │→ │  Portfolio      │   │
│  │   Engine   │  │   Engine    │  │  Manager        │   │
│  └─────────────┘  └─────────────┘  └─────────────────┘   │
│         ↓                ↓                ↓                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐   │
│  │   ML        │  │   VaR      │  │  Order          │   │
│  │   Models    │  │   CVaR     │  │  Execution      │   │
│  └─────────────┘  └─────────────┘  └─────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
ai-trading-system/
│
├── src/
│   ├── core/
│   │   ├── engine.py              # Trading engine orchestrator
│   │   ├── event_bus.py           # Event-driven messaging
│   │   ├── state_manager.py       # SQLite persistence
│   │   ├── portfolio/             # Portfolio management
│   │   ├── execution/             # Order execution
│   │   └── risk/
│   │       ├── institutional_risk_engine.py  # VaR/CVaR/Monte Carlo
│   │       ├── volatility_models.py          # GARCH/EGARCH
│   │       ├── fat_tail_risk.py              # Student-t/EVT
│   │       └── multiasset_cvar.py            # Portfolio optimization
│   │
│   ├── live/                      # Live trading modules
│   ├── strategy/                   # Trading strategies
│   ├── ml_model/                  # ML models
│   └── automl/                    # AutoML
│
├── tests/                         # Test suite
├── dashboard/                    # Dash dashboard
├── config.py                      # Configuration
├── main.py                       # Entry point
└── requirements.txt               # Dependencies
```

---

## 🧠 Machine Learning

### Signal Generation

- **Random Forest** - Ensemble tree-based signals
- **XGBoost** - Gradient boosting signals
- **Feature Engineering** - Technical indicators, sentiment, on-chain
- **Walk-Forward Validation** - Out-of-sample testing

### Training

```python
from src.ml_model import EnsembleModel

model = EnsembleModel(n_estimators=100)
model.fit(X_train, y_train)
signals = model.predict(X_test)
```

---

## ⚠️ Risk Management

### Institutional-Grade Features

| Module | Description |
|--------|-------------|
| **VaR** | Value at Risk (Historical, Parametric, Monte Carlo) |
| **CVaR** | Conditional Expected Shortfall |
| **GARCH** | Conditional volatility modeling |
| **Fat-Tail** | Student-t distribution, EVT |
| **Stress Test** | Custom market crash scenarios |
| **Risk Parity** | Equal risk contribution allocation |

### Usage

```python
from src.core.risk import InstitutionalRiskEngine

risk = InstitutionalRiskEngine(confidence=0.95)
report = risk.full_risk_report(returns)

print(f"VaR 95%: {report['historical_var']:.2%}")
print(f"CVaR: {report['expected_shortfall']:.2%}")
print(f"Monte Carlo 5%: {report['monte_carlo']['p5']:.2%}")
```

---

## 📊 Dashboard

Real-time monitoring with Dash/Plotly:

- Portfolio positions
- P&L tracking
- Risk metrics
- Signal visualization
- Monte Carlo distributions

Run dashboard:
```bash
python main.py --mode dashboard
```

Access at: `http://localhost:8050`

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure

Copy `.env.example` to `.env` and add your API keys:

```env
BINANCE_API_KEY=your_key
BINANCE_SECRET_KEY=your_secret
NEWS_API_KEY=your_key
```

### 3. Run Modes

```bash
# Dashboard
python main.py --mode dashboard

# Paper Trading
python main.py --mode simulate --assets BTCUSDT,ETHUSDT

# Live Trading (Testnet)
python main.py --mode live --assets BTCUSDT,ETHUSDT --simulation

# Backtest
python main.py --mode backtest --symbol BTCUSDT --days 365
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_paper_trading.py -v
```

---

## 📈 Risk Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| MAX_DRAWDOWN | 20% | Kill-switch threshold |
| STOPLOSS | 2x ATR | Stop loss multiplier |
| TAKEPROFIT | 3x ATR | Take profit multiplier |
| MAX_POSITION | 30% | Max position size |

---

## 🔒 Safety Features

- ✅ Kill-switch at max drawdown
- ✅ Position size limits
- ✅ Order retry logic with exponential backoff
- ✅ SQLite state persistence
- ✅ Paper trading validation tests

---

## 📝 License

MIT License - See LICENSE file.

---

## 👤 Author

**Quantum AI Trading System**

- GitHub: [ballales1984-wq](https://github.com/ballales1984-wq)
- Built with Python, Scikit-learn, XGBoost, Dash, SQLite

---

## ⚡ Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python 3.10+ |
| ML | Scikit-learn, XGBoost |
| Risk | SciPy, NumPy |
| Trading | Binance API |
| Dashboard | Dash, Plotly |
| Database | SQLite |
| Orchestration | Event Bus |

---

*Built for professional quantitative trading with institutional-grade risk management.*
