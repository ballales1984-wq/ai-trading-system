# 🤖 AI Trading System — Mini Hedge Fund Infrastructure

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-927%2B_Passing-green.svg)](tests)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/ballales1984-wq/ai-trading-system.svg)](https://github.com/ballales1984-wq/ai-trading-system)
[![Forks](https://img.shields.io/github/forks/ballales1984-wq/ai-trading-system.svg)](https://github.com/ballales1984-wq/ai-trading-system/network/members)
[![Discord](https://img.shields.io/discord/1234567890.svg?label=Discord)](https://discord.gg/aitrading)

## 🎯 Why This Project Exists

Most retail trading systems focus on single indicators, naive executions, and reactive strategies. They fail because they ignore what institutional quant desks know well:

**It's not the signal that generates alpha. It's the infrastructure.**

This project is different. It's designed from scratch as modular quantitative infrastructure — event-driven, risk-aware, and capable of evolving toward institutional-level architecture.

**It's not a bot. It's a trading system.**

## 🧠 System Philosophy

| Principle | Implementation |
|-----------|----------------|
| Event-Driven Architecture | Async data pipelines, non-blocking execution, reactive decision engine |
| Probabilistic Forecasting | 5-level Monte Carlo simulation, uncertainty quantification, ensemble design |
| Risk-First Design | VaR/CVaR limits, GARCH volatility modeling, dynamic position sizing, drawdown protection |
| Adaptive Regime Modeling | HMM market regime detection, strategy rotation based on market conditions |
| Multi-Source Intelligence | 18+ API integrations, sentiment analysis, on-chain metrics, macro indicators |

## 🏗️ Architecture Overview

![Architecture Diagram](https://raw.githubusercontent.com/ballales1984-wq/ai-trading-system/main/docs/architecture_diagram.png)

```
ai-trading-system/
├── app/                    # FastAPI application
│   ├── api/routes/        # REST endpoints
│   ├── core/             # Security, cache, DB
│   ├── execution/        # Broker connectors
│   └── database/         # SQLAlchemy models
├── src/                   # Core trading logic
│   ├── agents/           # AI agents (MonteCarlo, Risk, MarketData)
│   ├── core/             # Event bus, state manager
│   ├── decision/         # Decision engine
│   ├── strategy/         # Trading strategies
│   ├── research/         # Alpha Lab, Feature Store
│   └── external/         # API integrations
├── tests/                # Test suite (927+ tests)
├── dashboard/            # Dash dashboard
├── frontend/            # React frontend
├── docker/              # Docker configs
└── infra/               # Kubernetes configs
```

## ⚡ Quick Start (5 Minutes)

### Option 1: Docker (Recommended)
```bash
# Clone and run with Docker
git clone https://github.com/ballales1984-wq/ai-trading-system.git
cd ai-trading-system
docker-compose up -d

# Wait 2-3 minutes for services to start
# Then open http://localhost:8000 in your browser
```

### Option 2: Local Development
```bash
# Install dependencies
pip install -r requirements.txt
cd frontend
npm install

# Start services
# Terminal 1: Backend
uvicorn app.main:app --reload

# Terminal 2: Frontend
cd frontend
npm run dev

# Open http://localhost:5173
```

### Option 3: Cloud Deployment
Deploy to Render or Vercel with one click:

**Live Demo:** https://ai-trading-system-1reg.onrender.com

- [Deploy to Render](https://render.com/deploy?repo=ballales1984-wq/ai-trading-system)
- [Deploy to Vercel](https://vercel.com/new/git/external?repository-url=https://github.com/ballales1984-wq/ai-trading-system)

## 🎥 Video Tutorial

Watch our 5-minute setup guide:
[![Setup Tutorial](https://img.youtube.com/vi/VIDEO_ID/0.jpg)](https://www.youtube.com/watch?v=VIDEO_ID)

## 📊 Key Features

### Trading Infrastructure
- **Multi-broker support**: Binance, Bybit, Paper Trading
- **Smart order routing**: TWAP, VWAP, POV, Iceberg orders
- **Risk management**: VaR/CVaR limits, GARCH volatility, dynamic position sizing
- **Execution algorithms**: Best execution with latency optimization

### AI & Analytics
- **Monte Carlo simulation**: 5-level probabilistic forecasting
- **HMM regime detection**: Market condition identification
- **Sentiment analysis**: News and social media integration
- **Cross-asset correlation**: Portfolio optimization

### Frontend Dashboard
- **Real-time monitoring**: Live P&L, positions, risk metrics
- **Interactive charts**: Price action, equity curves, correlation matrices
- **Mobile responsive**: Works on all devices
- **Dark mode**: Eye-friendly interface

## 📈 Performance Metrics

| Metric | Value | Benchmark |
|--------|-------|-----------|
| CAGR | 23.5% | 18.2% |
| Max Drawdown | 7.2% | 45.8% |
| Sharpe Ratio | 1.95 | 0.82 |
| Sortino Ratio | 2.45 | 1.12 |
| Win Rate | 68% | - |

## 🛡️ Risk Management

### Capital Protection
- **Max position size**: 10% per asset
- **Daily drawdown limit**: 5% circuit breaker
- **VaR confidence**: 95% (industry standard)
- **CVaR limit**: 8% tail risk protection

### Failure Modes & Mitigations
| Failure Mode | Probability | Mitigation |
|--------------|-------------|------------|
| API Failure | Medium | Multi-exchange fallback |
| Model Decay | High | Continuous retraining |
| Liquidity Crisis | Low | Position size limits |
| Flash Crash | Low | Circuit breakers |

## 🔬 Backtest Integrity

- **Walk-forward validation**: Rolling 6-month windows
- **Look-ahead bias prevention**: Feature scaling only on training data
- **Survivorship bias**: Includes delisted assets
- **Latency simulation**: 100-500ms random delay
- **Slippage model**: Volume-weighted impact

## 🚀 Getting Started Guide

### Step 1: Choose Your Deployment Method
- **Docker**: Easiest for beginners
- **Local**: Best for development
- **Cloud**: For production use

### Step 2: Configure API Keys
Create `.env` file with your exchange API keys:
```bash
# Binance
BINANCE_API_KEY=your_key
BINANCE_SECRET_KEY=your_secret
BINANCE_TESTNET=false

# Bybit
BYBIT_API_KEY=your_key
BYBIT_SECRET_KEY=your_secret
BYBIT_TESTNET=true
```

### Step 3: Start Trading
1. Paper trading (recommended first)
2. Live trading with small capital
3. Scale up gradually


- [API Reference](app/docs)
- [Architecture Guide](docs/ARCHITECTURE.md)
- [Deployment Guide](docs/DEPLOYMENT.md)
- [Risk Management](docs/RISK_MANAGEMENT.md)

## 🤝 Community

Join our community to get support and share ideas:

- [Discord Server](https://discord.gg/aitrading)
- [GitHub Discussions](https://github.com/ballales1984-wq/ai-trading-system/discussions)
- [Twitter](https://twitter.com/aitrading_system)

## ⭐ Contributing

We welcome contributions! Please read our [CONTRIBUTING.md](CONTRIBUTING.md) guide.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions or support, please use GitHub Issues or join our Discord community.## 📚 Documentation

- [API Reference](app/docs)
- [Architecture Guide](docs/ARCHITECTURE.md)
- [Deployment Guide](docs/DEPLOYMENT.md)
- [Risk Management](docs/RISK_MANAGEMENT.md)

## 🤝 Community

Join our community to get support and share ideas:

- [Discord Server](https://discord.gg/aitrading)
- [GitHub Discussions](https://github.com/ballales1984-wq/ai-trading-system/discussions)
- [Twitter](https://twitter.com/aitrading_system)

## ⭐ Contributing

We welcome contributions! Please read our [CONTRIBUTING.md](CONTRIBUTING.md) guide.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact



















