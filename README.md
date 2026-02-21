🤖 AI Trading System — Mini Hedge Fund

Performance Badge Highlights


🎬 Live Demo & Dashboard Snapsh
Portfolio in tempo reale, P&L, posizioni, segnali, volatilità e drawdown charts.

Performance simulata Monte Carlo 5 livelli.

⚠️ Disclaimer

Software solo a scopo educativo e di ricerca.
Non costituisce consulenza finanziaria. Il trading comporta rischio significativo di perdita.

🎯 Vision

Costruire un’infrastruttura modulare AI-driven, evolutiva da retail bot a architettura quantitativa istituzionale, scalabile e ottimizzata per performance real-time.

🚀 Quick Start
git clone https://github.com/ballales1984-wq/ai-trading-system.git
cd ai-trading-system
pip install -r requirements.txt
python dashboard.py  # http://127.0.0.1:8050
python -m uvicorn app.main:app --reload  # http://127.0.0.1:8000/docs
docker-compose up -d  # optional Docker
🏗️ Architettura
External APIs (18+) → API Registry → Central Database
                            ↓
                    Analysis Engine
               (Technical + Sentiment + Events)
                            ↓
                  Monte Carlo Engine (5 Levels)
                            ↓
                   Decision Engine
                 (BUY/SELL/HOLD + Confidence)
                            ↓
                 Execution Engine → Exchanges
                            ↓
                  Dashboard + Alerts + Logs
🔹 Features

Multi-API Ingestion: Binance, CoinGecko, Alpha Vantage, NewsAPI, Twitter/X, GDELT, Trading Economics…

Monte Carlo 5 Levels: Base, Conditional, Adaptive, Multi-Factor, Semantic History

Decision Engine Ensemble: Technical 30%, Momentum 25%, Correlation 20%, Sentiment 15%, ML 10%

Risk Management: VaR, CVaR, GARCH, drawdown limits

Execution Engine: Best execution, TCA, paper/live trading, Binance/Bybit/OKX/IB

Dashboard: Portfolio, P&L, Sharpe/Sortino, volatility, order book, trade history

🆚 Compared to Retail Bots
Feature	AI Trading System	Typical Retail Bot
Monte Carlo 5-level	✅	❌
Multi-API ingestion	✅	⚠️ Limited
Institutional Risk	✅	❌
ML Ensemble	✅	⚠️ Basic
Event-driven Architecture	✅	❌
🧪 Backtesting & Performance
Metric	Value
CAGR	23.5%
Max Drawdown	7.2%
Sharpe Ratio	1.95
Sortino Ratio	2.45
Win Rate	68%
Profit Factor	1.85

I valori sono simulati su dati storici per scopi di testing e ricerca.

☁️ Deployment

Local / VPS / Docker Swarm

Cloud-ready (AWS / GCP)

Modular & scalable

⚙️ Configurazione API Keys

Crea un file .env:

# Required
BINANCE_API_KEY=your_key
BINANCE_SECRET_KEY=your_secret
USE_BINANCE_TESTNET=true

# Optional
NEWSAPI_KEY=your_newsapi_key
ALPHA_VANTAGE_API_KEY=your_av_key
📁 Struttura Progetto
ai-trading-system/
├── main.py
├── dashboard.py
├── app/
│   ├── main.py
│   ├── api/routes/
│   ├── core/
│   ├── execution/
│   └── database/
├── src/
│   ├── external/
│   ├── core/
│   ├── decision/
│   ├── strategy/
│   ├── agents/
│   └── ml_enhanced.py
├── docker/
├── tests/
└── docs/
👨‍💻 Author

Alessio Ballini — Quantitative Developer | Python Engineer | AI Trading Systems
GitHub
 | LinkedIn

📄 Licenza

MIT License — vedi LICENSE
