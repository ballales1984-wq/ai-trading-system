# Crypto Commodity Trading System

An experimental AI-powered trading system for cryptocurrency and commodity-linked assets with real-time market analysis.

## Features

### Core Features
- **Data Collection**: Real-time crypto prices from Binance API (28+ trading pairs)
- **Technical Analysis**: RSI, EMA, SMA, Bollinger Bands, MACD, VWAP, Stochastic, ATR
- **Cross-Market Analysis**: Correlations between crypto and commodity assets
- **News/Sentiment**: Market sentiment analysis with Fear & Greed index
- **Decision Engine**: Probabilistic trading signals with risk management
- **Trading Simulator**: Paper trading with portfolio tracking

### Dashboard Features
- **Interactive Charts**: Candlestick charts with multiple indicators
- **Timeframe Selector**: 1h, 4h, 1d timeframes
- **MACD & RSI Panels**: Technical oscillator subplots
- **Volume & ATR**: Trading volume with volatility indicator
- **Portfolio Performance**: Equity curve tracking
- **Binance Market Data**: Real-time prices, volume, dominance
- **Pairs Comparison**: Top 10 USDT pairs by volume
- **Live Clock**: Real-time system clock

## Supported Trading Pairs

### Cryptocurrencies
BTC, ETH, XRP, SOL, ADA, DOT, AVAX, MATIC, BNB, DOGE, LINK, ATOM, UNI, LTC, NEAR, APT, ARB, OP, INJ, SUI, SEI, TIA

### Commodity Tokens
PAXG (Gold), XAUT (Gold), STETH (Ethereum), FXS (Frax)

## Installation

```bash
# Clone the repository
git clone https://github.com/ballales1984-wq/ai-trading-system.git
cd ai-trading-system

# Install dependencies
pip install -r requirements.txt

# Copy environment file
copy .env.example .env
```

## Configuration

Edit `.env` file to add your API keys:

```env
# Binance API (for real data)
BINANCE_API_KEY=your_key
BINANCE_SECRET_KEY=your_secret

# News API (optional)
NEWS_API_KEY=your_key

# Simulation mode
SIMULATION_MODE=false
```

## Usage

```bash
# Start dashboard (default)
python main.py --mode dashboard

# Generate signals
python main.py --mode signals

# Run analysis
python main.py --mode analysis --symbol BTC/USDT

# Backtest
python main.py --mode backtest

# Auto trading (paper)
python main.py --mode auto

# Portfolio control
python main.py --mode portfolio --portfolio-action check
```

## Dashboard

Open http://localhost:8050 in your browser to view the interactive dashboard.

### Dashboard Sections:
1. **Signals Summary** - Total/Buy/Sell/Hold signals
2. **Portfolio** - Balance, PnL, Win Rate
3. **Price Chart** - Candlesticks + EMA + Bollinger + VWAP
4. **MACD** - Moving Average Convergence Divergence
5. **RSI/Stochastic** - Momentum oscillators
6. **Volume/ATR** - Trading activity
7. **Trading Signals** - Top 10 opportunities
8. **Pairs Comparison** - Top 10 by volume
9. **Binance Market** - Real-time BTC/ETH data

## Project Structure

```
├── config.py              # Configuration settings
├── data_collector.py     # Data collection from Binance
├── technical_analysis.py  # Technical indicators
├── sentiment_news.py      # News and sentiment analysis
├── decision_engine.py     # Trading signals generation
├── trading_simulator.py   # Paper trading simulator
├── auto_trader.py        # Auto trading bot
├── binance_research.py   # Binance research API
├── dashboard.py          # Interactive Dash dashboard
├── main.py              # CLI entry point
└── requirements.txt      # Dependencies
```

## Risk Warning

This is an experimental system for educational purposes. Do not use with real money without proper testing and risk management.

## License

MIT License
