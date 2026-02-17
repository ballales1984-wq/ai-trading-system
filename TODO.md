# TODO - Crypto Commodity Trading System

## Completed ✅

- [x] Project structure setup
- [x] Configuration module (config.py)
- [x] Data collector module with simulated data
- [x] Technical analysis module (RSI, EMA, Bollinger, MACD, etc.)
- [x] Sentiment/news analysis module
- [x] Decision engine with probabilistic signals
- [x] Dashboard module (Plotly/Dash ready)
- [x] Main entry point with CLI
- [x] Testing and validation
- [x] Enable real Binance data mode
- [x] Added ADX and VWAP indicators

## Next Steps (Optional Enhancements)

- [ ] Install full dependencies: `pip install dash plotly ccxt`
- [ ] Add API keys to .env for real data
- [ ] Connect to live exchange data
- [ ] Add NewsAPI key for real news
- [ ] Backtest with historical data
- [ ] Add machine learning models for predictions

## Running the System

```bash
# Console signals (real data if API keys configured)
venv\Scripts\python.exe main.py --mode signals

# Test mode
venv\Scripts\python.exe main.py --mode test

# Dashboard (requires dash/plotly)
venv\Scripts\python.exe main.py --mode dashboard

# Analysis mode
venv\Scripts\python.exe main.py --mode analysis --symbol BTC/USDT
```

## Configuration

Copy `.env.example` to `.env` and add your API keys:

```bash
# Binance (get from binance.com)
BINANCE_API_KEY=your_key
BINANCE_SECRET_KEY=your_secret

# News (get from newsapi.org)
NEWS_API_KEY=your_key
```
