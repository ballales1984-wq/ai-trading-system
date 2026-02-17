# Crypto Commodity Trading System

Modular experimental system for crypto + commodity-linked trading signals.

## Features

- **Data Collection**: Crypto prices (BTC, ETH, XRP), commodity-linked stablecoins (PAX Gold, etc.)
- **Technical Analysis**: RSI, EMA, Bollinger Bands, volatility metrics
- **Cross-Market Analysis**: Correlations between crypto and real assets
- **News/Sentiment**: Geopolitical and market news integration
- **Decision Engine**: Probabilistic trading signals with risk management
- **Dashboard**: Interactive visualizations

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

## Structure

```
crypto_commodity_trading/
├── config.py           # Configuration settings
├── data_collector.py   # Data collection from APIs
├── technical_analysis.py  # Technical indicators
├── sentiment_news.py   # News and sentiment analysis
├── decision_engine.py  # Trading signals generation
├── dashboard.py        # Interactive dashboard
├── main.py            # Main entry point
└── requirements.txt   # Dependencies
```
