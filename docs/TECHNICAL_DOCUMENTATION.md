# AI Trading System - Technical Documentation

## Backtest Integrity Framework

Our backtesting system ensures statistical validity and prevents overfitting.

### Methodology

| Component | Implementation | Purpose |
|-----------|----------------|---------|
| Data Quality | Point-in-time data, no look-ahead bias | Ensure historical accuracy |
| Transaction Costs | 0.1% taker fee + 0.5 bps slippage | Realistic execution |
| Market Impact | Volume-based impact model | Large order simulation |
| Slippage Modeling | Bid-ask spread + volatility adjustment | Realistic fills |
| Risk-Free Rate | 5% annual | Sharpe/Sortino calculation |

### Walk-Forward Analysis

```
WALK-FORWARD OPTIMIZATION
--- In-Sample 1 ---|--- In-Sample 2 ---|--- In-Sample 3 ---
     (3 years)          (3 years)           (3 years)
         ████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
                        |                   |
                        ▼                   ▼
                   Out-of-Sample 1    Out-of-Sample 2
                      (1 year)          (1 year)
                      ░░░░░░░          ░░░░░░░
Process: Optimize on IS -> Test on OOS -> Repeat
Result: Robust parameters that work forward
```

### Performance Metrics

| Metric | Description | Threshold |
|--------|-------------|-----------|
| CAGR | Compound Annual Growth Rate | > 15% |
| Sharpe Ratio | Risk-adjusted return | > 1.5 |
| Sortino Ratio | Downside risk-adjusted | > 2.0 |
| Max Drawdown | Largest peak-to-trough | < 20% |
| Calmar Ratio | CAGR / Max DD | > 2.0 |
| Win Rate | Profitable trades % | > 55% |
| Profit Factor | Gross profit / Gross loss | > 1.5 |
| Risk of Ruin | Probability of 50% drawdown | < 5% |

---

## Alpha Sources (18+ Data Sources)

### Price Data
| Source | Type | Purpose |
|--------|------|---------|
| Binance | WebSocket | Real-time OHLCV, Order Book |
| CoinGecko | REST API | Market data, coin info |
| Alpha Vantage | REST API | Technical indicators |

### Sentiment
| Source | Type | Purpose |
|--------|------|---------|
| NewsAPI | REST API | Financial news headlines |
| Twitter/X | Streaming | Social sentiment |
| Reddit | REST API | Market sentiment |

### Macro
| Source | Type | Purpose |
|--------|------|---------|
| Trading Economics | REST API | GDP, inflation, rates |
| FRED | REST API | US economic data |

### On-Chain
| Source | Type | Purpose |
|--------|------|---------|
| Glassnode | REST API | Blockchain metrics |
| CoinGlass | REST API | Liquidations, funding |

### Alternative
| Source | Type | Purpose |
|--------|------|---------|
| GDELT | Periodic | Global events |
| Weather API | REST API | Weather correlation |
| SEC EDGAR | REST API | Corporate filings |

---

## Risk of Ruin Calculator

The system includes a sophisticated Risk of Ruin calculator based on Kelly Criterion.

### Implementation

Risk of Ruin Formula:
P_ruin = ((1 - W) / (1 + W)) ^ N

Where:
- W = Win rate (e.g., 0.60)
- W = (avg_win * win_rate - avg_loss * (1 - win_rate)) / avg_loss
- N = Number of consecutive losses to ruin (capital / position_size)

### Risk of Ruin Matrix

| Win Rate | Avg Win/Loss | Position Size | Risk of Ruin |
|----------|--------------|---------------|--------------|
| 55% | 1.5 | 5% | 12.3% |
| 55% | 1.5 | 2% | 0.8% |
| 60% | 1.5 | 5% | 2.1% |
| 60% | 1.5 | 2% | 0.05% |
| 65% | 2.0 | 5% | 0.3% |
| 65% | 2.0 | 2% | 0.001% |
| 70% | 2.0 | 5% | 0.02% |
| 70% | 2.0 | 2% | < 0.001% |

### Risk Controls

- Max Position Size: 10% of portfolio
- Max Daily Loss: 5% of portfolio
- Max Drawdown: 15% (auto-trading stop)
- VaR Confidence: 95%
- Kelly Fraction: Max 25% of optimal

---

## System Data Flow

External Data Layer:
- Binance (WebSocket)
- CoinGecko (REST)
- NewsAPI (REST)
- Twitter (Streaming)
- Alpha Vantage (REST)
- GDELT (Periodic)

Event Bus (Async Pub/Sub):
- MARKET_DATA -> SIGNALS -> DECISION -> EXECUTION -> ORDERS

Core Engines:
- Alpha Lab (Research)
- Decision Engine (5-Question)
- Risk Engine (VaR/CVaR/GARCH)

Execution Layer:
- TWAP Algorithm
- VWAP Algorithm
- POV Algorithm
- Adaptive Algorithm

Broker Connectors:
- Binance (Live)
- Bybit (Live)
- Interactive Brokers
- Paper Trading

