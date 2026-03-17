# Risk Book Documentation

## Overview

The **Risk Book** is the central component for risk management in the AI Trading System. It provides comprehensive tracking of positions, exposures, limits, and drawdowns.

## Location

- **Module**: `app/risk/risk_book.py`
- **Integration**: Used by `UnifiedDecisionEngine` and `AutonomousQuantAgent`

## Features

### Position Tracking

The Risk Book maintains a real-time ledger of all open positions:

```python
from app.risk.risk_book import RiskBook, RiskLimits, Position, PositionSide

# Configure limits
limits = RiskLimits(
    max_position_pct=0.10,       # Max 10% per position
    max_daily_drawdown_pct=0.05, # Max 5% daily drawdown
    var_95_limit=0.08,          # VaR 95% limit
    cvar_95_limit=0.10,         # CVaR 95% limit
)

# Create Risk Book
risk_book = RiskBook(limits)
risk_book.register_equity(100000.0)

# Update position
position = Position(
    symbol="BTCUSDT",
    quantity=0.1,
    avg_price=50000,
    side=PositionSide.LONG
)
risk_book.update_position(position)
```

### Risk Limits

| Limit | Default | Description |
|-------|---------|-------------|
| `max_position_pct` | 10% | Maximum position size as % of equity |
| `max_daily_drawdown_pct` | 5% | Maximum daily drawdown allowed |
| `var_95_limit` | 8% | Value at Risk (95% confidence) |
| `cvar_95_limit` | 10% | Conditional VaR (95% confidence) |

### Key Methods

#### Position Management

- `update_position(position)` - Add/update a position
- `get_position(symbol)` - Get position for a symbol
- `get_all_positions()` - Get all open positions
- `close_position(symbol)` - Close a position

#### Risk Checks

- `check_position_limit(symbol, prices, equity)` - Check if position is within limits
- `daily_drawdown_ok(equity)` - Check if drawdown is within limits
- `get_exposure_pct(prices)` - Calculate total exposure %
- `calculate_var()` - Calculate VaR
- `calculate_cvar()` - Calculate CVaR

### Integration with Decision Engine

The Risk Book is integrated into the Unified Decision Engine:

```python
from src.decision.unified_engine import UnifiedDecisionEngine

engine = UnifiedDecisionEngine()

# Decision with risk validation
result = engine.decide(
    symbol="BTCUSDT",
    current_price=50000,
    signals={"technical": 0.8, "sentiment": 0.7}
)
# Risk limits are automatically checked
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/risk/limits` | GET | Get current risk limits |
| `/api/v1/risk/positions` | GET | Get all positions |
| `/api/v1/risk/exposure` | GET | Get exposure metrics |
| `/api/v1/risk/validate` | POST | Validate a trade |

### Monitoring (Grafana)

The Risk Book exports metrics to Prometheus:

| Metric | Type | Description |
|--------|------|-------------|
| `riskbook_equity` | Gauge | Current equity |
| `riskbook_drawdown_percent` | Gauge | Current drawdown % |
| `riskbook_exposure_percent` | Gauge | Total exposure % |
| `riskbook_position_count` | Gauge | Number of open positions |
| `riskbook_var_95` | Gauge | VaR 95% |
| `riskbook_cvar_95` | Gauge | CVaR 95% |
| `riskbook_signals_blocked_total` | Counter | Blocked signals count |

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Autonomous Agent                       │
│  ┌─────────────────────────────────────────────────┐   │
│  │           UnifiedDecisionEngine                 │   │
│  │  ┌─────────────────────────────────────────┐   │   │
│  │  │            RiskBook                     │   │   │
│  │  │  - Position tracking                    │   │   │
│  │  │  - Exposure limits                      │   │   │
│  │  │  - Drawdown monitoring                 │   │   │
│  │  │  - VaR/CVaR calculation                │   │   │
│  │  └─────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## Error Handling

The Risk Book raises exceptions for critical conditions:

- `PositionLimitExceeded` - Position exceeds limit
- `DrawdownExceeded` - Drawdown exceeds limit
- `VaRExceeded` - VaR exceeds limit

## Best Practices

1. **Always initialize equity** before trading
2. **Check limits before executing** trades
3. **Monitor drawdown** in real-time
4. **Review VaR/CVaR** regularly
5. **Use the Risk Book in all decision flows**
