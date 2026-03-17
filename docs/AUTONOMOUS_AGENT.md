# Autonomous Quant Agent Documentation

## Overview

The **Autonomous Quant Agent** is a Level 5 AI agent that orchestrates all system components to make autonomous trading decisions. It combines:

- **Market Regime Detection** (HMM)
- **Volatility Forecasting** (GARCH)
- **Risk Simulation** (Monte Carlo)
- **Portfolio Optimization** (MPT)
- **Risk Management** (Risk Book)
- **Model Selection** (Model Registry)

## Location

- **Module**: `src/agents/autonomous_quant_agent.py`
- **CLI Entry**: `src/agents/__main__.py`
- **API Routes**: `app/api/routes/agents.py`

## ⚠️ Important Limitations

**This agent does NOT execute trades autonomously.** It generates:
- Analysis reports
- Action proposals
- Risk assessments

All trades require human approval through the API or CLI before execution.

## Usage

### CLI Usage

```bash
# Daily report for a symbol
python -m src.agents BTCUSDT

# With action proposals
python -m src.agents BTCUSDT --proposals

# Multiple symbols
python -m src.agents BTCUSDT,ETHUSDT,SOLUSDT

# Portfolio status only
python -m src.agents --status
```

### API Usage

```python
from fastapi import HTTPClient

# Get daily report
response = client.get("/api/v1/agents/autonomous/report/BTCUSDT")

# Get action proposals
response = client.get("/api/v1/agents/autonomous/proposals/BTCUSDT")

# Get portfolio status
response = client.get("/api/v1/agents/autonomous/portfolio")

# Execute action (requires approval)
response = client.post("/api/v1/agents/autonomous/execute", json={
    "symbol": "BTCUSDT",
    "action": "buy",
    "size": 0.1,
    "reason": "Bull regime with high confidence"
})
```

### Python Usage

```python
from src.agents.autonomous_quant_agent import AutonomousQuantAgent, AgentConfig

# Configure agent
config = AgentConfig(
    default_symbols=["BTCUSDT", "ETHUSDT"],
    max_position_pct=0.10,
    max_drawdown_pct=0.05,
    regime_confidence_threshold=0.7,
    mc_simulations=5000,
    mc_days_ahead=30,
)

# Create agent
agent = AutonomousQuantAgent(config)

# Get daily report
report = agent.daily_report("BTCUSDT")

# Get action proposals
proposals = agent.propose_actions("BTCUSDT")

# Get portfolio status
status = agent.get_portfolio_status()
```

## Report Structure

The daily report contains:

```json
{
  "timestamp": "2026-03-17T12:00:00",
  "trading_mode": "active",
  "regime": {
    "regime": "bull",
    "confidence": 0.85,
    "volatility": 0.025
  },
  "monte_carlo": {
    "mean_price": 52000,
    "median_price": 51800,
    "percentile_5": 48000,
    "percentile_95": 56000
  },
  "portfolio": {
    "equity": 102500,
    "pnl": 2500,
    "pnl_pct": 2.5,
    "position_count": 3
  },
  "risk": {
    "within_limits": true,
    "var_95": 0.06,
    "cvar_95": 0.08,
    "drawdown_pct": 1.2
  },
  "models": {
    "champion": {
      "name": "price_prediction_v2",
      "version": "2.1.0",
      "metrics": {"mape": 0.03, "rmse": 150}
    }
  }
}
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_position_pct` | 10% | Maximum position size |
| `max_drawdown_pct` | 5% | Maximum daily drawdown |
| `default_symbols` | [BTCUSDT, ETHUSDT] | Default symbols to analyze |
| `regime_confidence_threshold` | 0.7 | Minimum regime confidence |
| `mc_simulations` | 5000 | Monte Carlo paths |
| `mc_days_ahead` | 30 | Forecast horizon |

## Trading Modes

| Mode | Description |
|------|-------------|
| `active` | Normal trading mode |
| `close_only` | Only close existing positions |
| `paused` | No new trades, monitoring only |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   Autonomous Quant Agent                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  HMM Regime  │  │   GARCH      │  │  Monte Carlo │          │
│  │  Detection   │  │  Volatility  │  │  Simulation │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Portfolio  │  │   RiskBook   │  │ ModelRegistry│          │
│  │  Optimizer  │  │  Management  │  │    (ML)      │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                     Decision Engine                              │
│  - Signal aggregation                                           │
│  - Risk validation                                              │
│  - Position sizing                                               │
└─────────────────────────────────────────────────────────────────┘
```

## Integration with Decision Engine

The agent uses the Unified Decision Engine for final decision-making:

```python
from src.decision.unified_engine import UnifiedDecisionEngine

engine = UnifiedDecisionEngine()

# Agent uses engine for risk-validated decisions
result = engine.decide(
    symbol="BTCUSDT",
    current_price=50000,
    signals={"technical": 0.8, "sentiment": 0.7}
)
```

## Monitoring

The agent exports metrics to Prometheus:

| Metric | Type | Description |
|--------|------|-------------|
| `agent_decisions_total` | Counter | Total decisions made |
| `agent_proposals_total` | Counter | Total proposals generated |
| `agent_trading_mode` | Gauge | Current trading mode |
| `agent_last_report_time` | Gauge | Last report timestamp |

## Best Practices

1. **Always review proposals** before execution
2. **Monitor risk metrics** in Grafana
3. **Set appropriate limits** for your risk tolerance
4. **Use human-in-the-loop** for large positions
5. **Review daily reports** to understand agent behavior
