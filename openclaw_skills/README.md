# OpenClaw + ai-trading-system Integration

> **Level 1**: Conversational Frontend → FastAPI Backend  
> **Level 2**: Multi-Agent Orchestration (Research → Quant → Risk → Execute)

## Quick Start

### Prerequisites

1. **ai-trading-system running** on port 8000:
   ```bash
   cd c:/ai-trading-system
   python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

2. **OpenClaw framework** installed (see OpenClaw docs)

### Option 1: Basic Skill Integration (Level 1)

#### Step 1: Import the Skill
Copy [`quant_trading_assistant.md`](quant_trading_assistant.md) to your OpenClaw skills directory.

#### Step 2: Configure API Credentials
Edit [`skill.yaml`](skill.yaml) with your credentials:
```yaml
auth:
  credentials:
    username: "admin"
    password: "admin123"
```

#### Step 3: Test the Skill
Start a conversation:
- "What's my current portfolio risk?"
- "Run a Monte Carlo simulation"
- "Can I buy 0.5 BTC at $65,000?"

### Option 2: Multi-Agent Setup (Level 2)

#### Step 1: Load Multi-Agent Config
Copy [`multi_agent_config.yaml`](multi_agent_config.yaml) to your OpenClaw config.

#### Step 2: Start the Agents
Each agent will handle specific responsibilities:
- **Research Agent** → Market data & sentiment
- **Quant Analyst Agent** → Regime detection, Monte Carlo
- **Risk Gate Agent** → Validates all trades
- **Executor Agent** → Executes approved trades

#### Step 3: Trigger Workflows
- "Analyze BTC and execute if risk-approved"
- "Run full portfolio risk review"

---

## Available API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/auth/login` | POST | Get JWT token |
| `/api/v1/portfolio/summary/dual` | GET | Real + simulated portfolio |
| `/api/v1/portfolio/positions` | GET | Current positions |
| `/api/v1/portfolio/performance` | GET | Performance metrics |
| `/api/v1/risk/metrics` | GET | VaR, CVaR, volatility |
| `/api/v1/risk/limits` | GET | Risk limits status |
| `/api/v1/risk/check_order` | POST | Pre-trade risk check |
| `/api/v1/risk/var/monte_carlo` | GET | Monte Carlo VaR |
| `/api/v1/risk/stress_test` | GET | Stress scenarios |
| `/api/v1/risk/correlation` | GET | Correlation matrix |
| `/api/v1/market/prices` | GET | Market prices |
| `/api/v1/orders` | GET/POST | Order management |

---

## Python API Wrapper

Use [`api_wrapper.py`](api_wrapper.py) for direct Python access:

```python
from openclaw_skills.api_wrapper import QuantTradingAPI

# Initialize
api = QuantTradingAPI(base_url="http://localhost:8000/api/v1")
api.login("admin", "admin123")

# Get risk metrics
risk = api.get_risk_metrics()
print(f"VaR: ${risk['var_1d']:,.2f}")

# Check trade
result = api.check_order_risk("BTCUSDT", "BUY", 0.5, 65000)
print(f"Approved: {result['approved']}")

# Run Monte Carlo
mc = api.run_monte_carlo(simulations=10000, confidence=0.95)
print(f"Worst case (5%): ${mc['worst_case']:,.2f}")
```

### CLI Usage
```bash
# Get risk metrics
python openclaw_skills/api_wrapper.py risk

# Analyze a trade
python openclaw_skills/api_wrapper.py analyze --symbol BTCUSDT --side BUY --quantity 0.5 --price 65000
```

---

## Conversation Examples

### Portfolio Analysis
```
User:  "What's my current risk exposure?"

Agent: "Based on the latest risk metrics:
        - VaR (1-day, 95%): $12,500
        - CVaR: $18,750
        - Portfolio volatility: 25% annualized
        - Sharpe ratio: 1.85
        - Current leverage: 1.2x

        Your largest position is BTCUSDT at 65% concentration risk.
        Would you like me to run a rebalancing analysis?"
```

### Trade Risk Check
```
User:  "I want to buy 1 BTC at $65,000"

Agent: "Let me check the risk gates...

        Risk check results:
        - Risk score: 65/100 ✓
        - Estimated market impact: $65 (0.1%)
        - Concentration: Would increase to 72% ⚠️
        - VaR impact: +$8,500 (within limits) ✓

        Recommendation: Approved, but consider reducing to 0.8 BTC
        to stay under 70% concentration limit. Execute?"
```

### Monte Carlo Analysis
```
User:  "What's my worst-case scenario next month?"

Agent: "Running 10,000 Monte Carlo simulations...

        Results (95% confidence):
        - Median outcome: +$15,200
        - 5th percentile (worst case): -$28,500
        - 95th percentile (best case): +$58,000
        - Expected Shortfall: -$42,000

        The distribution suggests a 5% chance of losing more than
        $28,500. Your current risk limits can absorb this.
        
        Would you like to see stress test scenarios?"
```

### Trade Rejection
```
User:  "Buy 5 BTC at market"

Agent: "❌ Order Rejected

        Reason: Concentration limit would be exceeded
        - Current BTC exposure: 65% ($65,250)
        - Additional: 100% ($217,000)
        - Total: 165% (limit: 100%)

        Suggestions:
        1. Reduce to 0.35 BTC maximum
        2. Or close some existing BTC positions first
        3. Consider diversifying into ETH or SOL

        Would you like me to calculate the optimal position size?"
```

---

## Safety Guards

### Hard Blocks (Auto-Reject)
- Concentration > 100%
- Leverage > 3x
- VaR limit exceeded
- Risk score > 80

### Soft Warnings (Require ACK)
- Concentration 70-100%
- Large orders > $50,000
- High volatility periods

### Human-in-the-Loop
- Orders > $10,000 require confirmation
- Emergency stop: "STOP ALL TRADES"

---

## MiniMax API Integration (AI-Powered Agents)

> **Note**: MiniMax-M2 requires ~16GB VRAM to run locally. For systems with limited VRAM (like 5GB), use the API instead.

### Getting Started

#### Step 1: Get Your Free API Key
1. Go to [platform.minimax.io](https://platform.minimax.io/)
2. Register for a free account
3. Create a new project
4. Generate an API key (starts with `ey`)
5. Note your Group ID from the dashboard

#### Step 2: Configure Environment Variables

Add to your `.env` file:
```bash
MINIMAX_API_KEY=your_api_key_here
MINIMAX_GROUP_ID=your_group_id_here
```

Or set environment variables:
```bash
# Windows
set MINIMAX_API_KEY=your_key
set MINIMAX_GROUP_ID=your_group

# Linux/Mac
export MINIMAX_API_KEY=your_key
export MINIMAX_GROUP_ID=your_group
```

#### Step 3: Test the Connection
```bash
cd c:/ai-trading-system
python openclaw_skills/test_minimax_connection.py
```

Or with inline credentials:
```bash
python openclaw_skills/test_minimax_connection.py --api-key "your_key" --group-id "your_group"
```

### Using MiniMax in Your Code

```python
from openclaw_skills.minimax_connector import (
    MiniMaxClient,
    MiniMaxConfig,
    OpenClawMiniMaxBridge
)

# Simple usage
config = MiniMaxConfig()  # Loads from env vars
client = MiniMaxClient(config)

response = client.chat_completion(
    messages=[
        {"role": "system", "content": "You are a trading assistant"},
        {"role": "user", "content": "Analyze BTC market"}
    ]
)
print(response['choices'][0]['message']['content'])

# Trading-specific analysis
analysis = client.analyze_market(
    symbol="BTCUSDT",
    market_data={"price": 67500, "volume": 1000000},
    analysis_type="regime"
)

# OpenClaw integration
bridge = OpenClawMiniMaxBridge(config)
response = bridge.process_agent_message(
    agent_name="research_agent",
    message="What's the market sentiment for BTC?",
    context={"price": 67500}
)
```

### Available Models

| Model | Description | Use Case |
|-------|-------------|----------|
| `MiniMax-M2` | General purpose (recommended) | Trading analysis, general queries |
| `abab6.5s-chat` | Fast model | Simple queries, high throughput |
| `abab6.5g-chat` | General chat | Balanced performance |

### Configuration Options

In `skill.yaml` or `multi_agent_config.yaml`:
```yaml
llm:
  provider: "minimax"
  model: "MiniMax-M2"
  temperature: 0.7
  max_tokens: 2048
  # Fallback if API fails
  fallback_to_keyword_matching: true
```

---

## Files Overview

| File | Purpose |
|------|---------|
| `quant_trading_assistant.md` | OpenClaw skill definition |
| `skill.yaml` | Skill configuration |
| `multi_agent_config.yaml` | Multi-agent orchestration |
| `api_wrapper.py` | Python API wrapper |
| `minimax_connector.py` | MiniMax API connector |
| `test_minimax_connection.py` | Connection test script |
| `README.md` | This file |

---

## Production Considerations

### Security
- Change default credentials (`admin/admin123`)
- Use API keys with limited scope
- Enable HTTPS in production
- Restrict network access via firewall

### Monitoring
- All API calls logged to audit
- Risk metrics tracked in DB
- Dashboard for human oversight

### Deployment
- ai-trading-system: Docker/K8s on secure network
- OpenClaw: Separate VPS or container
- Use VPN or scoped API keys for communication

---

## Next Steps

1. **Start with Level 1** - Test the skill with simple queries
2. **Add multi-agent** - Scale to orchestrated team
3. **Production hardening** - Add HTTPS, custom credentials, monitoring
4. **Connect live execution** - Move from paper to live trading

---

*Generated for ai-trading-system + OpenClaw integration*
