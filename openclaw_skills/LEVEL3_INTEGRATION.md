# OpenClaw Level 3: Direct Quant Execution

> **Advanced Integration**: Run quant models directly in OpenClaw

Unlike Levels 1-2 which call the FastAPI backend, Level 3 executes Python quant code **directly** in the OpenClaw agent's environment.

## What Changed?

| Level | Method | Latency | Libraries |
|-------|--------|---------|-----------|
| Level 1 | HTTP API calls | ~100ms | None (calls backend) |
| Level 2 | Multi-agent orchestration | ~200ms | None (calls backend) |
| **Level 3** | Direct Python execution | **~10ms** | **numpy, pandas, hmmlearn, arch, pypfopt** |

## New Skills (Direct Execution)

### 1. HMM Regime Detection
**File**: [`hmm_regime_detect.py`](hmm_regime_detect.py)

Detects market regimes using Hidden Markov Models.

```python
from hmm_regime_detect import detect_regimes, format_output

result = detect_regimes("BTCUSDT", n_states=3, lookback_days=90)
print(format_output(result))
```

**Output**:
- Current regime (bear/sideways/bull)
- Regime probabilities
- Transition matrix
- Recommendations

### 2. Monte Carlo Paths
**File**: [`monte_carlo_paths.py`](monte_carlo_paths.py)

Generates thousands of price paths using Geometric Brownian Motion.

```python
from monte_carlo_paths import generate_price_paths, format_output

result = generate_price_paths(
    initial_price=50000,
    expected_return=0.001,
    volatility=0.03,
    n_paths=5000,
    days_ahead=30
)
print(format_output(result))
```

**Output**:
- Percentile distribution (P1-P99)
- VaR and CVaR
- Probability of profit
- Risk metrics

### 3. GARCH Volatility
**File**: [`garch_volatility.py`](garch_volatility.py)

Fits GARCH volatility models and forecasts.

```python
from garch_volatility import fit_garch, format_output

result = fit_garch("BTCUSDT", p=1, q=1, forecast_horizon=5)
print(format_output(result))
```

**Output**:
- Fitted parameters (omega, alpha, beta)
- Current annualized volatility
- Volatility forecast
- VaR calculations

### 4. Portfolio Optimizer
**File**: [`portfolio_optimizer.py`](portfolio_optimizer.py)

Modern Portfolio Theory optimization.

```python
from portfolio_optimizer import optimize_portfolio, format_output

result = optimize_portfolio(
    assets=["BTC", "ETH", "SOL", "SPY", "GLD"],
    objective="max_sharpe"
)
print(format_output(result))
```

**Output**:
- Optimal weights
- Expected return & volatility
- Sharpe ratio
- Efficient frontier points

## Dependencies

Install required libraries:

```bash
pip install numpy pandas hmmlearn arch pypfopt
```

Or use the existing requirements:
```bash
pip install -r requirements.stable.txt
```

## OpenClaw Integration

### Option A: Direct Python Execution
If OpenClaw supports Python sandbox execution:

```yaml
skill:
  name: quant_exec
  type: python_local
  file: openclaw_skills/hmm_regime_detect.py
  function: detect_regimes
```

### Option B: CLI Wrapper
Execute via subprocess:

```python
import subprocess

def run_skill(skill_name, *args):
    result = subprocess.run(
        ["python", f"openclaw_skills/{skill_name}.py", *args],
        capture_output=True, text=True
    )
    return result.stdout
```

## Example Conversations

### HMM Regime Detection
```
User: "What regime is BTC in?"

Agent: 🔍 HMM Regime Analysis: BTCUSDT

Current Regime: BULL (State 2)
Confidence: 87.3%

Regime Probabilities:
- 🐻 Bear: 5.2%
- 📊 Sideways: 12.1%
- 🐂 Bull: 82.7%

Volatility: Normal

📋 Recommendation:
Increase exposure, momentum strategy, trend following
```

### Monte Carlo
```
User: "Simulate 5000 paths for my 0.5 BTC position"

Agent: 🎲 Monte Carlo Simulation: 5,000 Paths

Parameters:
- Initial Price: $50,000
- Expected Daily Return: 0.1%
- Volatility: 3%
- Horizon: 30 days

Key Percentiles:
| Percentile | Price | Return |
|------------|-------|--------|
| P5 (Worst) | $42,100 | -15.8% |
| P50 (Median) | $51,500 | +3.0% |
| P95 (Best) | $62,800 | +25.6% |

Risk Metrics:
- VaR (95%): $3,850 (7.7%)
- Prob. Profit: 57.2%
```

### Portfolio Optimization
```
User: "Optimize my portfolio: BTC, ETH, SOL, SPY"

Agent: 💼 Portfolio Optimization: Max Sharpe

Optimal Weights:
- BTC: 35.0% ████████████████
- ETH: 25.0% ████████████
- SOL: 15.0% ████████
- SPY: 25.0% ████████████

Portfolio Metrics:
- Expected Return (Annual): 24.5%
- Volatility (Annual): 18.2%
- Sharpe Ratio: 1.24
```

## Security Considerations

⚠️ **Important**: Running arbitrary Python code in agents has risks:

1. **Sandboxing**: Use isolated environments
2. **Resource limits**: Cap simulation iterations
3. **Dependencies**: Pin versions
4. **Timeouts**: Prevent infinite loops
5. **Logging**: Track all executions

## Comparison

| Aspect | Level 1-2 (API) | Level 3 (Direct) |
|--------|-----------------|------------------|
| Latency | 100-200ms | 10-50ms |
| Dependencies | None | Full quant stack |
| Security | API sandbox | Code sandbox |
| Flexibility | API limits | Full Python |
| Complexity | Low | Medium |

## When to Use Each Level

- **Level 1-2**: Production, risk-averse, API-first
- **Level 3**: Research, low-latency, custom models
