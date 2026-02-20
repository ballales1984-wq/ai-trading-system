# AI Trading System - Agent Architecture

## Overview

The AI Trading System implements a **multi-agent architecture** where specialized agents work together to analyze markets, manage risk, and execute trades. This document describes the agent system architecture.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     SUPERVISOR AGENT                            │
│  (Orchestrates all agents, manages system state)               │
└───────────────────────────┬─────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│ MARKET DATA   │   │ MONTE CARLO   │   │    RISK       │
│    AGENT      │   │    AGENT      │   │    AGENT      │
│               │   │               │   │               │
│ - Price fetch │   │ - Simulation  │   │ - VaR/CVaR    │
│ - Streaming   │──▶│ - 5 Levels    │──▶│ - Alerts      │
│ - Caching     │   │ - Forecasting │   │ - Limits      │
└───────────────┘   └───────────────┘   └───────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │     EVENT BUS         │
                │  (Pub/Sub Backbone)   │
                └───────────────────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │   STATE MANAGER       │
                │  (Shared State)       │
                └───────────────────────┘
```

## Core Components

### 1. EventBus

Central event dispatcher for pub/sub communication between agents.

```python
from src.core.event_bus import EventBus, Event, EventType

# Create event bus
bus = EventBus()

# Subscribe to events
async def on_market_data(event: Event):
    print(f"Price update: {event.data}")

bus.subscribe(EventType.MARKET_DATA, on_market_data)

# Publish events
await bus.publish(Event(
    event_type=EventType.MARKET_DATA,
    data={"symbol": "BTCUSDT", "price": 42000}
))
```

### 2. StateManager

Thread-safe shared state management with persistence.

```python
from src.core.state_manager import StateManager

state = StateManager(snapshot_path="data/state.json")

# Set values
state.set("price:BTCUSDT", 42000)

# Get values
price = state.get("price:BTCUSDT")

# Persist to disk
state.snapshot()
```

### 3. BaseAgent

Abstract base class for all agents.

```python
from src.agents.base_agent import BaseAgent

class MyAgent(BaseAgent):
    async def run(self):
        # Main agent logic
        while self._running:
            # Do work
            await asyncio.sleep(1)
    
    async def on_start(self):
        # Initialization
        pass
    
    async def on_stop(self):
        # Cleanup
        pass
```

## Agent Implementations

### MarketDataAgent

Fetches and distributes real-time market data.

```python
from src.agents.agent_marketdata import MarketDataAgent

config = {
    "symbols": ["BTCUSDT", "ETHUSDT"],
    "interval_sec": 5,
    "sources": ["binance", "coingecko"],
}

agent = MarketDataAgent(
    name="market_data",
    event_bus=bus,
    state_manager=state,
    config=config
)

await agent.start()
```

**Features:**
- Multi-symbol support
- Multiple data sources
- Price caching and history
- Automatic reconnection

### MonteCarloAgent

Advanced Monte Carlo simulation engine.

```python
from src.agents.agent_montecarlo import MonteCarloAgent, SimulationLevel

config = {
    "symbols": ["BTCUSDT"],
    "n_paths": 1000,
    "n_steps": 50,
    "levels": [
        SimulationLevel.LEVEL_1_BASE,
        SimulationLevel.LEVEL_2_CONDITIONAL,
    ],
}

agent = MonteCarloAgent(
    name="montecarlo",
    event_bus=bus,
    state_manager=state,
    config=config
)
```

**Simulation Levels:**
1. **Level 1 - Base**: Geometric Brownian Motion
2. **Level 2 - Conditional**: Event-conditioned paths
3. **Level 3 - Adaptive**: RL from past accuracy
4. **Level 4 - Multi-Factor**: Cross-correlations, regime switching
5. **Level 5 - Semantic**: Pattern matching, black swan detection

### RiskAgent

Institutional risk management.

```python
from src.agents.agent_risk import RiskAgent

config = {
    "symbols": ["BTCUSDT"],
    "var_confidence": 0.95,
    "max_var_threshold": 0.05,
    "max_drawdown_threshold": 0.10,
}

agent = RiskAgent(
    name="risk",
    event_bus=bus,
    state_manager=state,
    config=config
)
```

**Risk Metrics:**
- Value at Risk (VaR) - Historical, Parametric, Monte Carlo
- Conditional VaR (CVaR / Expected Shortfall)
- Maximum Drawdown
- Sharpe Ratio
- Sortino Ratio
- Beta

### SupervisorAgent

Orchestrates all agents.

```python
from src.agents.agent_supervisor import SupervisorAgent

config = {
    "mode": "paper_trading",
    "health_check_interval": 30,
}

supervisor = SupervisorAgent(
    name="supervisor",
    event_bus=bus,
    state_manager=state,
    config=config
)

# Register agents
supervisor.register_agent(market_data_agent)
supervisor.register_agent(montecarlo_agent)
supervisor.register_agent(risk_agent)

# Start all agents
await supervisor.start()
```

## Strategy System

### BaseStrategy

Abstract base class for trading strategies.

```python
from src.strategy.base_strategy import BaseStrategy, Signal, SignalAction

class MyStrategy(BaseStrategy):
    def get_required_data(self):
        return ["prices", "volumes"]
    
    def generate_signal(self, context):
        prices = context["prices"]
        
        # Your logic here
        if prices[-1] > prices[-2]:
            return Signal(
                symbol=context["symbol"],
                action=SignalAction.BUY,
                confidence=0.7,
                strength=SignalStrength.MODERATE,
                price=prices[-1],
                quantity=None,
                stop_loss=self.calculate_stop_loss(prices[-1], SignalAction.BUY),
                take_profit=self.calculate_take_profit(prices[-1], SignalAction.BUY),
                reason="Price increase",
                strategy=self.name,
                timestamp=datetime.now(),
            )
        
        return None
```

### MomentumStrategy

Trend-following momentum strategy.

```python
from src.strategy.momentum import MomentumStrategy

config = {
    "lookback_period": 14,
    "momentum_threshold": 0.02,
    "volume_threshold": 1.5,
    "use_ma_filter": True,
    "ma_fast": 10,
    "ma_slow": 30,
}

strategy = MomentumStrategy(name="momentum", config=config)

# Generate signal
signal = strategy.generate_signal({
    "symbol": "BTCUSDT",
    "prices": price_history,
    "volumes": volume_history,
})
```

## AutoML Evolution

### EvolutionEngine

Genetic algorithm for parameter optimization.

```python
from src.automl.evolution import EvolutionEngine, EvolutionConfig, create_param_ranges

config = EvolutionConfig(
    population_size=20,
    elite_size=4,
    mutation_rate=0.15,
    generations=10,
)

engine = EvolutionEngine(config)

# Initialize population
param_ranges = create_param_ranges("momentum")
engine.initialize_population(param_ranges)

# Define evaluation function
def evaluate(params):
    # Backtest with these parameters
    return backtest_score

# Run evolution
best = engine.evolve(evaluate)
print(f"Best params: {best.params}")
print(f"Best fitness: {best.fitness}")
```

## Event Types

| Event Type | Description |
|------------|-------------|
| `MARKET_DATA` | Price/volume updates |
| `SIGNAL_GENERATED` | New trading signal |
| `SIGNAL_EXECUTED` | Signal sent to execution |
| `ORDER_PLACED` | Order submitted |
| `ORDER_FILLED` | Order executed |
| `RISK_ALERT` | Risk threshold breach |
| `ENGINE_STARTED` | Agent started |
| `ENGINE_STOPPED` | Agent stopped |

## Running the System

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run main engine
python main.py

# Run dashboard
python dashboard.py
```

### Docker

```bash
# Build image
docker build -t ai-trading-system .

# Run container
docker run -d \
  -e BINANCE_API_KEY=your_key \
  -e BINANCE_SECRET_KEY=your_secret \
  -p 8000:8000 -p 8050:8050 \
  ai-trading-system
```

### Kubernetes

```bash
# Apply all manifests
kubectl apply -f infra/k8s/

# Check status
kubectl get pods -l app=ai-trading-system

# View logs
kubectl logs -f deployment/ai-trading-system
```

## Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_agents.py -v

# Run with coverage
pytest --cov=src tests/
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `BINANCE_API_KEY` | Binance API key | Required |
| `BINANCE_SECRET_KEY` | Binance secret | Required |
| `TRADING_MODE` | paper/live | paper |
| `LOG_LEVEL` | Logging level | INFO |
| `MAX_POSITION_SIZE` | Max position fraction | 0.1 |
| `RISK_PER_TRADE` | Risk per trade | 0.02 |

## License

Apache 2.0 - See LICENSE file for details.
