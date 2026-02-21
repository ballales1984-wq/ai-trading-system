# AI Trading System - Technical Architecture Overview

## Document Information
- **Version**: 1.0
- **Last Updated**: February 2026
- **Author**: System Architecture Team

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Overview](#2-system-overview)
3. [Architecture Layers](#3-architecture-layers)
4. [Core Components](#4-core-components)
5. [Data Flow Architecture](#5-data-flow-architecture)
6. [API Architecture](#6-api-architecture)
7. [Database Architecture](#7-database-architecture)
8. [Risk Management System](#8-risk-management-system)
9. [Execution Engine](#9-execution-engine)
10. [Strategy Framework](#10-strategy-framework)
11. [Infrastructure & Deployment](#11-infrastructure--deployment)
12. [Security Architecture](#12-security-architecture)
13. [Technology Stack](#13-technology-stack)

---

## 1. Executive Summary

The AI Trading System is a professional-grade, multi-asset trading platform designed for algorithmic trading across cryptocurrency, commodity, and traditional financial markets. The system implements an event-driven architecture with institutional-grade risk management, real-time market data processing, and multi-broker execution capabilities.

### Key Features
- **Multi-Asset Support**: Crypto, forex, stocks, futures, commodities
- **Multi-Strategy Execution**: Momentum, mean-reversion, custom strategies
- **Institutional Risk Management**: VaR, CVaR, Monte Carlo simulations
- **Real-Time Processing**: WebSocket streams, event-driven architecture
- **Multi-Broker Support**: Binance, Interactive Brokers, Bybit, paper trading
- **ML Integration**: XGBoost models, regime detection (HMM), sentiment analysis

---

## 2. System Overview

### High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AI TRADING SYSTEM                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │   Frontend   │    │   REST API   │    │  Dashboard   │                   │
│  │   (React)    │◄──►│  (FastAPI)   │◄──►│   (Dash)     │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│         │                   │                   │                            │
│         └───────────────────┼───────────────────┘                            │
│                             │                                                │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                        CORE ENGINE LAYER                              │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │   │
│  │  │   Trading   │  │   Event     │  │    State    │  │   Signal   │  │   │
│  │  │   Engine    │  │    Bus      │  │  Manager    │  │ Generator  │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                             │                                                │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                       DOMAIN SERVICES LAYER                           │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │   │
│  │  │    Risk     │  │  Portfolio  │  │  Execution  │  │   Market   │  │   │
│  │  │   Engine    │  │   Manager   │  │   Engine    │  │ Data Feed  │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                             │                                                │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                       INFRASTRUCTURE LAYER                            │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │   │
│  │  │  Database   │  │   Broker    │  │   Cache     │  │  Logging   │  │   │
│  │  │ (PostgreSQL)│  │ Connectors  │  │   (Redis)   │  │  System    │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Architecture Layers

### 3.1 Presentation Layer

| Component | Technology | Purpose |
|-----------|------------|---------|
| React Frontend | React + TypeScript | Modern web UI for trading operations |
| Dash Dashboard | Plotly Dash | Real-time analytics and visualization |
| REST API | FastAPI | Programmatic access to all system functions |
| Landing Page | HTML/CSS/JS | Marketing and waitlist management |

### 3.2 Application Layer

| Component | Location | Purpose |
|-----------|----------|---------|
| Trading Engine | [`src/core/engine.py`](src/core/engine.py) | Central orchestrator for all trading operations |
| Event Bus | [`src/core/event_bus.py`](src/core/event_bus.py) | Pub/sub communication between components |
| Decision Engine | [`decision_engine.py`](decision_engine.py) | Signal generation and strategy coordination |
| Auto Trader | [`auto_trader.py`](auto_trader.py) | Automated trading bot |

### 3.3 Domain Layer

| Component | Location | Purpose |
|-----------|----------|---------|
| Risk Engine | [`app/risk/risk_engine.py`](app/risk/risk_engine.py) | VaR, CVaR, position limits |
| Execution Engine | [`app/execution/execution_engine.py`](app/execution/execution_engine.py) | Order execution with retry logic |
| Portfolio Manager | [`app/portfolio/performance.py`](app/portfolio/performance.py) | Position tracking, P&L calculation |
| Strategy Framework | [`app/strategies/`](app/strategies/) | Trading strategy implementations |

### 3.4 Infrastructure Layer

| Component | Location | Purpose |
|-----------|----------|---------|
| Database | [`app/database/`](app/database/) | PostgreSQL/SQLite persistence |
| Broker Connectors | [`app/execution/broker_connector.py`](app/execution/broker_connector.py) | Multi-broker abstraction |
| Cache | [`app/core/cache.py`](app/core/cache.py) | Redis caching layer |
| Logging | [`app/core/logging.py`](app/core/logging.py) | Structured logging system |

---

## 4. Core Components

### 4.1 Trading Engine

The [`TradingEngine`](src/core/engine.py:75) is the central orchestrator that coordinates all trading components through an event-driven architecture.

```python
class TradingEngine:
    """
    Central trading engine orchestrator.
    Coordinates all components via event bus.
    """
    
    def __init__(self, config: EngineConfig):
        self.event_bus = EventBus()
        self.state_manager = StateManager()
        self.broker = None
        self.risk_manager = None
        self.signal_generator = None
        self.portfolio_manager = None
```

**Engine States:**
- `STOPPED` - Engine not running
- `INITIALIZING` - Starting up, connecting to brokers
- `RUNNING` - Active trading mode
- `PAUSED` - Temporarily halted
- `STOPPING` - Graceful shutdown in progress
- `ERROR` - Error state requiring intervention

**Trading Modes:**
- `BACKTEST` - Historical data simulation
- `PAPER` - Simulated trading with real data
- `LIVE` - Real money trading

### 4.2 Event Bus

The [`EventBus`](src/core/event_bus.py:99) provides decoupled communication between all system components.

**Event Types:**

| Category | Events |
|----------|--------|
| Market | `MARKET_DATA`, `TICKER_UPDATE` |
| Signal | `SIGNAL_GENERATED`, `SIGNAL_EXECUTED`, `SIGNAL_REJECTED` |
| Order | `ORDER_PLACED`, `ORDER_FILLED`, `ORDER_CANCELLED`, `ORDER_REJECTED` |
| Position | `POSITION_OPENED`, `POSITION_CLOSED`, `POSITION_UPDATED` |
| Risk | `RISK_CHECK_PASSED`, `RISK_CHECK_FAILED`, `RISK_ALERT`, `EMERGENCY_EXIT` |
| System | `ENGINE_STARTED`, `ENGINE_STOPPED`, `ENGINE_ERROR`, `STATE_SAVED` |

```python
class EventBus:
    """Central event bus for pub/sub communication."""
    
    def subscribe(self, event_type: EventType, handler: EventHandler)
    async def publish(self, event: Event)
    def get_event_history(self, event_type: EventType, limit: int) -> List[Event]
```

### 4.3 Decision Engine

The [`DecisionEngine`](decision_engine.py:146) generates probabilistic trading signals by combining multiple analysis types.

**Signal Components:**
- Technical Analysis Score (RSI, MACD, Bollinger Bands)
- Momentum Score (Price momentum, volume)
- Sentiment Score (News, social media analysis)
- Correlation Score (Cross-asset correlations)
- Volatility Score (ATR, volatility regime)
- ML Score (XGBoost predictions)
- Regime Score (HMM market regime detection)

**5-Question Framework:**
1. **What** - Asset selection based on opportunity scoring
2. **Why** - Macro + sentiment reasoning
3. **How Much** - Position sizing based on risk
4. **When** - Monte Carlo timing analysis
5. **Risk** - Comprehensive risk assessment

---

## 5. Data Flow Architecture

### 5.1 Market Data Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Exchange   │────►│  WebSocket  │────►│  Data Feed  │────►│   Event     │
│   APIs      │     │   Stream    │     │   Manager   │     │    Bus      │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                                                                  │
                         ┌────────────────────────────────────────┘
                         │
                         ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Technical  │────►│  Decision   │────►│    Risk     │────►│  Execution  │
│  Analysis   │     │   Engine    │     │   Engine    │     │   Engine    │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                                                                   │
                         ┌─────────────────────────────────────────┘
                         │
                         ▼
                    ┌─────────────┐
                    │   Broker    │
                    │  Connector  │
                    └─────────────┘
```

### 5.2 Order Execution Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Signal    │────►│   Risk      │────►│  Position   │────►│  Execution  │
│  Generated  │     │   Check     │     │   Sizing    │     │   Engine    │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                                                                   │
                         ┌─────────────────────────────────────────┘
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
    ┌──────────┐  ┌──────────┐  ┌──────────┐
    │ Binance  │  │  Bybit   │  │   IB     │
    │Connector │  │Connector │  │Connector │
    └──────────┘  └──────────┘  └──────────┘
```

---

## 6. API Architecture

### 6.1 REST API Endpoints

The FastAPI application in [`app/main.py`](app/main.py) exposes the following endpoint groups:

| Router | Prefix | Purpose |
|--------|--------|---------|
| Health | `/api/health` | System health checks |
| Orders | `/api/orders` | Order management |
| Portfolio | `/api/portfolio` | Portfolio and positions |
| Strategy | `/api/strategy` | Strategy management |
| Risk | `/api/risk` | Risk metrics and limits |
| Market | `/api/market` | Market data access |
| Waitlist | `/api/waitlist` | User waitlist management |

### 6.2 API Request/Response Models

**Order Creation:**
```python
class OrderCreate(BaseModel):
    symbol: str
    side: str  # BUY or SELL
    order_type: str = "MARKET"
    quantity: float
    price: Optional[float]
    stop_price: Optional[float]
    time_in_force: str = "GTC"
    strategy_id: Optional[str]
    broker: str = "binance"
```

**Portfolio Summary:**
```python
class PortfolioSummary(BaseModel):
    total_value: float
    cash_balance: float
    market_value: float
    total_pnl: float
    unrealized_pnl: float
    realized_pnl: float
    daily_pnl: float
    daily_return_pct: float
    leverage: float
    buying_power: float
    num_positions: int
```

---

## 7. Database Architecture

### 7.1 Database Models

The system uses SQLAlchemy ORM with models defined in [`app/database/models.py`](app/database/models.py).

**Core Tables:**

| Table | Purpose |
|-------|---------|
| `prices` | Historical and real-time OHLCV data |
| `orders` | Trading orders with status tracking |
| `trades` | Executed trade records |
| `positions` | Current position snapshots |
| `signals` | Generated trading signals |
| `portfolio_snapshots` | Portfolio state history |
| `news` | News articles with sentiment scores |
| `macro_events` | Economic calendar events |
| `energy_records` | Energy commodity data from EIA |

**Entity Relationships:**

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   orders    │────►│   trades    │     │  positions  │
│             │     │             │     │             │
│ order_id    │     │ order_id FK │     │ symbol      │
│ symbol      │     │ symbol      │     │ quantity    │
│ side        │     │ price       │     │ entry_price │
│ quantity    │     │ pnl         │     │ unrealized  │
│ status      │     │ timestamp   │     │ leverage    │
└─────────────┘     └─────────────┘     └─────────────┘
       │
       │
       ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   signals   │     │portfolio_   │     │   prices    │
│             │     │ snapshots   │     │             │
│ symbol      │     │             │     │ symbol      │
│ action      │     │ total_equity│     │ timestamp   │
│ confidence  │     │ drawdown    │     │ OHLCV       │
│ executed    │     │ sharpe      │     │ volume      │
└─────────────┘     └─────────────┘     └─────────────┘
```

### 7.2 TimescaleDB Integration

For time-series data, the system supports TimescaleDB hypertables:

```python
# app/database/timescale_models.py
class PriceTick(Base):
    """High-frequency price ticks for TimescaleDB."""
    __tablename__ = "price_ticks"
    
    time = Column(DateTime, primary_key=True)
    symbol = Column(String, primary_key=True)
    price = Column(Float)
    volume = Column(Float)
```

---

## 8. Risk Management System

### 8.1 Risk Engine Architecture

The [`RiskEngine`](app/risk/risk_engine.py:241) provides institutional-grade risk management.

**Risk Metrics Calculated:**

| Metric | Description |
|--------|-------------|
| VaR (95%, 99%) | Value at Risk - maximum expected loss |
| CVaR | Conditional VaR - expected shortfall |
| Max Drawdown | Largest peak-to-trough decline |
| Sharpe Ratio | Risk-adjusted return |
| Sortino Ratio | Downside risk-adjusted return |
| Beta | Market correlation |
| Leverage | Total exposure / equity |

**VaR Calculation Methods:**

```python
class VaRCalculator:
    def historical_var(returns, confidence, horizon) -> float
    def parametric_var(returns, confidence, horizon) -> float
    def monte_carlo_var(returns, confidence, horizon, n_simulations) -> float
    def cvar(returns, confidence, horizon) -> float
```

### 8.2 Risk Limits

```python
class RiskEngine:
    def __init__(
        self,
        max_var_pct: float = 0.02,      # 2% VaR limit
        max_cvar_pct: float = 0.05,     # 5% CVaR limit
        max_leverage: float = 10.0,      # 10x max leverage
        max_position_pct: float = 0.25,  # 25% max position
        max_sector_pct: float = 0.30,    # 30% max sector exposure
    ):
```

### 8.3 Order Risk Check

Every order passes through risk validation:

```python
def check_order_risk(symbol, side, quantity, price, portfolio) -> RiskCheckResult:
    """
    Checks:
    1. Position size limit
    2. VaR impact
    3. Leverage limit
    4. Concentration risk
    """
```

---

## 9. Execution Engine

### 9.1 Broker Connector Pattern

The system uses an abstract broker interface for multi-broker support:

```python
class BrokerConnector(ABC):
    @abstractmethod
    async def connect(self) -> bool
    
    @abstractmethod
    async def place_order(self, order: BrokerOrder) -> BrokerOrder
    
    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> bool
    
    @abstractmethod
    async def get_balance(self) -> List[AccountBalance]
    
    @abstractmethod
    async def get_positions(self) -> List[Position]
```

**Implemented Connectors:**

| Broker | Class | Features |
|--------|-------|----------|
| Binance | [`BinanceConnector`](app/execution/broker_connector.py:184) | Spot, Futures, Testnet |
| Bybit | [`BybitConnector`](app/execution/broker_connector.py:456) | Derivatives, Testnet |
| Interactive Brokers | [`IBConnector`](app/execution/connectors/ib_connector.py) | Stocks, Options, Futures |
| Paper | [`PaperConnector`](app/execution/connectors/paper_connector.py) | Simulation |

### 9.2 Execution Flow

```python
class ExecutionEngine:
    async def execute_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "market",
        validate_risk: bool = True,
    ) -> ExecutionResult:
        """
        1. Validate with risk engine
        2. Create order object
        3. Execute with retry logic
        4. Track execution statistics
        """
```

**Retry Configuration:**
- Max retries: 3
- Base delay: 1.0s
- Max delay: 30.0s
- Exponential backoff with jitter

---

## 10. Strategy Framework

### 10.1 Base Strategy Class

All strategies inherit from [`BaseStrategy`](app/strategies/base_strategy.py:64):

```python
class BaseStrategy(ABC):
    def __init__(self, config: StrategyConfig):
        self.name = config.name
        self.enabled = config.enabled
        self.parameters = config.parameters
    
    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> Optional[TradingSignal]
    
    def calculate_position_size(self, signal, account_balance, risk_pct) -> float
    
    def validate_signal(self, signal: TradingSignal) -> bool
    
    def get_indicators(self, data: pd.DataFrame) -> pd.DataFrame
```

### 10.2 Implemented Strategies

| Strategy | Location | Type |
|----------|----------|------|
| Momentum | [`app/strategies/momentum.py`](app/strategies/momentum.py) | Trend following |
| Mean Reversion | [`app/strategies/mean_reversion.py`](app/strategies/mean_reversion.py) | Counter-trend |
| Multi-Strategy | [`app/strategies/multi_strategy.py`](app/strategies/multi_strategy.py) | Combined |

### 10.3 Signal Structure

```python
@dataclass
class TradingSignal:
    symbol: str
    direction: SignalDirection  # LONG, SHORT, FLAT
    confidence: float  # 0-1
    entry_price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    timestamp: datetime
    strategy_name: str
    metadata: Dict
```

---

## 11. Infrastructure & Deployment

### 11.1 Docker Configuration

**Production Docker Compose:** [`docker-compose.production.yml`](docker-compose.production.yml)

```yaml
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://...
      - REDIS_URL=redis://...
  
  dashboard:
    build: .
    ports:
      - "8050:8050"
  
  postgres:
    image: postgres:15
    volumes:
      - postgres_data:/var/lib/postgresql/data
  
  redis:
    image: redis:7-alpine
```

### 11.2 Directory Structure

```
ai-trading-system/
├── app/                    # FastAPI application
│   ├── api/routes/         # API endpoints
│   ├── core/               # Core utilities
│   ├── database/           # Database models
│   ├── execution/          # Execution engine
│   ├── market_data/        # Market data handling
│   ├── portfolio/          # Portfolio management
│   ├── risk/               # Risk management
│   └── strategies/         # Trading strategies
├── src/                    # Core trading engine
│   └── core/               # Engine, event bus, state
├── dashboard/              # Dash dashboard
├── frontend/               # React frontend
├── data/                   # Data storage
├── logs/                   # Log files
├── tests/                  # Test suite
└── config.py               # Configuration
```

---

## 12. Security Architecture

### 12.1 Authentication & Authorization

```python
# app/core/rbac.py
class RoleBasedAccessControl:
    ROLES = {
        "admin": ["*"],
        "trader": ["read", "trade"],
        "analyst": ["read"],
        "viewer": ["read:portfolio"]
    }
```

### 12.2 API Security

- **Rate Limiting**: [`app/core/rate_limiter.py`](app/core/rate_limiter.py)
- **Input Validation**: Pydantic models
- **CORS**: Configured origins
- **API Key Management**: Environment variables

### 12.3 Secrets Management

```python
# Environment variables (never hardcoded)
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY')
DATABASE_URL = os.getenv('DATABASE_URL')
```

---

## 13. Technology Stack

### 13.1 Backend

| Component | Technology | Version |
|-----------|------------|---------|
| Language | Python | 3.11+ |
| Web Framework | FastAPI | 0.100+ |
| ORM | SQLAlchemy | 2.0+ |
| Database | PostgreSQL / TimescaleDB | 15+ |
| Cache | Redis | 7+ |
| Task Queue | asyncio | Built-in |

### 13.2 Data Science

| Component | Technology |
|-----------|------------|
| Data Processing | Pandas, NumPy |
| Machine Learning | XGBoost, scikit-learn |
| Statistical Analysis | SciPy |
| Visualization | Plotly, Matplotlib |

### 13.3 Frontend

| Component | Technology |
|-----------|------------|
| Web UI | React + TypeScript |
| Dashboard | Plotly Dash |
| Styling | Tailwind CSS |

### 13.4 Infrastructure

| Component | Technology |
|-----------|------------|
| Containerization | Docker |
| Orchestration | Docker Compose |
| Logging | Python logging + structured JSON |
| Monitoring | Built-in health checks |

---

## Appendix A: Configuration Reference

Key configuration options from [`config.py`](config.py):

```python
# Trading Configuration
DEFAULT_EXCHANGE = 'binance'
DEFAULT_TIMEFRAME = '1h'
MAX_CANDLES = 500

# Risk Management
DECISION_SETTINGS = {
    'min_signal_confidence': 0.55,
    'max_position_size': 0.1,
    'stop_loss_percent': 0.02,
    'take_profit_percent': 0.05,
}

# Technical Indicators
INDICATOR_SETTINGS = {
    'rsi_period': 14,
    'ema_short': 12,
    'ema_medium': 26,
    'bb_period': 20,
}
```

---

## Appendix B: Key Files Reference

| File | Purpose |
|------|---------|
| [`main.py`](main.py) | CLI entry point |
| [`app/main.py`](app/main.py) | FastAPI application |
| [`src/core/engine.py`](src/core/engine.py) | Trading engine |
| [`src/core/event_bus.py`](src/core/event_bus.py) | Event system |
| [`decision_engine.py`](decision_engine.py) | Signal generation |
| [`app/risk/risk_engine.py`](app/risk/risk_engine.py) | Risk management |
| [`app/execution/broker_connector.py`](app/execution/broker_connector.py) | Broker abstraction |
| [`app/database/models.py`](app/database/models.py) | Data models |
| [`config.py`](config.py) | System configuration |

---

*Document generated from codebase analysis - AI Trading System © 2026*
