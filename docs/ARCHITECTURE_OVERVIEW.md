# Architecture Overview

## System Architecture Diagram

```mermaid
flowchart TB
    subgraph User["User Layer"]
        UI[Web UI / Trading Dashboard]
        API[REST API]
        CLI[CLI Client]
    end

    subgraph OpenClaw["OpenClaw Skills Layer"]
        IR[Intent Router]
        SK1[HMM Regime Detection]
        SK2[GARCH Volatility]
        SK3[Monte Carlo Sim]
        SK4[Portfolio Optimizer]
        CS[Composed Strategies]
    end

    subgraph Decision["Decision Engine"]
        RE[Rule Engine]
        SE[Strategy Executor]
        RB[Risk Book]
    end

    subgraph Execution["Execution Layer"]
        OE[Order Engine]
        BC[Broker Connectors]
        Bin[Binance]
        IB[Interactive Brokers]
    end

    subgraph Data["Data Layer"]
        DB[(PostgreSQL/TimescaleDB)]
        Cache[(Redis Cache)]
        WS[WebSocket Feed]
    end

    subgraph Research["Research & ML"]
        MR[Model Registry]
        DF[Data Feeds]
        Train[Training Pipeline]
    end

    User --> API
    API --> OpenClaw
    OpenClaw --> IR
    IR --> SK1
    IR --> SK2
    IR --> SK3
    IR --> SK4
    CS -.-> IR
    
    OpenClaw --> Decision
    Decision --> RB
    Decision --> SE
    
    SE --> Execution
    Execution --> Bin
    Execution --> IB
    
    Execution --> Data
    Data --> DB
    Data --> Cache
    WS --> Data
    
    Research --> MR
    MR -.-> Decision
    DF --> Research
```

## Data Flow Diagram

```mermaid
sequenceDiagram
    participant U as User
    participant API as REST API
    participant OC as OpenClaw
    participant DE as Decision Engine
    participant RB as Risk Book
    participant EX as Execution
    participant DB as Database

    U->>API: "What regime is BTC?"
    API->>OC: route_intent("regime_analysis", {symbol: "BTC"})
    OC->>OC: IntentRouter.parse()
    OC->>OC: Load HMM skill
    OC->>OC: Execute detect_regimes()
    OC-->>API: {current_state: "bull", confidence: 0.85}
    API-->>U: Display regime analysis

    U->>API: "Optimize portfolio: BTC, ETH, SOL"
    API->>OC: route_intent("portfolio_optimization", {assets: [...]})
    OC-->>API: {weights: {BTC: 0.5, ETH: 0.3, SOL: 0.2}}

    U->>API: "Execute buy order"
    API->>DE: validate_order(order)
    DE->>RB: check_position_limit()
    RB-->>DE: true
    DE->>EX: execute_order()
    EX->>DB: log_order()
    EX-->>API: {order_id: "12345", status: "filled"}
    API-->>U: Order filled!
```

## Component Interaction

```mermaid
flowchart LR
    subgraph Input
        Msg[User Message]
    end

    subgraph Processing
        NLP[Intent Classification]
        Entity[Entity Extraction]
        Skill[Skill Selection]
        Exec[Skill Execution]
    end

    subgraph Output
        Result[Structured Result]
        Action[Trading Action]
    end

    Msg --> NLP
    NLP --> Entity
    Entity --> Skill
    Skill --> Exec
    Exec --> Result
    Result --> Action
```

## Risk Management Flow

```mermaid
flowchart TD
    Start[New Order Request] --> Check1{Position Limit?}
    Check1 -->|No| Check2{Drawdown OK?}
    Check2 -->|No| Reject[Reject Order]
    Check2 -->|Yes| Check3{VaR OK?}
    Check3 -->|No| Reject
    Check3 -->|Yes| Check4{Leverage OK?}
    Check4 -->|No| Reject
    Check4 -->|Yes| Execute[Execute Order]
    
    Reject --> Log[Log & Alert]
    Execute --> Confirm[Confirm Execution]
    Confirm --> Monitor[Monitor Position]
```

## Technology Stack

```mermaid
flowchart TB
    subgraph Frontend["Frontend"]
        React[React + TypeScript]
        Tailwind[Tailwind CSS]
        Recharts[Recharts]
    end

    subgraph Backend["Backend"]
        FastAPI[FastAPI]
        Pydantic[Pydantic]
        SQLAlchemy[SQLAlchemy]
    end

    subgraph Infrastructure["Infrastructure"]
        K8s[Kubernetes]
        Docker[Docker]
        Nginx[Nginx]
        Prometheus[Prometheus]
        Grafana[Grafana]
    end

    subgraph Data["Data Layer"]
        Postgres[PostgreSQL]
        Timescale[TimescaleDB]
        Redis[Redis]
    end
```

---

## Module Dependencies

| Module | Depends On | Used By |
|--------|------------|---------|
| intent_router | skill_registry | API, Composed Strategies |
| skill_registry | registry_config.yaml | All skills |
| risk_book | None | Decision Engine |
| model_registry | None | Research, Decision |
| composed_strategies | intent_router | API, CLI |

---

## Configuration Files

- `openclaw_skills/registry_config.yaml` - Skill definitions
- `openclaw_skills/skill.yaml` - OpenClaw integration
- `src/risk/limits.json` - Risk parameters
- `data/model_registry.json` - ML model versions

---

*Last updated: 2024*
*Version: 3.0*
