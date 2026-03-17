# Model Registry Documentation

## Overview

The **Model Registry** provides versioning and lifecycle management for ML models. It supports the **Champion/Challenger** pattern for A/B testing and gradual rollout.

## Location

- **Module**: `src/research/model_registry.py`
- **Integration**: Used by `UnifiedDecisionEngine` and `AutonomousQuantAgent`

## Features

### Model Versioning

- Track multiple versions of the same model
- Store metadata (training date, metrics, parameters)
- Compare performance across versions

### Champion/Challenger

- Designate a "champion" model for production
- Register "challenger" models for testing
- Promote challengers based on performance

### Model Lifecycle

```
Draft → Training → Validating → Staged → Champion/Challenger → Archived
```

## Usage

### Basic Usage

```python
from src.research.model_registry import ModelRegistry, ModelMeta

registry = ModelRegistry()

# Register a new model version
meta = ModelMeta(
    name="price_prediction",
    version="1.0.0",
    model_type="gradient_boosting",
    metrics={
        "mape": 0.05,
        "rmse": 200,
        "sharpe": 1.2
    },
    training_date="2026-01-15",
    parameters={
        "n_estimators": 100,
        "learning_rate": 0.1
    }
)

model_id = registry.register_model(meta)
```

### Champion/Challenger

```python
# Register challenger
challenger_meta = ModelMeta(
    name="price_prediction",
    version="2.0.0",
    model_type="neural_network",
    metrics={"mape": 0.03, "rmse": 150},
    ...
)
registry.register_model(challenger_meta)

# Promote challenger to champion
registry.promote_to_champion("price_prediction", "2.0.0")

# Get champion for inference
champion = registry.get_champion("price_prediction")
```

### Model Comparison

```python
# Compare all versions
versions = registry.get_versions("price_prediction")

for v in versions:
    print(f"{v.version}: MAPE={v.metrics['mape']}")
```

## Data Structure

### ModelMeta

```python
@dataclass
class ModelMeta:
    name: str                    # Model name
    version: str                 # Semantic version
    model_type: str              # Model type (e.g., "xgboost", "lstm")
    status: ModelStatus          # Lifecycle status
    metrics: Dict[str, float]    # Performance metrics
    training_date: str          # Training date
    parameters: Dict[str, Any]   # Model parameters
    description: str = ""        # Optional description
    tags: List[str] = field(default_factory=list)  # Tags
```

### ModelStatus

| Status | Description |
|--------|-------------|
| `draft` | Initial draft |
| `training` | Currently training |
| `validating` | In validation |
| `staged` | Ready for testing |
| `champion` | Production model |
| `challenger` | A/B testing |
| `archived` | Deprecated |

## Integration with ML Pipeline

### Training Script

```python
# In train_ml.py
from src.research.model_registry import ModelRegistry, ModelMeta

def train_and_register():
    # Train model
    model = train_model(...)
    
    # Evaluate
    metrics = evaluate(model, test_data)
    
    # Register
    meta = ModelMeta(
        name="price_prediction",
        version="2.1.0",
        metrics=metrics,
        ...
    )
    registry.register_model(meta)
    
    # Auto-promote if better than champion
    if is_better_than_champion(metrics):
        registry.promote_to_champion("price_prediction", "2.1.0")
```

### Inference

```python
# In ml_predictor.py or decision engine
from src.research.model_registry import ModelRegistry

registry = ModelRegistry()
champion = registry.get_champion("price_prediction")

if champion:
    load_model(champion.model_path)
else:
    # Fallback to default
    load_default_model()
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/models` | GET | List all models |
| `/api/v1/models/{name}` | GET | Get model versions |
| `/api/v1/models/{name}/champion` | GET | Get champion |
| `/api/v1/models` | POST | Register model |
| `/api/v1/models/{name}/promote` | POST | Promote to champion |
| `/api/v1/models/{name}/archive` | POST | Archive model |

## Monitoring (Grafana)

| Metric | Type | Description |
|--------|------|-------------|
| `model_registry_total_models` | Gauge | Total registered models |
| `model_registry_champion_versions` | Gauge | Champion version per model |
| `model_registry_promotions_total` | Counter | Total promotions |
| `model_registry_inferences_total` | Counter | Inferences by model |

## Best Practices

1. **Always evaluate** before promoting to champion
2. **Keep baselines** - don't remove old models immediately
3. **Track metadata** - training date, parameters, dataset
4. **Use semantic versioning** - MAJOR.MINOR.PATCH
5. **Monitor drift** - compare live vs backtest performance

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Model Registry                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   price_prediction                                          │
│   ├── v1.0.0 (archived)                                     │
│   ├── v1.5.0 (challenger)  ←── Current challenger          │
│   └── v2.0.0 (champion)     ←── Production model           │
│                                                              │
│   sentiment_analysis                                        │
│   ├── v1.0.0 (champion)                                    │
│   └── v1.1.0 (staged)                                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
            │                           │
            ▼                           ▼
    ┌───────────────┐          ┌───────────────┐
    │  ML Training  │          │   Inference   │
    │    Pipeline   │          │    Engine     │
    └───────────────┘          └───────────────┘
```
