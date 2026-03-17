"""
Model Registry Module
=====================
Centralized model versioning and lifecycle management.

This module provides a ModelRegistry class for:
- Tracking model versions and metadata
- Managing champion/challenger/retired model states
- Persisting model metadata to JSON
- Listing and querying models by status

Usage:
    from src.research.model_registry import ModelRegistry, ModelMeta
    
    registry = ModelRegistry("data/model_registry.json")
    
    # Register a new model
    meta = ModelMeta(
        name="hmm_regime_detector",
        version="1.0.0",
        trained_at=datetime.utcnow(),
        dataset_id="BTC_USDT_1H_90D",
        metrics={"accuracy": 0.85, "sharpe": 1.2},
        status="champion"
    )
    registry.register(meta)
    
    # Get champion model
    champion = registry.get_champion("hmm_regime_detector")
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
from pathlib import Path


class ModelStatus(str):
    """Model status enumeration."""
    CHAMPION = "champion"
    CHALLENGER = "challenger"
    RETIRED = "retired"
    TRAINING = "training"


@dataclass
class ModelMeta:
    """
    Model metadata container.
    
    Attributes:
        name: Model name/identifier
        version: Semantic version string
        trained_at: Training completion timestamp
        dataset_id: ID of training dataset
        metrics: Performance metrics dictionary
        status: Model status (champion/challenger/retired/training)
        description: Optional model description
        hyperparameters: Optional hyperparameter configuration
    """
    name: str
    version: str
    trained_at: datetime
    dataset_id: str
    metrics: Dict[str, float]
    status: str = "challenger"
    description: Optional[str] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate model metadata."""
        if self.status not in ["champion", "challenger", "retired", "training"]:
            raise ValueError(f"Invalid status: {self.status}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = {
            "name": self.name,
            "version": self.version,
            "trained_at": self.trained_at.isoformat(),
            "dataset_id": self.dataset_id,
            "metrics": self.metrics,
            "status": self.status,
        }
        if self.description:
            data["description"] = self.description
        if self.hyperparameters:
            data["hyperparameters"] = self.hyperparameters
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMeta":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            version=data["version"],
            trained_at=datetime.fromisoformat(data["trained_at"]),
            dataset_id=data["dataset_id"],
            metrics=data["metrics"],
            status=data.get("status", "challenger"),
            description=data.get("description"),
            hyperparameters=data.get("hyperparameters"),
        )


class ModelRegistry:
    """
    Centralized model registry for version management.
    
    Provides:
    - Persistent storage of model metadata
    - Champion/challenger model tracking
    - Model promotion and retirement workflows
    """
    
    def __init__(self, path: str = "data/model_registry.json"):
        """
        Initialize ModelRegistry.
        
        Args:
            path: Path to JSON file for persistence
        """
        self.path = Path(path)
        self._models: Dict[str, ModelMeta] = {}
        self._load()
    
    def _load(self) -> None:
        """Load models from persistent storage."""
        if not self.path.exists():
            return
        
        try:
            raw = json.loads(self.path.read_text())
            for key, value in raw.items():
                self._models[key] = ModelMeta.from_dict(value)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Warning: Failed to load model registry: {e}")
    
    def _save(self) -> None:
        """Save models to persistent storage."""
        # Ensure directory exists
        self.path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            key: model.to_dict()
            for key, model in self._models.items()
        }
        
        self.path.write_text(json.dumps(data, indent=2))
    
    def _make_key(self, name: str, version: str) -> str:
        """Create unique key for model."""
        return f"{name}:{version}"
    
    def register(self, meta: ModelMeta) -> None:
        """
        Register a new model version.
        
        Args:
            meta: Model metadata
        """
        key = self._make_key(meta.name, meta.version)
        self._models[key] = meta
        self._save()
    
    def get(self, name: str, version: str) -> Optional[ModelMeta]:
        """
        Get a specific model version.
        
        Args:
            name: Model name
            version: Model version
            
        Returns:
            ModelMeta if found, None otherwise
        """
        key = self._make_key(name, version)
        return self._models.get(key)
    
    def get_champion(self, name: str) -> Optional[ModelMeta]:
        """
        Get the champion model for a given name.
        
        The champion is the most recently trained model with status "champion".
        
        Args:
            name: Model name
            
        Returns:
            Champion ModelMeta if exists, None otherwise
        """
        candidates = [
            m for m in self._models.values() 
            if m.name == name and m.status == ModelStatus.CHAMPION
        ]
        
        if not candidates:
            return None
        
        return max(candidates, key=lambda m: m.trained_at)
    
    def get_challengers(self, name: str) -> List[ModelMeta]:
        """
        Get all challenger models for a given name.
        
        Args:
            name: Model name
            
        Returns:
            List of challenger models
        """
        return [
            m for m in self._models.values()
            if m.name == name and m.status == ModelStatus.CHALLENGER
        ]
    
    def promote_to_champion(self, name: str, version: str) -> bool:
        """
        Promote a model to champion status.
        
        This will demote any existing champion to challenger.
        
        Args:
            name: Model name
            version: Model version
            
        Returns:
            True if promotion successful, False if model not found
        """
        key = self._make_key(name, version)
        
        if key not in self._models:
            return False
        
        # Demote existing champion
        for model in self._models.values():
            if model.name == name and model.status == ModelStatus.CHAMPION:
                model.status = ModelStatus.CHALLENGER
        
        # Promote new champion
        self._models[key].status = ModelStatus.CHAMPION
        self._save()
        
        return True
    
    def retire_model(self, name: str, version: str) -> bool:
        """
        Retire a model version.
        
        Args:
            name: Model name
            version: Model version
            
        Returns:
            True if retirement successful, False if model not found
        """
        key = self._make_key(name, version)
        
        if key not in self._models:
            return False
        
        self._models[key].status = ModelStatus.RETIRED
        self._save()
        
        return True
    
    def list_models(
        self, 
        name: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[ModelMeta]:
        """
        List models with optional filtering.
        
        Args:
            name: Optional name filter
            status: Optional status filter
            
        Returns:
            List of matching models
        """
        models = list(self._models.values())
        
        if name:
            models = [m for m in models if m.name == name]
        
        if status:
            models = [m for m in models if m.status == status]
        
        # Sort by trained_at descending (most recent first)
        models.sort(key=lambda m: m.trained_at, reverse=True)
        
        return models
    
    def get_all_names(self) -> List[str]:
        """
        Get all unique model names.
        
        Returns:
            List of unique model names
        """
        return list(set(m.name for m in self._models.values()))
    
    def get_metrics_summary(self, name: str) -> Dict[str, Any]:
        """
        Get metrics summary for a model.
        
        Args:
            name: Model name
            
        Returns:
            Dictionary with metrics from champion and challenger
        """
        champion = self.get_champion(name)
        challengers = self.get_challengers(name)
        
        result = {
            "name": name,
            "champion": champion.metrics if champion else None,
            "champion_version": champion.version if champion else None,
            "challengers": [
                {"version": c.version, "metrics": c.metrics}
                for c in challengers
            ],
            "total_versions": len(self._models),
        }
        
        return result
    
    def delete(self, name: str, version: str) -> bool:
        """
        Delete a model from the registry.
        
        Args:
            name: Model name
            version: Model version
            
        Returns:
            True if deletion successful
        """
        key = self._make_key(name, version)
        
        if key in self._models:
            del self._models[key]
            self._save()
            return True
        
        return False
    
    def __repr__(self) -> str:
        names = self.get_all_names()
        return f"ModelRegistry({len(names)} models: {', '.join(names)})"


# Convenience function for quick access
_default_registry: Optional[ModelRegistry] = None


def get_model_registry(path: str = "data/model_registry.json") -> ModelRegistry:
    """
    Get the default model registry instance.
    
    Args:
        path: Path to registry file
        
    Returns:
        ModelRegistry instance
    """
    global _default_registry
    
    if _default_registry is None:
        _default_registry = ModelRegistry(path)
    
    return _default_registry
