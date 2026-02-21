"""
Feature Store Module
===================
Centralized feature engineering and storage for ML models.

Author: AI Trading System
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import hashlib
import json
import asyncio
from collections import deque

logger = logging.getLogger(__name__)


class FeatureType(Enum):
    """Feature types."""
    PRICE = "price"
    VOLUME = "volume"
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    SENTIMENT = "sentiment"
    MACRO = "macro"
    DERIVED = "derived"
    LAGGED = "lagged"
    ROLLING = "rolling"


class AggregationType(Enum):
    """Aggregation types for feature computation."""
    SUM = "sum"
    MEAN = "mean"
    MEDIAN = "median"
    STD = "std"
    MIN = "min"
    MAX = "max"
    LAST = "last"
    FIRST = "first"


@dataclass
class FeatureDefinition:
    """Feature definition."""
    name: str
    feature_type: FeatureType
    description: str
    source: str  # e.g., 'price', 'volume', 'technical'
    computation_fn: Optional[Callable] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    
    def compute_hash(self) -> str:
        """Compute unique hash for this feature."""
        config = {
            "name": self.name,
            "type": self.feature_type.value,
            "source": self.source,
            "params": self.parameters,
        }
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]


@dataclass
class FeatureVector:
    """Feature vector with metadata."""
    feature_names: List[str]
    values: np.ndarray
    timestamp: datetime
    symbol: str
    source: str = "unknown"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "source": self.source,
            "features": dict(zip(self.feature_names, self.values.tolist())),
        }


class FeatureComputer:
    """
    Feature Computer
    ==============
    Computes features from raw data.
    """
    
    def __init__(self):
        """Initialize feature computer."""
        self._computations: Dict[str, Callable] = {}
        self._register_default_computations()
    
    def _register_default_computations(self):
        """Register default computation functions."""
        # Price-based features
        self._computations["returns"] = self._compute_returns
        self._computations["log_returns"] = self._compute_log_returns
        self._computations["price_change"] = self._compute_price_change
        
        # Volume-based features
        self._computations["volume_ratio"] = self._compute_volume_ratio
        self._computations["volume_ma_ratio"] = self._compute_volume_ma_ratio
        
        # Technical features
        self._computations["rsi"] = self._compute_rsi
        self._computations["macd"] = self._compute_macd
        self._computations["bollinger_position"] = self._compute_bollinger_position
        
        # Rolling features
        self._computations["rolling_mean"] = self._compute_rolling_mean
        self._computations["rolling_std"] = self._compute_rolling_std
        self._computations["rolling_min"] = self._compute_rolling_min
        self._computations["rolling_max"] = self._compute_rolling_max
    
    def _compute_returns(self, data: pd.Series, **kwargs) -> pd.Series:
        """Compute simple returns."""
        return data.pct_change()
    
    def _compute_log_returns(self, data: pd.Series, **kwargs) -> pd.Series:
        """Compute log returns."""
        return np.log(data / data.shift(1))
    
    def _compute_price_change(self, data: pd.Series, **kwargs) -> pd.Series:
        """Compute absolute price change."""
        return data.diff()
    
    def _compute_volume_ratio(self, data: pd.Series, **kwargs) -> pd.Series:
        """Compute volume ratio vs previous."""
        return data / data.shift(1)
    
    def _compute_volume_ma_ratio(self, data: pd.Series, **kwargs) -> pd.Series:
        """Compute volume vs moving average."""
        window = kwargs.get("window", 20)
        return data / data.rolling(window).mean()
    
    def _compute_rsi(self, data: pd.Series, **kwargs) -> pd.Series:
        """Compute RSI."""
        window = kwargs.get("window", 14)
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _compute_macd(self, data: pd.Series, **kwargs) -> pd.Series:
        """Compute MACD."""
        fast = kwargs.get("fast", 12)
        slow = kwargs.get("slow", 26)
        signal = kwargs.get("signal", 9)
        
        ema_fast = data.ewm(span=fast).mean()
        ema_slow = data.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        
        return macd - signal_line
    
    def _compute_bollinger_position(self, data: pd.Series, **kwargs) -> pd.Series:
        """Compute Bollinger Band position."""
        window = kwargs.get("window", 20)
        num_std = kwargs.get("num_std", 2)
        
        ma = data.rolling(window).mean()
        std = data.rolling(window).std()
        
        upper = ma + num_std * std
        lower = ma - num_std * std
        
        return (data - lower) / (upper - lower)
    
    def _compute_rolling_mean(self, data: pd.Series, **kwargs) -> pd.Series:
        """Compute rolling mean."""
        window = kwargs.get("window", 20)
        return data.rolling(window).mean()
    
    def _compute_rolling_std(self, data: pd.Series, **kwargs) -> pd.Series:
        """Compute rolling standard deviation."""
        window = kwargs.get("window", 20)
        return data.rolling(window).std()
    
    def _compute_rolling_min(self, data: pd.Series, **kwargs) -> pd.Series:
        """Compute rolling minimum."""
        window = kwargs.get("window", 20)
        return data.rolling(window).min()
    
    def _compute_rolling_max(self, data: pd.Series, **kwargs) -> pd.Series:
        """Compute rolling maximum."""
        window = kwargs.get("window", 20)
        return data.rolling(window).max()
    
    def compute(
        self,
        feature_def: FeatureDefinition,
        data: pd.DataFrame,
    ) -> pd.Series:
        """Compute feature from data."""
        computation = self._computations.get(feature_def.name)
        
        if computation is None:
            raise ValueError(f"Unknown computation: {feature_def.name}")
        
        # Get source data
        if feature_def.source not in data.columns:
            raise ValueError(f"Source column not found: {feature_def.source}")
        
        source_data = data[feature_def.source]
        
        # Compute
        result = computation(source_data, **feature_def.parameters)
        
        return result
    
    def register_computation(self, name: str, fn: Callable):
        """Register custom computation function."""
        self._computations[name] = fn
        logger.info(f"Registered computation: {name}")


class FeatureStore:
    """
    Feature Store
    ============
    Centralized feature storage and retrieval.
    """
    
    def __init__(self, max_history: int = 10000):
        """Initialize feature store."""
        self.max_history = max_history
        self._features: Dict[str, Dict[str, deque]] = {}  # symbol -> feature_name -> deque
        self._feature_definitions: Dict[str, FeatureDefinition] = {}
        self._computer = FeatureComputer()
        
        # Online feature cache
        self._online_cache: Dict[str, Dict[str, float]] = {}
        
        logger.info("Feature Store initialized")
    
    def register_feature(self, feature_def: FeatureDefinition):
        """Register a feature definition."""
        self._feature_definitions[feature_def.name] = feature_def
        logger.info(f"Registered feature: {feature_def.name}")
    
    def register_features(self, feature_defs: List[FeatureDefinition]):
        """Register multiple features."""
        for feature_def in feature_defs:
            self.register_feature(feature_def)
    
    def compute_features(
        self,
        data: pd.DataFrame,
        feature_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Compute features from raw data."""
        if feature_names is None:
            feature_names = list(self._feature_definitions.keys())
        
        result = data.copy()
        
        for feature_name in feature_names:
            feature_def = self._feature_definitions.get(feature_name)
            
            if feature_def is None:
                logger.warning(f"Feature not registered: {feature_name}")
                continue
            
            try:
                result[feature_name] = self._computer.compute(feature_def, data)
            except Exception as e:
                logger.error(f"Error computing {feature_name}: {e}")
        
        return result
    
    def store_features(
        self,
        symbol: str,
        feature_vector: FeatureVector,
    ):
        """Store feature vector for symbol."""
        if symbol not in self._features:
            self._features[symbol] = {}
        
        for i, feature_name in enumerate(feature_vector.feature_names):
            if feature_name not in self._features[symbol]:
                self._features[symbol][feature_name] = deque(maxlen=self.max_history)
            
            self._features[symbol][feature_name].append({
                "timestamp": feature_vector.timestamp,
                "value": feature_vector.values[i],
            })
        
        # Update online cache
        self._online_cache[symbol] = dict(zip(
            feature_vector.feature_names,
            feature_vector.values.tolist()
        ))
    
    def get_features(
        self,
        symbol: str,
        feature_names: List[str],
        lookback: Optional[int] = None,
    ) -> pd.DataFrame:
        """Get historical features for symbol."""
        if symbol not in self._features:
            return pd.DataFrame()
        
        result = {"timestamp": []}
        
        for feature_name in feature_names:
            result[feature_name] = []
        
        if symbol in self._features:
            for feature_name in feature_names:
                if feature_name in self._features[symbol]:
                    data = self._features[symbol][feature_name]
                    
                    if lookback:
                        data = list(data)[-lookback:]
                    
                    for item in data:
                        result["timestamp"].append(item["timestamp"])
                        result[feature_name].append(item["value"])
        
        if not result["timestamp"]:
            return pd.DataFrame()
        
        df = pd.DataFrame(result)
        df = df.sort_values("timestamp")
        
        return df
    
    def get_online_features(self, symbol: str) -> Dict[str, float]:
        """Get latest features for symbol."""
        return self._online_cache.get(symbol, {})


# Predefined feature sets
def get_price_features() -> List[FeatureDefinition]:
    """Get price-based features."""
    return [
        FeatureDefinition(
            name="returns",
            feature_type=FeatureType.PRICE,
            description="Simple returns",
            source="close",
            parameters={},
        ),
        FeatureDefinition(
            name="log_returns",
            feature_type=FeatureType.PRICE,
            description="Logarithmic returns",
            source="close",
            parameters={},
        ),
        FeatureDefinition(
            name="price_change",
            feature_type=FeatureType.PRICE,
            description="Absolute price change",
            source="close",
            parameters={},
        ),
    ]


def get_volume_features() -> List[FeatureDefinition]:
    """Get volume-based features."""
    return [
        FeatureDefinition(
            name="volume_ratio",
            feature_type=FeatureType.VOLUME,
            description="Volume ratio vs previous",
            source="volume",
            parameters={},
        ),
        FeatureDefinition(
            name="volume_ma_ratio",
            feature_type=FeatureType.VOLUME,
            description="Volume vs moving average",
            source="volume",
            parameters={"window": 20},
        ),
    ]


def get_technical_features() -> List[FeatureDefinition]:
    """Get technical indicator features."""
    return [
        FeatureDefinition(
            name="rsi",
            feature_type=FeatureType.TECHNICAL,
            description="Relative Strength Index",
            source="close",
            parameters={"window": 14},
        ),
        FeatureDefinition(
            name="macd",
            feature_type=FeatureType.TECHNICAL,
            description="MACD histogram",
            source="close",
            parameters={"fast": 12, "slow": 26, "signal": 9},
        ),
        FeatureDefinition(
            name="bollinger_position",
            feature_type=FeatureType.TECHNICAL,
            description="Bollinger Band position",
            source="close",
            parameters={"window": 20, "num_std": 2},
        ),
    ]


def get_rolling_features() -> List[FeatureDefinition]:
    """Get rolling window features."""
    return [
        FeatureDefinition(
            name="rolling_mean_5",
            feature_type=FeatureType.ROLLING,
            description="5-period rolling mean",
            source="close",
            parameters={"window": 5},
        ),
        FeatureDefinition(
            name="rolling_mean_20",
            feature_type=FeatureType.ROLLING,
            description="20-period rolling mean",
            source="close",
            parameters={"window": 20},
        ),
        FeatureDefinition(
            name="rolling_std_20",
            feature_type=FeatureType.ROLLING,
            description="20-period rolling std",
            source="close",
            parameters={"window": 20},
        ),
    ]


# Default feature store instance
default_feature_store = FeatureStore()

# Register default features
default_feature_store.register_features(get_price_features())
default_feature_store.register_features(get_volume_features())
default_feature_store.register_features(get_technical_features())
default_feature_store.register_features(get_rolling_features())

