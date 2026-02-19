"""
Enhanced ML Model with Feature Importance and Ensemble Methods
=============================================================
Advanced ML utilities for trading signal generation including:
- SHAP-based feature importance
- Permutation importance
- Voting ensemble
- Stacking ensemble
- Weighted ensemble with model confidence

Author: AI Trading System
Version: 2.0.0
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
from datetime import datetime
import warnings
from abc import ABC, abstractmethod

from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
    VotingClassifier,
    StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
from sklearn.inspection import permutation_importance

import joblib
import json
from pathlib import Path


@dataclass
class ModelMetrics:
    """Container for comprehensive model metrics"""
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion_matrix: np.ndarray
    classification_report: str
    training_time: float
    

@dataclass
class FeatureImportanceResult:
    """Container for feature importance analysis"""
    # Built-in feature importance (from tree-based models)
    builtin_importance: Dict[str, float]
    
    # Permutation importance (model-agnostic)
    permutation_importance: Dict[str, float]
    
    # SHAP-based importance (if available)
    shap_importance: Optional[Dict[str, float]] = None
    
    # Aggregated/ranked importance
    aggregated_rank: List[Tuple[str, float]] = field(default_factory=list)
    
    def get_top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get top N most important features"""
        if not self.aggregated_rank:
            # Use builtin if no aggregation
            return sorted(self.builtin_importance.items(), key=lambda x: x[1], reverse=True)[:n]
        return self.aggregated_rank[:n]


@dataclass
class EnsemblePrediction:
    """Container for ensemble prediction results"""
    predictions: np.ndarray
    probabilities: np.ndarray
    individual_predictions: Dict[str, np.ndarray]
    individual_probabilities: Dict[str, np.ndarray]
    model_weights: Dict[str, float]
    confidence: float
    agreement: float  # How much models agree


class BaseMLModel(ABC):
    """Abstract base class for ML models"""
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseMLModel':
        """Fit the model"""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities"""
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance"""
        pass


class EnhancedRandomForest(BaseMLModel):
    """
    Enhanced Random Forest with feature importance analysis.
    """
    
    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 8,
        min_samples_split: int = 10,
        min_samples_leaf: int = 4,
        max_features: str = 'sqrt',
        class_weight: str = 'balanced',
        random_state: int = 42,
        n_jobs: int = -1
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs
        )
        
        self.scaler = RobustScaler()
        self.feature_names: List[str] = []
        self.is_fitted = False
        self._n_classes: int = 2
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'EnhancedRandomForest':
        """Fit the model with scaling"""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self._n_classes = len(np.unique(y))
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def get_feature_importance(self) -> Dict[str, float]:
        if not self.is_fitted:
            return {}
        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance))
    
    def get_permutation_importance(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        n_repeats: int = 10
    ) -> Dict[str, float]:
        """Calculate permutation importance"""
        X_scaled = self.scaler.transform(X)
        result = permutation_importance(
            self.model, X_scaled, y, 
            n_repeats=n_repeats, 
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        return dict(zip(self.feature_names, result.importances_mean))
    
    def set_feature_names(self, names: List[str]):
        self.feature_names = names


class EnhancedGradientBoosting(BaseMLModel):
    """
    Enhanced Gradient Boosting with feature importance.
    """
    
    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        random_state: int = 42
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )
        
        self.scaler = RobustScaler()
        self.feature_names: List[str] = []
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'EnhancedGradientBoosting':
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def get_feature_importance(self) -> Dict[str, float]:
        if not self.is_fitted:
            return {}
        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance))
    
    def set_feature_names(self, names: List[str]):
        self.feature_names = names


class EnhancedExtraTrees(BaseMLModel):
    """
    Extra Trees Classifier - often works well for financial data.
    """
    
    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 10,
        min_samples_split: int = 5,
        class_weight: str = 'balanced',
        random_state: int = 42,
        n_jobs: int = -1
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        self.model = ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs
        )
        
        self.scaler = RobustScaler()
        self.feature_names: List[str] = []
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'EnhancedExtraTrees':
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def get_feature_importance(self) -> Dict[str, float]:
        if not self.is_fitted:
            return {}
        return dict(zip(self.feature_names, self.model.feature_importances_))
    
    def set_feature_names(self, names: List[str]):
        self.feature_names = names


class AdvancedEnsemble:
    """
    Advanced ensemble with multiple strategies:
    - Voting (hard/soft)
    - Stacking
    - Weighted averaging
    - Dynamic weight adjustment
    """
    
    def __init__(
        self,
        models: Optional[List[BaseMLModel]] = None,
        ensemble_type: str = 'weighted',  # 'voting', 'stacking', 'weighted'
        weights: Optional[List[float]] = None,
        use_performance_weights: bool = True
    ):
        self.ensemble_type = ensemble_type
        self.models: List[BaseMLModel] = models or []
        self.model_names: List[str] = []
        self.weights = weights
        self.use_performance_weights = use_performance_weights
        
        # Performance tracking for dynamic weighting
        self.model_performance: Dict[str, float] = {}
        
        # For stacking
        self.meta_learner: Optional[Any] = None
        self.stacking_models: List[BaseMLModel] = []
        
    def add_model(self, model: BaseMLModel, name: str):
        """Add a model to the ensemble"""
        self.models.append(model)
        self.model_names.append(name)
        
    def _compute_performance_weights(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        cv: int = 3
    ) -> List[float]:
        """Compute weights based on cross-validation performance"""
        weights = []
        
        for model in self.models:
            try:
                scores = cross_val_score(
                    model, X, y, cv=cv, scoring='f1'
                )
                avg_score = np.mean(scores)
                self.model_performance[self.model_names[self.models.index(model)]] = avg_score
                weights.append(avg_score)
            except:
                weights.append(0.1)  # Default weight
        
        # Normalize weights
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
        else:
            weights = [1.0 / len(weights)] * len(weights)
            
        return weights
    
    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> 'AdvancedEnsemble':
        """Fit all models in the ensemble"""
        
        if self.ensemble_type == 'stacking':
            # Fit base models
            for model in self.models:
                model.fit(X, y)
            
            # Get base model predictions for meta-learner
            base_preds = []
            for model in self.models:
                if X_val is not None:
                    preds = model.predict_proba(X_val)
                    base_preds.append(preds[:, 1] if preds.shape[1] > 1 else preds)
            
            # Fit meta-learner
            if base_preds:
                meta_features = np.column_stack(base_preds)
                self.meta_learner = LogisticRegression(random_state=42)
                self.meta_learner.fit(meta_features, y_val if y_val is not None else y)
                
        elif self.ensemble_type == 'weighted':
            # Compute performance-based weights if enabled
            if self.use_performance_weights and X_val is not None and y_val is not None:
                self.weights = self._compute_performance_weights(X_val, y_val)
            
            # Fit all models
            for model in self.models:
                model.fit(X, y)
                
        else:  # voting
            for model in self.models:
                model.fit(X, y)
                
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions"""
        
        if self.ensemble_type == 'stacking':
            # Get base model predictions
            base_preds = []
            for model in self.models:
                proba = model.predict_proba(X)
                base_preds.append(proba[:, 1] if proba.shape[1] > 1 else proba)
            
            meta_features = np.column_stack(base_preds)
            return self.meta_learner.predict(meta_features)
            
        elif self.ensemble_type == 'weighted':
            if self.weights is None:
                self.weights = [1.0 / len(self.models)] * len(self.models)
            
            weighted_proba = np.zeros(X.shape[0])
            
            for i, model in enumerate(self.models):
                proba = model.predict_proba(X)
                if proba.shape[1] > 1:
                    weighted_proba += self.weights[i] * proba[:, 1]
                else:
                    weighted_proba += self.weights[i] * proba.flatten()
            
            return (weighted_proba > 0.5).astype(int)
            
        else:  # voting (soft)
            avg_proba = np.zeros(X.shape[0])
            for model in self.models:
                proba = model.predict_proba(X)
                if proba.shape[1] > 1:
                    avg_proba += proba[:, 1]
                else:
                    avg_proba += proba.flatten()
            
            avg_proba /= len(self.models)
            return (avg_proba > 0.5).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities from ensemble"""
        
        if self.ensemble_type == 'weighted':
            if self.weights is None:
                self.weights = [1.0 / len(self.models)] * len(self.models)
            
            weighted_proba = np.zeros((X.shape[0], 2))
            
            for i, model in enumerate(self.models):
                proba = model.predict_proba(X)
                if proba.shape[1] > 1:
                    weighted_proba[:, 1] += self.weights[i] * proba[:, 1]
                    weighted_proba[:, 0] += self.weights[i] * proba[:, 0]
                else:
                    weighted_proba[:, 1] += self.weights[i] * proba.flatten()
                    weighted_proba[:, 0] += self.weights[i] * (1 - proba.flatten())
            
            # Normalize
            weighted_proba /= sum(self.weights)
            return weighted_proba
            
        else:
            # For voting, average probabilities
            avg_proba = np.zeros((X.shape[0], 2))
            for model in self.models:
                proba = model.predict_proba(X)
                if proba.shape[1] > 1:
                    avg_proba += proba
                else:
                    avg_proba[:, 1] += proba.flatten()
                    avg_proba[:, 0] += (1 - proba.flatten())
            
            avg_proba /= len(self.models)
            return avg_proba
    
    def predict_ensemble_detailed(self, X: np.ndarray) -> EnsemblePrediction:
        """Get detailed ensemble predictions with individual model outputs"""
        
        individual_preds = {}
        individual_probas = {}
        
        for name, model in zip(self.model_names, self.models):
            individual_preds[name] = model.predict(X)
            proba = model.predict_proba(X)
            individual_probas[name] = proba[:, 1] if proba.shape[1] > 1 else proba.flatten()
        
        # Calculate agreement
        pred_matrix = np.column_stack(list(individual_preds.values()))
        agreement = np.mean(
            np.all(pred_matrix == pred_matrix[:, [0]], axis=1)
        )
        
        # Calculate confidence (spread of probabilities)
        prob_matrix = np.column_stack(list(individual_probas.values()))
        confidence = 1.0 - np.std(prob_matrix, axis=1)
        
        final_proba = self.predict_proba(X)
        
        return EnsemblePrediction(
            predictions=self.predict(X),
            probabilities=final_proba[:, 1] if final_proba.shape[1] > 1 else final_proba.flatten(),
            individual_predictions=individual_preds,
            individual_probabilities=individual_probas,
            model_weights=dict(zip(self.model_names, self.weights or [1.0/len(self.models)]*len(self.models))),
            confidence=np.mean(confidence),
            agreement=agreement
        )


class FeatureImportanceAnalyzer:
    """
    Comprehensive feature importance analysis.
    """
    
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        
    def analyze(
        self,
        model: BaseMLModel,
        X: np.ndarray,
        y: np.ndarray,
        n_permutation_repeats: int = 10
    ) -> FeatureImportanceResult:
        """Perform comprehensive feature importance analysis"""
        
        # Built-in importance (for tree-based models)
        builtin_importance = model.get_feature_importance()
        
        # Permutation importance
        try:
            perm_importance = self._compute_permutation_importance(
                model, X, y, n_permutation_repeats
            )
        except Exception as e:
            warnings.warn(f"Permutation importance failed: {e}")
            perm_importance = builtin_importance.copy()
        
        # Try SHAP importance (if available)
        shap_importance = None
        try:
            shap_importance = self._compute_shap_importance(model, X)
        except ImportError:
            pass  # SHAP not installed
        
        # Aggregate rankings
        aggregated = self._aggregate_importances(
            builtin_importance, 
            perm_importance, 
            shap_importance
        )
        
        return FeatureImportanceResult(
            builtin_importance=builtin_importance,
            permutation_importance=perm_importance,
            shap_importance=shap_importance,
            aggregated_rank=aggregated
        )
    
    def _compute_permutation_importance(
        self,
        model: BaseMLModel,
        X: np.ndarray,
        y: np.ndarray,
        n_repeats: int = 10
    ) -> Dict[str, float]:
        """Compute permutation importance"""
        result = permutation_importance(
            model, X, y, 
            n_repeats=n_repeats, 
            random_state=42,
            n_jobs=-1
        )
        return dict(zip(self.feature_names, result.importances_mean))
    
    def _compute_shap_importance(
        self,
        model: BaseMLModel,
        X: np.ndarray
    ) -> Optional[Dict[str, float]]:
        """Compute SHAP-based importance"""
        try:
            import shap
            
            # Create explainer based on model type
            if hasattr(model, 'model'):
                explainer = shap.TreeExplainer(model.model)
                shap_values = explainer.shap_values(X)
                
                # Use mean absolute SHAP values
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # For binary classification
                
                mean_abs_shap = np.abs(shap_values).mean(axis=0)
                return dict(zip(self.feature_names, mean_abs_shap))
                
        except Exception:
            return None
    
    def _aggregate_importances(
        self,
        builtin: Dict[str, float],
        permutation: Dict[str, float],
        shap: Optional[Dict[str, float]]
    ) -> List[Tuple[str, float]]:
        """Aggregate multiple importance measures"""
        
        # Normalize each importance measure
        def normalize(d: Dict[str, float]) -> Dict[str, float]:
            if not d:
                return {}
            max_val = max(d.values())
            if max_val > 0:
                return {k: v / max_val for k, v in d.items()}
            return d
        
        norm_builtin = normalize(builtin)
        norm_perm = normalize(permutation)
        norm_shap = normalize(shap) if shap else {}
        
        # Average rankings
        aggregated = {}
        all_features = set(norm_builtin.keys()) | set(norm_perm.keys())
        
        for feat in all_features:
            scores = []
            if feat in norm_builtin:
                scores.append(norm_builtin[feat])
            if feat in norm_perm:
                scores.append(norm_perm[feat])
            if feat in norm_shap:
                scores.append(norm_shap[feat])
            
            if scores:
                aggregated[feat] = np.mean(scores)
        
        # Sort by aggregated importance
        return sorted(aggregated.items(), key=lambda x: x[1], reverse=True)
    
    def get_feature_groups(self) -> Dict[str, List[str]]:
        """Group features by category"""
        groups = {
            'returns': [],
            'technical': [],
            'momentum': [],
            'volume': [],
            'volatility': [],
            'trend': [],
            'other': []
        }
        
        for feat in self.feature_names:
            if 'return' in feat.lower() or 'ret_' in feat.lower():
                groups['returns'].append(feat)
            elif 'rsi' in feat.lower() or 'macd' in feat.lower() or 'bb' in feat.lower():
                groups['technical'].append(feat)
            elif 'momentum' in feat.lower() or 'roc' in feat.lower():
                groups['momentum'].append(feat)
            elif 'volume' in feat.lower() or 'obv' in feat.lower():
                groups['volume'].append(feat)
            elif 'vol' in feat.lower() or 'atr' in feat.lower():
                groups['volatility'].append(feat)
            elif 'sma' in feat.lower() or 'ema' in feat.lower():
                groups['trend'].append(feat)
            else:
                groups['other'].append(feat)
        
        return {k: v for k, v in groups.items() if v}


class MLModelPipeline:
    """
    Complete ML pipeline for trading signal generation.
    """
    
    def __init__(
        self,
        model_type: str = 'ensemble',
        feature_config: Optional[Dict] = None
    ):
        self.model_type = model_type
        self.feature_config = feature_config or {}
        
        # Initialize models
        self.rf = EnhancedRandomForest()
        self.gb = EnhancedGradientBoosting()
        self.et = EnhancedExtraTrees()
        
        # Ensemble
        self.ensemble = AdvancedEnsemble(
            models=[self.rf, self.gb, self.et],
            ensemble_type='weighted'
        )
        
        # Feature importance analyzer
        self.importance_analyzer: Optional[FeatureImportanceAnalyzer] = None
        
        # Training history
        self.training_history: List[Dict] = []
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features from raw data"""
        features = pd.DataFrame(index=df.index)
        
        # Returns features
        if 'close' in df.columns:
            for lag in [1, 2, 3, 5, 10, 20]:
                features[f'ret_{lag}'] = df['close'].pct_change(lag)
            
            features['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # Volatility features
        for window in [5, 10, 20, 50]:
            if 'close' in df.columns:
                features[f'vol_{window}'] = df['close'].pct_change().rolling(window).std()
        
        # Moving averages
        for window in [10, 20, 50, 100, 200]:
            if 'close' in df.columns:
                features[f'sma_{window}'] = df['close'].rolling(window).mean()
                features[f'sma_ratio_{window}'] = df['close'] / features[f'sma_{window}']
        
        # Technical indicators
        if 'rsi' in df.columns:
            features['rsi'] = df['rsi']
            features['rsi_ma'] = df['rsi'].rolling(5).mean()
        
        if 'macd' in df.columns:
            features['macd'] = df['macd']
            features['macd_signal'] = df['macd_signal']
        
        if 'bb_position' in df.columns:
            features['bb_position'] = df['bb_position']
        
        # Momentum
        for period in [5, 10, 20]:
            if 'close' in df.columns:
                features[f'momentum_{period}'] = df['close'].pct_change(period)
        
        # Volume
        if 'volume' in df.columns:
            features['volume'] = df['volume']
            features['volume_ma'] = df['volume'].rolling(5).mean()
            features['volume_ratio'] = df['volume'] / features['volume_ma']
        
        # ATR
        if 'atr' in df.columns:
            features['atr'] = df['atr']
            features['atr_pct'] = df['atr'] / df['close']
        
        # Clean up
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(method='ffill').fillna(0)
        
        return features
    
    def create_target(
        self,
        df: pd.DataFrame,
        horizon: int = 1,
        threshold: float = 0.0
    ) -> pd.Series:
        """Create binary target variable"""
        future_return = df['close'].pct_change(horizon).shift(-horizon)
        return (future_return > threshold).astype(int)
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare train/test data"""
        X = self.prepare_features(df)
        y = self.create_target(df)
        
        # Align
        valid_idx = X.index.intersection(y.index)
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]
        
        # Remove NaN
        valid = ~y.isna()
        X = X[valid]
        y = y[valid]
        
        # Time series split
        split_idx = int(len(X) * (1 - test_size))
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        # Set feature names
        feature_names = X.columns.tolist()
        self.rf.set_feature_names(feature_names)
        self.gb.set_feature_names(feature_names)
        self.et.set_feature_names(feature_names)
        self.importance_analyzer = FeatureImportanceAnalyzer(feature_names)
        
        return X_train.values, X_test.values, y_train.values, y_test.values
    
    def train(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2
    ) -> Dict[str, Any]:
        """Train the complete pipeline"""
        
        X_train, X_test, y_train, y_test = self.prepare_data(df, test_size)
        
        # Train individual models
        import time
        start_time = time.time()
        
        self.rf.fit(X_train, y_train)
        self.gb.fit(X_train, y_train)
        self.et.fit(X_train, y_train)
        
        # Train ensemble
        self.ensemble.fit(X_train, y_train, X_test, y_test)
        
        training_time = time.time() - start_time
        
        # Evaluate
        results = self.evaluate(X_test, y_test)
        results['training_time'] = training_time
        
        # Store history
        self.training_history.append({
            'timestamp': datetime.now().isoformat(),
            'metrics': results,
            'n_features': X_train.shape[1],
            'n_samples': len(y_train)
        })
        
        return results
    
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, Any]:
        """Evaluate models"""
        
        results = {}
        
        # Individual model evaluation
        for name, model in [('rf', self.rf), ('gb', self.gb), ('et', self.et)]:
            y_pred = model.predict(X)
            y_proba = model.predict_proba(X)[:, 1]
            
            results[name] = {
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred, zero_division=0),
                'recall': recall_score(y, y_pred, zero_division=0),
                'f1': f1_score(y, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y, y_proba)
            }
        
        # Ensemble evaluation
        y_pred_ens = self.ensemble.predict(X)
        y_proba_ens = self.ensemble.predict_proba(X)[:, 1]
        
        results['ensemble'] = {
            'accuracy': accuracy_score(y, y_pred_ens),
            'precision': precision_score(y, y_pred_ens, zero_division=0),
            'recall': recall_score(y, y_pred_ens, zero_division=0),
            'f1': f1_score(y, y_pred_ens, zero_division=0),
            'roc_auc': roc_auc_score(y, y_proba_ens),
            'weights': self.ensemble.weights
        }
        
        return results
    
    def analyze_feature_importance(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> FeatureImportanceResult:
        """Analyze feature importance"""
        if self.importance_analyzer is None:
            raise ValueError("Must call train() first")
        
        return self.importance_analyzer.analyze(self.rf, X, y)
    
    def predict_signals(
        self,
        df: pd.DataFrame,
        buy_threshold: float = 0.55,
        sell_threshold: float = 0.45
    ) -> pd.Series:
        """Generate trading signals"""
        
        X = self.prepare_features(df).values
        
        proba = self.ensemble.predict_proba(X)[:, 1]
        
        signals = pd.Series('HOLD', index=df.index)
        signals[proba > buy_threshold] = 'BUY'
        signals[proba < sell_threshold] = 'SELL'
        
        return signals
    
    def save(self, filepath: str):
        """Save pipeline to disk"""
        data = {
            'rf': self.rf,
            'gb': self.gb,
            'et': self.et,
            'ensemble': self.ensemble,
            'feature_config': self.feature_config,
            'training_history': self.training_history
        }
        joblib.dump(data, filepath)
    
    def load(self, filepath: str):
        """Load pipeline from disk"""
        data = joblib.load(filepath)
        self.rf = data['rf']
        self.gb = data['gb']
        self.et = data['et']
        self.ensemble = data['ensemble']
        self.feature_config = data.get('feature_config', {})
        self.training_history = data.get('training_history', [])


def create_ensemble_fitness(
    data: pd.DataFrame,
    initial_capital: float = 10000
) -> callable:
    """
    Create fitness function for strategy optimization.
    
    Args:
        data: Price data
        initial_capital: Starting capital
        
    Returns:
        Fitness function
    """
    
    def fitness(genome: Any) -> float:
        """Calculate fitness score for a genome"""
        try:
            # This is a placeholder - implement actual backtest
            # based on genome parameters
            return 0.5  # Default neutral score
        except:
            return 0.0
    
    return fitness
