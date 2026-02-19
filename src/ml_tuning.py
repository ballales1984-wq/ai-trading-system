# src/ml_tuning.py
"""
ML Model Tuning and Optimization
================================
Hyperparameter optimization for trading ML models.

Features:
- Grid search with cross-validation
- Time-series aware cross-validation
- Feature importance analysis
- Model performance tracking
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import logging
import warnings

from sklearn.model_selection import (
    GridSearchCV, 
    RandomizedSearchCV,
    TimeSeriesSplit,
    cross_val_score
)
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

logger = logging.getLogger(__name__)


@dataclass
class TuningResult:
    """Container for tuning results."""
    model_name: str
    best_params: Dict[str, Any]
    best_score: float
    cv_results: Dict[str, Any]
    metrics: Dict[str, float]
    training_time: float
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class MLTuner:
    """
    ML Model Hyperparameter Tuner.
    
    Optimizes trading ML models using time-series cross-validation.
    """
    
    # Default parameter grids for different model types
    PARAM_GRIDS = {
        'random_forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        },
        'gradient_boosting': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'subsample': [0.8, 0.9, 1.0]
        },
        'extra_trees': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    }
    
    def __init__(
        self,
        n_splits: int = 5,
        scoring: str = 'f1_weighted',
        n_jobs: int = -1,
        verbose: int = 1,
        random_state: int = 42
    ):
        """
        Initialize ML Tuner.
        
        Args:
            n_splits: Number of cross-validation splits
            scoring: Scoring metric for optimization
            n_jobs: Number of parallel jobs
            verbose: Verbosity level
            random_state: Random seed for reproducibility
        """
        self.n_splits = n_splits
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state
        self.results: List[TuningResult] = []
        
    def _get_model(self, model_type: str):
        """Get model instance by type."""
        models = {
            'random_forest': RandomForestClassifier(random_state=self.random_state),
            'gradient_boosting': GradientBoostingClassifier(random_state=self.random_state),
            'extra_trees': ExtraTreesClassifier(random_state=self.random_state)
        }
        
        if model_type not in models:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(models.keys())}")
        
        return models[model_type]
    
    def _get_param_grid(self, model_type: str) -> Dict:
        """Get parameter grid for model type."""
        if model_type not in self.PARAM_GRIDS:
            raise ValueError(f"No param grid for: {model_type}")
        return self.PARAM_GRIDS[model_type]
    
    def _calculate_metrics(
        self, 
        model, 
        X_test: np.ndarray, 
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        y_pred = model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
        
        # Add ROC AUC for binary classification
        try:
            if len(np.unique(y_test)) == 2:
                y_proba = model.predict_proba(X_test)[:, 1]
                metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
        except Exception:
            pass
            
        return metrics
    
    def tune_model(
        self,
        model_type: str,
        X: np.ndarray,
        y: np.ndarray,
        param_grid: Optional[Dict] = None,
        search_type: str = 'random',
        n_iter: int = 50
    ) -> TuningResult:
        """
        Tune a model using cross-validation.
        
        Args:
            model_type: Type of model ('random_forest', 'gradient_boosting', 'extra_trees')
            X: Feature matrix
            y: Target labels
            param_grid: Custom parameter grid (optional)
            search_type: 'grid' or 'random'
            n_iter: Number of iterations for random search
            
        Returns:
            TuningResult with best parameters and metrics
        """
        import time
        start_time = time.time()
        
        # Get model and param grid
        model = self._get_model(model_type)
        if param_grid is None:
            param_grid = self._get_param_grid(model_type)
        
        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        
        # Perform search
        if search_type == 'grid':
            search = GridSearchCV(
                model,
                param_grid,
                cv=tscv,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                verbose=self.verbose
            )
        else:
            search = RandomizedSearchCV(
                model,
                param_grid,
                n_iter=n_iter,
                cv=tscv,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                random_state=self.random_state
            )
        
        # Fit search
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            search.fit(X, y)
        
        # Calculate metrics on best model
        # Use last split for test
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        best_model = search.best_estimator_
        best_model.fit(X_train, y_train)
        metrics = self._calculate_metrics(best_model, X_test, y_test)
        
        training_time = time.time() - start_time
        
        result = TuningResult(
            model_name=model_type,
            best_params=search.best_params_,
            best_score=search.best_score_,
            cv_results={
                'mean_test_score': search.cv_results_['mean_test_score'].tolist(),
                'std_test_score': search.cv_results_['std_test_score'].tolist()
            },
            metrics=metrics,
            training_time=training_time
        )
        
        self.results.append(result)
        
        logger.info(
            f"Tuned {model_type}: best_score={search.best_score_:.4f}, "
            f"params={search.best_params_}, time={training_time:.2f}s"
        )
        
        return result
    
    def tune_all_models(
        self,
        X: np.ndarray,
        y: np.ndarray,
        search_type: str = 'random',
        n_iter: int = 30
    ) -> Dict[str, TuningResult]:
        """
        Tune all available models.
        
        Args:
            X: Feature matrix
            y: Target labels
            search_type: 'grid' or 'random'
            n_iter: Iterations for random search
            
        Returns:
            Dictionary of model_name -> TuningResult
        """
        results = {}
        
        for model_type in self.PARAM_GRIDS.keys():
            logger.info(f"Tuning {model_type}...")
            try:
                result = self.tune_model(
                    model_type, X, y,
                    search_type=search_type,
                    n_iter=n_iter
                )
                results[model_type] = result
            except Exception as e:
                logger.error(f"Failed to tune {model_type}: {e}")
        
        return results
    
    def get_best_model(self) -> Tuple[str, TuningResult]:
        """Get the best performing model from all tuning results."""
        if not self.results:
            return None, None
        
        best = max(self.results, key=lambda r: r.best_score)
        return best.model_name, best
    
    def save_results(self, filepath: str):
        """Save tuning results to JSON."""
        import json
        
        data = {
            'results': [
                {
                    'model_name': r.model_name,
                    'best_params': r.best_params,
                    'best_score': r.best_score,
                    'metrics': r.metrics,
                    'training_time': r.training_time,
                    'timestamp': r.timestamp.isoformat() if r.timestamp else None
                }
                for r in self.results
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved tuning results to {filepath}")


def optimize_model_for_trading(
    df: pd.DataFrame,
    target_col: str = 'target',
    feature_cols: Optional[List[str]] = None,
    model_type: str = 'random_forest'
) -> Tuple[Any, TuningResult]:
    """
    Convenience function to optimize a model for trading.
    
    Args:
        df: DataFrame with features and target
        target_col: Name of target column
        feature_cols: List of feature columns (optional, uses all except target)
        model_type: Type of model to optimize
        
    Returns:
        Tuple of (trained_model, tuning_result)
    """
    # Prepare data
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_col]
    
    X = df[feature_cols].values
    y = df[target_col].values
    
    # Remove NaN
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[mask]
    y = y[mask]
    
    # Tune model
    tuner = MLTuner(n_splits=5)
    result = tuner.tune_model(model_type, X, y, n_iter=20)
    
    # Train final model with best params
    if model_type == 'random_forest':
        model = RandomForestClassifier(**result.best_params, random_state=42)
    elif model_type == 'gradient_boosting':
        model = GradientBoostingClassifier(**result.best_params, random_state=42)
    elif model_type == 'extra_trees':
        model = ExtraTreesClassifier(**result.best_params, random_state=42)
    
    model.fit(X, y)
    
    return model, result


if __name__ == "__main__":
    # Example usage
    print("ML Model Tuning Module")
    print("=" * 50)
    print("\nAvailable model types:")
    for mt in MLTuner.PARAM_GRIDS.keys():
        print(f"  - {mt}")
    print("\nUsage:")
    print("  tuner = MLTuner(n_splits=5)")
    print("  result = tuner.tune_model('random_forest', X, y)")
    print("  best_name, best_result = tuner.get_best_model()")
