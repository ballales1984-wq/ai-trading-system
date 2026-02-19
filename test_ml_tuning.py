"""
Tests for ML Model Tuning Module
================================
Tests hyperparameter optimization functionality.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from unittest.mock import Mock, patch
import warnings

# Suppress warnings during tests
warnings.filterwarnings('ignore')


class TestMLTunerImports:
    """Test module imports."""
    
    def test_import_ml_tuning(self):
        """Test that ml_tuning module can be imported."""
        from src.ml_tuning import MLTuner, TuningResult, optimize_model_for_trading
        assert MLTuner is not None
        assert TuningResult is not None
        assert optimize_model_for_trading is not None
    
    def test_param_grids_defined(self):
        """Test that parameter grids are defined."""
        from src.ml_tuning import MLTuner
        
        assert 'random_forest' in MLTuner.PARAM_GRIDS
        assert 'gradient_boosting' in MLTuner.PARAM_GRIDS
        assert 'extra_trees' in MLTuner.PARAM_GRIDS
        
        # Check random forest has expected params
        rf_grid = MLTuner.PARAM_GRIDS['random_forest']
        assert 'n_estimators' in rf_grid
        assert 'max_depth' in rf_grid


class TestTuningResult:
    """Test TuningResult dataclass."""
    
    def test_tuning_result_creation(self):
        """Test creating a TuningResult."""
        from src.ml_tuning import TuningResult
        
        result = TuningResult(
            model_name='random_forest',
            best_params={'n_estimators': 100, 'max_depth': 10},
            best_score=0.85,
            cv_results={'mean_test_score': [0.8, 0.85]},
            metrics={'accuracy': 0.87, 'f1': 0.86},
            training_time=12.5
        )
        
        assert result.model_name == 'random_forest'
        assert result.best_params['n_estimators'] == 100
        assert result.best_score == 0.85
        assert result.timestamp is not None  # Auto-set
    
    def test_tuning_result_with_timestamp(self):
        """Test TuningResult with explicit timestamp."""
        from src.ml_tuning import TuningResult
        
        ts = datetime(2026, 1, 1, 12, 0, 0)
        result = TuningResult(
            model_name='test',
            best_params={},
            best_score=0.5,
            cv_results={},
            metrics={},
            training_time=1.0,
            timestamp=ts
        )
        
        assert result.timestamp == ts


class TestMLTuner:
    """Test MLTuner class."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample classification data."""
        np.random.seed(42)
        X = np.random.randn(200, 5)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        return X, y
    
    @pytest.fixture
    def tuner(self):
        """Create MLTuner instance."""
        from src.ml_tuning import MLTuner
        return MLTuner(n_splits=3, n_jobs=1, verbose=0)
    
    def test_tuner_initialization(self, tuner):
        """Test MLTuner initialization."""
        assert tuner.n_splits == 3
        assert tuner.scoring == 'f1_weighted'
        assert tuner.n_jobs == 1
        assert tuner.verbose == 0
        assert tuner.random_state == 42
        assert tuner.results == []
    
    def test_get_model(self, tuner):
        """Test _get_model method."""
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        
        rf = tuner._get_model('random_forest')
        assert isinstance(rf, RandomForestClassifier)
        
        gb = tuner._get_model('gradient_boosting')
        assert isinstance(gb, GradientBoostingClassifier)
    
    def test_get_model_invalid(self, tuner):
        """Test _get_model with invalid type."""
        with pytest.raises(ValueError):
            tuner._get_model('invalid_model')
    
    def test_get_param_grid(self, tuner):
        """Test _get_param_grid method."""
        grid = tuner._get_param_grid('random_forest')
        assert 'n_estimators' in grid
        assert 'max_depth' in grid
    
    def test_calculate_metrics(self, tuner, sample_data):
        """Test _calculate_metrics method."""
        X, y = sample_data
        from sklearn.ensemble import RandomForestClassifier
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X[:150], y[:150])
        
        metrics = tuner._calculate_metrics(model, X[150:], y[150:])
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 0 <= metrics['accuracy'] <= 1
    
    def test_tune_model_random_search(self, tuner, sample_data):
        """Test tune_model with random search."""
        X, y = sample_data
        
        # Use small param grid for speed
        param_grid = {
            'n_estimators': [10, 20],
            'max_depth': [3, 5]
        }
        
        result = tuner.tune_model(
            'random_forest', X, y,
            param_grid=param_grid,
            search_type='random',
            n_iter=2
        )
        
        assert result.model_name == 'random_forest'
        assert 'n_estimators' in result.best_params
        assert result.best_score >= 0
        assert len(tuner.results) == 1
    
    def test_tune_model_grid_search(self, tuner, sample_data):
        """Test tune_model with grid search."""
        X, y = sample_data
        
        param_grid = {
            'n_estimators': [10, 20],
            'max_depth': [3, 5]
        }
        
        result = tuner.tune_model(
            'random_forest', X, y,
            param_grid=param_grid,
            search_type='grid'
        )
        
        assert result.model_name == 'random_forest'
        assert result.training_time > 0
    
    def test_get_best_model(self, tuner, sample_data):
        """Test get_best_model method."""
        X, y = sample_data
        
        # Tune multiple models
        param_grid = {'n_estimators': [10], 'max_depth': [3]}
        
        tuner.tune_model('random_forest', X, y, param_grid=param_grid, n_iter=1)
        tuner.tune_model('extra_trees', X, y, param_grid=param_grid, n_iter=1)
        
        best_name, best_result = tuner.get_best_model()
        
        assert best_name in ['random_forest', 'extra_trees']
        assert best_result is not None
    
    def test_get_best_model_empty(self, tuner):
        """Test get_best_model with no results."""
        best_name, best_result = tuner.get_best_model()
        assert best_name is None
        assert best_result is None


class TestOptimizeModelForTrading:
    """Test optimize_model_for_trading function."""
    
    def test_optimize_basic(self):
        """Test basic optimization."""
        from src.ml_tuning import optimize_model_for_trading
        
        # Create sample DataFrame
        np.random.seed(42)
        df = pd.DataFrame({
            'feature1': np.random.randn(150),
            'feature2': np.random.randn(150),
            'feature3': np.random.randn(150),
            'target': np.random.randint(0, 2, 150)
        })
        
        model, result = optimize_model_for_trading(
            df,
            target_col='target',
            model_type='random_forest'
        )
        
        assert model is not None
        assert result is not None
        assert hasattr(model, 'predict')
    
    def test_optimize_with_feature_cols(self):
        """Test optimization with explicit feature columns."""
        from src.ml_tuning import optimize_model_for_trading
        
        np.random.seed(42)
        df = pd.DataFrame({
            'f1': np.random.randn(150),
            'f2': np.random.randn(150),
            'f3': np.random.randn(150),
            'target': np.random.randint(0, 2, 150)
        })
        
        model, result = optimize_model_for_trading(
            df,
            target_col='target',
            feature_cols=['f1', 'f2'],
            model_type='extra_trees'
        )
        
        assert model is not None


class TestSaveResults:
    """Test saving tuning results."""
    
    def test_save_results(self, tmp_path):
        """Test save_results method."""
        from src.ml_tuning import MLTuner, TuningResult
        
        tuner = MLTuner(n_splits=2, n_jobs=1)
        
        # Add a mock result
        tuner.results.append(TuningResult(
            model_name='test_model',
            best_params={'n': 10},
            best_score=0.9,
            cv_results={},
            metrics={'accuracy': 0.9},
            training_time=1.0
        ))
        
        filepath = str(tmp_path / "tuning_results.json")
        tuner.save_results(filepath)
        
        import json
        with open(filepath) as f:
            data = json.load(f)
        
        assert 'results' in data
        assert len(data['results']) == 1
        assert data['results'][0]['model_name'] == 'test_model'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
