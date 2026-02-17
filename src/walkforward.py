"""
Walk-Forward Optimization Module
================================
Professional walk-forward validation for time series models.
- Rolling window optimization
- Expanding window validation
- Performance tracking across windows

Author: AI Trading System
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from datetime import datetime


@dataclass
class WindowResult:
    """Result for a single walk-forward window"""
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    train_score: float
    test_score: float
    num_trades: int
    return_pct: float
    max_drawdown: float


class WalkForwardOptimizer:
    """
    Walk-forward optimization for time series models.
    
    Implements:
    - Rolling window (fixed size train/test)
    - Expanding window (growing train set)
    - Purged walk-forward (avoid look-ahead bias)
    """
    
    def __init__(
        self,
        train_size: float = 0.7,
        n_windows: Optional[int] = None,
        step_size: Optional[int] = None
    ):
        """
        Initialize walk-forward optimizer.
        
        Parameters:
        -----------
        train_size : float
            Proportion of each window for training
        n_windows : int, optional
            Number of windows (if None, calculated from data)
        step_size : int, optional
            Step size between windows (default: test size)
        """
        self.train_size = train_size
        self.n_windows = n_windows
        self.step_size = step_size
        self.results: List[WindowResult] = []
    
    def rolling_walk_forward(
        self,
        df: pd.DataFrame,
        model_factory: Callable,
        metric_func: Optional[Callable] = None
    ) -> Tuple[pd.DataFrame, List[WindowResult]]:
        """
        Run rolling walk-forward optimization.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Full dataset
        model_factory : callable
            Function that creates and returns a fitted model
        metric_func : callable, optional
            Function to calculate custom metric
        
        Returns:
        --------
        Tuple[pd.DataFrame, List[WindowResult]] : (predictions, results)
        """
        n = len(df)
        
        # Calculate window sizes
        window_size = n // (self.n_windows or 5)
        train_size = int(window_size * self.train_size)
        test_size = window_size - train_size
        
        if self.step_size is None:
            self.step_size = test_size
        
        all_predictions = []
        self.results = []
        
        for start in range(0, n - window_size, self.step_size):
            train_end = start + train_size
            test_end = min(start + window_size, n)
            
            # Split data
            train_df = df.iloc[start:train_end]
            test_df = df.iloc[train_end:test_end]
            
            if len(train_df) < 50 or len(test_df) < 10:
                continue
            
            # Train model
            try:
                model = model_factory()
                
                # Fit model (assumes model has fit method)
                if hasattr(model, 'fit'):
                    model.fit(train_df)
                
                # Get predictions
                if hasattr(model, 'predict'):
                    predictions = model.predict(test_df)
                elif hasattr(model, 'predict_proba'):
                    predictions = model.predict_proba(test_df)[:, 1]
                else:
                    continue
                
                # Store predictions
                test_df_copy = test_df.copy()
                test_df_copy['prediction'] = predictions
                all_predictions.append(test_df_copy)
                
                # Calculate metrics
                if metric_func:
                    score = metric_func(test_df, predictions)
                else:
                    # Default: accuracy for classification
                    if hasattr(model, 'model'):
                        y_true = (test_df['close'].pct_change().shift(-1) > 0).dropna()
                        y_pred = pd.Series(predictions[:len(y_true)], index=y_true.index)
                        score = (y_true == y_pred).mean()
                    else:
                        score = 0.5
                
                # Calculate returns
                returns = test_df['close'].pct_change().dropna()
                if len(predictions) > len(returns):
                    pred_returns = predictions[1:len(returns)+1]
                else:
                    pred_returns = predictions
                
                strategy_returns = returns.values * (np.array(pred_returns[:len(returns)]) - 0.5) * 2
                
                # Calculate metrics
                return_pct = strategy_returns.sum() * 100
                
                # Max drawdown
                equity = (1 + pd.Series(strategy_returns)).cumprod()
                peak = equity.expanding().max()
                drawdown = (equity - peak) / peak
                max_dd = drawdown.min() * 100
                
                window_result = WindowResult(
                    train_start=start,
                    train_end=train_end,
                    test_start=train_end,
                    test_end=test_end,
                    train_score=0.5,  # Would need train metric
                    test_score=score,
                    num_trades=len(test_df),
                    return_pct=return_pct,
                    max_drawdown=max_dd
                )
                
                self.results.append(window_result)
                
            except Exception as e:
                print(f"Error in window {start}-{test_end}: {e}")
                continue
        
        # Combine all predictions
        if all_predictions:
            combined = pd.concat(all_predictions)
            return combined.sort_index(), self.results
        
        return pd.DataFrame(), self.results
    
    def expanding_walk_forward(
        self,
        df: pd.DataFrame,
        model_factory: Callable,
        min_train_size: int = 200,
        test_size: int = 50
    ) -> Tuple[pd.DataFrame, List[WindowResult]]:
        """
        Run expanding window walk-forward.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Full dataset
        model_factory : callable
            Function that creates and returns a fitted model
        min_train_size : int
            Minimum training size
        test_size : int
            Test window size
        
        Returns:
        --------
        Tuple[pd.DataFrame, List[WindowResult]] : (predictions, results)
        """
        all_predictions = []
        self.results = []
        
        for train_end in range(min_train_size, len(df) - test_size, test_size):
            train_df = df.iloc[:train_end]
            test_df = df.iloc[train_end:train_end + test_size]
            
            try:
                model = model_factory()
                
                if hasattr(model, 'fit'):
                    model.fit(train_df)
                
                if hasattr(model, 'predict'):
                    predictions = model.predict(test_df)
                elif hasattr(model, 'predict_proba'):
                    predictions = model.predict_proba(test_df)[:, 1]
                else:
                    continue
                
                test_df_copy = test_df.copy()
                test_df_copy['prediction'] = predictions
                all_predictions.append(test_df_copy)
                
                # Simple metrics
                returns = test_df['close'].pct_change().dropna()
                strategy_returns = returns.values * (np.array(predictions[:len(returns)]) - 0.5) * 2
                return_pct = strategy_returns.sum() * 100
                
                equity = (1 + pd.Series(strategy_returns)).cumprod()
                peak = equity.expanding().max()
                drawdown = (equity - peak) / peak
                max_dd = drawdown.min() * 100
                
                window_result = WindowResult(
                    train_start=0,
                    train_end=train_end,
                    test_start=train_end,
                    test_end=train_end + test_size,
                    train_score=0.5,
                    test_score=0.5,
                    num_trades=len(test_df),
                    return_pct=return_pct,
                    max_drawdown=max_dd
                )
                
                self.results.append(window_result)
                
            except Exception as e:
                print(f"Error in expanding window: {e}")
                continue
        
        if all_predictions:
            combined = pd.concat(all_predictions)
            return combined.sort_index(), self.results
        
        return pd.DataFrame(), self.results
    
    def get_summary(self) -> Dict:
        """Get summary statistics across all windows."""
        if not self.results:
            return {}
        
        returns = [r.return_pct for r in self.results]
        scores = [r.test_score for r in self.results]
        drawdowns = [r.max_drawdown for r in self.results]
        
        return {
            'n_windows': len(self.results),
            'avg_return': np.mean(returns),
            'total_return': np.sum(returns),
            'win_rate': len([r for r in returns if r > 0]) / len(returns),
            'avg_score': np.mean(scores),
            'avg_drawdown': np.mean(drawdowns),
            'max_drawdown': np.min(drawdowns),
            'best_return': np.max(returns),
            'worst_return': np.min(returns)
        }


def purge_walk_forward(
    df: pd.DataFrame,
    train_size: int,
    test_size: int,
    purge_size: int = 5
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Generate purged walk-forward splits to reduce look-ahead bias.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data
    train_size : int
        Training window size
    test_size : int
        Test window size
    purge_size : int
        Gap between train and test
    
    Returns:
    --------
    List[Tuple[pd.DataFrame, pd.DataFrame]] : List of (train, test) pairs
    """
    splits = []
    n = len(df)
    
    for start in range(0, n - train_size - test_size - purge_size, test_size):
        train_end = start + train_size
        test_start = train_end + purge_size
        test_end = min(test_start + test_size, n)
        
        train = df.iloc[start:train_end]
        test = df.iloc[test_start:test_end]
        
        splits.append((train, test))
    
    return splits


def calculate_ols_score(y_true: pd.Series, y_pred: np.ndarray) -> float:
    """
    Calculate directional accuracy score.
    
    Parameters:
    -----------
    y_true : pd.Series
        True returns
    y_pred : np.ndarray
        Predicted probabilities
    
    Returns:
    --------
    float : Directional accuracy (0-1)
    """
    if len(y_true) != len(y_pred):
        y_pred = y_pred[:len(y_true)]
    
    true_direction = (y_true > 0).astype(int).values
    pred_direction = (y_pred > 0.5).astype(int)
    
    return (true_direction == pred_direction).mean()
