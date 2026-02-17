"""
Utilities Module for Crypto Trading System
==========================================
Common utilities including:
- Walk-forward validation
- Data preprocessing
- Feature engineering
- Date/time utilities
- Performance tracking

Author: AI Trading System
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional, Callable
from datetime import datetime, timedelta
import json
import os


def walk_forward_split(
    df: pd.DataFrame,
    train_size: float = 0.7,
    test_size: Optional[float] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and test sets for walk-forward validation.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data to split
    train_size : float
        Proportion of data for training (0.0 to 1.0)
    test_size : float, optional
        Test size (if different from remaining)
    
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame] : (train_data, test_data)
    """
    split_idx = int(len(df) * train_size)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    
    return train, test


def rolling_window_split(
    df: pd.DataFrame,
    train_window: int,
    test_window: int,
    step: int = 1
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Generate multiple train/test splits for walk-forward validation.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data to split
    train_window : int
        Number of periods for training
    test_window : int
        Number of periods for testing
    step : int
        Step size between windows
    
    Returns:
    --------
    List[Tuple[pd.DataFrame, pd.DataFrame]] : List of (train, test) pairs
    """
    splits = []
    start = 0
    
    while start + train_window + test_window <= len(df):
        train_end = start + train_window
        test_end = train_end + test_window
        
        train = df.iloc[start:train_end]
        test = df.iloc[train_end:test_end]
        
        splits.append((train, test))
        start += step
    
    return splits


def expanding_window_split(
    df: pd.DataFrame,
    min_train_size: int,
    test_size: int
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Generate expanding window splits (train grows over time).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data to split
    min_train_size : int
        Minimum initial training size
    test_size : int
        Test window size
    
    Returns:
    --------
    List[Tuple[pd.DataFrame, pd.DataFrame]] : List of (train, test) pairs
    """
    splits = []
    
    for train_end in range(min_train_size, len(df) - test_size + 1, test_size):
        train = df.iloc[:train_end]
        test = df.iloc[train_end:train_end + test_size]
        
        if len(test) > 0:
            splits.append((train, test))
    
    return splits


def cross_validation_score(
    df: pd.DataFrame,
    signal_func: Callable,
    metric_func: Callable,
    n_splits: int = 5,
    train_ratio: float = 0.8
) -> List[float]:
    """
    Run cross-validation and calculate scores for each fold.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data for validation
    signal_func : callable
        Function to generate signals
    metric_func : callable
        Function to calculate metric
    n_splits : int
        Number of CV splits
    train_ratio : float
        Train/test split ratio
    
    Returns:
    --------
    List[float] : Scores for each fold
    """
    scores = []
    fold_size = len(df) // n_splits
    
    for i in range(n_splits):
        # Calculate test period
        test_start = i * fold_size
        test_end = min((i + 1) * fold_size, len(df))
        
        # Combine data before and after test period for training
        if i == 0:
            train = df.iloc[:int(test_end * train_ratio)]
        elif i == n_splits - 1:
            train = df.iloc[int(test_start * train_ratio):]
        else:
            train = pd.concat([
                df.iloc[:int(test_start * train_ratio)],
                df.iloc[int(test_end * train_ratio):]
            ])
        
        test = df.iloc[test_start:test_end]
        
        # Generate signals and calculate metric
        try:
            signals = signal_func(train)
            metric = metric_func(test, signals)
            scores.append(metric)
        except Exception as e:
            print(f"Error in fold {i}: {e}")
            scores.append(0.0)
    
    return scores


def normalize_features(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = 'zscore'
) -> pd.DataFrame:
    """
    Normalize feature columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input data
    columns : List[str], optional
        Columns to normalize (default: all numeric)
    method : str
        Normalization method: 'zscore', 'minmax', 'robust'
    
    Returns:
    --------
    pd.DataFrame : Normalized data
    """
    result = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if method == 'zscore':
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                result[col] = (df[col] - mean) / std
        elif method == 'minmax':
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val > min_val:
                result[col] = (df[col] - min_val) / (max_val - min_val)
        elif method == 'robust':
            median = df[col].median()
            iqr = df[col].quantile(0.75) - df[col].quantile(0.25)
            if iqr > 0:
                result[col] = (df[col] - median) / iqr
    
    return result


def add_lag_features(
    df: pd.DataFrame,
    columns: List[str],
    lags: List[int] = [1, 2, 3, 5, 10]
) -> pd.DataFrame:
    """
    Add lagged features to DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input data
    columns : List[str]
        Columns to create lags for
    lags : List[int]
        Lag periods
    
    Returns:
    --------
    pd.DataFrame : DataFrame with lag features
    """
    result = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        for lag in lags:
            result[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    return result


def add_rolling_features(
    df: pd.DataFrame,
    columns: List[str],
    windows: List[int] = [5, 10, 20, 50],
    operations: List[str] = ['mean', 'std', 'min', 'max']
) -> pd.DataFrame:
    """
    Add rolling window features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input data
    columns : List[str]
        Columns to create rolling features for
    windows : List[int]
        Window sizes
    operations : List[str]
        Rolling operations
    
    Returns:
    --------
    pd.DataFrame : DataFrame with rolling features
    """
    result = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        for window in windows:
            if 'mean' in operations:
                result[f'{col}_roll_mean_{window}'] = df[col].rolling(window).mean()
            if 'std' in operations:
                result[f'{col}_roll_std_{window}'] = df[col].rolling(window).std()
            if 'min' in operations:
                result[f'{col}_roll_min_{window}'] = df[col].rolling(window).min()
            if 'max' in operations:
                result[f'{col}_roll_max_{window}'] = df[col].rolling(window).max()
    
    return result


def create_date_features(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    """
    Create date-based features from datetime column.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input data
    date_col : str
        Name of datetime column
    
    Returns:
    --------
    pd.DataFrame : DataFrame with date features
    """
    result = df.copy()
    
    if date_col not in df.columns:
        return result
    
    # Ensure datetime type
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        result[date_col] = pd.to_datetime(df[date_col])
    
    # Extract features
    result['day_of_week'] = result[date_col].dt.dayofweek
    result['day_of_month'] = result[date_col].dt.day
    result['month'] = result[date_col].dt.month
    result['quarter'] = result[date_col].dt.quarter
    result['year'] = result[date_col].dt.year
    result['week_of_year'] = result[date_col].dt.isocalendar().week
    
    # Cyclical features
    result['day_sin'] = np.sin(2 * np.pi * result['day_of_week'] / 7)
    result['day_cos'] = np.cos(2 * np.pi * result['day_of_week'] / 7)
    result['month_sin'] = np.sin(2 * np.pi * result['month'] / 12)
    result['month_cos'] = np.cos(2 * np.pi * result['month'] / 12)
    
    return result


def save_results(results: Dict, filepath: str) -> None:
    """
    Save backtest results to JSON file.
    
    Parameters:
    -----------
    results : Dict
        Results dictionary
    filepath : str
        Output file path
    """
    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        return obj
    
    serializable_results = convert_to_serializable(results)
    
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)


def load_results(filepath: str) -> Dict:
    """
    Load backtest results from JSON file.
    
    Parameters:
    -----------
    filepath : str
        Input file path
    
    Returns:
    --------
    Dict : Loaded results
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def ensure_directory(filepath: str) -> None:
    """
    Ensure directory exists for file path.
    
    Parameters:
    -----------
    filepath : str
        File path
    """
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)


class PerformanceTracker:
    """Track and log performance metrics during backtesting."""
    
    def __init__(self):
        self.metrics = {}
        self.history = []
    
    def update(self, timestamp: pd.Timestamp, metrics: Dict) -> None:
        """Update metrics at a specific timestamp."""
        self.metrics = metrics
        self.history.append({
            'timestamp': timestamp,
            **metrics
        })
    
    def get_history(self) -> pd.DataFrame:
        """Get metrics history as DataFrame."""
        return pd.DataFrame(self.history)
    
    def summary(self) -> Dict:
        """Get summary statistics."""
        if not self.history:
            return {}
        
        df = pd.DataFrame(self.history)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        return {
            'total_periods': len(self.history),
            'metrics': df[numeric_cols].describe().to_dict() if len(numeric_cols) > 0 else {}
        }


def format_currency(value: float, currency: str = 'USD') -> str:
    """Format value as currency string."""
    return f"{currency} {value:,.2f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format value as percentage string."""
    return f"{value * 100:.{decimals}f}%"


def calculate_position_size(
    capital: float,
    risk_per_trade: float,
    entry_price: float,
    stop_loss: float
) -> float:
    """
    Calculate position size based on risk management.
    
    Parameters:
    -----------
    capital : float
        Available capital
    risk_per_trade : float
        Risk per trade as fraction of capital
    entry_price : float
        Entry price
    stop_loss : float
        Stop loss price
    
    Returns:
    --------
    float : Position size (units to buy/sell)
    """
    risk_amount = capital * risk_per_trade
    price_risk = abs(entry_price - stop_loss)
    
    if price_risk == 0:
        return 0
    
    return risk_amount / price_risk
