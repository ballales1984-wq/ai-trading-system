"""
Data Loader Module for Crypto Trading System
=============================================
Professional data loading from various sources:
- CSV files
- Binance API (via ccxt)
- SQLite databases
- API endpoints

Author: AI Trading System
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta
import os
import ccxt
from pathlib import Path


class DataLoader:
    """Unified data loader for multiple sources."""
    
    def __init__(self, exchange_id: str = 'binance'):
        """
        Initialize data loader.
        
        Parameters:
        -----------
        exchange_id : str
            Exchange ID for ccxt (default: 'binance')
        """
        self.exchange = getattr(ccxt, exchange_id)()
        self.exchange.enableRateLimit = True
    
    def load_csv(
        self,
        filepath: str,
        date_col: str = 'date',
        parse_dates: bool = True
    ) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Parameters:
        -----------
        filepath : str
            Path to CSV file
        date_col : str
            Name of date column
        parse_dates : bool
            Whether to parse dates
        
        Returns:
        --------
        pd.DataFrame : Loaded data
        """
        df = pd.read_csv(filepath)
        
        if parse_dates and date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
        
        return df
    
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = '1d',
        since: Optional[int] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from exchange.
        
        Parameters:
        -----------
        symbol : str
            Trading symbol (e.g., 'BTC/USDT')
        timeframe : str
            Timeframe (e.g., '1m', '1h', '1d')
        since : int, optional
            Start timestamp (ms)
        limit : int
            Number of candles to fetch
        
        Returns:
        --------
        pd.DataFrame : OHLCV data
        """
        ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since, limit)
        
        df = pd.DataFrame(
            ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def fetch_ohlcv_to_file(
        self,
        symbol: str,
        timeframe: str = '1d',
        days: int = 365,
        output_dir: str = 'data'
    ) -> str:
        """
        Fetch OHLCV data and save to CSV.
        
        Parameters:
        -----------
        symbol : str
            Trading symbol
        timeframe : str
            Timeframe
        days : int
            Number of days to fetch
        output_dir : str
            Output directory
        
        Returns:
        --------
        str : Path to saved file
        """
        since = self.exchange.milliseconds() - (days * 24 * 60 * 60 * 1000)
        
        # Fetch in batches if needed
        all_ohlcv = []
        max_retries = 3
        
        for _ in range(max_retries):
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since)
                all_ohlcv = ohlcv
                break
            except Exception as e:
                print(f"Error fetching data: {e}")
                continue
        
        df = pd.DataFrame(
            all_ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Create filename
        symbol_clean = symbol.replace('/', '_')
        filename = f"{symbol_clean}_{timeframe}.csv"
        filepath = os.path.join(output_dir, filename)
        
        df.to_csv(filepath)
        
        return filepath
    
    def load_multiple_symbols(
        self,
        symbols: List[str],
        timeframe: str = '1d',
        days: int = 365
    ) -> Dict[str, pd.DataFrame]:
        """
        Load data for multiple symbols.
        
        Parameters:
        -----------
        symbols : List[str]
            List of trading symbols
        timeframe : str
            Timeframe
        days : int
            Number of days
        
        Returns:
        --------
        Dict[str, pd.DataFrame] : Symbol -> DataFrame mapping
        """
        data = {}
        
        for symbol in symbols:
            try:
                df = self.fetch_ohlcv(symbol, timeframe, days=days)
                data[symbol] = df
            except Exception as e:
                print(f"Error loading {symbol}: {e}")
                continue
        
        return data


def load_crypto_data(
    symbol: str = 'BTC/USDT',
    timeframe: str = '1d',
    days: int = 365
) -> pd.DataFrame:
    """
    Convenience function to load crypto data.
    
    Parameters:
    -----------
    symbol : str
        Trading symbol
    timeframe : str
        Timeframe
    days : int
        Number of days
    
    Returns:
    --------
    pd.DataFrame : OHLCV data
    """
    loader = DataLoader()
    
    # Try to load from local file first
    symbol_clean = symbol.replace('/', '_')
    filepath = f"data/{symbol_clean}_{timeframe}.csv"
    
    if os.path.exists(filepath):
        return loader.load_csv(filepath)
    
    # Otherwise fetch from exchange
    return loader.fetch_ohlcv(symbol, timeframe, days=days)


def combine_timeframes(
    data: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Combine data from different timeframes.
    
    Parameters:
    -----------
    data : Dict[str, pd.DataFrame]
        Dictionary of timeframe -> DataFrame
    
    Returns:
    --------
    pd.DataFrame : Combined DataFrame
    """
    # Ensure all dataframes are aligned by index
    combined = None
    
    for timeframe, df in data.items():
        df_copy = df.copy()
        for col in df_copy.columns:
            df_copy[f'{col}_{timeframe}'] = df_copy[col]
            df_copy.drop(col, axis=1, inplace=True)
        
        if combined is None:
            combined = df_copy
        else:
            combined = combined.join(df_copy, how='outer')
    
    return combined


def resample_data(
    df: pd.DataFrame,
    timeframe: str = '1H',
    agg_func: str = 'last'
) -> pd.DataFrame:
    """
    Resample data to different timeframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input data
    timeframe : str
        Target timeframe (e.g., '1H', '4H', '1D')
    agg_func : str
        Aggregation function
    
    Returns:
    --------
    pd.DataFrame : Resampled data
    """
    agg_rules = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    
    return df.resample(timeframe).agg(agg_rules).dropna()


def validate_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate data quality.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data to validate
    
    Returns:
    --------
    Tuple[bool, List[str]] : (is_valid, list of issues)
    """
    issues = []
    
    # Check for required columns
    required = ['open', 'high', 'low', 'close']
    missing = [col for col in required if col not in df.columns]
    if missing:
        issues.append(f"Missing columns: {missing}")
    
    # Check for nulls
    null_counts = df[required].isnull().sum()
    if null_counts.any():
        issues.append(f"Null values found: {null_counts.to_dict()}")
    
    # Check for price consistency
    if 'high' in df.columns and 'low' in df.columns:
        invalid = df['high'] < df['low']
        if invalid.any():
            issues.append(f"High < Low in {invalid.sum()} rows")
    
    # Check for negative prices
    for col in required:
        if col in df.columns and (df[col] <= 0).any():
            issues.append(f"Non-positive {col} values found")
    
    return len(issues) == 0, issues


def fill_missing_data(
    df: pd.DataFrame,
    method: str = 'forward'
) -> pd.DataFrame:
    """
    Fill missing data in DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input data
    method : str
        Fill method: 'forward', 'backward', 'interpolate'
    
    Returns:
    --------
    pd.DataFrame : Filled data
    """
    result = df.copy()
    
    if method == 'forward':
        result.fillna(method='ffill', inplace=True)
    elif method == 'backward':
        result.fillna(method='bfill', inplace=True)
    elif method == 'interpolate':
        result.interpolate(method='linear', inplace=True)
    
    # Fill any remaining NaN
    result.fillna(method='ffill', inplace=True)
    result.fillna(method='bfill', inplace=True)
    
    return result
