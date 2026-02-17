# src/models/ensemble.py
"""
Ensemble Model for ML Signal Generation
Combines multiple ML models for more robust signals
"""

import numpy as np
from typing import List, Dict, Any


class EnsembleSignalModel:
    """
    Ensemble of ML models for generating more robust signals.
    Combines model probabilities using weighted average.
    """
    
    def __init__(self, models: List[Any], weights: List[float] = None):
        """
        Initialize the ensemble.
        
        Args:
            models: List of ML models (must have predict_proba method)
            weights: List of weights for each model (default: equal weights)
        """
        self.models = models
        self.weights = weights or [1 / len(models)] * len(models)
        
        # Normalize weights
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]
    
    def fit(self, df):
        """Fit all models in the ensemble"""
        for model in self.models:
            if hasattr(model, 'fit'):
                model.fit(df)
    
    def predict_proba(self, df) -> np.ndarray:
        """
        Get weighted average of probabilities from all models.
        
        Returns:
            Array of shape (n_samples, n_classes) with [prob_neg, prob_neutral, prob_pos]
        """
        prob_list = []
        
        for model in self.models:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(df)
                prob_list.append(proba)
        
        if not prob_list:
            # Return default probabilities if no model has predict_proba
            n_samples = len(df)
            return np.array([[0.33, 0.34, 0.33]] * n_samples)
        
        prob_array = np.array(prob_list)
        weights = np.array(self.weights).reshape(-1, 1, 1)
        
        # Weighted average of probabilities
        weighted_proba = np.sum(prob_array * weights, axis=0)
        
        return weighted_proba
    
    def predict_signals(self, df, threshold_buy: float = 0.55, threshold_sell: float = 0.55) -> np.ndarray:
        """
        Generate trading signals based on ensemble probabilities.
        
        Args:
            df: DataFrame with features
            threshold_buy: Minimum probability for BUY signal
            threshold_sell: Minimum probability for SELL signal
        
        Returns:
            Array of signals: 1 (BUY), -1 (SELL), 0 (HOLD)
        """
        proba = self.predict_proba(df)
        
        # Assuming proba columns are [prob_neg, prob_neutral, prob_pos]
        # For binary: [prob_sell, prob_buy]
        if proba.shape[1] == 2:
            buy_prob = proba[:, 1]
            sell_prob = proba[:, 0]
        elif proba.shape[1] == 3:
            # Three-class: [sell, hold, buy]
            buy_prob = proba[:, 2]
            sell_prob = proba[:, 0]
        else:
            buy_prob = proba[:, -1]
            sell_prob = proba[:, 0]
        
        signals = []
        
        for b, s in zip(buy_prob, sell_prob):
            if b > threshold_buy:
                signals.append(1)  # BUY
            elif s > threshold_sell:
                signals.append(-1)  # SELL
            else:
                signals.append(0)  # HOLD
        
        return np.array(signals)
    
    def get_confidence(self, df) -> np.ndarray:
        """
        Get confidence level for each prediction.
        
        Returns:
            Array of confidence values (0-1)
        """
        proba = self.predict_proba(df)
        
        # Confidence = max probability
        confidence = np.max(proba, axis=1)
        
        return confidence


# ---------------------------------------------------------
# MODEL LOADERS
# ---------------------------------------------------------

def load_model(model_type: str = "ensemble"):
    """
    Load the ML model to use for signals.
    
    Options:
    - 'rf' → Random Forest
    - 'xgb' → XGBoost  
    - 'ensemble' → RF + XGB
    """
    from src.ml_model import MLSignalModel
    from src.ml_model_xgb import XGBSignalModel
    
    if model_type == "rf":
        return MLSignalModel("random_forest")
    
    if model_type == "xgb":
        return XGBSignalModel()
    
    if model_type == "ensemble":
        rf = MLSignalModel("random_forest")
        xgb = XGBSignalModel()
        return EnsembleSignalModel(models=[rf, xgb], weights=[0.5, 0.5])
    
    raise ValueError(f"Unknown model type: {model_type}")
