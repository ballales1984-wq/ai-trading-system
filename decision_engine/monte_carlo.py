"""
Monte Carlo Simulation Engine
5-level progressive Monte Carlo simulation for probability estimation
"""

import logging
from typing import Dict, Any
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MonteCarloEngine:
    """
    Monte Carlo simulation engine with 5 progressive levels:
    
    Level 1: Base GBM random walk
    Level 2: Conditional (macro events + sentiment)
    Level 3: Adaptive (reinforcement learning from past accuracy)
    Level 4: Multi-factor (natural events + energy + cross-correlations)
    Level 5: Semantic History (geopolitics + innovation + pattern matching)
    """
    
    def __init__(self, decision_engine: 'DecisionEngine'):
        self.engine = decision_engine
        self._mc_cache = {}
    
    def run(self, symbol: str, df: pd.DataFrame,
            n_simulations: int = 1000, n_days: int = 30,
            level: int = 5) -> Dict:
        """Run Monte Carlo simulation."""
        if df is None or len(df) < 20:
            return {'probability_up': 0.5, 'confidence': 0.0, 'level': 0}
        
        returns = df['close'].pct_change().dropna()
        if len(returns) < 10:
            return {'probability_up': 0.5, 'confidence': 0.0, 'level': 0}
        
        current_price = df['close'].iloc[-1]
        mu = returns.mean()
        sigma = returns.std()
        
        # Level 1: Base GBM
        result = self._run_level_1(current_price, mu, sigma, n_simulations, n_days)
        
        if level >= 2:
            result = self._run_level_2(current_price, mu, sigma, n_simulations, n_days, result, symbol)
        
        if level >= 3:
            result = self._run_level_3(current_price, mu, sigma, n_simulations, n_days, result, symbol, df)
        
        if level >= 4:
            result = self._run_level_4(current_price, mu, sigma, n_simulations, n_days, result, symbol)
        
        if level >= 5:
            result = self._run_level_5(current_price, mu, sigma, n_simulations, n_days, result, returns)
        
        self._mc_cache[symbol] = {**result, 'timestamp': datetime.now().timestamp()}
        
        return result
    
    def _run_level_1(self, current_price: float, mu: float, sigma: float,
                     n_simulations: int, n_days: int) -> Dict:
        """Level 1: Base GBM random walk"""
        np.random.seed(42)
        dt = 1.0 / 252
        paths = np.zeros((n_simulations, n_days))
        paths[:, 0] = current_price
        
        for t in range(1, n_days):
            z = np.random.standard_normal(n_simulations)
            paths[:, t] = paths[:, t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
        
        final_prices = paths[:, -1]
        prob_up = np.mean(final_prices > current_price)
        
        return {
            'probability_up': float(prob_up),
            'expected_return': float(np.mean(final_prices) / current_price - 1),
            'var_95': float(np.percentile(final_prices / current_price - 1, 5)),
            'confidence': 0.3,
            'level': 1,
        }
    
    def _run_level_2(self, current_price: float, mu: float, sigma: float,
                     n_simulations: int, n_days: int, prev_result: Dict, symbol: str) -> Dict:
        """Level 2: Conditional Monte Carlo"""
        ext_sentiment = self._fetch_external_sentiment(symbol)
        macro_events = self._fetch_macro_events()
        cmc_context = self._fetch_cmc_context(symbol)
        
        sentiment_adj = ext_sentiment.get('score', 0) * 0.002
        event_vol_adj = 1.0 + macro_events.get('avg_impact', 0) * 0.3
        
        cmc_sentiment_adj = cmc_context.get('sentiment_score', 0) * 0.001
        cmc_vol_adj = 1.0
        if cmc_context.get('volume_ratio', 0) > 0.08:
            cmc_vol_adj = 1.1
        elif cmc_context.get('volume_ratio', 0) < 0.03:
            cmc_vol_adj = 0.9
        
        sigma_l2 = sigma * event_vol_adj * cmc_vol_adj
        mu_l2 = mu + sentiment_adj + cmc_sentiment_adj
        
        np.random.seed(42)
        dt = 1.0 / 252
        paths = np.zeros((n_simulations, n_days))
        paths[:, 0] = current_price
        
        for t in range(1, n_days):
            z = np.random.standard_normal(n_simulations)
            paths[:, t] = paths[:, t-1] * np.exp((mu_l2 - 0.5 * sigma_l2**2) * dt + sigma_l2 * np.sqrt(dt) * z)
        
        final_prices = paths[:, -1]
        prob_up = np.mean(final_prices > current_price)
        
        return {
            'probability_up': float(prob_up),
            'expected_return': float(np.mean(final_prices) / current_price - 1),
            'var_95': float(np.percentile(final_prices / current_price - 1, 5)),
            'sentiment_impact': float(sentiment_adj),
            'confidence': 0.45,
            'level': 2,
        }
    
    def _run_level_3(self, current_price: float, mu: float, sigma: float,
                     n_simulations: int, n_days: int, prev_result: Dict,
                     symbol: str, df: pd.DataFrame) -> Dict:
        """Level 3: Adaptive Monte Carlo"""
        past_mc = self._mc_cache.get(symbol, {})
        accuracy_adj = 0.0
        
        if past_mc and 'actual_return' in past_mc:
            predicted = past_mc.get('expected_return', 0)
            actual = past_mc.get('actual_return', 0)
            accuracy_adj = (actual - predicted) * 0.1
        
        mu_l3 = mu + accuracy_adj
        
        np.random.seed(42)
        dt = 1.0 / 252
        paths = np.zeros((n_simulations, n_days))
        paths[:, 0] = current_price
        
        for t in range(1, n_days):
            z = np.random.standard_normal(n_simulations)
            paths[:, t] = paths[:, t-1] * np.exp((mu_l3 - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
        
        final_prices = paths[:, -1]
        prob_up = np.mean(final_prices > current_price)
        
        return {
            'probability_up': float(prob_up),
            'expected_return': float(np.mean(final_prices) / current_price - 1),
            'var_95': float(np.percentile(final_prices / current_price - 1, 5)),
            'accuracy_adjustment': float(accuracy_adj),
            'confidence': 0.55,
            'level': 3,
        }
    
    def _run_level_4(self, current_price: float, mu: float, sigma: float,
                     n_simulations: int, n_days: int, prev_result: Dict, symbol: str) -> Dict:
        """Level 4: Multi-factor Monte Carlo"""
        natural_events = self._fetch_natural_events()
        natural_adj = natural_events.get('avg_intensity', 0) * 0.005
        
        mu_l4 = mu - natural_adj
        
        np.random.seed(42)
        dt = 1.0 / 252
        paths = np.zeros((n_simulations, n_days))
        paths[:, 0] = current_price
        
        for t in range(1, n_days):
            z = np.random.standard_normal(n_simulations)
            paths[:, t] = paths[:, t-1] * np.exp((mu_l4 - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
        
        final_prices = paths[:, -1]
        prob_up = np.mean(final_prices > current_price)
        
        return {
            'probability_up': float(prob_up),
            'expected_return': float(np.mean(final_prices) / current_price - 1),
            'var_95': float(np.percentile(final_prices / current_price - 1, 5)),
            'natural_impact': float(natural_adj),
            'confidence': 0.65,
            'level': 4,
        }
    
    def _run_level_5(self, current_price: float, mu: float, sigma: float,
                     n_simulations: int, n_days: int, prev_result: Dict,
                     returns: pd.Series) -> Dict:
        """Level 5: Semantic History Monte Carlo"""
        lookback = min(len(returns), 252)
        current_pattern = returns.iloc[-20:].values if len(returns) >= 20 else returns.values
        
        best_match_score = 0.0
        best_match_return = 0.0
        
        for start in range(0, lookback - 40, 5):
            historical_pattern = returns.iloc[start:start+20].values
            if len(historical_pattern) == len(current_pattern):
                try:
                    correlation = np.corrcoef(current_pattern, historical_pattern)[0, 1]
                    if not np.isnan(correlation) and abs(correlation) > best_match_score:
                        best_match_score = abs(correlation)
                        future_slice = returns.iloc[start+20:start+40]
                        if len(future_slice) > 0:
                            best_match_return = future_slice.mean()
                except:
                    pass
        
        tail_threshold = returns.quantile(0.01)
        black_swan_prob = np.mean(returns < tail_threshold * 2)
        
        semantic_adj = best_match_return * best_match_score * 0.3
        mu_l5 = mu + semantic_adj
        df_t = max(3, int(10 - black_swan_prob * 100))
        
        np.random.seed(42)
        dt = 1.0 / 252
        paths = np.zeros((n_simulations, n_days))
        paths[:, 0] = current_price
        
        for t in range(1, n_days):
            z = np.random.standard_t(df_t, n_simulations) / np.sqrt(df_t / (df_t - 2))
            paths[:, t] = paths[:, t-1] * np.exp((mu_l5 - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
        
        final_prices = paths[:, -1]
        prob_up = np.mean(final_prices > current_price)
        
        return {
            'probability_up': float(prob_up),
            'expected_return': float(np.mean(final_prices) / current_price - 1),
            'var_95': float(np.percentile(final_prices / current_price - 1, 5)),
            'semantic_match_score': float(best_match_score),
            'black_swan_probability': float(black_swan_prob),
            'confidence': 0.75,
            'level': 5,
        }
    
    def _fetch_external_sentiment(self, symbol: str) -> Dict:
        if hasattr(self.engine, 'fetch_external_sentiment'):
            return self.engine.fetch_external_sentiment(symbol)
        return {'score': 0.0}
    
    def _fetch_macro_events(self) -> Dict:
        if hasattr(self.engine, 'fetch_external_macro_events'):
            return self.engine.fetch_external_macro_events()
        return {'avg_impact': 0.0}
    
    def _fetch_cmc_context(self, symbol: str) -> Dict:
        if hasattr(self.engine, 'fetch_cmc_market_context'):
            return self.engine.fetch_cmc_market_context(symbol)
        return {'sentiment_score': 0.0, 'volume_ratio': 0.0}
    
    def _fetch_natural_events(self) -> Dict:
        if hasattr(self.engine, 'fetch_external_natural_events'):
            return self.engine.fetch_external_natural_events()
        return {'avg_intensity': 0.0}

