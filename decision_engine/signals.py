"""
Signal Generation Module
Contains: SignalGenerator class for combining factors and generating trading signals
"""

import logging
from typing import Dict, List, Any, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class SignalGenerator:
    """
    Signal generator that combines technical analysis, sentiment, and other factors
    to generate trading signals.
    """
    
    def __init__(self, settings: Dict[str, Any]):
        """
        Initialize signal generator.
        
        Args:
            settings: Decision settings dictionary
        """
        self.settings = settings
    
    def combine_factors(self, 
                       technical: 'TechnicalAnalysis',
                       sentiment: Dict,
                       correlations: Dict,
                       volatility_score: float,
                       ml_score: float = 0.5,
                       mc_score: float = 0.5,
                       regime_score: float = 0.5) -> float:
        """
        Combine all factors into a single score.
        
        Args:
            technical: TechnicalAnalysis object
            sentiment: Sentiment dictionary
            correlations: Correlation dictionary
            volatility_score: Volatility score (0-1)
            ml_score: ML prediction score (0-1)
            mc_score: Monte Carlo probability (0-1)
            regime_score: HMM regime score (-1 bear to +1 bull)
            
        Returns:
            Combined score (0-1)
        """
        weights = self.settings['weights']
        
        # Normalize sentiment score to 0-1
        sentiment_score = (sentiment.get('combined_score', 0) + 1) / 2
        
        # Correlation score - preserve direction, don't use abs!
        # Positive correlation = follow market direction
        # Negative correlation = hedge/contrarian
        correlation_raw = correlations.get('avg_correlation', 0)
        # Convert -1/+1 to 0/1 for scoring
        correlation_score = (correlation_raw + 1) / 2
        
        # Normalize regime score from -1/+1 to 0/1
        regime_norm = (regime_score + 1) / 2
        
        # Weights for ML, Monte Carlo, and Regime (from config or defaults)
        ml_weight = self.settings.get('ml_weight', 0.10)
        mc_weight = self.settings.get('mc_weight', 0.10)
        regime_weight = self.settings.get('regime_weight', 0.10)
        base_weight = 1.0 - ml_weight - mc_weight - regime_weight
        
        # Calculate base score (including regime)
        base_score = (
            technical.technical_score * weights['technical'] +
            technical.momentum_score * weights['momentum'] +
            correlation_score * weights['correlation'] +
            sentiment_score * weights['sentiment'] +
            volatility_score * weights['volatility'] +
            regime_norm * weights.get('regime', 0.10)  # Default 10% weight for regime
        )
        
        # Blend with ML, Monte Carlo, and Regime scores
        score = (base_score * base_weight + 
                 ml_score * ml_weight + 
                 mc_score * mc_weight + 
                 regime_norm * regime_weight)
        
        # Normalize score to center around 0.5
        # This reduces bias by shifting the mean to 0.5
        normalized_score = self._normalize_score(score)
        
        return normalized_score
    
    def _normalize_score(self, score: float) -> float:
        """
        Normalize score to reduce bias toward BUY.
        Adaptive compression - only compresses noisy middle zone.
        
        Args:
            score: Raw score (0-1)
            
        Returns:
            Normalized score (0-1)
        """
        # Adaptive normalization - only compress middle zone
        if 0.40 <= score <= 0.65:
            # Compress noisy middle zone (where most false signals occur)
            # This is where the bias is most problematic
            return 0.5 + (score - 0.5) * 0.5
        else:
            # Keep strong signals (high confidence) mostly unchanged
            return score
    
    def determine_action(self, score: float) -> str:
        """
        Determine action based on combined score.
        
        Wide HOLD zone (20%) to reduce overtrading and bias.
        
        Args:
            score: Combined score (0-1)
            
        Returns:
            Action: 'BUY', 'SELL', or 'HOLD'
        """
        # Get thresholds from settings
        strong_buy = self.settings.get('strong_signal_threshold', 0.70)
        buy_threshold = self.settings.get('min_signal_confidence', 0.60)
        sell_threshold = self.settings.get('sell_threshold', 0.40)
        strong_sell = self.settings.get('strong_sell_threshold', 0.30)
        
        # Strong BUY (score >= 0.70)
        if score >= strong_buy:
            return 'BUY'
        
        # Moderate BUY (score >= 0.60)
        if score >= buy_threshold:
            return 'BUY'
        
        # HOLD zone (0.40 < score < 0.60) - 20% of range
        if score > sell_threshold and score < buy_threshold:
            return 'HOLD'
        
        # Moderate SELL (score > 0.30)
        if score > strong_sell:
            return 'SELL'
        
        # Strong SELL (score <= 0.30)
        return 'SELL'
    
    def calculate_confidence(self,
                            technical: 'TechnicalAnalysis',
                            sentiment: Dict,
                            volatility_score: float,
                            ml_confidence: float = 0.5) -> float:
        """
        Calculate confidence level for the signal.
        
        Args:
            technical: TechnicalAnalysis object
            sentiment: Sentiment dictionary
            volatility_score: Volatility score (0-1)
            ml_confidence: ML confidence (0-1)
            
        Returns:
            Confidence level (0-1)
        """
        # Base confidence from technical analysis
        confidence = technical.technical_score
        
        # Adjust for sentiment confidence
        sentiment_conf = sentiment.get('confidence', 0.5)
        confidence = (confidence + sentiment_conf) / 2
        
        # Adjust for volatility (lower volatility = higher confidence)
        confidence = (confidence + volatility_score) / 2
        
        # Blend with ML confidence
        if ml_confidence > 0:
            confidence = confidence * 0.7 + ml_confidence * 0.3
        
        return max(0, min(1, confidence))
    
    def get_strength_label(self, confidence: float) -> str:
        """
        Convert confidence to strength label.
        
        Args:
            confidence: Confidence level (0-1)
            
        Returns:
            Strength label: 'STRONG', 'MODERATE', or 'WEAK'
        """
        if confidence >= 0.7:
            return 'STRONG'
        elif confidence >= 0.55:
            return 'MODERATE'
        return 'WEAK'
    
    def calculate_price_levels(self, 
                              current_price: float, 
                              atr: float,
                              action: str) -> Tuple[float, float]:
        """
        Calculate stop loss and take profit levels.
        
        Args:
            current_price: Current asset price
            atr: Average True Range
            action: Trading action
            
        Returns:
            Tuple of (stop_loss, take_profit)
        """
        stop_loss_pct = self.settings['stop_loss_percent']
        take_profit_pct = self.settings['take_profit_percent']
        
        # Use ATR if available for dynamic stop loss
        if atr > 0:
            atr_stop = atr / current_price
            if atr_stop < stop_loss_pct:
                stop_loss_pct = atr_stop
        
        if action == 'BUY':
            stop_loss = current_price * (1 - stop_loss_pct)
            take_profit = current_price * (1 + take_profit_pct)
        elif action == 'SELL':
            stop_loss = current_price * (1 + stop_loss_pct)
            take_profit = current_price * (1 - take_profit_pct)
        else:
            stop_loss = 0
            take_profit = 0
        
        return stop_loss, take_profit
    
    def calculate_position_size(self, 
                             confidence: float, 
                             risk_percent: float) -> float:
        """
        Calculate position size based on confidence and risk.
        
        Args:
            confidence: Signal confidence (0-1)
            risk_percent: Risk percentage
            
        Returns:
            Position size as decimal (0-1)
        """
        max_position = self.settings['max_position_size']
        
        # Scale position size by confidence
        position = max_position * confidence
        
        return position
    
    def generate_reason(self, 
                      technical: 'TechnicalAnalysis',
                      sentiment: Dict, 
                      action: str) -> str:
        """
        Generate human-readable reason for the signal.
        
        Args:
            technical: TechnicalAnalysis object
            sentiment: Sentiment dictionary
            action: Trading action
            
        Returns:
            Human-readable reason string
        """
        reasons = []
        
        # Technical reasons
        if technical.trend == 'bullish':
            reasons.append("bullish trend")
        elif technical.trend == 'bearish':
            reasons.append("bearish trend")
        
        if technical.rsi_signal == 'buy' and technical.rsi < 30:
            reasons.append("RSI oversold")
        elif technical.rsi_signal == 'sell' and technical.rsi > 70:
            reasons.append("RSI overbought")
        
        if technical.macd_histogram > 0:
            reasons.append("MACD bullish")
        else:
            reasons.append("MACD bearish")
        
        # Sentiment reasons
        sentiment_score = sentiment.get('combined_score', 0)
        if sentiment_score > 0.3:
            reasons.append("positive sentiment")
        elif sentiment_score < -0.3:
            reasons.append("negative sentiment")
        
        if not reasons:
            reasons.append("neutral conditions")
        
        return f"{action}: {', '.join(reasons)}"
    
    def rank_signals(self, signals: List[Any]) -> List[Any]:
        """
        Rank signals by confidence.
        
        Args:
            signals: List of trading signals
            
        Returns:
            Sorted list of signals
        """
        return sorted(signals, key=lambda x: x.confidence, reverse=True)
    
    def filter_signals(self, 
                      signals: List[Any], 
                      min_confidence: float = None) -> List[Any]:
        """
        Filter signals by minimum confidence.
        
        Args:
            signals: List of trading signals
            min_confidence: Minimum confidence threshold
            
        Returns:
            Filtered list of signals
        """
        if min_confidence is None:
            min_confidence = self.settings['min_signal_confidence']
        
        return [s for s in signals if s.confidence >= min_confidence]
    
    def suggest_hedges(self, 
                      signal: Any,
                      all_signals: List[Any]) -> List[Any]:
        """
        Suggest hedging positions based on correlations.
        
        Args:
            signal: Primary signal
            all_signals: All generated signals
            
        Returns:
            List of hedge signals
        """
        hedges = []
        
        for other in all_signals:
            if other.symbol == signal.symbol:
                continue
            
            if signal.action == 'BUY' and other.action == 'SELL':
                hedges.append(other)
            elif signal.action == 'SELL' and other.action == 'BUY':
                hedges.append(other)
        
        hedges.sort(key=lambda x: x.confidence, reverse=True)
        
        return hedges[:3]

