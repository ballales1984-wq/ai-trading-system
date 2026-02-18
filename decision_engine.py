"""
Decision Engine Module
Generates probabilistic trading signals by combining all analysis types
"""

import logging
import random
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import json

import pandas as pd
import numpy as np

import config
from data_collector import DataCollector, MarketData, CorrelationData
from technical_analysis import TechnicalAnalyzer, TechnicalAnalysis
from sentiment_news import SentimentAnalyzer, SentimentData
from ml_predictor import PricePredictor, get_ml_predictor

# Configure logging
logger = logging.getLogger(__name__)


# ==================== DATA STRUCTURES ====================

@dataclass
class TradingSignal:
    """Represents a trading signal for an asset"""
    symbol: str
    asset_type: str  # 'crypto' or 'commodity'
    
    # Signal information
    action: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float  # 0-1
    strength: str  # 'STRONG', 'MODERATE', 'WEAK'
    
    # Price information
    current_price: float
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    risk_reward_ratio: float = 0.0
    
    # Component scores
    technical_score: float = 0.5
    momentum_score: float = 0.5
    sentiment_score: float = 0.5
    correlation_score: float = 0.5
    volatility_score: float = 0.5
    ml_score: float = 0.5  # ML prediction score (blackbox agent)
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    timeframe: str = "1h"
    reason: str = ""
    
    # Risk metrics
    position_size: float = 0.0
    risk_percent: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'symbol': self.symbol,
            'asset_type': self.asset_type,
            'action': self.action,
            'confidence': f"{self.confidence:.1%}",
            'strength': self.strength,
            'current_price': self.current_price,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'risk_reward_ratio': f"{self.risk_reward_ratio:.2f}",
            'technical_score': f"{self.technical_score:.1%}",
            'momentum_score': f"{self.momentum_score:.1%}",
            'sentiment_score': f"{self.sentiment_score:.1%}",
            'correlation_score': f"{self.correlation_score:.1%}",
            'volatility_score': f"{self.volatility_score:.1%}",
            'timestamp': self.timestamp.isoformat(),
            'timeframe': self.timeframe,
            'reason': self.reason,
            'position_size': f"{self.position_size:.1%}",
            'risk_percent': f"{self.risk_percent:.1%}"
        }


@dataclass
class PortfolioState:
    """Current portfolio state"""
    total_value: float = 10000.0
    cash: float = 10000.0
    positions: Dict[str, Dict] = field(default_factory=dict)  # symbol -> {quantity, avg_price}
    
    # Performance metrics
    total_pnl: float = 0.0
    daily_pnl: float = 0.0
    win_rate: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'total_value': self.total_value,
            'cash': self.cash,
            'positions': self.positions,
            'total_pnl': self.total_pnl,
            'daily_pnl': self.daily_pnl,
            'win_rate': f"{self.win_rate:.1%}"
        }


# ==================== DECISION ENGINE CLASS ====================

class DecisionEngine:
    """
    Decision engine that combines technical analysis, sentiment, and correlations
    to generate probabilistic trading signals.
    """
    
    def __init__(self, data_collector: DataCollector = None, 
                 sentiment_analyzer: SentimentAnalyzer = None):
        """
        Initialize the decision engine.
        
        Args:
            data_collector: DataCollector instance
            sentiment_analyzer: SentimentAnalyzer instance
        """
        self.data_collector = data_collector or DataCollector(simulation=True)
        self.sentiment_analyzer = sentiment_analyzer or SentimentAnalyzer()
        self.technical_analyzer = TechnicalAnalyzer()
        
        # Initialize ML Predictor (Blackbox Agent)
        self.ml_predictor = get_ml_predictor()
        self.ml_enabled = True
        
        self.settings = config.DECISION_SETTINGS
        self.portfolio = PortfolioState()
        
        # Cache for analysis results
        self.analysis_cache = {}
        self.cache_duration = 60  # seconds
        
        logger.info("DecisionEngine initialized")
    
    # ==================== ML COORDINATION (BLACKBOX AGENT) ====================
    
    def train_ml_model(self, symbol: str = None, df: pd.DataFrame = None) -> bool:
        """
        Train the ML predictor (blackbox agent) on historical data.
        
        Args:
            symbol: Symbol to train on (if df not provided)
            df: DataFrame with OHLCV data
            
        Returns:
            True if training succeeded, False otherwise
        """
        try:
            if df is None and symbol:
                # Fetch data if not provided
                df = self.data_collector.fetch_ohlcv(symbol, config.DEFAULT_TIMEFRAME, 200)
            
            if df is None or df.empty:
                logger.warning("No data available for ML training")
                return False
            
            # Train the ML model
            result = self.ml_predictor.train(df)
            if result:
                logger.info(f"ML model trained successfully for {symbol or 'default'}")
                self.ml_enabled = True
                return True
            return False
        except Exception as e:
            logger.error(f"ML training failed: {e}")
            return False
    
    def enable_ml(self, enabled: bool = True):
        """Enable or disable ML integration (blackbox agent coordination)"""
        self.ml_enabled = enabled
        logger.info(f"ML integration {'enabled' if enabled else 'disabled'}")
    
    def is_ml_ready(self) -> bool:
        """Check if ML predictor is ready for inference"""
        return self.ml_predictor.is_trained and self.ml_enabled
    
    def get_ml_prediction(self, symbol: str) -> Optional[Dict]:
        """
        Get ML prediction for a symbol (direct blackbox agent query).
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Prediction dict with 'prediction', 'confidence', or None
        """
        if not self.is_ml_ready():
            return None
        
        try:
            df = self.data_collector.fetch_ohlcv(symbol, config.DEFAULT_TIMEFRAME, 100)
            if df is not None:
                return self.ml_predictor.predict(df)
        except Exception as e:
            logger.warning(f"ML prediction failed for {symbol}: {e}")
        
        return None
    
    # ==================== SIGNAL GENERATION ====================
    
    def generate_signals(self, symbols: List[str] = None) -> List[TradingSignal]:
        """
        Generate trading signals for specified symbols.
        
        Args:
            symbols: List of trading symbols (uses config defaults if None)
            
        Returns:
            List of TradingSignal objects
        """
        if symbols is None:
            symbols = self.data_collector.get_supported_symbols()
        
        signals = []
        
        # Separate crypto and commodity symbols
        crypto_symbols = [s for s in symbols if s in config.CRYPTO_SYMBOLS.values()]
        commodity_symbols = [s for s in symbols if s in config.COMMODITY_TOKENS.values()]
        
        # Generate signals for each asset
        for symbol in symbols:
            try:
                signal = self._generate_signal(symbol)
                if signal:
                    signals.append(signal)
            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")
        
        # Sort by confidence
        signals.sort(key=lambda x: x.confidence, reverse=True)
        
        return signals
    
    def _generate_signal(self, symbol: str) -> Optional[TradingSignal]:
        """Generate a single trading signal for a symbol"""
        
        # Determine asset type
        asset_type = 'crypto' if symbol in config.CRYPTO_SYMBOLS.values() else 'commodity'
        
        # Get market data
        market_data = self.data_collector.fetch_market_data(symbol)
        
        if market_data.current_price <= 0:
            return None
        
        # Get technical analysis
        df = self.data_collector.fetch_ohlcv(symbol, config.DEFAULT_TIMEFRAME, 100)
        
        if df is None or df.empty:
            return None
        
        technical_analysis = self.technical_analyzer.analyze(df, symbol)
        
        # Get ML Prediction (Blackbox Agent Coordination)
        ml_score = 0.5
        ml_confidence = 0.0
        if self.ml_enabled and self.ml_predictor.is_trained:
            try:
                prediction = self.ml_predictor.predict(df)
                if prediction:
                    # Convert ML prediction to score (1=BULLISH, 0=NEUTRAL, -1=BEARISH)
                    ml_confidence = prediction.get('confidence', 0.5)
                    ml_direction = prediction.get('prediction', 0)  # -1, 0, or 1
                    ml_score = (ml_direction + 1) / 2  # Convert -1,0,1 to 0,0.5,1
                    logger.info(f"ML Prediction for {symbol}: direction={ml_direction}, confidence={ml_confidence}")
            except Exception as e:
                logger.warning(f"ML prediction failed for {symbol}: {e}")
        
        # Get sentiment analysis
        asset_name = symbol.split('/')[0]
        sentiment = self.sentiment_analyzer.get_combined_sentiment(asset_name)
        
        # Get correlations with other assets
        correlations = self._analyze_correlations(symbol, symbols=[
            s for s in self.data_collector.get_supported_symbols() if s != symbol
        ][:5])
        
        # Calculate volatility score
        volatility_score = self._calculate_volatility_score(df, technical_analysis)
        
        # Combine all factors (including ML prediction)
        combined_score = self._combine_factors(
            technical_analysis,
            sentiment,
            correlations,
            volatility_score,
            ml_score  # Include ML prediction in scoring
        )
        
        # Generate signal
        action = self._determine_action(combined_score)
        confidence = self._calculate_confidence(
            technical_analysis,
            sentiment,
            correlations,
            volatility_score,
            ml_confidence  # Include ML confidence
        )
        
        # Create trading signal
        signal = TradingSignal(
            symbol=symbol,
            asset_type=asset_type,
            action=action,
            confidence=confidence,
            strength=self._get_strength_label(confidence),
            current_price=market_data.current_price,
            technical_score=technical_analysis.technical_score,
            momentum_score=technical_analysis.momentum_score,
            sentiment_score=(sentiment['combined_score'] + 1) / 2,  # Convert -1,1 to 0,1
            correlation_score=correlations.get('avg_correlation', 0.5),
            volatility_score=volatility_score,
            ml_score=ml_score,  # Blackbox agent prediction score
            reason=self._generate_reason(technical_analysis, sentiment, action)
        )
        
        # Calculate price levels
        signal.entry_price = market_data.current_price
        signal.stop_loss, signal.take_profit = self._calculate_price_levels(
            market_data.current_price,
            technical_analysis.atr,
            action
        )
        
        # Risk management
        if action != 'HOLD':
            signal.risk_percent = self.settings['stop_loss_percent']
            signal.risk_reward_ratio = (
                abs(signal.take_profit - signal.entry_price) / 
                abs(signal.entry_price - signal.stop_loss)
                if signal.stop_loss > 0 else 0
            )
            signal.position_size = self._calculate_position_size(
                confidence,
                signal.risk_percent
            )
        
        return signal
    
    def _analyze_correlations(self, symbol: str, symbols: List[str]) -> Dict:
        """Analyze correlations with other assets"""
        if not symbols:
            return {'avg_correlation': 0.5, 'correlations': {}}
        
        correlations = {}
        
        for other_symbol in symbols[:5]:
            try:
                corr_data = self.data_collector.calculate_correlation(
                    symbol, other_symbol, self.settings['correlation_lookback']
                )
                correlations[other_symbol] = corr_data.correlation
            except:
                pass
        
        if not correlations:
            return {'avg_correlation': 0.5, 'correlations': {}}
        
        avg_corr = sum(correlations.values()) / len(correlations)
        
        return {
            'avg_correlation': avg_corr,
            'correlations': correlations
        }
    
    def _calculate_volatility_score(self, df: pd.DataFrame, 
                                   analysis: TechnicalAnalysis) -> float:
        """Calculate volatility score (0-1, lower is better for stability)"""
        try:
            returns = df['close'].pct_change().dropna()
            
            if len(returns) < 2:
                return 0.5
            
            # Calculate volatility (standard deviation of returns)
            volatility = returns.std()
            
            # Annualize if we have hourly data
            volatility = volatility * (24 ** 0.5)
            
            # Convert to score (0-1, where 1 = very volatile)
            # Typical crypto volatility: 0.5 to 2.0 annualized
            score = min(1.0, volatility / 1.5)
            
            return 1 - score  # Invert so higher = more stable
            
        except:
            return 0.5
    
    def _combine_factors(self, technical: TechnicalAnalysis, 
                        sentiment: Dict,
                        correlations: Dict,
                        volatility_score: float,
                        ml_score: float = 0.5) -> float:
        """Combine all factors into a single score (including ML prediction)"""
        weights = self.settings['weights']
        
        # Normalize sentiment score to 0-1
        sentiment_score = (sentiment['combined_score'] + 1) / 2
        
        # Correlation score (use absolute value to consider negative correlations)
        correlation_score = abs(correlations.get('avg_correlation', 0))
        
        # Calculate weighted score (include ML prediction)
        ml_weight = 0.15  # Weight for ML prediction
        base_score = (
            technical.technical_score * weights['technical'] +
            technical.momentum_score * weights['momentum'] +
            correlation_score * weights['correlation'] +
            sentiment_score * weights['sentiment'] +
            volatility_score * weights['volatility']
        )
        
        # Blend ML score with base score
        score = base_score * (1 - ml_weight) + ml_score * ml_weight
        
        return max(0, min(1, score))
    
    def _determine_action(self, score: float) -> str:
        """Determine action based on combined score"""
        strong_buy = self.settings['strong_signal_threshold']
        buy_threshold = self.settings['min_signal_confidence']
        
        if score >= strong_buy:
            return 'BUY'
        elif score >= buy_threshold:
            return 'BUY' if score > 0.5 else 'SELL'
        elif score <= (1 - strong_buy):
            return 'SELL'
        elif score <= (1 - buy_threshold):
            return 'SELL' if score < 0.5 else 'BUY'
        
        return 'HOLD'
    
    def _calculate_confidence(self, technical: TechnicalAnalysis,
                             sentiment: Dict,
                             correlations: Dict,
                             volatility_score: float,
                             ml_confidence: float = 0.5) -> float:
        """Calculate confidence level for the signal (including ML prediction)"""
        # Base confidence from technical analysis
        confidence = technical.technical_score
        
        # Adjust for sentiment confidence
        sentiment_conf = sentiment.get('confidence', 0.5)
        confidence = (confidence + sentiment_conf) / 2
        
        # Adjust for volatility (lower volatility = higher confidence)
        confidence = (confidence + volatility_score) / 2
        
        # Blend with ML confidence (if available)
        if ml_confidence > 0:
            confidence = confidence * 0.7 + ml_confidence * 0.3
        
        return max(0, min(1, confidence))
    
    def _get_strength_label(self, confidence: float) -> str:
        """Convert confidence to strength label"""
        if confidence >= 0.7:
            return 'STRONG'
        elif confidence >= 0.55:
            return 'MODERATE'
        return 'WEAK'
    
    def _calculate_price_levels(self, current_price: float, atr: float,
                                action: str) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels"""
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
    
    def _calculate_position_size(self, confidence: float, 
                                risk_percent: float) -> float:
        """Calculate position size based on confidence and risk"""
        max_position = self.settings['max_position_size']
        
        # Scale position size by confidence
        position = max_position * confidence
        
        return position
    
    def _generate_reason(self, technical: TechnicalAnalysis,
                       sentiment: Dict, action: str) -> str:
        """Generate human-readable reason for the signal"""
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
    
    # ==================== RANKING & FILTERING ====================
    
    def rank_signals(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """
        Rank signals by various criteria.
        
        Args:
            signals: List of trading signals
            
        Returns:
            Sorted list of signals
        """
        # Sort by confidence first
        ranked = sorted(signals, key=lambda x: x.confidence, reverse=True)
        
        return ranked
    
    def filter_signals(self, signals: List[TradingSignal], 
                      min_confidence: float = None) -> List[TradingSignal]:
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
    
    def get_top_signals(self, signals: List[TradingSignal], 
                       n: int = 5) -> List[TradingSignal]:
        """Get top N signals"""
        return signals[:n]
    
    # ==================== HEDGE RECOMMENDATIONS ====================
    
    def suggest_hedges(self, signal: TradingSignal, 
                       all_signals: List[TradingSignal]) -> List[TradingSignal]:
        """
        Suggest hedging positions based on correlations.
        
        Args:
            signal: Primary signal
            all_signals: All generated signals
            
        Returns:
            List of hedge signals
        """
        hedges = []
        
        # Look for negatively correlated assets
        for other in all_signals:
            if other.symbol == signal.symbol:
                continue
            
            # If signal is BUY, suggest hedges that are SELL
            if signal.action == 'BUY' and other.action == 'SELL':
                hedges.append(other)
            # If signal is SELL, suggest hedges that are BUY
            elif signal.action == 'SELL' and other.action == 'BUY':
                hedges.append(other)
        
        # Sort by confidence and return top 3
        hedges.sort(key=lambda x: x.confidence, reverse=True)
        
        return hedges[:3]
    
    # ==================== OUTPUT FORMATTING ====================
    
    def format_signal_display(self, signal: TradingSignal) -> str:
        """Format signal for console display"""
        emoji = {
            'BUY': '🟢',
            'SELL': '🔴',
            'HOLD': '⚪'
        }.get(signal.action, '⚪')
        
        return f"""
{emoji} {signal.symbol} - {signal.action} ({signal.strength})
   Price: ${signal.current_price:,.2f}
   Confidence: {signal.confidence:.1%}
   Technical: {signal.technical_score:.1%} | Momentum: {signal.momentum_score:.1%}
   Sentiment: {signal.sentiment_score:.1%} | Volatility: {signal.volatility_score:.1%}
   Entry: ${signal.entry_price:,.2f} | SL: ${signal.stop_loss:,.2f} | TP: ${signal.take_profit:,.2f}
   R/R: {signal.risk_reward_ratio:.2f} | Position: {signal.position_size:.1%}
   Reason: {signal.reason}
"""
    
    def generate_signal_report(self, signals: List[TradingSignal]) -> str:
        """Generate a formatted signal report"""
        report = []
        report.append("=" * 70)
        report.append("TRADING SIGNALS REPORT")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 70)
        
        # Group by action
        buy_signals = [s for s in signals if s.action == 'BUY']
        sell_signals = [s for s in signals if s.action == 'SELL']
        hold_signals = [s for s in signals if s.action == 'HOLD']
        
        report.append(f"\n📈 BUY SIGNALS ({len(buy_signals)})")
        report.append("-" * 40)
        
        for signal in buy_signals[:5]:
            report.append(f"  {signal.symbol}: {signal.confidence:.1%} confidence")
            report.append(f"    Entry: ${signal.entry_price:,.2f} | SL: ${signal.stop_loss:,.2f}")
            report.append(f"    R/R: {signal.risk_reward_ratio:.2f} | Position: {signal.position_size:.1%}")
        
        report.append(f"\n📉 SELL SIGNALS ({len(sell_signals)})")
        report.append("-" * 40)
        
        for signal in sell_signals[:5]:
            report.append(f"  {signal.symbol}: {signal.confidence:.1%} confidence")
            report.append(f"    Entry: ${signal.entry_price:,.2f} | SL: ${signal.stop_loss:,.2f}")
        
        if hold_signals:
            report.append(f"\n⚪ HOLD ({len(hold_signals)})")
            report.append("-" * 40)
            for signal in hold_signals[:3]:
                report.append(f"  {signal.symbol}: {signal.confidence:.1%} confidence")
        
        return "\n".join(report)
    
    def export_signals(self, signals: List[TradingSignal], filepath: str):
        """Export signals to JSON file"""
        data = {
            'generated_at': datetime.now().isoformat(),
            'signals': [s.to_dict() for s in signals]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Signals exported to {filepath}")


# ==================== STANDALONE FUNCTIONS ====================

def generate_all_signals() -> List[TradingSignal]:
    """Quick function to generate all signals"""
    engine = DecisionEngine()
    return engine.generate_signals()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*70)
    print("DECISION ENGINE TEST")
    print("="*70)
    
    # Initialize
    engine = DecisionEngine()
    
    # Generate signals
    print("\n🔄 Generating trading signals...")
    signals = engine.generate_signals()
    
    # Display report
    print(engine.generate_signal_report(signals))
    
    # Top signals
    print("\n🏆 TOP SIGNALS:")
    top = engine.get_top_signals(signals, 3)
    for signal in top:
        print(engine.format_signal_display(signal))
    
    print("\n✅ Test complete!")

