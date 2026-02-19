"""
Decision Engine Module
Generates probabilistic trading signals by combining all analysis types
"""

import logging
import random
import asyncio
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

try:
    from src.external import (
        APIRegistry, APICategory, NormalizedRecord,
        create_full_registry,
    )
    EXTERNAL_APIS_AVAILABLE = True
except ImportError:
    EXTERNAL_APIS_AVAILABLE = False

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
                 sentiment_analyzer: SentimentAnalyzer = None,
                 api_registry: 'APIRegistry' = None):
        """
        Initialize the decision engine.
        
        Args:
            data_collector: DataCollector instance
            sentiment_analyzer: SentimentAnalyzer instance
            api_registry: External API registry (src.external)
        """
        self.data_collector = data_collector or DataCollector(simulation=True)
        self.sentiment_analyzer = sentiment_analyzer or SentimentAnalyzer()
        self.technical_analyzer = TechnicalAnalyzer()
        
        # Initialize ML Predictor (Blackbox Agent)
        self.ml_predictor = get_ml_predictor()
        self.ml_enabled = True
        
        # Initialize External API Registry
        self.api_registry = api_registry
        if self.api_registry is None and EXTERNAL_APIS_AVAILABLE:
            try:
                self.api_registry = create_full_registry()
                logger.info(f"External API registry loaded: {self.api_registry.summary()['total_clients']} clients")
            except Exception as e:
                logger.warning(f"Failed to create API registry: {e}")
                self.api_registry = None
        
        self.settings = config.DECISION_SETTINGS
        self.portfolio = PortfolioState()
        
        # Cache for analysis results
        self.analysis_cache = {}
        self.cache_duration = 60  # seconds
        
        # External data cache
        self._external_sentiment_cache = {}  # symbol -> {data, timestamp}
        self._external_events_cache = {}     # region -> {data, timestamp}
        self._external_natural_cache = {}    # region -> {data, timestamp}
        self._external_cache_ttl = 300       # 5 minutes
        
        # CoinMarketCap client for global market context
        self._cmc_client = None
        self._cmc_cache = {}  # key -> {data, timestamp}
        try:
            from src.external.coinmarketcap_client import CoinMarketCapClient
            self._cmc_client = CoinMarketCapClient()
            if self._cmc_client.test_connection():
                logger.info("CoinMarketCap API connected")
            else:
                logger.warning("CoinMarketCap API key invalid or not set")
                self._cmc_client = None
        except Exception as e:
            logger.warning(f"CoinMarketCap client not available: {e}")
        
        # Monte Carlo simulation results cache
        self._mc_cache = {}  # symbol -> {results, timestamp}
        
        logger.info("DecisionEngine initialized (external APIs: %s)",
                    'enabled' if self.api_registry else 'disabled')
    
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
        return hasattr(self.ml_predictor, 'is_trained') and self.ml_predictor.is_trained and self.ml_enabled
    
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
        if self.ml_enabled and hasattr(self.ml_predictor, 'is_trained') and self.ml_predictor.is_trained:
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
        
        # Get external sentiment from APIs (NewsAPI, Benzinga, Twitter, GDELT)
        ext_sentiment = self.fetch_external_sentiment(symbol)
        if ext_sentiment['sources'] > 0:
            # Blend internal and external sentiment
            internal_score = sentiment.get('combined_score', 0)
            external_score = ext_sentiment['score']
            ext_weight = min(0.5, ext_sentiment['confidence'])
            sentiment['combined_score'] = internal_score * (1 - ext_weight) + external_score * ext_weight
            sentiment['confidence'] = max(sentiment.get('confidence', 0.5), ext_sentiment['confidence'])
            sentiment['external_sources'] = ext_sentiment['sources']
        
        # Get correlations with other assets
        correlations = self._analyze_correlations(symbol, symbols=[
            s for s in self.data_collector.get_supported_symbols() if s != symbol
        ][:5])
        
        # Calculate volatility score
        volatility_score = self._calculate_volatility_score(df, technical_analysis)
        
        # Run Monte Carlo simulation (all 5 levels)
        mc_results = self.run_monte_carlo(symbol, df, n_simulations=500, n_days=14, level=5)
        mc_score = mc_results.get('probability_up', 0.5)
        mc_confidence = mc_results.get('confidence', 0.0)
        
        # Combine all factors (including ML prediction and Monte Carlo)
        combined_score = self._combine_factors(
            technical_analysis,
            sentiment,
            correlations,
            volatility_score,
            ml_score,
            mc_score  # Include Monte Carlo probability
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
                        ml_score: float = 0.5,
                        mc_score: float = 0.5) -> float:
        """Combine all factors into a single score (including ML prediction and Monte Carlo)"""
        weights = self.settings['weights']
        
        # Normalize sentiment score to 0-1
        sentiment_score = (sentiment['combined_score'] + 1) / 2
        
        # Correlation score (use absolute value to consider negative correlations)
        correlation_score = abs(correlations.get('avg_correlation', 0))
        
        # Calculate weighted score (include ML prediction and Monte Carlo)
        ml_weight = 0.10  # Weight for ML prediction
        mc_weight = 0.10  # Weight for Monte Carlo probability
        base_weight = 1.0 - ml_weight - mc_weight
        
        base_score = (
            technical.technical_score * weights['technical'] +
            technical.momentum_score * weights['momentum'] +
            correlation_score * weights['correlation'] +
            sentiment_score * weights['sentiment'] +
            volatility_score * weights['volatility']
        )
        
        # Blend ML score and MC score with base score
        score = base_score * base_weight + ml_score * ml_weight + mc_score * mc_weight
        
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
    
    # ==================== EXTERNAL API DATA FETCHING ====================
    
    def _run_async(self, coro):
        """Run an async coroutine from sync context."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, coro)
                    return future.result(timeout=30)
            else:
                return loop.run_until_complete(coro)
        except RuntimeError:
            return asyncio.run(coro)
    
    def fetch_external_sentiment(self, symbol: str) -> Dict:
        """
        Fetch sentiment from external APIs (NewsAPI, Benzinga, Twitter, GDELT).
        Returns aggregated sentiment score and confidence.
        """
        if not self.api_registry:
            return {'score': 0.0, 'confidence': 0.0, 'sources': 0}
        
        # Check cache
        now = datetime.now().timestamp()
        cached = self._external_sentiment_cache.get(symbol)
        if cached and (now - cached['timestamp']) < self._external_cache_ttl:
            return cached['data']
        
        try:
            query = symbol.replace('USDT', '').replace('/', '')
            records = self._run_async(self.api_registry.fetch_sentiment(query, limit=20))
            
            if not records:
                return {'score': 0.0, 'confidence': 0.0, 'sources': 0}
            
            # Aggregate sentiment scores weighted by source reliability
            total_weight = 0.0
            weighted_score = 0.0
            for rec in records:
                score = rec.payload.get('sentiment_score', 0.0)
                client = self.api_registry.get_client(rec.source_api)
                weight = client.weight if client else 1.0
                confidence = rec.confidence
                w = weight * confidence
                weighted_score += score * w
                total_weight += w
            
            avg_score = weighted_score / total_weight if total_weight > 0 else 0.0
            avg_confidence = min(1.0, len(records) / 10.0)  # More sources = higher confidence
            
            result = {
                'score': max(-1.0, min(1.0, avg_score)),
                'confidence': avg_confidence,
                'sources': len(records),
                'records': records,
            }
            
            self._external_sentiment_cache[symbol] = {'data': result, 'timestamp': now}
            logger.info(f"External sentiment for {symbol}: score={avg_score:.3f}, sources={len(records)}")
            return result
            
        except Exception as e:
            logger.warning(f"External sentiment fetch failed for {symbol}: {e}")
            return {'score': 0.0, 'confidence': 0.0, 'sources': 0}
    
    def fetch_external_macro_events(self, region: str = 'global') -> Dict:
        """
        Fetch macro economic events from external APIs.
        Returns event impact scores for Monte Carlo conditioning.
        """
        if not self.api_registry:
            return {'events': [], 'avg_impact': 0.0, 'high_impact_count': 0}
        
        now = datetime.now().timestamp()
        cached = self._external_events_cache.get(region)
        if cached and (now - cached['timestamp']) < self._external_cache_ttl:
            return cached['data']
        
        try:
            records = self._run_async(self.api_registry.fetch_macro_events(region=region, days_ahead=7))
            
            events = []
            high_impact = 0
            for rec in records:
                impact = rec.payload.get('impact', 'low')
                impact_score = {'high': 1.0, 'medium': 0.5, 'low': 0.2}.get(impact, 0.1)
                if impact == 'high':
                    high_impact += 1
                events.append({
                    'event': rec.payload.get('event', ''),
                    'country': rec.payload.get('country', ''),
                    'impact': impact,
                    'impact_score': impact_score,
                    'timestamp': rec.timestamp,
                })
            
            avg_impact = sum(e['impact_score'] for e in events) / len(events) if events else 0.0
            
            result = {
                'events': events,
                'avg_impact': avg_impact,
                'high_impact_count': high_impact,
            }
            
            self._external_events_cache[region] = {'data': result, 'timestamp': now}
            logger.info(f"Macro events ({region}): {len(events)} events, {high_impact} high-impact")
            return result
            
        except Exception as e:
            logger.warning(f"Macro events fetch failed: {e}")
            return {'events': [], 'avg_impact': 0.0, 'high_impact_count': 0}
    
    def fetch_external_natural_events(self, region: str = 'global') -> Dict:
        """
        Fetch natural/climate events from external APIs.
        Returns event data for Monte Carlo multi-factor conditioning.
        """
        if not self.api_registry:
            return {'events': [], 'avg_intensity': 0.0}
        
        now = datetime.now().timestamp()
        cached = self._external_natural_cache.get(region)
        if cached and (now - cached['timestamp']) < self._external_cache_ttl:
            return cached['data']
        
        try:
            records = self._run_async(self.api_registry.fetch_natural_events(region=region))
            
            events = []
            for rec in records:
                event_type = rec.payload.get('event_type', 'normal')
                intensity = rec.payload.get('intensity', 0.0)
                if event_type != 'normal' and intensity > 0:
                    events.append({
                        'type': event_type,
                        'intensity': intensity,
                        'region': rec.payload.get('region', region),
                        'timestamp': rec.timestamp,
                    })
            
            avg_intensity = sum(e['intensity'] for e in events) / len(events) if events else 0.0
            
            result = {'events': events, 'avg_intensity': avg_intensity}
            self._external_natural_cache[region] = {'data': result, 'timestamp': now}
            return result
            
        except Exception as e:
            logger.warning(f"Natural events fetch failed: {e}")
            return {'events': [], 'avg_intensity': 0.0}
    
    def fetch_cmc_market_context(self, symbol: str = '') -> Dict:
        """
        Fetch global crypto market context from CoinMarketCap.
        Used in Monte Carlo Level 2+ for market-wide drift/volatility adjustments.
        
        Returns:
            dict with btc_dominance, market_sentiment, volume_ratio,
            coin-specific data (rank, market_cap_dominance, percent changes)
        """
        if not self._cmc_client:
            return {
                'btc_dominance': 0.0, 'sentiment': 'neutral', 'sentiment_score': 0.0,
                'volume_ratio': 0.0, 'coin_data': None,
            }
        
        now = datetime.now().timestamp()
        cache_key = f'cmc_{symbol}'
        cached = self._cmc_cache.get(cache_key)
        if cached and (now - cached['timestamp']) < self._external_cache_ttl:
            return cached['data']
        
        try:
            # Global metrics
            global_data = self._cmc_client.get_global_metrics()
            sentiment_data = self._cmc_client.get_market_sentiment_proxy()
            
            result = {
                'btc_dominance': global_data.get('btc_dominance', 0.0),
                'eth_dominance': global_data.get('eth_dominance', 0.0),
                'total_market_cap': global_data.get('total_market_cap', 0.0),
                'total_volume_24h': global_data.get('total_volume_24h', 0.0),
                'active_cryptos': global_data.get('active_cryptocurrencies', 0),
                'defi_volume_24h': global_data.get('defi_volume_24h', 0.0),
                'stablecoin_volume_24h': global_data.get('stablecoin_volume_24h', 0.0),
                'sentiment': sentiment_data.get('sentiment', 'neutral'),
                'sentiment_score': (sentiment_data.get('score', 50) - 50) / 50,  # normalize to -1..1
                'volume_ratio': sentiment_data.get('total_volume_24h', 0) / max(sentiment_data.get('total_market_cap', 1), 1),
                'coin_data': None,
            }
            
            # Coin-specific data
            if symbol:
                coin_sym = symbol.replace('USDT', '').replace('/', '').replace('USD', '')
                coin_data = self._cmc_client.get_quote(coin_sym)
                if coin_data:
                    result['coin_data'] = {
                        'rank': coin_data.get('rank'),
                        'market_cap_dominance': coin_data.get('market_cap_dominance', 0),
                        'percent_change_1h': coin_data.get('percent_change_1h', 0),
                        'percent_change_24h': coin_data.get('percent_change_24h', 0),
                        'percent_change_7d': coin_data.get('percent_change_7d', 0),
                        'percent_change_30d': coin_data.get('percent_change_30d', 0),
                        'volume_24h': coin_data.get('volume_24h', 0),
                    }
            
            self._cmc_cache[cache_key] = {'data': result, 'timestamp': now}
            logger.info(
                f"CMC context: sentiment={result['sentiment']}, "
                f"BTC dom={result['btc_dominance']:.1f}%%, "
                f"market cap=${result['total_market_cap']/1e12:.2f}T"
            )
            return result
            
        except Exception as e:
            logger.warning(f"CMC market context fetch failed: {e}")
            return {
                'btc_dominance': 0.0, 'sentiment': 'neutral', 'sentiment_score': 0.0,
                'volume_ratio': 0.0, 'coin_data': None,
            }
    
    # ==================== MONTE CARLO SIMULATION ENGINE ====================
    
    def run_monte_carlo(self, symbol: str, df: pd.DataFrame,
                        n_simulations: int = 1000, n_days: int = 30,
                        level: int = 5) -> Dict:
        """
        Run Monte Carlo simulation with 5 progressive levels.
        
        Level 1: Base GBM random walk
        Level 2: Conditional (macro events + sentiment)
        Level 3: Adaptive (reinforcement learning from past accuracy)
        Level 4: Multi-factor (natural events + energy + cross-correlations)
        Level 5: Semantic History (geopolitics + innovation + pattern matching)
        
        Args:
            symbol: Trading symbol
            df: OHLCV DataFrame
            n_simulations: Number of simulation paths
            n_days: Days to simulate forward
            level: Maximum MC level to run (1-5)
            
        Returns:
            Dict with simulation results, probabilities, and confidence
        """
        if df is None or len(df) < 20:
            return {'probability_up': 0.5, 'confidence': 0.0, 'level': 0}
        
        returns = df['close'].pct_change().dropna()
        if len(returns) < 10:
            return {'probability_up': 0.5, 'confidence': 0.0, 'level': 0}
        
        current_price = df['close'].iloc[-1]
        mu = returns.mean()
        sigma = returns.std()
        
        # ---- Level 1: Base Monte Carlo (GBM) ----
        np.random.seed(42)
        dt = 1.0 / 252  # daily
        paths = np.zeros((n_simulations, n_days))
        paths[:, 0] = current_price
        
        for t in range(1, n_days):
            z = np.random.standard_normal(n_simulations)
            paths[:, t] = paths[:, t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
        
        final_prices_l1 = paths[:, -1]
        prob_up_l1 = np.mean(final_prices_l1 > current_price)
        
        if level < 2:
            return {
                'probability_up': float(prob_up_l1),
                'expected_return': float(np.mean(final_prices_l1) / current_price - 1),
                'var_95': float(np.percentile(final_prices_l1 / current_price - 1, 5)),
                'confidence': 0.3,
                'level': 1,
            }
        
        # ---- Level 2: Conditional Monte Carlo (events + sentiment + CMC market context) ----
        ext_sentiment = self.fetch_external_sentiment(symbol)
        macro_events = self.fetch_external_macro_events()
        cmc_context = self.fetch_cmc_market_context(symbol)
        
        sentiment_adj = ext_sentiment['score'] * 0.002  # sentiment drift adjustment
        event_vol_adj = 1.0 + macro_events['avg_impact'] * 0.3  # volatility increase near events
        
        # CMC market context adjustments
        cmc_sentiment_adj = cmc_context['sentiment_score'] * 0.001  # global market mood
        cmc_vol_adj = 1.0
        if cmc_context['volume_ratio'] > 0.08:  # high volume = higher volatility
            cmc_vol_adj = 1.1
        elif cmc_context['volume_ratio'] < 0.03:  # low volume = lower volatility
            cmc_vol_adj = 0.9
        
        # Coin-specific momentum from CMC
        coin_momentum_adj = 0.0
        if cmc_context.get('coin_data'):
            pct_7d = cmc_context['coin_data'].get('percent_change_7d', 0) or 0
            coin_momentum_adj = pct_7d / 100 * 0.001  # small drift from 7d trend
        
        sigma_l2 = sigma * event_vol_adj * cmc_vol_adj
        mu_l2 = mu + sentiment_adj + cmc_sentiment_adj + coin_momentum_adj
        
        for t in range(1, n_days):
            z = np.random.standard_normal(n_simulations)
            paths[:, t] = paths[:, t-1] * np.exp((mu_l2 - 0.5 * sigma_l2**2) * dt + sigma_l2 * np.sqrt(dt) * z)
        
        final_prices_l2 = paths[:, -1]
        prob_up_l2 = np.mean(final_prices_l2 > current_price)
        
        if level < 3:
            return {
                'probability_up': float(prob_up_l2),
                'expected_return': float(np.mean(final_prices_l2) / current_price - 1),
                'var_95': float(np.percentile(final_prices_l2 / current_price - 1, 5)),
                'sentiment_impact': float(sentiment_adj),
                'event_vol_multiplier': float(event_vol_adj),
                'cmc_sentiment': cmc_context['sentiment'],
                'cmc_btc_dominance': cmc_context['btc_dominance'],
                'confidence': 0.45,
                'level': 2,
            }
        
        # ---- Level 3: Adaptive Monte Carlo (learning from past) ----
        # Check past MC accuracy and adjust
        past_mc = self._mc_cache.get(symbol, {})
        accuracy_adj = 0.0
        if past_mc and 'actual_return' in past_mc:
            predicted = past_mc.get('expected_return', 0)
            actual = past_mc.get('actual_return', 0)
            error = actual - predicted
            accuracy_adj = error * 0.1  # small correction based on past error
        
        mu_l3 = mu_l2 + accuracy_adj
        
        for t in range(1, n_days):
            z = np.random.standard_normal(n_simulations)
            # Dynamic volatility: use EWMA
            if t > 1:
                recent_vol = np.std(np.log(paths[:, t-1] / paths[:, t-2]))
                sigma_dynamic = 0.7 * sigma_l2 + 0.3 * recent_vol
            else:
                sigma_dynamic = sigma_l2
            paths[:, t] = paths[:, t-1] * np.exp((mu_l3 - 0.5 * sigma_dynamic**2) * dt + sigma_dynamic * np.sqrt(dt) * z)
        
        final_prices_l3 = paths[:, -1]
        prob_up_l3 = np.mean(final_prices_l3 > current_price)
        
        if level < 4:
            return {
                'probability_up': float(prob_up_l3),
                'expected_return': float(np.mean(final_prices_l3) / current_price - 1),
                'var_95': float(np.percentile(final_prices_l3 / current_price - 1, 5)),
                'accuracy_adjustment': float(accuracy_adj),
                'confidence': 0.55,
                'level': 3,
            }
        
        # ---- Level 4: Multi-Factor Monte Carlo ----
        natural_events = self.fetch_external_natural_events()
        natural_adj = natural_events['avg_intensity'] * 0.005  # commodity impact
        
        # Cross-asset correlation factor
        corr_factor = 1.0
        try:
            other_symbols = [s for s in self.data_collector.get_supported_symbols() if s != symbol][:3]
            for other in other_symbols:
                corr_data = self.data_collector.calculate_correlation(symbol, other, 30)
                if abs(corr_data.correlation) > 0.7:
                    corr_factor *= (1.0 + abs(corr_data.correlation) * 0.1)
        except Exception:
            pass
        
        sigma_l4 = sigma_l2 * corr_factor
        mu_l4 = mu_l3 - natural_adj  # natural events typically increase uncertainty
        
        # Regime switching: detect if we're in high/low vol regime
        recent_vol = returns.iloc[-20:].std() if len(returns) >= 20 else sigma
        long_vol = returns.std()
        regime_ratio = recent_vol / long_vol if long_vol > 0 else 1.0
        sigma_l4 *= regime_ratio
        
        for t in range(1, n_days):
            z = np.random.standard_normal(n_simulations)
            paths[:, t] = paths[:, t-1] * np.exp((mu_l4 - 0.5 * sigma_l4**2) * dt + sigma_l4 * np.sqrt(dt) * z)
        
        final_prices_l4 = paths[:, -1]
        prob_up_l4 = np.mean(final_prices_l4 > current_price)
        
        if level < 5:
            return {
                'probability_up': float(prob_up_l4),
                'expected_return': float(np.mean(final_prices_l4) / current_price - 1),
                'var_95': float(np.percentile(final_prices_l4 / current_price - 1, 5)),
                'natural_impact': float(natural_adj),
                'regime_ratio': float(regime_ratio),
                'confidence': 0.65,
                'level': 4,
            }
        
        # ---- Level 5: Semantic History Monte Carlo ----
        # Pattern matching: find similar historical periods
        lookback = min(len(returns), 252)
        current_pattern = returns.iloc[-20:].values if len(returns) >= 20 else returns.values
        
        best_match_score = 0.0
        best_match_return = 0.0
        
        for start in range(0, lookback - 40, 5):
            historical_pattern = returns.iloc[start:start+20].values
            if len(historical_pattern) == len(current_pattern):
                correlation = np.corrcoef(current_pattern, historical_pattern)[0, 1]
                if not np.isnan(correlation) and abs(correlation) > best_match_score:
                    best_match_score = abs(correlation)
                    # What happened after this historical pattern?
                    future_slice = returns.iloc[start+20:start+40]
                    if len(future_slice) > 0:
                        best_match_return = future_slice.mean()
        
        # Black swan detection: check for extreme tail events
        tail_threshold = returns.quantile(0.01)
        black_swan_prob = np.mean(returns < tail_threshold * 2)
        
        # Adjust final simulation with semantic history
        semantic_adj = best_match_return * best_match_score * 0.3
        mu_l5 = mu_l4 + semantic_adj
        
        # Add fat-tail component (Student-t instead of normal)
        df_t = max(3, int(10 - black_swan_prob * 100))  # degrees of freedom
        
        for t in range(1, n_days):
            z = np.random.standard_t(df_t, n_simulations) / np.sqrt(df_t / (df_t - 2))
            paths[:, t] = paths[:, t-1] * np.exp((mu_l5 - 0.5 * sigma_l4**2) * dt + sigma_l4 * np.sqrt(dt) * z)
        
        final_prices_l5 = paths[:, -1]
        prob_up_l5 = np.mean(final_prices_l5 > current_price)
        
        # Store for adaptive learning (Level 3 next time)
        mc_result = {
            'probability_up': float(prob_up_l5),
            'expected_return': float(np.mean(final_prices_l5) / current_price - 1),
            'var_95': float(np.percentile(final_prices_l5 / current_price - 1, 5)),
            'cvar_95': float(np.mean(final_prices_l5[final_prices_l5 < np.percentile(final_prices_l5, 5)] / current_price - 1)),
            'semantic_match_score': float(best_match_score),
            'semantic_adjustment': float(semantic_adj),
            'black_swan_probability': float(black_swan_prob),
            'fat_tail_df': df_t,
            'all_levels': {
                'l1_prob': float(prob_up_l1),
                'l2_prob': float(prob_up_l2),
                'l3_prob': float(prob_up_l3),
                'l4_prob': float(prob_up_l4),
                'l5_prob': float(prob_up_l5),
            },
            'confidence': 0.75,
            'level': 5,
        }
        
        self._mc_cache[symbol] = {**mc_result, 'timestamp': datetime.now().timestamp()}
        return mc_result
    
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

