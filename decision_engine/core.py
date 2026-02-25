"""
Core Data Structures and DecisionEngine Class
Contains: TradingSignal, PortfolioState, DecisionEngine
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

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
    regime_score: float = 0.5  # HMM regime detection score (-1 bear to +1 bull)
    regime_name: str = "Unknown"  # Bull, Bear, Sideways
    
    # NEW: 5-Question Framework Scores
    what_score: float = 0.5      # Question 1: What to buy/sell
    why_score: float = 0.5       # Question 2: Reason score (0.6*Macro + 0.4*Sentiment)
    how_much_score: float = 0.5  # Question 3: Position size
    when_score: float = 0.5      # Question 4: Timing score (Monte Carlo)
    risk_score: float = 0.5      # Question 5: Risk check score
    
    # Macro and Sentiment breakdown for Question 2
    macro_score: float = 0.5     # Macro economic score
    reason_sentiment_score: float = 0.5  # Sentiment for reason calculation
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    timeframe: str = "1h"
    reason: str = ""
    
    # Risk metrics
    position_size: float = 0.0
    risk_percent: float = 0.0
    var_95: float = 0.0          # Value at Risk 95%
    cvar_95: float = 0.0         # Conditional VaR 95%
    
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
        self.settings = config.DECISION_SETTINGS
        self.portfolio = PortfolioState()
        
        # Cache for analysis results
        self.analysis_cache = {}
        self.cache_duration = 60  # seconds
        
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
        
        # HMM Regime Detector
        self._hmm_detector = None
        self._hmm_cache = {}  # symbol -> {regime_result, timestamp}
        self._hmm_cache_ttl = 3600  # 1 hour
        self._init_hmm()
        
        logger.info("DecisionEngine initialized")
    
    def _init_hmm(self):
        """Initialize HMM Regime Detector"""
        try:
            from src.hmm_regime import HMMRegimeDetector, RegimeAwareSignalGenerator
            self._hmm_detector = HMMRegimeDetector(n_regimes=3)
            self._regime_generator = RegimeAwareSignalGenerator(regime_detector=self._hmm_detector)
            logger.info("HMM Regime Detector initialized")
        except ImportError:
            logger.warning("HMM Regime Detector not available")
        except Exception as e:
            logger.warning(f"Failed to initialize HMM Regime Detector: {e}")
    
    # ==================== ML COORDINATION ====================
    
    def train_ml_model(self, symbol: str = None, df: pd.DataFrame = None) -> bool:
        """Train the ML predictor on historical data."""
        try:
            if df is None and symbol:
                df = self.data_collector.fetch_ohlcv(symbol, config.DEFAULT_TIMEFRAME, 200)
            
            if df is None or df.empty:
                logger.warning("No data available for ML training")
                return False
            
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
        """Enable or disable ML integration"""
        self.ml_enabled = enabled
        logger.info(f"ML integration {'enabled' if enabled else 'disabled'}")
    
    def is_ml_ready(self) -> bool:
        """Check if ML predictor is ready for inference"""
        return hasattr(self.ml_predictor, 'is_trained') and self.ml_predictor.is_trained and self.ml_enabled
    
    def get_ml_prediction(self, symbol: str) -> Optional[Dict]:
        """Get ML prediction for a symbol."""
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
        """Generate trading signals for specified symbols."""
        if symbols is None:
            symbols = self.data_collector.get_supported_symbols()
        
        signals = []
        
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
        from .signals import SignalGenerator
        from .monte_carlo import MonteCarloEngine
        
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
        
        # Get ML Prediction
        ml_score = 0.5
        ml_confidence = 0.0
        if self.is_ml_ready():
            try:
                prediction = self.ml_predictor.predict(df)
                if prediction:
                    ml_confidence = prediction.get('confidence', 0.5)
                    ml_direction = prediction.get('prediction', 0)
                    ml_score = (ml_direction + 1) / 2
            except Exception as e:
                logger.warning(f"ML prediction failed for {symbol}: {e}")
        
        # Get sentiment analysis
        asset_name = symbol.split('/')[0]
        sentiment = self.sentiment_analyzer.get_combined_sentiment(asset_name)
        
        # Get correlations
        correlations = self._analyze_correlations(symbol, [
            s for s in self.data_collector.get_supported_symbols() if s != symbol
        ][:5])
        
        # Calculate volatility score
        volatility_score = self._calculate_volatility_score(df, technical_analysis)
        
        # Get HMM Regime
        regime_score, regime_name = self._get_regime(df, symbol)
        
        # Run Monte Carlo
        mc_engine = MonteCarloEngine(self)
        mc_results = mc_engine.run(symbol, df, level=5)
        mc_score = mc_results.get('probability_up', 0.5)
        
        # Combine factors
        generator = SignalGenerator(self.settings)
        combined_score = generator.combine_factors(
            technical_analysis,
            sentiment,
            correlations,
            volatility_score,
            ml_score,
            mc_score
        )
        
        # Determine action
        action = generator.determine_action(combined_score)
        confidence = generator.calculate_confidence(
            technical_analysis,
            sentiment,
            volatility_score,
            ml_confidence
        )
        
        # Create trading signal
        signal = TradingSignal(
            symbol=symbol,
            asset_type=asset_type,
            action=action,
            confidence=confidence,
            strength=generator.get_strength_label(confidence),
            current_price=market_data.current_price,
            technical_score=technical_analysis.technical_score,
            momentum_score=technical_analysis.momentum_score,
            sentiment_score=(sentiment['combined_score'] + 1) / 2,
            correlation_score=correlations.get('avg_correlation', 0.5),
            volatility_score=volatility_score,
            ml_score=ml_score,
            regime_score=regime_score,
            regime_name=regime_name,
            reason=generator.generate_reason(technical_analysis, sentiment, action)
        )
        
        # Calculate price levels
        signal.entry_price = market_data.current_price
        signal.stop_loss, signal.take_profit = generator.calculate_price_levels(
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
            signal.position_size = generator.calculate_position_size(
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
            
            volatility = returns.std() * (24 ** 0.5)
            score = min(1.0, volatility / 1.5)
            
            return 1 - score
            
        except:
            return 0.5
    
    def _get_regime(self, df: pd.DataFrame, symbol: str) -> Tuple[float, str]:
        """Get HMM regime for symbol"""
        regime_score = 0.0
        regime_name = "Unknown"
        
        if self._hmm_detector and len(df) > 50:
            try:
                from src.hmm_regime import get_regime_score
                returns = df['close'].pct_change().dropna().values
                volatility = returns.rolling(20).std().dropna().values if hasattr(returns, 'rolling') else None
                
                if not self._hmm_detector.is_fitted:
                    self._hmm_detector.fit(returns[-100:], volatility[-100:] if volatility is not None else None)
                
                regime_result = self._hmm_detector.predict(returns, volatility)
                regime_score = get_regime_score(regime_result)
                regime_name = regime_result.current_regime.regime_name
            except Exception as e:
                logger.warning(f"HMM regime detection failed for {symbol}: {e}")
        
        return regime_score, regime_name
    
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
        import json
        data = {
            'generated_at': datetime.now().isoformat(),
            'signals': [s.to_dict() for s in signals]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Signals exported to {filepath}")
    
    def get_top_signals(self, signals: List[TradingSignal], n: int = 5) -> List[TradingSignal]:
        """Get top N signals"""
        return signals[:n]

