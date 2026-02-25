"""
5-Question Decision Framework
Complete decision pipeline using 5 questions: What, Why, How Much, When, Risk
"""

import logging
from typing import Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


class FiveQuestionFramework:
    """
    5-Question Decision Framework:
    
    Q1: What to buy/sell (ML + Technical)
    Q2: Why (Macro + Sentiment)
    Q3: How much (Position sizing)
    Q4: When (Monte Carlo timing)
    Q5: Risk control (VaR/CVaR)
    """
    
    def __init__(self, decision_engine: 'DecisionEngine'):
        self.engine = decision_engine
        self.settings = decision_engine.settings
        self.portfolio = decision_engine.portfolio
    
    def unified_decision(self, symbol: str) -> Dict:
        """Execute the complete 5-question decision pipeline."""
        logger.info(f"Processing unified decision for {symbol}")
        
        market_data = self.engine.data_collector.fetch_market_data(symbol)
        if market_data.current_price <= 0:
            return {'error': 'Invalid price', 'action': 'HOLD'}
        
        df = self.engine.data_collector.fetch_ohlcv(symbol, '1h', 100)
        
        # Q1: What
        what_result = self.answer_what(symbol, df)
        action = what_result.get('action', 'HOLD')
        
        if action == 'HOLD':
            return {
                'symbol': symbol,
                'action': 'HOLD',
                'reason': 'What score indicates HOLD',
                'what_score': what_result.get('what_score', 0),
                'prices': {'current': market_data.current_price}
            }
        
        # Q2: Why
        why_result = self.answer_why(symbol, df)
        
        # Q3: How much
        how_much_result = self.answer_how_much(
            symbol, why_result.get('why_score', 0.5), market_data.current_price
        )
        
        # Q4: When
        when_result = self.answer_when(symbol, df)
        
        # Q5: Risk
        risk_result = self.answer_risk(
            symbol, action, how_much_result.get('position_size', 0),
            market_data.current_price, when_result.get('when_score', 0.5)
        )
        
        if not risk_result.get('passed'):
            return {
                'symbol': symbol,
                'action': 'HOLD',
                'reason': f"Risk failed: {risk_result.get('reason', '')}",
                'risk_score': risk_result.get('risk_score', 0),
                'prices': {'current': market_data.current_price},
                'decision_flow': {
                    'what': what_result, 'why': why_result,
                    'how_much': how_much_result, 'when': when_result, 'risk': risk_result
                }
            }
        
        # Final decision
        final_confidence = (
            what_result.get('what_score', 0.5) * 0.25 +
            why_result.get('why_score', 0.5) * 0.25 +
            when_result.get('when_score', 0.5) * 0.25 +
            risk_result.get('risk_score', 0.5) * 0.25
        )
        
        stop_loss_pct = self.settings.get('stop_loss_percent', 0.02)
        take_profit_pct = self.settings.get('take_profit_percent', 0.05)
        
        if action == 'BUY':
            entry_price = market_data.current_price
            stop_loss = entry_price * (1 - stop_loss_pct)
            take_profit = entry_price * (1 + take_profit_pct)
        else:
            entry_price = market_data.current_price
            stop_loss = entry_price * (1 + stop_loss_pct)
            take_profit = entry_price * (1 - take_profit_pct)
        
        risk_reward = (take_profit - entry_price) / (entry_price - stop_loss) if stop_loss > 0 else 0
        
        return {
            'symbol': symbol,
            'action': action,
            'confidence': final_confidence,
            'strength': 'STRONG' if final_confidence >= 0.7 else 'MODERATE' if final_confidence >= 0.55 else 'WEAK',
            'position_size': how_much_result.get('position_size', 0),
            'prices': {
                'current': market_data.current_price,
                'entry': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_reward': risk_reward
            },
            'scores': {
                'what': what_result.get('what_score', 0),
                'why': why_result.get('why_score', 0),
                'how_much': how_much_result.get('how_much_score', 0),
                'when': when_result.get('when_score', 0),
                'risk': risk_result.get('risk_score', 0)
            },
            'risk_metrics': {
                'var_95': risk_result.get('var_95', 0),
                'cvar_95': risk_result.get('cvar_95', 0)
            }
        }
    
    def answer_what(self, symbol: str, df: pd.DataFrame = None) -> Dict:
        """Q1: What to buy/sell"""
        market_data = self.engine.data_collector.fetch_market_data(symbol)
        if market_data.current_price <= 0:
            return {'action': 'HOLD', 'what_score': 0.0}
        
        if df is None:
            df = self.engine.data_collector.fetch_ohlcv(symbol, '1h', 100)
        
        if df is None or df.empty:
            return {'action': 'HOLD', 'what_score': 0.0}
        
        technical_analysis = self.engine.technical_analyzer.analyze(df, symbol)
        
        ml_score = 0.5
        if self.engine.is_ml_ready():
            try:
                prediction = self.engine.ml_predictor.predict(df)
                if prediction:
                    ml_direction = prediction.get('prediction', 0)
                    ml_score = (ml_direction + 1) / 2
            except:
                pass
        
        what_score = ml_score * 0.6 + technical_analysis.technical_score * 0.4
        
        if what_score >= 0.65:
            action = 'BUY'
        elif what_score <= 0.35:
            action = 'SELL'
        else:
            action = 'HOLD'
        
        return {
            'action': action,
            'what_score': what_score,
            'ml_score': ml_score,
            'technical_score': technical_analysis.technical_score
        }
    
    def answer_why(self, symbol: str, df: pd.DataFrame = None) -> Dict:
        """Q2: Why (Macro + Sentiment)"""
        macro_events = self._fetch_macro()
        
        if macro_events.get('high_impact_count', 0) > 3:
            macro_score = -0.3
        elif macro_events.get('high_impact_count', 0) > 1:
            macro_score = 0.0
        else:
            macro_score = 0.3 + (1 - macro_events.get('avg_impact', 0)) * 0.4
        
        ext_sentiment = self._fetch_sentiment(symbol)
        sentiment_score = ext_sentiment.get('score', 0.0)
        
        raw_why = 0.6 * macro_score + 0.4 * sentiment_score
        why_score = (raw_why + 1) / 2
        
        return {
            'why_score': why_score,
            'macro_score': macro_score,
            'sentiment_score': sentiment_score,
            'reason': f"Macro: {macro_score:.2f}, Sentiment: {sentiment_score:.2f}"
        }
    
    def answer_how_much(self, symbol: str, why_score: float, current_price: float) -> Dict:
        """Q3: How much (position sizing)"""
        max_position = self.settings.get('max_position_size', 0.1)
        position_size = max_position * why_score
        total_value = getattr(self.portfolio, 'total_value', 10000)
        
        return {
            'how_much_score': why_score,
            'position_size': position_size,
            'position_value': position_size * total_value,
            'position_units': (position_size * total_value) / current_price if current_price > 0 else 0
        }
    
    def answer_when(self, symbol: str, df: pd.DataFrame = None) -> Dict:
        """Q4: When (Monte Carlo timing)"""
        if df is None:
            df = self.engine.data_collector.fetch_ohlcv(symbol, '1h', 100)
        
        if df is None or df.empty:
            return {'when_score': 0.5, 'probability_up': 0.5}
        
        from .monte_carlo import MonteCarloEngine
        mc = MonteCarloEngine(self.engine)
        mc_results = mc.run(symbol, df, n_simulations=500, n_days=14, level=5)
        
        return {
            'when_score': mc_results.get('probability_up', 0.5),
            'probability_up': mc_results.get('probability_up', 0.5),
            'confidence': mc_results.get('confidence', 0.5)
        }
    
    def answer_risk(self, symbol: str, action: str, position_size: float,
                   current_price: float, when_score: float) -> Dict:
        """Q5: Risk control"""
        returns_std = 0.02
        try:
            df = self.engine.data_collector.fetch_ohlcv(symbol, '1h', 100)
            if df is not None and len(df) > 20:
                returns = df['close'].pct_change().dropna()
                returns_std = returns.std() if len(returns) > 0 else 0.02
        except:
            pass
        
        total_value = getattr(self.portfolio, 'total_value', 10000)
        position_value = position_size * total_value
        var_95 = 1.65 * returns_std * position_value
        cvar_95 = 2.0 * returns_std * position_value
        
        checks_passed = True
        reasons = []
        
        max_pos = self.settings.get('max_position_size', 0.1)
        if position_size > max_pos:
            checks_passed = False
            reasons.append(f"Position {position_size:.1%} > max {max_pos:.1%}")
        
        var_pct = var_95 / total_value if total_value > 0 else 0
        if var_pct > 0.05:
            checks_passed = False
            reasons.append(f"VaR {var_pct:.1%} > 5%")
        
        if when_score < 0.4:
            checks_passed = False
            reasons.append(f"Timing {when_score:.1%} < 40%")
        
        risk_score = 0.5
        risk_score -= (position_size / max_pos) * 0.2 if max_pos > 0 else 0
        risk_score += when_score * 0.2
        risk_score = max(0, min(1, risk_score))
        
        return {
            'risk_score': risk_score,
            'passed': checks_passed,
            'reason': '; '.join(reasons) if reasons else 'All passed',
            'var_95': var_95,
            'cvar_95': cvar_95
        }
    
    def generate_signals(self, symbols: List[str] = None) -> List['TradingSignal']:
        """Generate signals using 5Q framework."""
        if symbols is None:
            symbols = self.engine.data_collector.get_supported_symbols()
        
        from .core import TradingSignal
        import config
        
        signals = []
        for symbol in symbols:
            try:
                decision = self.unified_decision(symbol)
                if decision.get('action') == 'HOLD':
                    continue
                
                asset_type = 'crypto' if symbol in config.CRYPTO_SYMBOLS.values() else 'commodity'
                signal = TradingSignal(
                    symbol=symbol,
                    asset_type=asset_type,
                    action=decision.get('action', 'HOLD'),
                    confidence=decision.get('confidence', 0.5),
                    strength=decision.get('strength', 'WEAK'),
                    current_price=decision['prices']['current'],
                    entry_price=decision['prices'].get('entry', 0),
                    stop_loss=decision['prices'].get('stop_loss', 0),
                    take_profit=decision['prices'].get('take_profit', 0),
                    risk_reward_ratio=decision['prices'].get('risk_reward', 0),
                    what_score=decision['scores'].get('what', 0.5),
                    why_score=decision['scores'].get('why', 0.5),
                    how_much_score=decision['scores'].get('how_much', 0.5),
                    when_score=decision['scores'].get('when', 0.5),
                    risk_score=decision['scores'].get('risk', 0.5),
                    position_size=decision.get('position_size', 0)
                )
                signals.append(signal)
            except Exception as e:
                logger.error(f"Error: {e}")
        
        return sorted(signals, key=lambda x: x.confidence, reverse=True)
    
    def _fetch_sentiment(self, symbol: str) -> Dict:
        if hasattr(self.engine, 'fetch_external_sentiment'):
            return self.engine.fetch_external_sentiment(symbol)
        return {'score': 0.0}
    
    def _fetch_macro(self) -> Dict:
        if hasattr(self.engine, 'fetch_external_macro_events'):
            return self.engine.fetch_external_macro_events()
        return {'high_impact_count': 0, 'avg_impact': 0.0}

