"""
External Data Fetcher Module
Handles external API data fetching: sentiment, macro events, CMC context
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ExternalDataFetcher:
    """
    External data fetcher for sentiment, macro events, and market context.
    """
    
    def __init__(self, decision_engine: 'DecisionEngine'):
        """
        Initialize external data fetcher.
        
        Args:
            decision_engine: DecisionEngine instance
        """
        self.engine = decision_engine
        self._external_sentiment_cache = {}
        self._external_events_cache = {}
        self._external_natural_cache = {}
        self._cmc_cache = {}
        self._cache_ttl = 300  # 5 minutes
    
    def fetch_sentiment(self, symbol: str) -> Dict:
        """Fetch external sentiment for a symbol."""
        now = datetime.now().timestamp()
        cached = self._external_sentiment_cache.get(symbol)
        
        if cached and (now - cached['timestamp']) < self._cache_ttl:
            return cached['data']
        
        if not hasattr(self.engine, 'api_registry') or not self.engine.api_registry:
            return {'score': 0.0, 'confidence': 0.0, 'sources': 0}
        
        try:
            query = symbol.replace('USDT', '').replace('/', '').replace('USD', '')
            
            # Try async fetch if available
            if hasattr(self.engine.api_registry, 'fetch_sentiment'):
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor() as pool:
                            future = pool.submit(
                                asyncio.run, 
                                self.engine.api_registry.fetch_sentiment(query, limit=20)
                            )
                            records = future.result(timeout=30)
                    else:
                            records = loop.run_until_complete(
                                self.engine.api_registry.fetch_sentiment(query, limit=20)
                            )
                except:
                    records = []
            else:
                records = []
            
            if not records:
                return {'score': 0.0, 'confidence': 0.0, 'sources': 0}
            
            # Aggregate scores
            total_weight = 0.0
            weighted_score = 0.0
            
            for rec in records:
                score = rec.payload.get('sentiment_score', 0.0) if hasattr(rec, 'payload') else 0.0
                confidence = rec.confidence if hasattr(rec, 'confidence') else 0.5
                weight = confidence
                weighted_score += score * weight
                total_weight += weight
            
            avg_score = weighted_score / total_weight if total_weight > 0 else 0.0
            avg_confidence = min(1.0, len(records) / 10.0)
            
            result = {
                'score': max(-1.0, min(1.0, avg_score)),
                'confidence': avg_confidence,
                'sources': len(records)
            }
            
            self._external_sentiment_cache[symbol] = {'data': result, 'timestamp': now}
            return result
            
        except Exception as e:
            logger.warning(f"External sentiment fetch failed: {e}")
            return {'score': 0.0, 'confidence': 0.0, 'sources': 0}
    
    def fetch_macro_events(self, region: str = 'global') -> Dict:
        """Fetch macro economic events."""
        now = datetime.now().timestamp()
        cached = self._external_events_cache.get(region)
        
        if cached and (now - cached['timestamp']) < self._cache_ttl:
            return cached['data']
        
        if not hasattr(self.engine, 'api_registry') or not self.engine.api_registry:
            return {'events': [], 'avg_impact': 0.0, 'high_impact_count': 0}
        
        try:
            if hasattr(self.engine.api_registry, 'fetch_macro_events'):
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor() as pool:
                            future = pool.submit(
                                asyncio.run,
                                self.engine.api_registry.fetch_macro_events(region=region, days_ahead=7)
                            )
                            records = future.result(timeout=30)
                    else:
                        records = loop.run_until_complete(
                            self.engine.api_registry.fetch_macro_events(region=region, days_ahead=7)
                        )
                except:
                    records = []
            else:
                records = []
            
            events = []
            high_impact = 0
            
            for rec in records:
                impact = rec.payload.get('impact', 'low') if hasattr(rec, 'payload') else 'low'
                impact_score = {'high': 1.0, 'medium': 0.5, 'low': 0.2}.get(impact, 0.1)
                if impact == 'high':
                    high_impact += 1
                events.append({
                    'event': rec.payload.get('event', '') if hasattr(rec, 'payload') else '',
                    'impact': impact,
                    'impact_score': impact_score
                })
            
            avg_impact = sum(e['impact_score'] for e in events) / len(events) if events else 0.0
            
            result = {
                'events': events,
                'avg_impact': avg_impact,
                'high_impact_count': high_impact
            }
            
            self._external_events_cache[region] = {'data': result, 'timestamp': now}
            return result
            
        except Exception as e:
            logger.warning(f"Macro events fetch failed: {e}")
            return {'events': [], 'avg_impact': 0.0, 'high_impact_count': 0}
    
    def fetch_cmc_context(self, symbol: str = '') -> Dict:
        """Fetch CoinMarketCap market context."""
        if not hasattr(self.engine, '_cmc_client') or not self.engine._cmc_client:
            return {
                'btc_dominance': 0.0,
                'sentiment': 'neutral',
                'sentiment_score': 0.0,
                'volume_ratio': 0.0,
                'coin_data': None
            }
        
        now = datetime.now().timestamp()
        cache_key = f'cmc_{symbol}'
        cached = self._cmc_cache.get(cache_key)
        
        if cached and (now - cached['timestamp']) < self._cache_ttl:
            return cached['data']
        
        try:
            client = self.engine._cmc_client
            global_data = client.get_global_metrics()
            sentiment_data = client.get_market_sentiment_proxy()
            
            result = {
                'btc_dominance': global_data.get('btc_dominance', 0.0),
                'eth_dominance': global_data.get('eth_dominance', 0.0),
                'total_market_cap': global_data.get('total_market_cap', 0.0),
                'total_volume_24h': global_data.get('total_volume_24h', 0.0),
                'sentiment': sentiment_data.get('sentiment', 'neutral'),
                'sentiment_score': (sentiment_data.get('score', 50) - 50) / 50,
                'volume_ratio': sentiment_data.get('total_volume_24h', 0) / max(sentiment_data.get('total_market_cap', 1), 1),
                'coin_data': None
            }
            
            # Coin-specific data
            if symbol:
                coin_sym = symbol.replace('USDT', '').replace('/', '').replace('USD', '')
                coin_data = client.get_quote(coin_sym)
                if coin_data:
                    result['coin_data'] = {
                        'rank': coin_data.get('rank'),
                        'percent_change_24h': coin_data.get('percent_change_24h', 0),
                        'percent_change_7d': coin_data.get('percent_change_7d', 0),
                        'volume_24h': coin_data.get('volume_24h', 0)
                    }
            
            self._cmc_cache[cache_key] = {'data': result, 'timestamp': now}
            return result
            
        except Exception as e:
            logger.warning(f"CMC context fetch failed: {e}")
            return {
                'btc_dominance': 0.0,
                'sentiment': 'neutral',
                'sentiment_score': 0.0,
                'volume_ratio': 0.0,
                'coin_data': None
            }
    
    def fetch_natural_events(self, region: str = 'global') -> Dict:
        """Fetch natural/climate events."""
        now = datetime.now().timestamp()
        cached = self._external_natural_cache.get(region)
        
        if cached and (now - cached['timestamp']) < self._cache_ttl:
            return cached['data']
        
        if not hasattr(self.engine, 'api_registry') or not self.engine.api_registry:
            return {'events': [], 'avg_intensity': 0.0}
        
        try:
            if hasattr(self.engine.api_registry, 'fetch_natural_events'):
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    records = loop.run_until_complete(
                        self.engine.api_registry.fetch_natural_events(region=region)
                    )
                except:
                    records = []
            else:
                records = []
            
            events = []
            for rec in records:
                event_type = rec.payload.get('event_type', 'normal') if hasattr(rec, 'payload') else 'normal'
                intensity = rec.payload.get('intensity', 0.0) if hasattr(rec, 'payload') else 0.0
                if event_type != 'normal' and intensity > 0:
                    events.append({
                        'type': event_type,
                        'intensity': intensity
                    })
            
            avg_intensity = sum(e['intensity'] for e in events) / len(events) if events else 0.0
            
            result = {'events': events, 'avg_intensity': avg_intensity}
            self._external_natural_cache[region] = {'data': result, 'timestamp': now}
            return result
            
        except Exception as e:
            logger.warning(f"Natural events fetch failed: {e}")
            return {'events': [], 'avg_intensity': 0.0}

