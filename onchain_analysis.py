"""
On-Chain Analysis Module
Provides on-chain metrics and whale tracking for crypto assets
"""

import random
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class OnChainMetrics:
    """On-chain metrics for an asset"""
    symbol: str
    timestamp: datetime
    
    # Exchange Flow
    exchange_inflow: float = 0.0
    exchange_outflow: float = 0.0
    net_exchange_flow: float = 0.0
    
    # Whale Activity
    large_transactions: int = 0
    whale_buy_count: int = 0
    whale_sell_count: int = 0
    
    # Network Health
    active_addresses: int = 0
    new_addresses: int = 0
    transaction_count: int = 0
    
    # Holder Metrics
    holder_count: int = 0
    new_holders_24h: int = 0
    
    # Miner Metrics
    hash_rate: float = 0.0
    difficulty: float = 0.0
    miner_revenue: float = 0.0
    
    # DeFi Metrics (for ETH)
    tvl: float = 0.0
    gas_price: float = 0.0


class OnChainAnalyzer:
    """
    Analyzes on-chain data for crypto assets.
    In production, would connect to APIs like Glassnode, IntoTheBlock, Chainalysis
    """
    
    def __init__(self):
        """Initialize the on-chain analyzer"""
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
        
        # Thresholds for whale detection (in USD)
        self.whale_threshold = 100000
        
        logger.info("OnChainAnalyzer initialized")
    
    def get_metrics(self, symbol: str, force_refresh: bool = False) -> OnChainMetrics:
        """
        Get on-chain metrics for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            force_refresh: Force refresh cached data
            
        Returns:
            OnChainMetrics object
        """
        # In production, fetch from APIs
        # For now, generate simulated data
        return self._generate_metrics(symbol)
    
    def _generate_metrics(self, symbol: str) -> OnChainMetrics:
        """Generate simulated on-chain metrics"""
        
        # Base values based on asset
        if 'BTC' in symbol:
            base_addresses = 1_000_000
            base_txs = 300_000
            base_hash_rate = 500_000_000_000  # EH/s
        elif 'ETH' in symbol:
            base_addresses = 500_000
            base_txs = 1_000_000
            base_hash_rate = 100_000_000  # TH/s
        else:
            base_addresses = 100_000
            base_txs = 50_000
            base_hash_rate = 1_000_000
        
        # Add some randomness
        active_addresses = int(base_addresses * random.uniform(0.9, 1.1))
        new_addresses = int(active_addresses * random.uniform(0.02, 0.05))
        transaction_count = int(base_txs * random.uniform(0.8, 1.2))
        
        # Exchange flows
        exchange_inflow = random.uniform(1000, 50000)
        exchange_outflow = random.uniform(1000, 50000)
        net_flow = exchange_outflow - exchange_inflow
        
        # Whale activity
        large_txs = random.randint(50, 500)
        whale_buys = random.randint(10, 100)
        whale_sells = random.randint(10, 100)
        
        return OnChainMetrics(
            symbol=symbol,
            timestamp=datetime.now(),
            exchange_inflow=exchange_inflow,
            exchange_outflow=exchange_outflow,
            net_exchange_flow=net_flow,
            large_transactions=large_txs,
            whale_buy_count=whale_buys,
            whale_sell_count=whale_sells,
            active_addresses=active_addresses,
            new_addresses=new_addresses,
            transaction_count=transaction_count,
            holder_count=active_addresses * 2,
            new_holders_24h=new_addresses,
            hash_rate=base_hash_rate * random.uniform(0.95, 1.05),
            difficulty=random.uniform(1000, 10000),
            miner_revenue=random.uniform(10, 50) * 1_000_000,
            tvl=random.uniform(1, 50) * 1_000_000_000 if 'ETH' in symbol else 0,
            gas_price=random.uniform(10, 100) if 'ETH' in symbol else 0
        )
    
    def analyze_whale_activity(self, symbol: str) -> Dict:
        """
        Analyze whale activity for a symbol.
        
        Returns:
            Dictionary with whale metrics
        """
        metrics = self.get_metrics(symbol)
        
        # Determine whale sentiment
        net_whale_activity = metrics.whale_buy_count - metrics.whale_sell_count
        
        if net_whale_activity > 20:
            sentiment = 'bullish'
        elif net_whale_activity < -20:
            sentiment = 'bearish'
        else:
            sentiment = 'neutral'
        
        return {
            'symbol': symbol,
            'timestamp': metrics.timestamp.isoformat(),
            'large_transactions': metrics.large_transactions,
            'whale_buy_count': metrics.whale_buy_count,
            'whale_sell_count': metrics.whale_sell_count,
            'net_activity': net_whale_activity,
            'sentiment': sentiment,
            'activity_level': 'high' if metrics.large_transactions > 200 else 'medium' if metrics.large_transactions > 100 else 'low'
        }
    
    def analyze_exchange_flows(self, symbol: str) -> Dict:
        """
        Analyze exchange flows.
        
        Returns:
            Dictionary with flow analysis
        """
        metrics = self.get_metrics(symbol)
        
        # Interpret flows
        if metrics.net_exchange_flow > 0:
            interpretation = 'bullish'  # More outflows = accumulation
        elif metrics.net_exchange_flow < 0:
            interpretation = 'bearish'  # More inflows = distribution
        else:
            interpretation = 'neutral'
        
        return {
            'symbol': symbol,
            'inflow': metrics.exchange_inflow,
            'outflow': metrics.exchange_outflow,
            'net_flow': metrics.net_exchange_flow,
            'interpretation': interpretation,
            'flow_ratio': metrics.exchange_outflow / metrics.exchange_inflow if metrics.exchange_inflow > 0 else 1.0
        }
    
    def analyze_network_health(self, symbol: str) -> Dict:
        """
        Analyze network health metrics.
        
        Returns:
            Dictionary with network health
        """
        metrics = self.get_metrics(symbol)
        
        # Calculate growth metrics
        address_growth = (metrics.new_addresses / metrics.active_addresses * 100) if metrics.active_addresses > 0 else 0
        
        # Health score
        health_score = 0.5
        
        # Positive indicators
        if address_growth > 3:
            health_score += 0.2
        if metrics.transaction_count > 100000:
            health_score += 0.15
        if metrics.net_exchange_flow > 0:
            health_score += 0.15
            
        return {
            'symbol': symbol,
            'active_addresses': metrics.active_addresses,
            'new_addresses_24h': metrics.new_addresses,
            'address_growth_pct': round(address_growth, 2),
            'transaction_count': metrics.transaction_count,
            'health_score': min(1.0, health_score),
            'health_status': 'excellent' if health_score > 0.8 else 'good' if health_score > 0.6 else 'fair' if health_score > 0.4 else 'poor'
        }
    
    def get_combined_analysis(self, symbol: str) -> Dict:
        """
        Get combined on-chain analysis.
        
        Returns:
            Dictionary with all on-chain metrics
        """
        whale = self.analyze_whale_activity(symbol)
        flows = self.analyze_exchange_flows(symbol)
        network = self.analyze_network_health(symbol)
        
        # Calculate overall score
        score = 0.5
        
        if whale['sentiment'] == 'bullish':
            score += 0.2
        elif whale['sentiment'] == 'bearish':
            score -= 0.2
            
        if flows['interpretation'] == 'bullish':
            score += 0.15
        elif flows['interpretation'] == 'bearish':
            score -= 0.15
            
        score += (network['health_score'] - 0.5) * 0.3
        
        return {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'whale_activity': whale,
            'exchange_flows': flows,
            'network_health': network,
            'overall_score': max(0, min(1, score)),
            'signal': 'BUY' if score > 0.65 else 'SELL' if score < 0.35 else 'HOLD'
        }


def get_onchain_analysis(symbol: str) -> Dict:
    """Get on-chain analysis for a symbol"""
    analyzer = OnChainAnalyzer()
    return analyzer.get_combined_analysis(symbol)


if __name__ == "__main__":
    # Test
    analyzer = OnChainAnalyzer()
    result = analyzer.get_combined_analysis("BTC/USDT")
    print(result)
