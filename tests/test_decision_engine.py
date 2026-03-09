"""
Test script for the Decision Engine with Monte Carlo integration.
"""

from src.decision import DecisionEngine
import logging

logging.basicConfig(level=logging.INFO)

# Test assets with strong signals
assets = [
    {
        'name': 'Oro',
        'sentiment_score': 0.8,
        'event_impact': 0.3,
        'trend_signal': 0.5,
        'news_score': 0.6,
        'rsi_score': 0.7,
        'macd_score': 0.6,
        'volatility_score': 0.2,
        'momentum_score': 0.5,
        'volume_score': 0.4,
        'price': 2000,
        'volatility_annual': 0.15,
        'expected_return': 0.08
    },
    {
        'name': 'BTC',
        'sentiment_score': 0.9,
        'event_impact': 0.5,
        'trend_signal': 0.7,
        'news_score': 0.8,
        'rsi_score': 0.8,
        'macd_score': 0.7,
        'volatility_score': 0.3,
        'momentum_score': 0.6,
        'volume_score': 0.5,
        'price': 95000,
        'volatility_annual': 0.6,
        'expected_return': 0.15
    },
    {
        'name': 'Rame',
        'sentiment_score': -0.7,
        'event_impact': -0.4,
        'trend_signal': -0.5,
        'news_score': -0.6,
        'rsi_score': -0.6,
        'macd_score': -0.5,
        'volatility_score': 0.3,
        'momentum_score': -0.4,
        'volume_score': -0.3,
        'price': 4.5,
        'volatility_annual': 0.25,
        'expected_return': -0.05
    }
]

# Create engine with default settings
engine = DecisionEngine(portfolio_balance=100000)

# Set lower threshold for testing
engine.filter.threshold = 0.4

# Generate orders
orders = engine.generate_orders(assets)

# Print results
print('\n' + '=' * 60)
print('ORDERS GENERATED')
print('=' * 60)
for order in orders:
    print(f"  {order['action']:4} {order['amount']:>10.2f} USDT of {order['asset']}")
    print(f"       Confidence: {order['confidence']:.2f}")
    print()

print('=' * 60)
print('PORTFOLIO SUMMARY')
print('=' * 60)
print(f"  Cash: {engine.portfolio['cash']:,.2f} USDT")
print(f"  Positions: {len(engine.portfolio['positions'])}")
for asset, amount in engine.portfolio['positions'].items():
    print(f"    - {asset}: {amount:,.2f} USDT")
print('=' * 60)
