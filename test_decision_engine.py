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

# Create engine with lower threshold for testing
engine = DecisionEngine(
    portfolio_balance=100000,
    threshold_confidence=0.4,
    max_risk_per_trade=0.02,
    monte_carlo_sims=500
)

# Run trading cycle
result = engine.run_trading_cycle(assets)

# Print results
print('\n' + '=' * 60)
print('ORDERS GENERATED')
print('=' * 60)
for order in result['orders']:
    mc = order['monte_carlo']
    print(f"  {order['action']:4} {order['amount']:>10.2f} USDT of {order['asset']}")
    print(f"       Confidence: {order['confidence']:.2f}")
    print(f"       Semantic Score: {order['semantic_score']:.2f}")
    print(f"       Numeric Score: {order['numeric_score']:.2f}")
    print(f"       VaR (95%): {mc['var']:.2%}")
    print(f"       Expected Shortfall: {mc['expected_shortfall']:.2%}")
    print(f"       Prob Profit: {mc['prob_profit']:.1%}")
    print()

print('=' * 60)
print('PORTFOLIO SUMMARY')
print('=' * 60)
summary = result['portfolio']
print(f"  Cash: {summary['cash']:,.2f} USDT")
print(f"  Positions: {summary['n_positions']}")
for asset, amount in summary['positions'].items():
    print(f"    - {asset}: {amount:,.2f} USDT")
print(f"  Total Value: {summary['total_value']:,.2f} USDT")
print('=' * 60)
