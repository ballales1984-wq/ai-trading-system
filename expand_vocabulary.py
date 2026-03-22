#!/usr/bin/env python3
"""
Semantic Vocabulary Expansion Script
===================================
Adds more financial/trading concepts with relationships and multilingual support.

Run: python expand_vocabulary.py
"""

import json
from datetime import datetime
from typing import List, Dict, Any

# New concepts to add (with related concepts linking)
NEW_CONCEPTS = [
    # === ON-CHAIN METRICS ===
    {
        "id": "on_chain_volume",
        "term": "On-Chain Volume",
        "definition": "Total value transferred on blockchain",
        "category": "onchain",
        "keywords": ["on-chain", "volume", "transfers", "transactions", "活性"],
        "related_concepts": ["volume", "liquidity"],
        "language": "en"
    },
    {
        "id": "active_addresses",
        "term": "Active Addresses",
        "definition": "Unique addresses participating in transactions",
        "category": "onchain",
        "keywords": ["active addresses", "unique wallets", "users", "participants"],
        "related_concepts": ["volume", "network_activity"],
        "language": "en"
    },
    {
        "id": "network_hashrate",
        "term": "Network Hash Rate",
        "definition": "Total computational power securing the network",
        "category": "onchain",
        "keywords": ["hashrate", "hash rate", "mining difficulty", "computational power"],
        "related_concepts": ["security", "network_activity"],
        "language": "en"
    },
    {
        "id": "exchange_reserves",
        "term": "Exchange Reserves",
        "definition": "Amount of crypto held on exchanges",
        "category": "onchain",
        "keywords": ["exchange reserves", "exchange balance", "exchange flows", "器"],
        "related_concepts": ["liquidity", "supply"],
        "language": "en"
    },
    {
        "id": "nvt_ratio",
        "term": "NVT Ratio",
        "definition": "Network Value to Transactions ratio - crypto PE ratio",
        "category": "onchain",
        "keywords": ["nvt", "network value", "pe ratio", "valuation"],
        "related_concepts": ["market_cap", "valuation"],
        "language": "en"
    },
    {
        "id": "miner_revenue",
        "term": "Miner Revenue",
        "definition": "Total revenue earned by miners/validators",
        "category": "onchain",
        "keywords": ["miner revenue", "validator rewards", "block rewards", "fees"],
        "related_concepts": ["staking", "security"],
        "language": "en"
    },
    
    # === DEFIV2 TERMS ===
    {
        "id": "flash_loan",
        "term": "Flash Loan",
        "definition": "Uncollateralized loan within single transaction",
        "category": "defi",
        "keywords": ["flash loan", "flash", "uncollateralized", "atomic"],
        "related_concepts": ["liquidity_pool", "arbitrage"],
        "language": "en"
    },
    {
        "id": "mev",
        "term": "MEV",
        "definition": "Maximal Extractable Value - value extracted from transaction ordering",
        "category": "defi",
        "keywords": ["mev", "maximal extractable", "miner extractable", "sandwich", "arbitrage"],
        "related_concepts": ["flash_loan", "gas", "priority_fee"],
        "language": "en"
    },
    {
        "id": "slippage",
        "term": "Slippage",
        "definition": "Difference between expected and actual execution price",
        "category": "defi",
        "keywords": ["slippage", "price impact", "execution", "fill"],
        "related_concepts": ["spread", "liquidity", "volume"],
        "language": "en"
    },
    {
        "id": "impermanent_loss_v2",
        "term": "Impermanent Loss (Pool)",
        "definition": "Value loss when providing liquidity vs holding",
        "category": "defi",
        "keywords": ["il", "impermanent loss", "divergence loss", "loss", "pool"],
        "related_concepts": ["liquidity_pool", "yield_farming"],
        "language": "en"
    },
    {
        "id": "apr_apy",
        "term": "APR vs APY",
        "definition": "Annual Percentage Rate vs Annual Percentage Yield",
        "category": "defi",
        "keywords": ["apr", "apy", "annual percentage", "yield", "compound", "compounding"],
        "related_concepts": ["yield_farming", "staking"],
        "language": "en"
    },
    {
        "id": "governance_token",
        "term": "Governance Token",
        "definition": "Token giving voting rights in protocol",
        "category": "defi",
        "keywords": ["governance", "dao", "voting", "proposal", "vote"],
        "related_concepts": ["token", "utility"],
        "language": "en"
    },
    {
        "id": "token_swap",
        "term": "Token Swap",
        "definition": "Exchange one token for another",
        "category": "defi",
        "keywords": ["swap", "exchange", "trade", "convert", "scambiare"],
        "related_concepts": ["liquidity_pool", "dex"],
        "language": "en"
    },
    {
        "id": "oracle_manipulation",
        "term": "Oracle Manipulation",
        "definition": "Artificially manipulating price feed data",
        "category": "defi",
        "keywords": ["oracle", "manipulation", "price feed", "data", "attack"],
        "related_concepts": ["security", "smart_contract"],
        "language": "en"
    },
    
    # === TRADING TERMS ===
    {
        "id": "order_book",
        "term": "Order Book",
        "definition": "List of pending buy and sell orders",
        "category": "trading",
        "keywords": ["order book", "bid", "ask", "depth", "ladder"],
        "related_concepts": ["spread", "liquidity", "market_maker"],
        "language": "en"
    },
    {
        "id": "market_maker",
        "term": "Market Maker",
        "definition": "Provides liquidity by placing bid/ask orders",
        "category": "trading",
        "keywords": ["market maker", "mm", "liquidity provider", "lp"],
        "related_concepts": ["spread", "order_book", "bid_ask"],
        "language": "en"
    },
    {
        "id": "bid_ask_spread",
        "term": "Bid-Ask Spread",
        "definition": "Difference between highest bid and lowest ask",
        "category": "trading",
        "keywords": ["spread", "bid ask", "tight spread", "wide spread"],
        "related_concepts": ["liquidity", "market_maker"],
        "language": "en"
    },
    {
        "id": "liquidation",
        "term": "Liquidation",
        "definition": "Forced closure of position due to margin call",
        "category": "trading",
        "keywords": ["liquidation", "liquidated", "margin call", "forced close"],
        "related_concepts": ["leverage", "margin", "stop_loss"],
        "language": "en"
    },
    {
        "id": "funding_rate",
        "term": "Funding Rate",
        "definition": "Periodic payment between long and short traders",
        "category": "trading",
        "keywords": ["funding", "funding rate", "funding fee", "periodic"],
        "related_concepts": ["futures", "perpetual", "leverage"],
        "language": "en"
    },
    {
        "id": "open_interest",
        "term": "Open Interest",
        "definition": "Total number of open derivative contracts",
        "category": "trading",
        "keywords": ["open interest", "oi", "positions", "contracts"],
        "related_concepts": ["futures", "volume", "market_sentiment"],
        "language": "en"
    },
    {
        "id": "delta_neutral",
        "term": "Delta Neutral",
        "definition": "Strategy with no directional exposure",
        "category": "trading",
        "keywords": ["delta neutral", "market neutral", "hedge", "hedged"],
        "related_concepts": ["hedge", "arbitrage", "risk_management"],
        "language": "en"
    },
    {
        "id": "grid_trading",
        "term": "Grid Trading",
        "definition": "Placing buy/sell orders at regular intervals",
        "category": "trading",
        "keywords": ["grid", "range", "buy grid", "sell grid", "bot"],
        "related_concepts": ["automated_trading", " DCA"],
        "language": "en"
    },
    {
        "id": "arbitrage",
        "term": "Arbitrage",
        "definition": "Profiting from price differences across markets",
        "category": "trading",
        "keywords": ["arbitrage", "arb", "price difference", "cross-exchange"],
        "related_concepts": ["mev", "exchange", "spread"],
        "language": "en"
    },
    
    # === RISK TERMS ===
    {
        "id": "max_drawdown",
        "term": "Maximum Drawdown",
        "definition": "Largest peak-to-trough decline",
        "category": "risk",
        "keywords": ["max drawdown", "mdd", "peak to trough", "worst decline"],
        "related_concepts": ["drawdown", "volatility", "risk"],
        "language": "en"
    },
    {
        "id": "calmar_ratio",
        "term": "Calmar Ratio",
        "definition": "Return divided by maximum drawdown",
        "category": "risk",
        "keywords": ["calmar", "return over dd", "risk adjusted"],
        "related_concepts": ["sharpe_ratio", "max_drawdown"],
        "language": "en"
    },
    {
        "id": "beta",
        "term": "Beta",
        "definition": "Measure of volatility relative to market",
        "category": "risk",
        "keywords": ["beta", "market beta", "sensitivity", "correlation"],
        "related_concepts": ["correlation", "volatility"],
        "language": "en"
    },
    {
        "id": "alpha",
        "term": "Alpha",
        "definition": "Excess return above benchmark",
        "category": "risk",
        "keywords": ["alpha", "excess return", "outperformance", "skill"],
        "related_concepts": ["beta", "sharpe_ratio"],
        "language": "en"
    },
    {
        "id": "volatility_clustering",
        "term": "Volatility Clustering",
        "definition": "High volatility tends to follow high volatility",
        "category": "risk",
        "keywords": ["volatility clustering", "heteroskedasticity", "garch"],
        "related_concepts": ["volatility", "garch", "risk"],
        "language": "en"
    },
    {
        "id": "correlation_breakdown",
        "term": "Correlation Breakdown",
        "definition": "Normally correlated assets move independently",
        "category": "risk",
        "keywords": ["correlation breakdown", "correlation break", "decorrelation"],
        "related_concepts": ["correlation", "diversification", "risk"],
        "language": "en"
    },
    
    # === MACRO TERMS ===
    {
        "id": " QE",
        "term": "Quantitative Easing",
        "definition": "Central bank asset purchase program",
        "category": "macro",
        "keywords": ["qe", "quantitative easing", "money printing", "asset purchase"],
        "related_concepts": ["inflation", "interest_rate", "liquidity"],
        "language": "en"
    },
    {
        "id": "tapering",
        "term": "Tapering",
        "definition": "Gradual reduction of monetary stimulus",
        "category": "macro",
        "keywords": ["tapering", "taper", "wind down", "reduce stimulus"],
        "related_concepts": ["QE", "interest_rate"],
        "language": "en"
    },
    {
        "id": "usd_index",
        "term": "US Dollar Index",
        "definition": "Value of USD against major currencies",
        "category": "macro",
        "keywords": ["dxy", "usd index", "dollar index", "usd"],
        "related_concepts": ["forex", "currency", "inflation"],
        "language": "en"
    },
    {
        "id": "vix",
        "term": "VIX",
        "definition": "CBOE Volatility Index - fear index",
        "category": "macro",
        "keywords": ["vix", "volatility index", "fear index", "market fear"],
        "related_concepts": ["volatility", "fear_greed", "market_sentiment"],
        "language": "en"
    },
    {
        "id": "yield_curve",
        "term": "Yield Curve",
        "definition": "Interest rates across different maturities",
        "category": "macro",
        "keywords": ["yield curve", "inversion", "steepening", "flattening"],
        "related_concepts": ["interest_rate", "bonds", "recession"],
        "language": "en"
    },
    
    # === SENTIMENT ===
    {
        "id": "whale_activity",
        "term": "Whale Activity",
        "definition": "Large transactions from big holders",
        "category": "sentiment",
        "keywords": ["whale", "whales", "large transaction", "big money", "巨鲸"],
        "related_concepts": ["volume", "on_chain_volume"],
        "language": "en"
    },
    {
        "id": "social_sentiment",
        "term": "Social Sentiment",
        "definition": "Overall mood from social media and news",
        "category": "sentiment",
        "keywords": ["social sentiment", "twitter", "reddit", "social media", "讨论"],
        "related_concepts": ["fear_greed", "bullish", "bearish"],
        "language": "en"
    },
    {
        "id": "coin_rain",
        "term": "Accumulation/Distribution",
        "definition": "Buying/selling pressure based on price and volume",
        "category": "sentiment",
        "keywords": ["accumulation", "distribution", "a/d", "accumulate", "distribute"],
        "related_concepts": ["volume", "obv", "sentiment"],
        "language": "en"
    },
    
    # === TECHNICAL ===
    {
        "id": "divergence",
        "term": "Divergence",
        "definition": "Price and indicator moving in opposite directions",
        "category": "technical",
        "keywords": ["divergence", "bullish div", "bearish div", "hidden"],
        "related_concepts": ["rsi", "macd", "momentum"],
        "language": "en"
    },
    {
        "id": "harmonic_patterns",
        "term": "Harmonic Patterns",
        "definition": "Recurring price patterns using Fibonacci ratios",
        "category": "technical",
        "keywords": ["harmonic", "gartley", "butterfly", "crab", "fibonacci"],
        "related_concepts": ["support", "resistance", "pattern"],
        "language": "en"
    },
    {
        "id": "ichimoku",
        "term": "Ichimoku Cloud",
        "definition": "Multi-component technical indicator",
        "category": "technical",
        "keywords": ["ichimoku", "cloud", "tenkan", "kijun", "senkou"],
        "related_concepts": ["support", "resistance", "trend"],
        "language": "en"
    },
    
    # === SMART CONTRACTS ===
    {
        "id": "smart_contract",
        "term": "Smart Contract",
        "definition": "Self-executing code on blockchain",
        "category": "crypto",
        "keywords": ["smart contract", "contract", "code", "protocol", "合约"],
        "related_concepts": ["defi", "token", "blockchain"],
        "language": "en"
    },
    {
        "id": "gas_fee",
        "term": "Gas Fee",
        "definition": "Transaction fee on blockchain networks",
        "category": "crypto",
        "keywords": ["gas", "fee", "gas fee", "network fee", "gwei"],
        "related_concepts": ["network", "transaction", "priority_fee"],
        "language": "en"
    },
    {
        "id": "bridge",
        "term": "Cross-Chain Bridge",
        "definition": "Protocol for transferring assets between blockchains",
        "category": "crypto",
        "keywords": ["bridge", "cross-chain", "bridge", "interoperability"],
        "related_concepts": ["layer2", "token_swap"],
        "language": "en"
    }
]


def expand_vocabulary():
    """Expand the vocabulary with new concepts"""
    
    # Load existing vocabulary
    try:
        with open('data/vocabulary/concepts.json', 'r', encoding='utf-8') as f:
            vocabulary = json.load(f)
    except FileNotFoundError:
        vocabulary = []
    
    # Get existing IDs
    existing_ids = {item['id'] for item in vocabulary}
    
    # Add new concepts
    added_count = 0
    for concept in NEW_CONCEPTS:
        if concept['id'] not in existing_ids:
            # Add metadata
            concept['examples'] = []
            concept['first_seen'] = datetime.now().isoformat()
            concept['last_updated'] = datetime.now().isoformat()
            concept['occurrence_count'] = 1
            concept['confidence'] = 1.0
            
            vocabulary.append(concept)
            added_count += 1
            print(f"Added: {concept['id']} - {concept['term']}")
    
    # Add relationships between existing concepts
    add_relationships(vocabulary)
    
    # Save updated vocabulary
    with open('data/vocabulary/concepts.json', 'w', encoding='utf-8') as f:
        json.dump(vocabulary, f, indent=2, ensure_ascii=False)
    
    print(f"\nOK - Added {added_count} new concepts")
    print(f"Total concepts: {len(vocabulary)}")
    
    return vocabulary


def add_relationships(vocabulary: List[Dict]):
    """Add related_concepts links between existing concepts"""
    
    # Build ID lookup
    concept_map = {item['id']: item for item in vocabulary}
    
    # Define relationships to add
    relationship_rules = [
        # Risk relationships
        ("sharpe_ratio", "sortino_ratio"),
        ("sharpe_ratio", "calmar_ratio"),
        ("sharpe_ratio", "beta"),
        ("var", "cvar"),
        ("var", "max_drawdown"),
        ("drawdown", "max_drawdown"),
        
        # Trading relationships
        ("long_position", "short_position"),
        ("long_position", "leverage"),
        ("short_position", "leverage"),
        ("stop_loss", "take_profit"),
        ("stop_loss", "liquidation"),
        ("leverage", "liquidation"),
        
        # Technical relationships
        ("rsi", "macd"),
        ("rsi", "divergence"),
        ("macd", "divergence"),
        ("moving_average", "support"),
        ("moving_average", "resistance"),
        
        # DeFi relationships
        ("staking", "yield_farming"),
        ("staking", "apr_apy"),
        ("liquidity_pool", "impermanent_loss"),
        ("liquidity_pool", "slippage"),
        ("flash_loan", "arbitrage"),
        ("flash_loan", "mev"),
        
        # Sentiment relationships
        ("bullish", "fear_greed"),
        ("bearish", "fear_greed"),
        ("volume", "whale_activity"),
        
        # Macro relationships
        ("inflation", "interest_rate"),
        ("inflation", "QE"),
        ("interest_rate", "tapering"),
        
        # On-chain relationships
        ("on_chain_volume", "active_addresses"),
        ("on_chain_volume", "nvt_ratio"),
    ]
    
    # Add relationships
    relationships_added = 0
    for source_id, target_id in relationship_rules:
        source = concept_map.get(source_id)
        target = concept_map.get(target_id)
        
        if source and target:
            # Add to source's related_concepts
            if target_id not in source.get('related_concepts', []):
                if 'related_concepts' not in source:
                    source['related_concepts'] = []
                source['related_concepts'].append(target_id)
                relationships_added += 1
            
            # Add bidirectional relationship
            if source_id not in target.get('related_concepts', []):
                if 'related_concepts' not in target:
                    target['related_concepts'] = []
                target['related_concepts'].append(source_id)
                relationships_added += 1
    
    print(f"Added {relationships_added} concept relationships")


if __name__ == "__main__":
    print("=" * 60)
    print("SEMANTIC VOCABULARY EXPANSION")
    print("=" * 60)
    expand_vocabulary()
    print("\n✓ Vocabulary expansion complete!")
