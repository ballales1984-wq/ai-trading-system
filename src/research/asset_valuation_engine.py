"""
Asset Valuation Engine
=====================
Complex algorithms for calculating the value of each asset type.

Supports:
- Cryptocurrencies
- Stocks/Equities
- Commodities
- Forex
- Bonds
- Real Estate
- Alternative Assets

Author: AI Trading System
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


# ============================================================================
# ASSET TYPES
# ============================================================================

class AssetType(Enum):
    """Asset type enumeration."""
    CRYPTO = "crypto"
    STOCK = "stock"
    COMMODITY = "commodity"
    FOREX = "forex"
    BOND = "bond"
    REAL_ESTATE = "real_estate"
    ALTERNATIVE = "alternative"


@dataclass
class Asset:
    """Asset data class."""
    symbol: str
    name: str
    asset_type: AssetType
    current_price: float
    volume_24h: float = 0.0
    market_cap: float = 0.0
    sector: str = ""
    industry: str = ""
    
    # Historical data
    price_history: List[float] = field(default_factory=list)
    volume_history: List[float] = field(default_factory=list)
    
    # Fundamental data
    pe_ratio: float = 0.0
    pb_ratio: float = 0.0
    dividend_yield: float = 0.0
    debt_to_equity: float = 0.0
    roe: float = 0.0
    
    # Risk metrics
    beta: float = 1.0
    volatility_30d: float = 0.0
    sharpe_ratio: float = 0.0
    
    # Additional info
    country: str = ""
    currency: str = "USD"
    exchange: str = ""


# ============================================================================
# BASE VALUATION ALGORITHM
# ============================================================================

class ValuationAlgorithm(ABC):
    """Abstract base class for valuation algorithms."""
    
    @abstractmethod
    def calculate(self, asset: Asset) -> Dict[str, Any]:
        """Calculate valuation metrics."""
        pass
    
    @abstractmethod
    def get_fair_value(self, asset: Asset) -> float:
        """Calculate fair value."""
        pass


# ============================================================================
# CRYPTOCURRENCY VALUATION
# ============================================================================

class CryptoValuation(ValuationAlgorithm):
    """
    Cryptocurrency Valuation Algorithm
    
    Methods:
    1. NVT Ratio (Network Value to Transactions)
    2. Stock-to-Flow Model
    3. Metcalfe's Law
    4. On-Chain Analysis
    5. Market Sentiment Integration
    """
    
    def __init__(self):
        self.nvt_baseline = 65  # Historical NVT baseline
        self.s2f_model_params = {
            'alpha': 0.4,  # Model coefficient
            'beta': 3.3,   # Stock-to-flow exponent
            'gamma': 14.6  # Time factor
        }
    
    def calculate(self, asset: Asset) -> Dict[str, Any]:
        """Calculate comprehensive crypto valuation."""
        
        results = {
            'symbol': asset.symbol,
            'asset_type': 'crypto',
            'timestamp': datetime.now().isoformat(),
        }
        
        # 1. NVT Analysis
        nvt = self._calculate_nvt(asset)
        results['nvt_ratio'] = nvt
        results['nvt_signal'] = self._nvt_signal(nvt)
        
        # 2. Stock-to-Flow
        s2f = self._calculate_stock_to_flow(asset)
        results['stock_to_flow'] = s2f
        results['s2f_price_model'] = self._s2f_price_model(s2f)
        
        # 3. Metcalfe's Law
        metcalfe = self._calculate_metcalfe(asset)
        results['metcalfe_value'] = metcalfe
        results['metcalfe_ratio'] = asset.market_cap / metcalfe if metcalfe > 0 else 0
        
        # 4. On-Chain Metrics
        results['on_chain_score'] = self._on_chain_score(asset)
        
        # 5. Volatility Analysis
        results['volatility_percentile'] = self._volatility_percentile(asset)
        
        # 6. Composite Valuation
        results['fair_value'] = self.get_fair_value(asset)
        results['undervaluation_score'] = self._undervaluation_score(asset, results)
        
        return results
    
    def get_fair_value(self, asset: Asset) -> float:
        """Calculate weighted fair value."""
        
        # Get individual valuations
        s2f_price = self._s2f_price_model(self._calculate_stock_to_flow(asset))
        metcalfe_price = self._calculate_metcalfe(asset) / (asset.market_cap / asset.current_price) if asset.market_cap > 0 else asset.current_price
        
        # Weighted average
        weights = {
            's2f': 0.35,
            'metcalfe': 0.30,
            'nvt': 0.20,
            'market': 0.15
        }
        
        fair_value = (
            weights['s2f'] * s2f_price +
            weights['metcalfe'] * metcalfe_price +
            weights['nvt'] * asset.current_price * (self.nvt_baseline / max(1, self._calculate_nvt(asset))) +
            weights['market'] * asset.current_price
        )
        
        return fair_value
    
    def _calculate_nvt(self, asset: Asset) -> float:
        """Network Value to Transactions ratio."""
        if asset.volume_24h > 0:
            # Annualized transaction volume
            annual_volume = asset.volume_24h * 365
            return asset.market_cap / annual_volume if annual_volume > 0 else self.nvt_baseline
        return self.nvt_baseline
    
    def _nvt_signal(self, nvt: float) -> str:
        """Interpret NVT ratio."""
        if nvt > 100:
            return "OVERVALUED"
        elif nvt < 40:
            return "UNDERVALUED"
        else:
            return "FAIR_VALUE"
    
    def _calculate_stock_to_flow(self, asset: Asset) -> float:
        """Calculate stock-to-flow ratio."""
        # Simplified S2F calculation
        # In production, would use actual circulating supply and issuance rate
        if asset.symbol == "BTC":
            # Bitcoin specific S2F
            return 56  # Approximate current S2F for Bitcoin
        elif asset.symbol == "ETH":
            return 25  # Post-merge Ethereum
        else:
            return 10  # Default for other cryptos
    
    def _s2f_price_model(self, s2f: float) -> float:
        """Stock-to-Flow price model."""
        params = self.s2f_model_params
        return np.exp(params['alpha'] * (s2f ** params['beta']) + params['gamma'])
    
    def _calculate_metcalfe(self, asset: Asset) -> float:
        """Metcalfe's Law valuation."""
        # V = a * n^2 where n is number of users
        # Approximate users from market cap
        implied_users = np.sqrt(asset.market_cap / 100)  # $100 per user assumption
        return implied_users ** 2 * 100
    
    def _on_chain_score(self, asset: Asset) -> float:
        """On-chain health score (0-100)."""
        # Simplified scoring based on available metrics
        score = 50  # Base score
        
        # Volume activity bonus
        if asset.volume_24h > asset.market_cap * 0.1:  # High turnover
            score += 15
        elif asset.volume_24h > asset.market_cap * 0.05:
            score += 10
        elif asset.volume_24h > asset.market_cap * 0.01:
            score += 5
        
        # Volatility penalty
        if asset.volatility_30d > 0.8:
            score -= 15
        elif asset.volatility_30d > 0.5:
            score -= 10
        elif asset.volatility_30d < 0.2:
            score += 10  # Stability bonus
        
        return max(0, min(100, score))
    
    def _volatility_percentile(self, asset: Asset) -> float:
        """Calculate volatility percentile."""
        if len(asset.price_history) < 30:
            return 50
        
        returns = pd.Series(asset.price_history).pct_change().dropna()
        current_vol = returns.std() * np.sqrt(365)  # Annualized
        
        # Historical volatility distribution
        rolling_vol = returns.rolling(30).std() * np.sqrt(365)
        percentile = (rolling_vol < current_vol).mean() * 100
        
        return percentile
    
    def _undervaluation_score(self, asset: Asset, metrics: Dict) -> float:
        """Calculate undervaluation score (0-100)."""
        score = 50
        
        # NVT signal
        if metrics['nvt_signal'] == "UNDERVALUED":
            score += 20
        elif metrics['nvt_signal'] == "OVERVALUED":
            score -= 20
        
        # Metcalfe ratio
        if metrics['metcalfe_ratio'] < 0.5:
            score += 15  # Undervalued by Metcalfe
        elif metrics['metcalfe_ratio'] > 2:
            score -= 15
        
        # On-chain score
        score += (metrics['on_chain_score'] - 50) * 0.3
        
        return max(0, min(100, score))


# ============================================================================
# STOCK VALUATION
# ============================================================================

class StockValuation(ValuationAlgorithm):
    """
    Stock/Equity Valuation Algorithm
    
    Methods:
    1. Discounted Cash Flow (DCF)
    2. Dividend Discount Model (DDM)
    3. P/E Relative Valuation
    4. P/B Value Investing
    5. EV/EBITDA
    6. Graham Number
    7. PEG Ratio
    """
    
    def __init__(self):
        self.risk_free_rate = 0.045  # 4.5% risk-free rate
        self.market_risk_premium = 0.06  # 6% market risk premium
    
    def calculate(self, asset: Asset) -> Dict[str, Any]:
        """Calculate comprehensive stock valuation."""
        
        results = {
            'symbol': asset.symbol,
            'asset_type': 'stock',
            'timestamp': datetime.now().isoformat(),
        }
        
        # 1. DCF Valuation
        results['dcf_value'] = self._dcf_valuation(asset)
        
        # 2. Dividend Discount Model
        results['ddm_value'] = self._ddm_valuation(asset)
        
        # 3. Relative Valuation
        results['pe_relative_value'] = self._pe_relative_valuation(asset)
        results['pb_relative_value'] = self._pb_relative_valuation(asset)
        
        # 4. EV/EBITDA
        results['ev_ebitda_value'] = self._ev_ebitda_valuation(asset)
        
        # 5. Graham Number
        results['graham_number'] = self._graham_number(asset)
        
        # 6. PEG Ratio
        results['peg_ratio'] = self._peg_ratio(asset)
        
        # 7. Composite Fair Value
        results['fair_value'] = self.get_fair_value(asset)
        
        # 8. Margin of Safety
        results['margin_of_safety'] = self._margin_of_safety(asset, results['fair_value'])
        
        # 9. Quality Score
        results['quality_score'] = self._quality_score(asset)
        
        return results
    
    def get_fair_value(self, asset: Asset) -> float:
        """Calculate weighted fair value."""
        
        values = []
        weights = []
        
        # DCF (highest weight if available)
        dcf = self._dcf_valuation(asset)
        if dcf > 0:
            values.append(dcf)
            weights.append(0.30)
        
        # DDM (for dividend stocks)
        ddm = self._ddm_valuation(asset)
        if ddm > 0:
            values.append(ddm)
            weights.append(0.20)
        
        # Relative valuations
        pe_val = self._pe_relative_valuation(asset)
        if pe_val > 0:
            values.append(pe_val)
            weights.append(0.20)
        
        pb_val = self._pb_relative_valuation(asset)
        if pb_val > 0:
            values.append(pb_val)
            weights.append(0.15)
        
        # Graham Number
        graham = self._graham_number(asset)
        if graham > 0:
            values.append(graham)
            weights.append(0.15)
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Weighted average
        fair_value = sum(v * w for v, w in zip(values, weights))
        
        return fair_value
    
    def _dcf_valuation(self, asset: Asset) -> float:
        """Discounted Cash Flow valuation."""
        # Simplified DCF
        # In production, would use detailed cash flow projections
        
        # Estimate free cash flow from market cap and ROE
        if asset.roe <= 0 or asset.market_cap <= 0:
            return 0
        
        # Assume FCF yield based on ROE
        fcf_yield = asset.roe * 0.3  # Conservative payout
        fcf = asset.market_cap * fcf_yield
        
        # Growth assumptions
        growth_rate = min(0.15, asset.roe * 0.5)  # Cap at 15%
        terminal_growth = 0.025  # 2.5% terminal growth
        
        # Discount rate (WACC approximation)
        cost_of_equity = self.risk_free_rate + asset.beta * self.market_risk_premium
        discount_rate = cost_of_equity
        
        # DCF calculation (10-year projection)
        present_value = 0
        for year in range(1, 11):
            fcf_year = fcf * (1 + growth_rate) ** year
            present_value += fcf_year / (1 + discount_rate) ** year
        
        # Terminal value
        terminal_value = fcf * (1 + terminal_growth) / (discount_rate - terminal_growth)
        terminal_pv = terminal_value / (1 + discount_rate) ** 10
        
        return present_value + terminal_pv
    
    def _ddm_valuation(self, asset: Asset) -> float:
        """Dividend Discount Model (Gordon Growth Model)."""
        if asset.dividend_yield <= 0:
            return 0
        
        # Current dividend
        dividend = asset.current_price * asset.dividend_yield
        
        # Cost of equity
        cost_of_equity = self.risk_free_rate + asset.beta * self.market_risk_premium
        
        # Dividend growth rate (sustainable growth)
        growth_rate = min(0.10, asset.roe * (1 - asset.dividend_yield))  # Retention * ROE
        
        if cost_of_equity <= growth_rate:
            return asset.current_price  # Invalid inputs
        
        # Gordon Growth Model
        fair_value = dividend * (1 + growth_rate) / (cost_of_equity - growth_rate)
        
        return fair_value
    
    def _pe_relative_valuation(self, asset: Asset) -> float:
        """P/E relative valuation."""
        if asset.pe_ratio <= 0:
            return 0
        
        # Industry average P/E (simplified)
        industry_pe = {
            'Technology': 25,
            'Healthcare': 20,
            'Financial': 12,
            'Consumer': 18,
            'Industrial': 15,
            'Energy': 10,
            'Utilities': 14,
            'Real Estate': 16,
        }
        
        avg_pe = industry_pe.get(asset.industry, 15)
        
        # Fair value based on industry P/E
        earnings_per_share = asset.current_price / asset.pe_ratio
        fair_value = earnings_per_share * avg_pe
        
        return fair_value
    
    def _pb_relative_valuation(self, asset: Asset) -> float:
        """P/B relative valuation."""
        if asset.pb_ratio <= 0:
            return 0
        
        # Industry average P/B
        industry_pb = {
            'Technology': 5,
            'Healthcare': 3,
            'Financial': 1,
            'Consumer': 3,
            'Industrial': 2,
            'Energy': 1.5,
            'Utilities': 1.5,
            'Real Estate': 1.5,
        }
        
        avg_pb = industry_pb.get(asset.industry, 2)
        
        # Book value per share
        book_value = asset.current_price / asset.pb_ratio
        fair_value = book_value * avg_pb
        
        return fair_value
    
    def _ev_ebitda_valuation(self, asset: Asset) -> float:
        """EV/EBITDA valuation."""
        # Simplified - would need actual EBITDA
        # Using market cap as proxy
        ev_ebitda_multiple = 10  # Industry average
        
        # Estimate EBITDA from market cap
        ebitda = asset.market_cap / ev_ebitda_multiple
        
        # Fair enterprise value
        fair_ev = ebitda * ev_ebitda_multiple
        
        return fair_ev
    
    def _graham_number(self, asset: Asset) -> float:
        """Graham Number calculation."""
        # Graham Number = √(22.5 × EPS × Book Value per Share)
        if asset.pe_ratio <= 0 or asset.pb_ratio <= 0:
            return 0
        
        eps = asset.current_price / asset.pe_ratio
        book_value = asset.current_price / asset.pb_ratio
        
        graham = np.sqrt(22.5 * eps * book_value)
        
        return graham
    
    def _peg_ratio(self, asset: Asset) -> float:
        """PEG ratio calculation."""
        if asset.pe_ratio <= 0:
            return 0
        
        # Estimate growth rate from ROE
        growth_rate = asset.roe * 100 if asset.roe > 0 else 10  # As percentage
        
        peg = asset.pe_ratio / growth_rate
        
        return peg
    
    def _margin_of_safety(self, asset: Asset, fair_value: float) -> float:
        """Calculate margin of safety."""
        if fair_value <= 0:
            return 0
        
        margin = (fair_value - asset.current_price) / fair_value
        return margin * 100  # As percentage
    
    def _quality_score(self, asset: Asset) -> float:
        """Calculate quality score (0-100)."""
        score = 50
        
        # ROE bonus
        if asset.roe > 0.20:
            score += 15
        elif asset.roe > 0.15:
            score += 10
        elif asset.roe > 0.10:
            score += 5
        elif asset.roe < 0:
            score -= 15
        
        # Debt penalty
        if asset.debt_to_equity > 2:
            score -= 15
        elif asset.debt_to_equity > 1:
            score -= 10
        elif asset.debt_to_equity < 0.5:
            score += 10
        
        # P/E reasonableness
        if 0 < asset.pe_ratio < 15:
            score += 10
        elif asset.pe_ratio > 40:
            score -= 10
        
        # Dividend bonus
        if asset.dividend_yield > 0.03:
            score += 5
        
        return max(0, min(100, score))


# ============================================================================
# COMMODITY VALUATION
# ============================================================================

class CommodityValuation(ValuationAlgorithm):
    """
    Commodity Valuation Algorithm
    
    Methods:
    1. Cost of Production Model
    2. Futures Curve Analysis
    3. Inventory Level Analysis
    4. Supply/Demand Balance
    5. Real Price Adjustment
    """
    
    def __init__(self):
        # Production costs by commodity
        self.production_costs = {
            'GOLD': 1200,    # $/oz
            'SILVER': 15,    # $/oz
            'OIL_WTI': 45,   # $/barrel
            'OIL_BRENT': 50, # $/barrel
            'NATURAL_GAS': 2.5,  # $/MMBtu
            'COPPER': 3.5,   # $/lb
            'WHEAT': 5.5,    # $/bushel
            'CORN': 4.0,     # $/bushel
        }
        
        # Historical average prices
        self.historical_avg = {
            'GOLD': 1400,
            'SILVER': 20,
            'OIL_WTI': 65,
            'OIL_BRENT': 70,
            'NATURAL_GAS': 3.0,
            'COPPER': 3.0,
            'WHEAT': 5.0,
            'CORN': 4.0,
        }
    
    def calculate(self, asset: Asset) -> Dict[str, Any]:
        """Calculate comprehensive commodity valuation."""
        
        symbol = asset.symbol.upper()
        
        results = {
            'symbol': asset.symbol,
            'asset_type': 'commodity',
            'timestamp': datetime.now().isoformat(),
        }
        
        # 1. Cost of Production Analysis
        results['production_cost'] = self.production_costs.get(symbol, 0)
        results['cost_margin'] = self._cost_margin(asset)
        
        # 2. Historical Price Analysis
        results['historical_percentile'] = self._historical_percentile(asset)
        results['real_price'] = self._real_price_adjustment(asset)
        
        # 3. Supply/Demand Score
        results['supply_demand_score'] = self._supply_demand_score(asset)
        
        # 4. Futures Curve Analysis
        results['futures_curve_signal'] = self._futures_curve_signal(asset)
        
        # 5. Fair Value
        results['fair_value'] = self.get_fair_value(asset)
        
        # 6. Investment Signal
        results['investment_signal'] = self._investment_signal(results)
        
        return results
    
    def get_fair_value(self, asset: Asset) -> float:
        """Calculate fair value for commodity."""
        
        symbol = asset.symbol.upper()
        
        # Weighted approach
        production_cost = self.production_costs.get(symbol, asset.current_price * 0.8)
        historical_avg = self.historical_avg.get(symbol, asset.current_price)
        
        # Fair value is weighted average of:
        # - Production cost (floor price)
        # - Historical average (mean reversion)
        # - Current price (market efficiency)
        
        fair_value = (
            0.30 * production_cost * 1.2 +  # Production cost + margin
            0.30 * historical_avg +
            0.40 * asset.current_price
        )
        
        return fair_value
    
    def _cost_margin(self, asset: Asset) -> float:
        """Calculate margin above production cost."""
        symbol = asset.symbol.upper()
        cost = self.production_costs.get(symbol, 0)
        
        if cost <= 0:
            return 0
        
        margin = (asset.current_price - cost) / cost
        return margin * 100  # As percentage
    
    def _historical_percentile(self, asset: Asset) -> float:
        """Calculate price percentile vs historical."""
        if len(asset.price_history) < 100:
            return 50
        
        prices = pd.Series(asset.price_history)
        percentile = (prices < asset.current_price).mean() * 100
        
        return percentile
    
    def _real_price_adjustment(self, asset: Asset) -> float:
        """Adjust for inflation (real price)."""
        # Simplified - would use CPI data
        inflation_rate = 0.03  # 3% annual inflation
        
        # Real price = nominal / (1 + inflation)^years
        # Assuming 5-year lookback
        real_price = asset.current_price / ((1 + inflation_rate) ** 5)
        
        return real_price
    
    def _supply_demand_score(self, asset: Asset) -> float:
        """Calculate supply/demand balance score."""
        # Simplified scoring based on price momentum and volume
        score = 50
        
        if len(asset.price_history) < 30:
            return score
        
        # Price momentum
        returns = pd.Series(asset.price_history).pct_change().dropna()
        momentum = returns.tail(30).mean() * 252  # Annualized
        
        if momentum > 0.10:
            score += 15  # Strong demand
        elif momentum > 0.05:
            score += 10
        elif momentum < -0.10:
            score -= 15  # Weak demand
        elif momentum < -0.05:
            score -= 10
        
        # Volume analysis
        if len(asset.volume_history) > 0:
            vol_change = asset.volume_history[-1] / np.mean(asset.volume_history[-30:])
            if vol_change > 1.5:
                score += 10  # High activity
            elif vol_change < 0.7:
                score -= 5  # Low activity
        
        return max(0, min(100, score))
    
    def _futures_curve_signal(self, asset: Asset) -> str:
        """Analyze futures curve shape."""
        # Simplified - would use actual futures data
        # Contango = bearish, Backwardation = bullish
        
        if len(asset.price_history) < 60:
            return "NEUTRAL"
        
        # Use price trend as proxy
        ma_20 = np.mean(asset.price_history[-20:])
        ma_60 = np.mean(asset.price_history[-60:])
        
        if ma_20 > ma_60 * 1.05:
            return "BACKWARDATION_BULLISH"
        elif ma_20 < ma_60 * 0.95:
            return "CONTANGO_BEARISH"
        else:
            return "NEUTRAL"
    
    def _investment_signal(self, metrics: Dict) -> str:
        """Generate investment signal."""
        score = 50
        
        # Cost margin contribution
        if metrics['cost_margin'] < 10:
            score -= 20  # Near production cost
        elif metrics['cost_margin'] > 50:
            score += 10  # Good margin
        
        # Historical percentile
        if metrics['historical_percentile'] < 20:
            score += 15  # Historically cheap
        elif metrics['historical_percentile'] > 80:
            score -= 15  # Historically expensive
        
        # Supply/demand
        score += (metrics['supply_demand_score'] - 50) * 0.3
        
        # Futures curve
        if metrics['futures_curve_signal'] == "BACKWARDATION_BULLISH":
            score += 10
        elif metrics['futures_curve_signal'] == "CONTANGO_BEARISH":
            score -= 10
        
        if score >= 70:
            return "STRONG_BUY"
        elif score >= 55:
            return "BUY"
        elif score >= 45:
            return "HOLD"
        elif score >= 30:
            return "SELL"
        else:
            return "STRONG_SELL"


# ============================================================================
# FOREX VALUATION
# ============================================================================

class ForexValuation(ValuationAlgorithm):
    """
    Forex Valuation Algorithm
    
    Methods:
    1. Purchasing Power Parity (PPP)
    2. Interest Rate Parity
    3. Balance of Payments
    4. Real Effective Exchange Rate
    """
    
    def __init__(self):
        # PPP rates (simplified)
        self.ppp_rates = {
            'EURUSD': 1.15,
            'GBPUSD': 1.25,
            'USDJPY': 105,
            'AUDUSD': 0.75,
            'USDCAD': 1.25,
        }
        
        # Interest rates by currency
        self.interest_rates = {
            'USD': 0.0525,  # 5.25%
            'EUR': 0.04,
            'GBP': 0.05,
            'JPY': 0.001,   # 0.1%
            'AUD': 0.041,
            'CAD': 0.045,
        }
    
    def calculate(self, asset: Asset) -> Dict[str, Any]:
        """Calculate forex valuation."""
        
        results = {
            'symbol': asset.symbol,
            'asset_type': 'forex',
            'timestamp': datetime.now().isoformat(),
        }
        
        # 1. PPP Valuation
        results['ppp_value'] = self._ppp_valuation(asset)
        results['ppp_deviation'] = self._ppp_deviation(asset)
        
        # 2. Interest Rate Parity
        results['irp_fair_value'] = self._interest_rate_parity(asset)
        results['carry_score'] = self._carry_score(asset)
        
        # 3. Momentum Score
        results['momentum_score'] = self._momentum_score(asset)
        
        # 4. Fair Value
        results['fair_value'] = self.get_fair_value(asset)
        
        return results
    
    def get_fair_value(self, asset: Asset) -> float:
        """Calculate fair value for forex pair."""
        
        ppp_value = self._ppp_valuation(asset)
        irp_value = self._interest_rate_parity(asset)
        
        # Weighted average
        fair_value = (
            0.50 * ppp_value +
            0.30 * irp_value +
            0.20 * asset.current_price
        )
        
        return fair_value
    
    def _ppp_valuation(self, asset: Asset) -> float:
        """Purchasing Power Parity valuation."""
        return self.ppp_rates.get(asset.symbol.upper(), asset.current_price)
    
    def _ppp_deviation(self, asset: Asset) -> float:
        """Calculate deviation from PPP."""
        ppp = self._ppp_valuation(asset)
        if ppp <= 0:
            return 0
        
        deviation = (asset.current_price - ppp) / ppp
        return deviation * 100  # As percentage
    
    def _interest_rate_parity(self, asset: Asset) -> float:
        """Interest Rate Parity fair value."""
        symbol = asset.symbol.upper()
        
        # Extract currencies
        if len(symbol) == 6:
            base = symbol[:3]
            quote = symbol[3:]
        else:
            return asset.current_price
        
        base_rate = self.interest_rates.get(base, 0.03)
        quote_rate = self.interest_rates.get(quote, 0.03)
        
        # IRP: F = S * (1 + r_quote) / (1 + r_base)
        # For spot fair value, we invert
        fair_value = asset.current_price * (1 + base_rate) / (1 + quote_rate)
        
        return fair_value
    
    def _carry_score(self, asset: Asset) -> float:
        """Calculate carry trade score."""
        symbol = asset.symbol.upper()
        
        if len(symbol) == 6:
            base = symbol[:3]
            quote = symbol[3:]
        else:
            return 50
        
        base_rate = self.interest_rates.get(base, 0.03)
        quote_rate = self.interest_rates.get(quote, 0.03)
        
        # Positive carry if base rate > quote rate
        carry = base_rate - quote_rate
        
        # Score based on carry
        score = 50 + carry * 500  # Scale carry to score
        
        return max(0, min(100, score))
    
    def _momentum_score(self, asset: Asset) -> float:
        """Calculate momentum score."""
        if len(asset.price_history) < 20:
            return 50
        
        # Calculate returns
        prices = pd.Series(asset.price_history)
        returns = prices.pct_change().dropna()
        
        # Momentum indicators
        ma_5 = prices.tail(5).mean()
        ma_20 = prices.tail(20).mean()
        
        score = 50
        
        if ma_5 > ma_20:
            score += 15
        else:
            score -= 15
        
        # Trend strength
        trend = returns.tail(10).mean()
        if trend > 0:
            score += 10
        else:
            score -= 10
        
        return max(0, min(100, score))


# ============================================================================
# MAIN VALUATION ENGINE
# ============================================================================

class AssetValuationEngine:
    """
    Main Asset Valuation Engine
    Routes to appropriate valuation algorithm based on asset type.
    """
    
    def __init__(self):
        self.crypto_valuation = CryptoValuation()
        self.stock_valuation = StockValuation()
        self.commodity_valuation = CommodityValuation()
        self.forex_valuation = ForexValuation()
    
    def value_asset(self, asset: Asset) -> Dict[str, Any]:
        """Value any asset based on its type."""
        
        if asset.asset_type == AssetType.CRYPTO:
            return self.crypto_valuation.calculate(asset)
        elif asset.asset_type == AssetType.STOCK:
            return self.stock_valuation.calculate(asset)
        elif asset.asset_type == AssetType.COMMODITY:
            return self.commodity_valuation.calculate(asset)
        elif asset.asset_type == AssetType.FOREX:
            return self.forex_valuation.calculate(asset)
        else:
            return {
                'symbol': asset.symbol,
                'asset_type': asset.asset_type.value,
                'error': 'Valuation not supported for this asset type',
                'fair_value': asset.current_price
            }
    
    def value_portfolio(self, assets: List[Asset]) -> pd.DataFrame:
        """Value a portfolio of assets."""
        
        results = []
        for asset in assets:
            valuation = self.value_asset(asset)
            valuation['current_price'] = asset.current_price
            valuation['market_cap'] = asset.market_cap
            results.append(valuation)
        
        df = pd.DataFrame(results)
        return df


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def main():
    """Example usage of valuation engine."""
    
    print("=" * 70)
    print("ASSET VALUATION ENGINE")
    print("=" * 70)
    
    engine = AssetValuationEngine()
    
    # Example assets
    assets = [
        # Crypto
        Asset(
            symbol="BTC",
            name="Bitcoin",
            asset_type=AssetType.CRYPTO,
            current_price=50000,
            market_cap=1e12,
            volume_24h=30e9,
            volatility_30d=0.6
        ),
        
        # Stock
        Asset(
            symbol="AAPL",
            name="Apple Inc",
            asset_type=AssetType.STOCK,
            current_price=180,
            market_cap=2.8e12,
            pe_ratio=28,
            pb_ratio=45,
            dividend_yield=0.005,
            roe=0.45,
            debt_to_equity=1.5,
            beta=1.2,
            industry="Technology"
        ),
        
        # Commodity
        Asset(
            symbol="GOLD",
            name="Gold",
            asset_type=AssetType.COMMODITY,
            current_price=2000,
            price_history=list(np.random.normal(1900, 100, 100))
        ),
        
        # Forex
        Asset(
            symbol="EURUSD",
            name="EUR/USD",
            asset_type=AssetType.FOREX,
            current_price=1.08,
            price_history=list(np.random.normal(1.08, 0.02, 100))
        ),
    ]
    
    # Value each asset
    for asset in assets:
        print(f"\n{'='*70}")
        print(f"VALUATION: {asset.symbol} ({asset.asset_type.value.upper()})")
        print("=" * 70)
        
        valuation = engine.value_asset(asset)
        
        print(f"\nCurrent Price: ${asset.current_price:,.2f}")
        print(f"Fair Value: ${valuation.get('fair_value', 0):,.2f}")
        
        # Calculate undervaluation
        fair_value = valuation.get('fair_value', asset.current_price)
        undervaluation = (fair_value - asset.current_price) / fair_value * 100
        print(f"Undervaluation: {undervaluation:.1f}%")
        
        print("\nDetailed Metrics:")
        for key, value in valuation.items():
            if key not in ['symbol', 'asset_type', 'timestamp', 'fair_value']:
                print(f"  {key}: {value}")
    
    # Portfolio valuation
    print("\n" + "=" * 70)
    print("PORTFOLIO SUMMARY")
    print("=" * 70)
    
    df = engine.value_portfolio(assets)
    print(df[['symbol', 'asset_type', 'current_price', 'fair_value']].to_string(index=False))
    
    return df


if __name__ == "__main__":
    main()
