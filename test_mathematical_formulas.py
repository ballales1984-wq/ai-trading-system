"""
Test Suite for Mathematical Formulas
=====================================
Testa la correttezza delle formule matematiche utilizzate nel sistema di trading.

Formule testate:
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Volatilità (annualizzata)
- Risk/Reward Ratio
- Position Sizing
- RSI
- MACD
- Bollinger Bands
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import pytest
from typing import Dict, List, Tuple


# ==================== FIXTURES ====================

@pytest.fixture
def sample_returns() -> pd.Series:
    """Ritorni giornalieri di esempio per i test."""
    np.random.seed(42)
    # Simulazione: media 0.001, std 0.02 (circa 25% volatilità annualizzata)
    returns = pd.Series(np.random.normal(0.001, 0.02, 252))
    return returns


@pytest.fixture
def sample_equity_curve() -> pd.Series:
    """Curva equity di esempio per il test del drawdown."""
    # Simulazione: equity che sale e scende
    equity = pd.Series([100, 105, 110, 108, 115, 120, 118, 125, 130, 128, 135, 140, 138, 142, 145])
    return equity


@pytest.fixture
def sample_prices() -> pd.DataFrame:
    """Prezzi di esempio per i test degli indicatori tecnici."""
    np.random.seed(42)
    # Prezzi con trend rialzista
    prices = 100 + np.cumsum(np.random.randn(100) * 2)
    df = pd.DataFrame({
        'open': prices + np.random.randn(100) * 0.5,
        'high': prices + abs(np.random.randn(100)) * 2,
        'low': prices - abs(np.random.randn(100)) * 2,
        'close': prices,
        'volume': np.random.randint(1000, 10000, 100)
    })
    return df


# ==================== TEST SHARPE RATIO ====================

class TestSharpeRatio:
    """Test per la formula dello Sharpe Ratio."""
    
    def test_sharpe_ratio_positive_returns(self, sample_returns):
        """Testa Sharpe Ratio con ritorni positivi."""
        # Formula attesa: (mean - risk_free/252) * sqrt(252) / std
        risk_free = 0.02  # 2% annualizzato
        periods_per_year = 252
        
        excess_returns = sample_returns - risk_free / periods_per_year
        expected_sharpe = np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()
        
        # Importa e testa la funzione
        from src.risk import sharpe_ratio
        actual_sharpe = sharpe_ratio(sample_returns, risk_free, periods_per_year)
        
        print(f"\n[Sharpe Ratio Test]")
        print(f"  Expected: {expected_sharpe:.4f}")
        print(f"  Actual: {actual_sharpe:.4f}")
        print(f"  Difference: {abs(expected_sharpe - actual_sharpe):.6f}")
        
        # Il test passa se i valori sono molto vicini
        assert abs(expected_sharpe - actual_sharpe) < 0.001, f"Sharpe ratio mismatch: {expected_sharpe} vs {actual_sharpe}"
    
    def test_sharpe_ratio_zero_returns(self):
        """Testa Sharpe Ratio con ritorni zero."""
        from src.risk import sharpe_ratio
        
        zero_returns = pd.Series([0.0] * 100)
        sharpe = sharpe_ratio(zero_returns)
        
        print(f"\n[Sharpe Ratio Zero Returns]")
        print(f"  Sharpe: {sharpe}")
        
        assert sharpe == 0.0, "Sharpe ratio should be 0 for zero returns"
    
    def test_sharpe_ratio_zero_std(self):
        """Testa Sharpe Ratio con deviazione standard zero."""
        from src.risk import sharpe_ratio
        
        constant_returns = pd.Series([0.001] * 100)
        sharpe = sharpe_ratio(constant_returns)
        
        print(f"\n[Sharpe Ratio Zero Std]")
        print(f"  Sharpe: {sharpe}")
        
        assert sharpe == 0.0, "Sharpe ratio should be 0 for zero std"


# ==================== TEST SORTINO RATIO ====================

class TestSortinoRatio:
    """Test per la formula del Sortino Ratio."""
    
    def test_sortino_ratio(self, sample_returns):
        """Testa Sortino Ratio."""
        risk_free = 0.02
        periods_per_year = 252
        
        from src.risk import sortino_ratio
        sortino = sortino_ratio(sample_returns, risk_free, periods_per_year)
        
        # Calcola expected manualmente
        excess_returns = sample_returns - risk_free / periods_per_year
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            expected_sortino = np.sqrt(periods_per_year) * excess_returns.mean() / downside_returns.std()
        else:
            expected_sortino = 0.0
        
        print(f"\n[Sortino Ratio Test]")
        print(f"  Expected: {expected_sortino:.4f}")
        print(f"  Actual: {sortino:.4f}")
        
        # Il test passa se i valori sono vicini (o entrambi 0)
        assert abs(expected_sortino - sortino) < 0.001 or (expected_sortino == 0 and sortino == 0)


# ==================== TEST MAX DRAWDOWN ====================

class TestMaxDrawdown:
    """Test per la formula del Maximum Drawdown."""
    
    def test_max_drawdown(self, sample_equity_curve):
        """Testa Maximum Drawdown."""
        from src.risk import max_drawdown
        
        max_dd, peak_idx, trough_idx = max_drawdown(sample_equity_curve)
        
        # Calcola expected manualmente
        peak = sample_equity_curve.expanding(min_periods=1).max()
        drawdown = (sample_equity_curve - peak) / peak
        expected_max_dd = drawdown.min()
        
        print(f"\n[Max Drawdown Test]")
        print(f"  Expected: {expected_max_dd:.4f}")
        print(f"  Actual: {max_dd:.4f}")
        print(f"  Peak Index: {peak_idx}, Trough Index: {trough_idx}")
        
        assert abs(expected_max_dd - max_dd) < 0.001, f"Max drawdown mismatch"
    
    def test_max_drawdown_empty(self):
        """Testa Maximum Drawdown con input vuoto."""
        from src.risk import max_drawdown
        
        empty_curve = pd.Series([])
        max_dd, peak_idx, trough_idx = max_drawdown(empty_curve)
        
        assert max_dd == 0.0
        assert peak_idx == 0
        assert trough_idx == 0
    
    def test_max_drawdown_no_drawdown(self):
        """Testa Maximum Drawdown senza drawdown (sempre in aumento)."""
        from src.risk import max_drawdown
        
        increasing_curve = pd.Series([100, 110, 120, 130, 140])
        max_dd, peak_idx, trough_idx = max_drawdown(increasing_curve)
        
        print(f"\n[Max Drawdown No Drawdown]")
        print(f"  Max DD: {max_dd}")
        
        assert max_dd == 0.0, "No drawdown should give 0"


# ==================== TEST VOLATILITÀ ====================

class TestVolatility:
    """Test per la formula della volatilità."""
    
    def test_volatility_calculation(self, sample_prices):
        """Testa il calcolo della volatilità annualizzata."""
        from technical_analysis import TechnicalAnalyzer
        
        analyzer = TechnicalAnalyzer()
        volatility = analyzer.calculate_volatility(sample_prices, period=20)
        
        # Calcola expected manualmente
        returns = sample_prices['close'].pct_change()
        expected_vol = returns.rolling(window=20).std().iloc[-1] * np.sqrt(365)
        
        print(f"\n[Volatility Test]")
        print(f"  Expected: {expected_vol:.4f}")
        print(f"  Actual: {volatility:.4f}")
        
        assert abs(expected_vol - volatility) < 0.001, f"Volatility mismatch"
    
    def test_volatility_formula(self):
        """Verifica la formula della volatilità: σ_annual = σ_daily * √252"""
        np.random.seed(42)
        daily_returns = np.random.normal(0.001, 0.02, 100)
        
        # Volatilità daily
        daily_vol = np.std(daily_returns)
        
        # Volatilità annualizzata
        annual_vol = daily_vol * np.sqrt(252)
        
        print(f"\n[Volatility Formula Test]")
        print(f"  Daily Vol: {daily_vol:.4f}")
        print(f"  Annual Vol (calculated): {annual_vol:.4f}")
        
        # Verifica che sia nell'ordine giusto (circa 30-40% per questo esempio)
        assert 0.2 < annual_vol < 0.5, "Annual volatility should be around 30-40%"


# ==================== TEST RSI ====================

class TestRSI:
    """Test per la formula dell'RSI."""
    
    def test_rsi_calculation(self, sample_prices):
        """Testa il calcolo dell'RSI."""
        from technical_analysis import TechnicalAnalyzer
        
        analyzer = TechnicalAnalyzer()
        rsi = analyzer.calculate_rsi(sample_prices, period=14)
        
        print(f"\n[RSI Test]")
        print(f"  RSI: {rsi:.2f}")
        
        # L'RSI deve essere tra 0 e 100
        assert 0 <= rsi <= 100, f"RSI should be between 0 and 100, got {rsi}"
    
    def test_rsi_formula(self):
        """Verifica la formula dell'RSI."""
        # RSI = 100 - (100 / (1 + RS))
        # RS = Average Gain / Average Loss
        
        gains = [1, 2, 0, 3, 1]
        losses = [0, 1, 2, 0, 1]
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        print(f"\n[RSI Formula Test]")
        print(f"  Avg Gain: {avg_gain}")
        print(f"  Avg Loss: {avg_loss}")
        print(f"  RSI: {rsi:.2f}")
        
        assert 0 <= rsi <= 100


# ==================== TEST MACD ====================

class TestMACD:
    """Test per la formula del MACD."""
    
    def test_macd_calculation(self, sample_prices):
        """Testa il calcolo del MACD."""
        from technical_analysis import TechnicalAnalyzer
        
        analyzer = TechnicalAnalyzer()
        macd_result = analyzer.calculate_macd(sample_prices)
        
        print(f"\n[MACD Test]")
        print(f"  MACD Line: {macd_result.get('macd', 0):.4f}")
        print(f"  Signal Line: {macd_result.get('signal', 0):.4f}")
        print(f"  Histogram: {macd_result.get('histogram', 0):.4f}")
        
        # Verifica che tutti i componenti siano presenti
        assert 'macd' in macd_result
        assert 'signal' in macd_result
        assert 'histogram' in macd_result


# ==================== TEST POSITION SIZING ====================

class TestPositionSizing:
    """Test per le formule di position sizing."""
    
    def test_volatility_position_sizing(self):
        """Testa il calcolo della size basato sulla volatilità."""
        from src.live.position_sizing import calculate_position_size
        
        capital = 10000
        price = 50000
        volatility = 0.02  # 2% daily volatility
        target_volatility = 0.02  # Target 2%
        
        size = calculate_position_size(
            capital=capital,
            price=price,
            volatility=volatility,
            target_volatility=target_volatility,
            max_position_pct=0.1
        )
        
        # Formula: size = capital * (target_vol / volatility) / price
        expected_size = capital * (target_volatility / volatility) / price
        
        print(f"\n[Position Sizing Test]")
        print(f"  Capital: ${capital}")
        print(f"  Price: ${price}")
        print(f"  Volatility: {volatility}")
        print(f"  Target Vol: {target_volatility}")
        print(f"  Expected Size: {expected_size:.6f}")
        print(f"  Actual Size: {size:.6f}")
        
        # Verifica che la size sia positiva
        assert size > 0, "Position size should be positive"
    
    def test_kelly_criterion(self):
        """Testa il calcolo del Kelly Criterion."""
        from src.core.dynamic_allocation import KellyCalculator
        
        kelly = KellyCalculator()
        
        # Test con win rate 60%, avg win 100, avg loss 50
        win_rate = 0.60
        avg_win = 100
        avg_loss = 50
        
        kelly_fraction = kelly.calculate_kelly(win_rate, avg_win, avg_loss)
        
        # Formula: Kelly% = W - (1-W)/R
        # dove R = avg_win / avg_loss
        ratio = avg_win / avg_loss
        expected_kelly = win_rate - (1 - win_rate) / ratio
        
        print(f"\n[Kelly Criterion Test]")
        print(f"  Win Rate: {win_rate}")
        print(f"  Avg Win: ${avg_win}")
        print(f"  Avg Loss: ${avg_loss}")
        print(f"  Kelly %: {kelly_fraction:.4f} ({kelly_fraction*100:.2f}%)")
        
        # Verifica che sia positivo
        assert kelly_fraction > 0, "Kelly should be positive for positive edge"


# ==================== TEST RISK/REWARD ====================

class TestRiskReward:
    """Test per la formula del Risk/Reward Ratio."""
    
    def test_risk_reward_ratio(self):
        """Testa il calcolo del Risk/Reward Ratio."""
        from src.kpi import risk_reward_ratio
        
        # Crea un DataFrame di trades
        trades = pd.DataFrame({
            'pnl': [100, -50, 150, -30, 200, -40, 80, -20]
        })
        
        rr_ratio = risk_reward_ratio(trades)
        
        # Calcola manualmente
        wins = trades[trades['pnl'] > 0]['pnl'].mean()
        losses = abs(trades[trades['pnl'] < 0]['pnl'].mean())
        
        expected_rr = wins / losses if losses > 0 else 0
        
        print(f"\n[Risk/Reward Test]")
        print(f"  Avg Win: {wins}")
        print(f"  Avg Loss: {losses}")
        print(f"  R/R Ratio: {rr_ratio:.2f}")
        
        assert rr_ratio > 0


# ==================== TEST LOGICAL MODULES ====================

class TestLogicalModules:
    """Test per i moduli logici del portafoglio."""
    
    def test_portfolio_value_calculation(self):
        """Testa il calcolo del valore totale del portafoglio."""
        from logical_portfolio_module import Portfolio
        
        portfolio = Portfolio(balances={"BTC": 1.0, "ETH": 10.0, "USDT": 50000})
        portfolio.set_price("BTC", 95000)
        portfolio.set_price("ETH", 3500)
        
        total_value = portfolio.total_value()
        
        # Valore atteso: 1*95000 + 10*3500 + 50000 = 95000 + 35000 + 50000 = 180000
        expected_value = 1 * 95000 + 10 * 3500 + 50000
        
        print(f"\n[Portfolio Value Test]")
        print(f"  BTC: 1.0 @ $95,000 = $95,000")
        print(f"  ETH: 10.0 @ $3,500 = $35,000")
        print(f"  USDT: $50,000")
        print(f"  Expected Total: ${expected_value}")
        print(f"  Actual Total: ${total_value}")
        
        assert total_value == expected_value
    
    def test_sentiment_calculation(self):
        """Testa il calcolo del sentiment."""
        from logical_portfolio_module import LogicalPortfolioEngine, Portfolio
        
        portfolio = Portfolio(balances={"BTC": 1.0})
        engine = LogicalPortfolioEngine(portfolio)
        
        # Test con parole bullish
        bullish_title = "Bitcoin Surges Past $95K on ETF Inflows"
        sentiment = engine._calculate_sentiment(bullish_title)
        
        print(f"\n[Sentiment Test - Bullish]")
        print(f"  Title: {bullish_title}")
        print(f"  Sentiment: {sentiment:.2f}")
        
        assert sentiment > 0, "Bullish news should have positive sentiment"
        
        # Test con parole bearish
        bearish_title = "Bitcoin Crashes Amid Regulation Concerns"
        sentiment_bear = engine._calculate_sentiment(bearish_title)
        
        print(f"\n[Sentiment Test - Bearish]")
        print(f"  Title: {bearish_title}")
        print(f"  Sentiment: {sentiment_bear:.2f}")
        
        assert sentiment_bear < 0, "Bearish news should have negative sentiment"
    
    def test_signal_determination(self):
        """Testa la determinazione del segnale."""
        from logical_portfolio_module import LogicalPortfolioEngine, Portfolio
        
        portfolio = Portfolio(balances={"BTC": 1.0})
        engine = LogicalPortfolioEngine(portfolio)
        
        # Test threshold
        assert engine._determine_signal(0.5) == "BUY", "High positive sentiment should be BUY"
        assert engine._determine_signal(-0.5) == "SELL", "High negative sentiment should be SELL"
        assert engine._determine_signal(0.0) == "HOLD", "Neutral sentiment should be HOLD"
        
        print(f"\n[Signal Determination Test]")
        print(f"  Sentiment 0.5 -> {engine._determine_signal(0.5)}")
        print(f"  Sentiment -0.5 -> {engine._determine_signal(-0.5)}")
        print(f"  Sentiment 0.0 -> {engine._determine_signal(0.0)}")


# ==================== TEST BOLLINGER BANDS ====================

class TestBollingerBands:
    """Test per le Bollinger Bands."""
    
    def test_bollinger_bands(self, sample_prices):
        """Testa il calcolo delle Bollinger Bands."""
        from technical_analysis import TechnicalAnalyzer
        
        analyzer = TechnicalAnalyzer()
        bb = analyzer.calculate_bollinger_bands(sample_prices)
        
        print(f"\n[Bollinger Bands Test]")
        print(f"  Upper Band: {bb.get('upper', 0):.2f}")
        print(f"  Middle Band (SMA): {bb.get('middle', 0):.2f}")
        print(f"  Lower Band: {bb.get('lower', 0):.2f}")
        
        # Verifica ordine
        assert bb['upper'] > bb['middle'] > bb['lower'], "Upper > Middle > Lower"


# ==================== MAIN ====================

if __name__ == "__main__":
    # Esegue tutti i test
    pytest.main([__file__, "-v", "--tb=short"])
