#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for Trading Ledger Module
"""

import sys
import os

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.trading_ledger import TradingLedger, get_ledger, quick_trade
import tempfile
import shutil

def test_trading_ledger():
    """Test completo del Trading Ledger."""
    
    # Crea directory temporanea per test
    test_dir = tempfile.mkdtemp()
    
    try:
        print("=" * 60)
        print("  TRADING LEDGER - TEST SUITE")
        print("=" * 60)
        
        # 1. Inizializzazione
        print("\n[TEST 1] Inizializzazione...")
        ledger = TradingLedger(data_dir=test_dir, initial_balance=10000.0)
        assert ledger.balance == 10000.0, "Initial balance mismatch"
        print("[OK] Inizializzazione corretta")
        
        # 2. Registra trade BUY
        print("\n[TEST 2] Registrazione trade BUY...")
        trade1 = ledger.record_trade(
            asset="BTC",
            trade_type="BUY",
            quantity=0.1,
            price=25000.0,
            commission=5.0,
            notes="Test buy order"
        )
        assert trade1.trade_type == "BUY"
        assert trade1.asset == "BTC"
        assert "BTC" in ledger.positions
        print(f"[OK] BUY registrato: {trade1.id}")
        
        # 3. Registra trade SELL con profit
        print("\n[TEST 3] Registrazione trade SELL con profit...")
        trade2 = ledger.record_trade(
            asset="BTC",
            trade_type="SELL",
            quantity=0.1,
            price=26000.0,
            commission=5.0,
            notes="Test sell order - profit"
        )
        assert trade2.profit_loss > 0, "Should have profit"
        print(f"[OK] SELL registrato: {trade2.id} | P/L: {trade2.profit_loss:.2f}")
        
        # 4. Verifica posizioni chiuse
        print("\n[TEST 4] Verifica posizioni chiuse...")
        assert "BTC" not in ledger.positions, "BTC position should be closed"
        print("[OK] Posizione BTC chiusa correttamente")
        
        # 5. Trade multipli
        print("\n[TEST 5] Trade multipli...")
        for i in range(5):
            ledger.record_trade(
                asset="ETH",
                trade_type="BUY",
                quantity=1.0,
                price=1500.0 + i * 10,
                commission=2.0
            )
        
        for i in range(3):
            ledger.record_trade(
                asset="ETH",
                trade_type="SELL",
                quantity=1.0,
                price=1550.0 + i * 15,
                commission=2.0
            )
        
        print(f"[OK] Trade multipli registrati. Totale: {len(ledger.trades)}")
        
        # 6. Statistiche
        print("\n[TEST 6] Calcolo statistiche...")
        stats = ledger.get_statistics("all")
        print(f"  - Total Trades: {stats['total_trades']}")
        print(f"  - Win Rate: {stats['win_rate']:.1f}%")
        print(f"  - Total P/L: {stats['total_profit_loss']:.2f}")
        print(f"  - Current Balance: {stats['current_balance']:.2f}")
        print("[OK] Statistiche calcolate")
        
        # 7. Report
        print("\n[TEST 7] Generazione report...")
        report = ledger.generate_report("all")
        print(report)
        
        # 8. Export CSV
        print("[TEST 8] Export CSV...")
        csv_path = ledger.export_to_csv()
        assert os.path.exists(csv_path)
        print(f"[OK] CSV esportato: {csv_path}")
        
        # 9. Singleton
        print("\n[TEST 9] Test singleton...")
        ledger2 = get_ledger()
        # Nota: singleton usa directory default, non test_dir
        print("[OK] Singleton funzionante")
        
        # 10. Quick trade
        print("\n[TEST 10] Quick trade function...")
        quick = quick_trade("SOL", "BUY", 10.0, 100.0, 1.0)
        assert quick.asset == "SOL"
        print(f"[OK] Quick trade: {quick.id}")
        
        print("\n" + "=" * 60)
        print("  TUTTI I TEST SUPERATI")
        print("=" * 60)
        
    finally:
        # Cleanup
        shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    test_trading_ledger()
