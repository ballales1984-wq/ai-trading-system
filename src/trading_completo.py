"""
Trading Complete Module
=====================
Integrated trading system with:
1. Trade recording (CSV)
2. Balance tracking
3. Profit/Loss calculation
4. Award system for logic and APIs

Usage:
    from src.trading_completo import (
        inizializza_registro,
        registra_trade,
        report_giornaliero,
        report_premi,
        get_balance,
        reset_registro
    )
    
    # Initialize
    inizializza_registro()
    
    # Record BUY
    registra_trade({
        "asset": "BTC",
        "tipo": "BUY",
        "quantita": 0.1,
        "prezzo": 25000,
        "commissione": 5,
        "api_usata": "NotizieCryptoAPI"
    })
    
    # Record SELL
    registra_trade({
        "asset": "BTC",
        "tipo": "SELL",
        "quantita": 0.1,
        "prezzo": 25500,
        "commissione": 5,
        "prezzo_acquisto": 25000,
        "api_usata": "NotizieCryptoAPI"
    })
    
    # Reports
    report_giornaliero()
    report_premi()
"""

import csv
import os
import json
import logging
from datetime import datetime, date
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path


logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# Default paths
DEFAULT_DATA_DIR = "data/trading"
DEFAULT_REGISTRO_FILE = f"{DEFAULT_DATA_DIR}/registro_trading.csv"
DEFAULT_SALDO_FILE = f"{DEFAULT_DATA_DIR}/saldo.txt"
DEFAULT_POSIZIONI_FILE = f"{DEFAULT_DATA_DIR}/posizioni.json"
DEFAULT_PREMI_FILE = f"{DEFAULT_DATA_DIR}/premi.json"


# Award configuration
AWARD_CONFIG = {
    "base_points_per_profit": 1.0,      # Points per unit of profit
    "bonus_logic_sell": 10,              # Bonus for correct SELL with profit
    "bonus_api": 5,                      # Bonus for API contribution
    "bonus_strategy_correct": 15,        # Bonus for correct strategy decision
    "penalty_wrong_direction": -5,       # Penalty for wrong direction
    "penalty_loss": -2,                   # Points removed per unit of loss
}


@dataclass
class Trade:
    """Trade data structure."""
    timestamp: str
    asset: str
    tipo: str  # BUY or SELL
    quantita: float
    prezzo: float
    commissione: float
    profit_loss: float
    saldo_totale: float
    api_usata: str
    punteggio_premio: float
    strategy: str = "unknown"
    prezzo_acquisto: Optional[float] = None


@dataclass
class Position:
    """Open position tracking."""
    asset: str
    quantita: float
    prezzo_acquisto: float
    data_acquisto: str
    api_usata: str
    strategy: str


# =============================================================================
# Global State
# =============================================================================

# In-memory storage
_premi: Dict[str, float = {}]
_posizioni: Dict[str, Position] = {}
_data_dir = DEFAULT_DATA_DIR


# =============================================================================
# File Operations
# =============================================================================

def set_data_dir(path: str) -> None:
    """Set custom data directory."""
    global _data_dir
    _data_dir = path
    os.makedirs(path, exist_ok=True)


def _get_registro_path() -> str:
    """Get registro file path."""
    return os.path.join(_data_dir, "registro_trading.csv")


def _get_saldo_path() -> str:
    """Get saldo file path."""
    return os.path.join(_data_dir, "saldo.txt")


def _get_posizioni_path() -> str:
    """Get posizioni file path."""
    return os.path.join(_data_dir, "posizioni.json")


def _get_premi_path() -> str:
    """Get premi file path."""
    return os.path.join(_data_dir, "premi.json")


# =============================================================================
# Balance Management
# =============================================================================

def get_balance() -> float:
    """
    Read current balance.
    
    Returns:
        Current balance in account currency
    """
    saldo_path = _get_saldo_path()
    try:
        with open(saldo_path, "r") as f:
            return float(f.read().strip())
    except (FileNotFoundError, ValueError):
        return 0.0


def set_balance(amount: float) -> None:
    """
    Set balance to specific amount.
    
    Args:
        amount: New balance
    """
    saldo_path = _get_saldo_path()
    os.makedirs(os.path.dirname(saldo_path), exist_ok=True)
    with open(saldo_path, "w") as f:
        f.write(str(amount))


def update_balance(profit_loss: float) -> float:
    """
    Update balance by profit/loss.
    
    Args:
        profit_loss: Amount to add (can be negative)
    
    Returns:
        New balance
    """
    saldo = get_balance()
    saldo += profit_loss
    set_balance(saldo)
    return saldo


def reset_balance(amount: float = 0.0) -> None:
    """Reset balance to specified amount."""
    set_balance(amount)


# =============================================================================
# Position Management
# =============================================================================

def apri_posizione(asset: str, quantita: float, prezzo: float, 
                   api_usata: str, strategy: str = "unknown") -> Position:
    """
    Open a new position.
    
    Args:
        asset: Trading asset
        quantita: Quantity
        prezzo: Purchase price
        api_usata: API that generated the signal
        strategy: Strategy used
    
    Returns:
        Created position
    """
    posizione = Position(
        asset=asset,
        quantita=quantita,
        prezzo_acquisto=prezzo,
        data_acquisto=datetime.now().isoformat(),
        api_usata=api_usata,
        strategy=strategy
    )
    
    _posizioni[asset] = posizione
    _save_posizioni()
    
    logger.info(f"Opened position: {quantita} {asset} at {prezzo}")
    return posizione


def chiudi_posizione(asset: str) -> Optional[Position]:
    """
    Close a position.
    
    Args:
        asset: Asset to close
    
    Returns:
        Closed position or None
    """
    if asset in _posizioni:
        posizione = _posizioni.pop(asset)
        _save_posizioni()
        logger.info(f"Closed position: {posizione.quantita} {asset}")
        return posizione
    return None


def get_posizione(asset: str) -> Optional[Position]:
    """Get open position for asset."""
    return _posizioni.get(asset)


def get_all_posizioni() -> Dict[str, Position]:
    """Get all open positions."""
    return dict(_posizioni)


def _save_posizioni() -> None:
    """Save positions to file."""
    path = _get_posizioni_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    data = {k: asdict(v) for k, v in _posizioni.items()}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _load_posizioni() -> None:
    """Load positions from file."""
    path = _get_posizioni_path()
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
            for k, v in data.items():
                _posizioni[k] = Position(**v)


# =============================================================================
# Award System
# =============================================================================

def get_awards() -> Dict[str, float]:
    """Get all awards."""
    return dict(_premi)


def get_total_awards() -> float:
    """Get total award points."""
    return sum(_premi.values())


def reset_awards() -> None:
    """Reset all awards."""
    global _premi
    _premi = {}
    _save_premi()


def _save_premi() -> None:
    """Save awards to file."""
    path = _get_premi_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(_premi, f, indent=2)


def _load_premi() -> None:
    """Load awards from file."""
    global _premi
    path = _get_premi_path()
    if os.path.exists(path):
        with open(path, "r") as f:
            _premi = json.load(f)


def assegna_premio(trade: Dict[str, Any]) -> float:
    """
    Assign award points based on trade result.
    
    Args:
        trade: Trade dictionary
    
    Returns:
        Award points assigned
    """
    config = AWARD_CONFIG
    
    profit_loss = trade.get("profit_loss", 0)
    tipo = trade.get("tipo", "").upper()
    api_usata = trade.get("api_usata", "")
    strategy = trade.get("strategy", "unknown")
    
    # Base: points from profit
    if profit_loss > 0:
        punteggio = profit_loss * config["base_points_per_profit"]
    else:
        punteggio = profit_loss * abs(config["penalty_loss"])
    
    # Bonus: SELL with profit = correct logic
    if tipo == "SELL" and profit_loss > 0:
        punteggio += config["bonus_logic_sell"]
    
    # Bonus: API contribution
    if api_usata and profit_loss > 0:
        punteggio += config["bonus_api"]
    
    # Bonus: Strategy correct
    if strategy and strategy != "unknown" and profit_loss > 0:
        punteggio += config["bonus_strategy_correct"]
    
    # Penalty: wrong direction
    if tipo == "BUY" and profit_loss < 0:
        punteggio += config["penalty_wrong_direction"]
    
    # Update global awards
    chiave = api_usata if api_usata else "logica_base"
    if chiave not in _premi:
        _premi[chiave] = 0
    _premi[chiave] += punteggio
    
    # Also track by strategy
    if strategy and strategy != "unknown":
        strategy_key = f"strategy_{strategy}"
        if strategy_key not in _premi:
            _premi[strategy_key] = 0
        _premi[strategy_key] += punteggio
    
    _save_premi()
    
    logger.info(f"🏆 Award assigned: {punteggio:.2f} points to {chiave}")
    return punteggio


# =============================================================================
# Trade Registration
# =============================================================================

def inizializza_registro() -> None:
    """Initialize trading registry."""
    os.makedirs(_data_dir, exist_ok=True)
    
    registro_path = _get_registro_path()
    
    if not os.path.exists(registro_path):
        with open(registro_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                "Data/Ora", "Asset", "Tipo", "Quantità", "Prezzo",
                "Commissione", "Profit/Loss", "Saldo Totale", "API Usata", 
                "Punti Premio", "Strategy", "Prezzo Acquisto"
            ])
        logger.info(f"Initialized trading registry: {registro_path}")
    
    # Load existing data
    _load_posizioni()
    _load_premi()


def registra_trade(trade: Dict[str, Any]) -> Trade:
    """
    Register a trade with full tracking.
    
    Args:
        trade: Dictionary with keys:
            - asset: Trading asset (e.g., "BTC")
            - tipo: "BUY" or "SELL"
            - quantita: Quantity
            - prezzo: Current price
            - commissione: Commission fee
            - prezzo_acquisto: Purchase price (for SELL)
            - api_usata: API that generated signal
            - strategy: Strategy name
    
    Returns:
        Trade object with all calculated fields
    """
    asset = trade.get("asset")
    tipo = trade.get("tipo", "").upper()
    quantita = trade.get("quantita")
    prezzo = trade.get("prezzo")
    commissione = trade.get("commissione", 0.0)
    prezzo_acquisto = trade.get("prezzo_acquisto")
    api_usata = trade.get("api_usata", "")
    strategy = trade.get("strategy", "unknown")
    
    # Calculate profit/loss
    if tipo == "SELL" and prezzo_acquisto is not None:
        profit_loss = (prezzo - prezzo_acquisto) * quantita - commissione
    else:
        # BUY or no previous price = just commission
        profit_loss = -commissione
    
    # Update trade dict
    trade["profit_loss"] = profit_loss
    
    # Assign award
    trade["punteggio"] = assegna_premio(trade)
    
    # Update balance
    saldo = update_balance(profit_loss)
    
    # Update positions
    if tipo == "BUY":
        apri_posizione(asset, quantita, prezzo, api_usata, strategy)
    elif tipo == "SELL":
        chiudi_posizione(asset)
    
    # Create trade object
    timestamp = datetime.now().isoformat()
    trade_obj = Trade(
        timestamp=timestamp,
        asset=asset,
        tipo=tipo,
        quantita=quantita,
        prezzo=prezzo,
        commissione=commissione,
        profit_loss=profit_loss,
        saldo_totale=saldo,
        api_usata=api_usata,
        punteggio_premio=trade["punteggio"],
        strategy=strategy,
        prezzo_acquisto=prezzo_acquisto
    )
    
    # Write to CSV
    registro_path = _get_registro_path()
    with open(registro_path, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            timestamp, asset, tipo, quantita, prezzo,
            commissione, profit_loss, saldo, api_usata,
            trade["punteggio"], strategy, prezzo_acquisto
        ])
    
    logger.info(f"✅ Trade registered: {tipo} {quantita} {asset} @ {prezzo}. "
                f"P/L: {profit_loss:.2f}, Award: {trade['punteggio']:.2f}, Balance: {saldo:.2f}")
    
    return trade_obj


def reset_registro() -> None:
    """Reset all trading data."""
    global _premi, _posizioni
    
    # Reset files
    set_balance(0.0)
    _premi = {}
    _posizioni = {}
    
    # Delete files
    for path in [_get_registro_path(), _get_posizioni_path(), _get_premi_path()]:
        if os.path.exists(path):
            os.remove(path)
    
    # Re-initialize
    inizializza_registro()
    
    logger.warning("Trading registry reset")


# =============================================================================
# Reports
# =============================================================================

def report_giornaliero(giorno: Optional[date] = None) -> List[Dict]:
    """
    Get daily trading report.
    
    Args:
        giorno: Date to report (default: today)
    
    Returns:
        List of trade dictionaries
    """
    if giorno is None:
        giorno = date.today()
    
    trades = []
    registro_path = _get_registro_path()
    
    try:
        with open(registro_path, "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                trade_date = row["Data/Ora"][:10]
                if trade_date == giorno.isoformat():
                    trades.append(row)
    except FileNotFoundError:
        logger.warning("No trades found")
    
    # Print report
    print(f"\n📊 Report Trading Giornaliero - {giorno}")
    print("=" * 80)
    
    if not trades:
        print("Nessun trade registrato oggi.")
    else:
        total_pnl = 0
        for t in trades:
            pnl = float(t["Profit/Loss"])
            total_pnl += pnl
            print(f"{t['Data/Ora']} | {t['Tipo']:4s} {t['Quantita']:8s} {t['Asset']:6s} "
                  f"@ {t['Prezzo']:10s} | P/L: {pnl:10.2f} | Award: {t['Punti Premio']:6.2f}")
        
        print("-" * 80)
        print(f"Total P/L: {total_pnl:.2f}")
        print(f"Total Trades: {len(trades)}")
    
    print("=" * 80)
    
    return trades


def report_premi() -> Dict[str, float]:
    """
    Print award report.
    
    Returns:
        Dictionary of awards by source
    """
    _load_premi()  # Ensure latest data
    
    print("\n🏆 Report Premi Logica/API")
    print("=" * 50)
    
    if not _premi:
        print("Nessun premio assegnato.")
    else:
        # Sort by points
        sorted_premi = sorted(_premi.items(), key=lambda x: x[1], reverse=True)
        
        for chiave, punti in sorted_premi:
            bar = "█" * int(min(punti / 10, 20))
            print(f"{chiave:30s} | {punti:8.2f} | {bar}")
        
        print("-" * 50)
        totale = sum(_premi.values())
        print(f"{'TOTALE':30s} | {totale:8.2f}")
    
    print("=" * 50)
    
    return dict(_premi)


def report_performance() -> Dict[str, Any]:
    """
    Get comprehensive performance report.
    
    Returns:
        Dictionary with performance metrics
    """
    saldo = get_balance()
    posizioni = get_all_posizioni()
    premi = get_awards()
    
    # Calculate stats from CSV
    registro_path = _get_registro_path()
    total_trades = 0
    total_pnl = 0
    winning_trades = 0
    losing_trades = 0
    
    try:
        with open(registro_path, "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                total_trades += 1
                pnl = float(row["Profit/Loss"])
                total_pnl += pnl
                if pnl > 0:
                    winning_trades += 1
                elif pnl < 0:
                    losing_trades += 1
    except FileNotFoundError:
        pass
    
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    report = {
        "balance": saldo,
        "open_positions": len(posizioni),
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "total_awards": sum(premi.values()),
        "awards_by_source": premi
    }
    
    # Print summary
    print("\n📈 Performance Report")
    print("=" * 50)
    print(f"Balance:          {saldo:.2f}")
    print(f"Open Positions:   {len(posizioni)}")
    print(f"Total Trades:     {total_trades}")
    print(f"Win Rate:         {win_rate:.1f}%")
    print(f"Total P/L:         {total_pnl:.2f}")
    print(f"Total Awards:     {sum(premi.values()):.2f}")
    print("=" * 50)
    
    return report


# =============================================================================
# Integration Functions
# =============================================================================

def create_trade_from_execution(execution_result: Dict, strategy: str) -> Trade:
    """
    Create and register trade from execution result.
    
    Args:
        execution_result: Execution result from broker
        strategy: Strategy name
    
    Returns:
        Registered trade
    """
    trade = {
        "asset": execution_result.get("symbol", "UNKNOWN"),
        "tipo": execution_result.get("side", "BUY"),
        "quantita": execution_result.get("quantity", 0),
        "prezzo": execution_result.get("price", 0),
        "commissione": execution_result.get("commission", 0),
        "prezzo_acquisto": execution_result.get("entry_price"),
        "api_usata": execution_result.get("signal_source", "system"),
        "strategy": strategy
    }
    
    return registra_trade(trade)


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    print("Testing Trading Complete Module...")
    
    # Initialize
    inizializza_registro()
    set_balance(10000)  # Starting balance
    
    # Example trades
    print("\n--- Recording Trades ---")
    
    # BUY BTC
    registra_trade({
        "asset": "BTC",
        "tipo": "BUY",
        "quantita": 0.1,
        "prezzo": 25000,
        "commissione": 5,
        "api_usata": "NotizieCryptoAPI",
        "strategy": "momentum"
    })
    
    # BUY ETH
    registra_trade({
        "asset": "ETH",
        "tipo": "BUY",
        "quantita": 1.0,
        "prezzo": 1500,
        "commissione": 3,
        "api_usata": "SentimentAPI",
        "strategy": "mean_reversion"
    })
    
    # SELL BTC (profit)
    registra_trade({
        "asset": "BTC",
        "tipo": "SELL",
        "quantita": 0.1,
        "prezzo": 25500,
        "commissione": 5,
        "prezzo_acquisto": 25000,
        "api_usata": "NotizieCryptoAPI",
        "strategy": "momentum"
    })
    
    # Reports
    report_giornaliero()
    report_premi()
    report_performance()
    
    print("\n✅ Test complete!")

