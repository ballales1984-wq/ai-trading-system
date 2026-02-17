"""
Trade Logger - Storico Operazioni
==================================
Registra ogni trade eseguito con tutti i dettagli.

Integrazione:
- Execution Layer → log automatico
- Dashboard → visualizzazione trades
- KPI → calcolo metriche
"""

import json
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class TradeLogger:
    """
    Logger per registrare ogni trade eseguito.
    """
    
    def __init__(self, storage_path: str = "data/trades.json"):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.trades = self._load_trades()
    
    def _load_trades(self) -> Dict:
        """Carica i trades da file."""
        try:
            if self.storage_path.exists():
                with open(self.storage_path, "r") as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Errore caricamento trades: {e}")
        return {}
    
    def _save_trades(self):
        """Salva i trades su file."""
        with open(self.storage_path, "w") as f:
            json.dump(self.trades, f, indent=2)
    
    def log_trade(
        self,
        user_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        equity_before: float,
        equity_after: float,
        order_type: str = "MARKET",
        fees: float = 0.0,
    ) -> Dict:
        """
        Registra un trade.
        
        Args:
            user_id: ID utente
            symbol: Simbolo (es. BTCUSDT)
            side: BUY o SELL
            quantity: Quantità
            price: Prezzo di esecuzione
            equity_before: Equity prima del trade
            equity_after: Equity dopo il trade
            order_type: Tipo ordine (MARKET, LIMIT, etc.)
            fees: Fee pagate
            
        Returns:
            Trade entry creata
        """
        if user_id not in self.trades:
            self.trades[user_id] = []
        
        pnl = equity_after - equity_before
        
        trade_entry = {
            "id": f"{user_id}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "notional": quantity * price,
            "order_type": order_type,
            "fees": fees,
            "equity_before": equity_before,
            "equity_after": equity_after,
            "pnl": pnl,
            "pnl_pct": (pnl / equity_before * 100) if equity_before > 0 else 0,
        }
        
        self.trades[user_id].append(trade_entry)
        self._save_trades()
        
        logger.info(f"📝 Trade registrato: {side} {quantity} {symbol} @ ${price}")
        
        return trade_entry
    
    def get_user_trades(
        self, 
        user_id: str, 
        symbol: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict]:
        """
        Recupera i trades di un utente.
        
        Args:
            user_id: ID utente
            symbol: Filtra per simbolo (opzionale)
            limit: Numero massimo trades
            
        Returns:
            Lista trades
        """
        trades = self.trades.get(user_id, [])
        
        if symbol:
            trades = [t for t in trades if t.get("symbol") == symbol]
        
        return trades[-limit:]
    
    def get_trades_by_date(
        self,
        user_id: str,
        start_date: str,
        end_date: str = None,
    ) -> List[Dict]:
        """
        Recupera trades in un intervallo di date.
        
        Args:
            user_id: ID utente
            start_date: Data inizio (ISO format)
            end_date: Data fine (ISO format, default now)
            
        Returns:
            Lista trades
        """
        if end_date is None:
            end_date = datetime.now().isoformat()
        
        trades = self.trades.get(user_id, [])
        
        return [
            t for t in trades 
            if start_date <= t.get("timestamp", "") <= end_date
        ]
    
    def get_trade_stats(self, user_id: str) -> Dict:
        """
        Statistiche base sui trades.
        
        Returns:
            Dizionario con stats
        """
        trades = self.trades.get(user_id, [])
        
        if not trades:
            return {
                "total_trades": 0,
                "total_volume": 0,
                "total_fees": 0,
            }
        
        total_volume = sum(t.get("notional", 0) for t in trades)
        total_fees = sum(t.get("fees", 0) for t in trades)
        
        return {
            "total_trades": len(trades),
            "buy_trades": len([t for t in trades if t.get("side") == "BUY"]),
            "sell_trades": len([t for t in trades if t.get("side") == "SELL"]),
            "total_volume": total_volume,
            "total_fees": total_fees,
            "avg_trade_size": total_volume / len(trades),
        }
    
    def clear_user_trades(self, user_id: str):
        """Elimina tutti i trades di un utente."""
        if user_id in self.trades:
            self.trades[user_id] = []
            self._save_trades()
            logger.info(f"🗑️ Trades eliminati per utente {user_id}")


# ======================
# ESEMPIO DI UTILIZZO
# ======================

if __name__ == "__main__":
    logger = TradeLogger()
    
    # Log some test trades
    logger.log_trade(
        user_id="test_user",
        symbol="BTCUSDT",
        side="BUY",
        quantity=0.01,
        price=50000,
        equity_before=10000,
        equity_after=9950,
        fees=10,
    )
    
    logger.log_trade(
        user_id="test_user",
        symbol="ETHUSDT",
        side="BUY",
        quantity=1,
        price=3000,
        equity_before=9950,
        equity_after=9900,
        fees=5,
    )
    
    # Get stats
    stats = logger.get_trade_stats("test_user")
    print(f"📊 Stats: {stats}")
