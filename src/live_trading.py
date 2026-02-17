"""
Live Trading Orchestrator
=========================
Orchestrates live trading: signals → execution → account management.

This module connects:
- Signal Engine (generates trading signals)
- Execution Layer (places orders on exchanges)
- Account Manager (tracks equity, PnL, fees)

Usage:
    orchestrator = LiveTradingOrchestrator()
    orchestrator.run_cycle()  # Run once per iteration
    # OR
    orchestrator.start()  # Start continuous loop
"""

import time
import logging
import schedule
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from src.execution import ExchangeClient, LiveTradingEngine
from src.account_manager import AccountManager, EquityTracker, PerformanceFeeCalculator, AccountSnapshot
from src.data_collector import DataCollector
from src.technical_analysis import TechnicalAnalysis
from src.signal_engine import SignalEngine

logger = logging.getLogger(__name__)


@dataclass
class TradeResult:
    """Risultato di un trade."""
    symbol: str
    signal: int
    order_result: Optional[Dict[str, Any]]
    success: bool
    timestamp: str
    error: str = ""


class LiveTradingOrchestrator:
    """
    Orchestratore per il trading live.
    
    Workflow:
    1. Carica dati mercato
    2. Genera segnali
    3. Esegue ordini
    4. Salva snapshot equity
    5. Log results
    """
    
    def __init__(
        self,
        data_dir: str = "data/accounts",
        symbols: List[str] = None,
        interval: str = "1h",
        testnet: bool = True,
    ):
        """
        Inizializza l'orchestratore.
        
        Args:
            data_dir: Directory per i dati account
            symbols: Lista simboli da tradare
            interval: Intervallo candele
            testnet: Usa testnet
        """
        self.data_dir = data_dir
        self.symbols = symbols or ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        self.interval = interval
        self.testnet = testnet
        
        # Inizializza componenti
        self.account_manager = AccountManager(data_dir=data_dir)
        self.equity_tracker = EquityTracker(data_dir=data_dir)
        self.fee_calculator = PerformanceFeeCalculator()
        
        self.data_collector = DataCollector()
        self.tech_analysis = TechnicalAnalysis()
        self.signal_engine = SignalEngine()
        
        # Clients per utente {user_id: ExchangeClient}
        self.exchange_clients: Dict[str, ExchangeClient] = {}
        
        # Stato
        self.is_running = False
        self.last_cycle = None
        
        logger.info(f"🚀 Live Trading Orchestrator inizializzato")
        logger.info(f"   Simboli: {self.symbols}")
        logger.info(f"   Testnet: {testnet}")
    
    def get_or_create_client(self, user_id: str) -> Optional[ExchangeClient]:
        """
        Ottiene o crea un client per l'utente.
        """
        if user_id in self.exchange_clients:
            return self.exchange_clients[user_id]
        
        user = self.account_manager.get_user(user_id)
        if not user:
            logger.error(f"Utente {user_id} non trovato")
            return None
        
        try:
            client = ExchangeClient(
                api_key=user.api_key,
                api_secret=user.api_secret,
                testnet=user.testnet,
                exchange=user.exchange,
            )
            
            # Valida API key
            if not client.validate_api_key():
                logger.error(f"API key non valida per {user_id}")
                return None
            
            self.exchange_clients[user_id] = client
            logger.info(f"✅ Client creato per {user.username}")
            return client
            
        except Exception as e:
            logger.error(f"Errore creazione client: {e}")
            return None
    
    def generate_signals(self, symbol: str) -> Dict[str, Any]:
        """
        Genera segnali per un simbolo.
        
        Returns:
            Dizionario con signal, confidence, etc.
        """
        try:
            # Carica dati
            df = self.data_collector.fetch_ohlcv(
                symbol=symbol,
                interval=self.interval,
                limit=200,
            )
            
            if df is None or len(df) < 100:
                logger.warning(f"Dati insufficienti per {symbol}")
                return {"signal": 0, "confidence": 0}
            
            # Calcola indicatori
            df = self.tech_analysis.add_indicators(df)
            
            # Genera segnali
            df = self.signal_engine.generate(df)
            
            # Prendi l'ultimo segnale
            latest = df.iloc[-1]
            
            return {
                "signal": int(latest.get("signal", 0)),
                "confidence": float(latest.get("signal_confidence", 0)),
                "price": float(latest.get("close", 0)),
                "timestamp": latest.name.isoformat() if hasattr(latest.name, 'isoformat') else str(latest.name),
            }
            
        except Exception as e:
            logger.error(f"Errore generazione segnali {symbol}: {e}")
            return {"signal": 0, "confidence": 0}
    
    def execute_trade(
        self,
        client: ExchangeClient,
        symbol: str,
        signal: int,
        risk_capital: float = 100,
    ) -> TradeResult:
        """
        Esegue un trade basato sul segnale.
        """
        timestamp = datetime.now().isoformat()
        
        if signal == 0:
            # Close position
            result = client.close_position(symbol)
            return TradeResult(
                symbol=symbol,
                signal=signal,
                order_result=result,
                success=result is not None,
                timestamp=timestamp,
            )
        
        # Execute signal
        side = "BUY" if signal == 1 else "SELL"
        price = client.get_price(symbol)
        quantity = risk_capital / price
        
        result = client.place_market_order(symbol, side, quantity)
        
        return TradeResult(
            symbol=symbol,
            signal=signal,
            order_result=result,
            success=result is not None,
            timestamp=timestamp,
            error="" if result else "Order failed",
        )
    
    def save_account_snapshot(self, user_id: str, client: ExchangeClient):
        """
        Salva uno snapshot dell'account.
        """
        try:
            balance = client.get_balance("USDT")
            positions = client.get_positions()
            unrealized_pnl = sum(p.get("unrealized_pnl", 0) for p in positions)
            
            # Calcola equity
            equity = balance + unrealized_pnl
            
            # Calcola daily PNL
            last_snapshot = self.equity_tracker.get_latest_snapshot(user_id)
            if last_snapshot:
                daily_pnl = equity - last_snapshot.equity
                daily_pnl_pct = (daily_pnl / last_snapshot.equity) * 100
            else:
                daily_pnl = 0
                daily_pnl_pct = 0
            
            snapshot = AccountSnapshot(
                user_id=user_id,
                timestamp=datetime.now().isoformat(),
                equity=equity,
                balance=balance,
                unrealized_pnl=unrealized_pnl,
                daily_pnl=daily_pnl,
                daily_pnl_pct=daily_pnl_pct,
                open_positions=len(positions),
            )
            
            self.equity_tracker.save_snapshot(snapshot)
            logger.info(f"📊 Snapshot salvato: ${equity:.2f}")
            
        except Exception as e:
            logger.error(f"Errore salvataggio snapshot: {e}")
    
    def run_cycle(self):
        """
        Esegue un ciclo completo di trading.
        
        Workflow:
        1. Per ogni utente attivo:
           a. Genera segnali per ogni simbolo
           b. Esegue trade
           c. Salva snapshot
        """
        logger.info("🔄 Inizio ciclo trading...")
        
        active_users = self.account_manager.list_active_users()
        
        for user in active_users:
            logger.info(f"👤 Elaboro utente: {user.username}")
            
            # Get/create exchange client
            client = self.get_or_create_client(user.user_id)
            if not client:
                continue
            
            # Process each symbol
            for symbol in self.symbols:
                # Generate signal
                signal_data = self.generate_signals(symbol)
                signal = signal_data.get("signal", 0)
                confidence = signal_data.get("confidence", 0)
                
                logger.info(f"   {symbol}: signal={signal}, confidence={confidence:.2f}")
                
                # Execute if confidence is high enough
                if confidence >= 0.6 and signal != 0:
                    risk_capital = user.max_risk_per_trade * client.get_balance("USDT")
                    
                    result = self.execute_trade(
                        client=client,
                        symbol=symbol,
                        signal=signal,
                        risk_capital=risk_capital,
                    )
                    
                    if result.success:
                        logger.info(f"   ✅ Trade eseguito: {result.side} {symbol}")
                    else:
                        logger.warning(f"   ❌ Trade fallito: {result.error}")
                
                # Close on flat signal
                elif signal == 0:
                    result = self.execute_trade(client, symbol, 0)
                    if result.success:
                        logger.info(f"   🔒 Posizione chiusa: {symbol}")
            
            # Save snapshot
            self.save_account_snapshot(user.user_id, client)
        
        self.last_cycle = datetime.now()
        logger.info("✅ Ciclo completato")
    
    def start(self, interval_minutes: int = 60):
        """
        Avvia il loop di trading continuo.
        
        Args:
            interval_minutes: Intervallo tra i cicli
        """
        self.is_running = True
        
        # Schedule runs
        schedule.every(interval_minutes).minutes.do(self.run_cycle)
        
        # Run immediately
        self.run_cycle()
        
        logger.info(f"🚀 Trading avviato (ogni {interval_minutes} minuti)")
        
        while self.is_running:
            schedule.run_pending()
            time.sleep(10)
    
    def stop(self):
        """Ferma il trading."""
        self.is_running = False
        logger.info("⏹️ Trading fermato")


class PortfolioManager:
    """
    Gestisce un portfolio multi-asset per un utente.
    Implementa various allocation strategies.
    """
    
    def __init__(self, allocation_strategy: str = "equal"):
        """
        Inizializza il portfolio manager.
        
        Args:
            allocation_strategy: 'equal', 'volatility_parity', 'risk_parity'
        """
        self.allocation_strategy = allocation_strategy
        
    def calculate_allocation(
        self,
        signals: Dict[str, Dict[str, Any]],
        total_capital: float,
    ) -> Dict[str, float]:
        """
        Calcola l'allocazione per ogni asset.
        
        Args:
            signals: {symbol: signal_data}
            total_capital: Capitale totale disponibile
            
        Returns:
            {symbol: allocated_capital}
        """
        active_symbols = [
            s for s, data in signals.items() 
            if data.get("signal", 0) != 0 and data.get("confidence", 0) >= 0.6
        ]
        
        if not active_symbols:
            return {}
        
        n_assets = len(active_symbols)
        
        if self.allocation_strategy == "equal":
            # Equal weight
            per_asset = total_capital / n_assets
            
        elif self.allocation_strategy == "volatility_parity":
            # Inverse volatility weighting
            volatilities = [
                signals[s].get("volatility", 0.02) for s in active_symbols
            ]
            inv_vol = [1/v if v > 0 else 1 for v in volatilities]
            total_inv_vol = sum(inv_vol)
            weights = [iv/total_inv_vol for iv in inv_vol]
            per_asset = [total_capital * w for w in weights]
            
        else:
            # Default: equal
            per_asset = total_capital / n_assets
        
        return {
            s: per_asset[i] if isinstance(per_asset, list) else per_asset
            for i, s in enumerate(active_symbols)
        }


# ======================
# ESEMPIO DI UTILIZZO
# ======================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Crea orchestrator
    orchestrator = LiveTradingOrchestrator(
        symbols=["BTCUSDT", "ETHUSDT"],
        interval="1h",
        testnet=True,
    )
    
    # Run single cycle (for testing)
    print("🧪 Test ciclo singolo...")
    orchestrator.run_cycle()
    
    print("✅ Test completato!")
