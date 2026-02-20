"""
Live Streaming Manager
======================
Integra WebSocket Binance con PortfolioManager per trading live multi-asset.

Day 1 Checklist:
- [x] WebSocket Binance per tutti gli asset
- [x] PortfolioManager.update_prices() a ogni tick
- [x] Log posizioni aperte e PnL
- [x] Stop-loss in tempo reale
"""

import logging
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field

from src.live.binance_multi_ws import BinanceMultiWebSocket
from src.core.portfolio.portfolio_manager import PortfolioManager, Position, PositionSide

logger = logging.getLogger(__name__)


@dataclass
class StopLossOrder:
    """Ordine stop-loss."""
    symbol: str
    stop_price: float
    quantity: float
    is_trailing: bool = False
    trail_pct: float = 0.0
    triggered: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    
    def check_trigger(self, current_price: float) -> bool:
        """Verifica se lo stop-loss è triggerato."""
        if self.triggered:
            return False
        
        if self.is_trailing:
            # Trailing stop: aggiorna stop price se il prezzo sale
            new_stop = current_price * (1 - self.trail_pct)
            if new_stop > self.stop_price:
                self.stop_price = new_stop
                logger.info(f"Trailing stop updated: {self.symbol} @ {self.stop_price:.4f}")
        
        if current_price <= self.stop_price:
            self.triggered = True
            logger.warning(f"🚨 Stop-loss triggered: {self.symbol} @ {self.stop_price:.4f}")
            return True
        
        return False


@dataclass
class PositionLog:
    """Log posizione per debug."""
    timestamp: datetime
    symbol: str
    side: str
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    pnl_pct: float
    stop_loss: Optional[float] = None


class LiveStreamingManager:
    """
    Manager per streaming live multi-asset.
    
    Integra:
    - BinanceMultiWebSocket per dati real-time
    - PortfolioManager per gestione posizioni
    - Stop-loss monitoring in tempo reale
    - Logging posizioni e PnL
    """
    
    def __init__(
        self,
        symbols: List[str],
        initial_balance: float = 100000,
        interval: str = "1m",
        testnet: bool = False,
        default_stop_loss_pct: float = 0.02,
        enable_trailing_stop: bool = True,
        trailing_stop_pct: float = 0.015,
        log_interval_seconds: int = 60,
    ):
        """
        Inizializza il Live Streaming Manager.
        
        Args:
            symbols: Lista simboli da monitorare
            initial_balance: Bilancio iniziale
            interval: Intervallo candele WebSocket
            testnet: Usa testnet Binance
            default_stop_loss_pct: Stop-loss default (2%)
            enable_trailing_stop: Abilita trailing stop
            trailing_stop_pct: Percentuale trailing stop (1.5%)
            log_interval_seconds: Intervallo log posizioni
        """
        self.symbols = [s.upper().replace('/', '') for s in symbols]
        self.testnet = testnet
        self.default_stop_loss_pct = default_stop_loss_pct
        self.enable_trailing_stop = enable_trailing_stop
        self.trailing_stop_pct = trailing_stop_pct
        self.log_interval_seconds = log_interval_seconds
        
        # Componenti
        self.websocket = BinanceMultiWebSocket(
            symbols=self.symbols,
            interval=interval,
            testnet=testnet
        )
        self.portfolio = PortfolioManager(
            initial_balance=initial_balance,
            max_position_pct=0.3,
            max_leverage=1.0
        )
        
        # Stop-loss orders
        self.stop_loss_orders: Dict[str, StopLossOrder] = {}
        
        # Callbacks
        self._on_price_update: Optional[Callable] = None
        self._on_stop_loss_triggered: Optional[Callable] = None
        self._on_position_change: Optional[Callable] = None
        
        # Stato
        self.is_running = False
        self._update_thread: Optional[threading.Thread] = None
        self._log_thread: Optional[threading.Thread] = None
        
        # Position logs history
        self.position_logs: List[PositionLog] = []
        
        logger.info(f"📊 LiveStreamingManager initialized")
        logger.info(f"   Symbols: {self.symbols}")
        logger.info(f"   Balance: {initial_balance:,.2f} USDT")
        logger.info(f"   Stop-loss: {default_stop_loss_pct:.2%}")
        logger.info(f"   Trailing stop: {enable_trailing_stop} ({trailing_stop_pct:.2%})")
    
    def set_callbacks(
        self,
        on_price_update: Optional[Callable[[str, float], None]] = None,
        on_stop_loss_triggered: Optional[Callable[[StopLossOrder, float], None]] = None,
        on_position_change: Optional[Callable[[Position], None]] = None,
    ):
        """Imposta callback per eventi."""
        self._on_price_update = on_price_update
        self._on_stop_loss_triggered = on_stop_loss_triggered
        self._on_position_change = on_position_change
    
    def start(self):
        """Avvia lo streaming live."""
        if self.is_running:
            logger.warning("Live streaming already running")
            return
        
        logger.info("🚀 Starting live streaming...")
        self.is_running = True
        
        # Avvia WebSocket
        self.websocket.start()
        
        # Avvia thread di update
        self._update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self._update_thread.start()
        
        # Avvia thread di logging
        self._log_thread = threading.Thread(target=self._log_loop, daemon=True)
        self._log_thread.start()
        
        logger.info("✅ Live streaming started")
    
    def stop(self):
        """Ferma lo streaming live."""
        logger.info("🛑 Stopping live streaming...")
        self.is_running = False
        
        # Ferma WebSocket
        self.websocket.stop()
        
        # Aspetta thread
        if self._update_thread:
            self._update_thread.join(timeout=5)
        if self._log_thread:
            self._log_thread.join(timeout=5)
        
        logger.info("✅ Live streaming stopped")
    
    def _update_loop(self):
        """Loop principale di update prezzi e stop-loss."""
        logger.info("Update loop started")
        
        while self.is_running:
            try:
                # Ottieni prezzi correnti
                prices = self.websocket.get_all_prices()
                
                if not prices:
                    time.sleep(0.1)
                    continue
                
                # Aggiorna prezzi nel portfolio
                self.portfolio.update_prices(prices)
                
                # Callback price update
                if self._on_price_update:
                    for symbol, price in prices.items():
                        try:
                            self._on_price_update(symbol, price)
                        except Exception as e:
                            logger.error(f"Error in price update callback: {e}")
                
                # Check stop-loss
                self._check_stop_losses(prices)
                
                time.sleep(0.1)  # 100ms update cycle
                
            except Exception as e:
                logger.error(f"Error in update loop: {e}")
                time.sleep(1)
        
        logger.info("Update loop stopped")
    
    def _check_stop_losses(self, prices: Dict[str, float]):
        """Verifica stop-loss per tutte le posizioni."""
        for symbol, stop_order in list(self.stop_loss_orders.items()):
            price = prices.get(symbol)
            if not price:
                continue
            
            if stop_order.check_trigger(price):
                # Stop-loss triggered
                if self._on_stop_loss_triggered:
                    try:
                        self._on_stop_loss_triggered(stop_order, price)
                    except Exception as e:
                        logger.error(f"Error in stop-loss callback: {e}")
                
                # Chiudi posizione
                self._execute_stop_loss(symbol, price)
    
    def _execute_stop_loss(self, symbol: str, price: float):
        """Esegui chiusura per stop-loss."""
        try:
            result = self.portfolio.close_position(symbol=symbol, price=price)
            logger.warning(
                f"🔴 Stop-loss executed: {symbol} @ {price:.4f} "
                f"PnL: {result['pnl']:.2f}"
            )
            
            # Rimuovi stop-loss
            if symbol in self.stop_loss_orders:
                del self.stop_loss_orders[symbol]
            
        except Exception as e:
            logger.error(f"Error executing stop-loss for {symbol}: {e}")
    
    def _log_loop(self):
        """Loop di logging posizioni."""
        logger.info("Log loop started")
        
        while self.is_running:
            try:
                self._log_positions()
                time.sleep(self.log_interval_seconds)
            except Exception as e:
                logger.error(f"Error in log loop: {e}")
                time.sleep(10)
        
        logger.info("Log loop stopped")
    
    def _log_positions(self):
        """Logga stato posizioni."""
        positions = self.portfolio.get_open_positions()
        
        if not positions:
            return
        
        prices = self.websocket.get_all_prices()
        
        log_lines = ["\n" + "=" * 60]
        log_lines.append(f"📊 POSITIONS LOG - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_lines.append("=" * 60)
        
        total_pnl = 0
        
        for pos in positions:
            price = prices.get(pos.symbol, pos.current_price)
            pnl_pct = (pos.unrealized_pnl / pos.notional_value * 100) if pos.notional_value > 0 else 0
            total_pnl += pos.unrealized_pnl
            
            stop_order = self.stop_loss_orders.get(pos.symbol)
            stop_price = stop_order.stop_price if stop_order else None
            
            # Crea log entry
            log_entry = PositionLog(
                timestamp=datetime.now(),
                symbol=pos.symbol,
                side=pos.side.value,
                quantity=pos.quantity,
                entry_price=pos.entry_price,
                current_price=price,
                unrealized_pnl=pos.unrealized_pnl,
                pnl_pct=pnl_pct,
                stop_loss=stop_price
            )
            self.position_logs.append(log_entry)
            
            # Format log line
            pnl_emoji = "🟢" if pos.unrealized_pnl >= 0 else "🔴"
            log_lines.append(
                f"{pnl_emoji} {pos.symbol}: {pos.side.value.upper()} {pos.quantity:.4f} "
                f"@ {pos.entry_price:.4f} → {price:.4f} "
                f"| PnL: {pos.unrealized_pnl:+.2f} ({pnl_pct:+.2f}%)"
            )
            if stop_price:
                log_lines.append(f"   Stop-loss: {stop_price:.4f}")
        
        # Portfolio summary
        metrics = self.portfolio.get_metrics()
        log_lines.append("-" * 60)
        log_lines.append(f"💰 Total Equity: {metrics.total_equity:,.2f} USDT")
        log_lines.append(f"📈 Unrealized PnL: {total_pnl:+,.2f} USDT")
        log_lines.append(f"📊 Realized PnL: {metrics.realized_pnl:+,.2f} USDT")
        log_lines.append(f"📉 Max Drawdown: {metrics.max_drawdown:.2%}")
        log_lines.append("=" * 60)
        
        logger.info("\n".join(log_lines))
    
    def open_position(
        self,
        symbol: str,
        side: str,
        quantity: Optional[float] = None,
        price: Optional[float] = None,
        stop_loss_pct: Optional[float] = None,
        use_trailing_stop: Optional[bool] = None,
    ) -> Position:
        """
        Apri una nuova posizione con stop-loss automatico.
        
        Args:
            symbol: Simbolo
            side: 'long' o 'short'
            quantity: Quantità (None = auto-calcolata)
            price: Prezzo ingresso (None = prezzo corrente)
            stop_loss_pct: Percentuale stop-loss
            use_trailing_stop: Usa trailing stop
            
        Returns:
            Position aperta
        """
        symbol = symbol.upper().replace('/', '')
        
        # Ottieni prezzo corrente se non specificato
        if price is None:
            price = self.websocket.get_price(symbol)
            if not price:
                raise ValueError(f"Cannot get price for {symbol}")
        
        # Calcola quantità se non specificata
        if quantity is None:
            stop_pct = stop_loss_pct or self.default_stop_loss_pct
            quantity = self.portfolio.calculate_position_size(
                symbol=symbol,
                entry_price=price,
                stop_loss_pct=stop_pct,
                confidence=1.0
            )
        
        # Apri posizione
        position = self.portfolio.open_position(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price
        )
        
        # Crea stop-loss
        sl_pct = stop_loss_pct or self.default_stop_loss_pct
        use_trailing = use_trailing_stop if use_trailing_stop is not None else self.enable_trailing_stop
        
        if side.upper() == 'LONG':
            stop_price = price * (1 - sl_pct)
        else:
            stop_price = price * (1 + sl_pct)
        
        stop_order = StopLossOrder(
            symbol=symbol,
            stop_price=stop_price,
            quantity=quantity,
            is_trailing=use_trailing,
            trail_pct=self.trailing_stop_pct if use_trailing else 0
        )
        self.stop_loss_orders[symbol] = stop_order
        
        logger.info(
            f"✅ Position opened: {symbol} {side.upper()} {quantity:.4f} @ {price:.4f} "
            f"Stop-loss: {stop_price:.4f} ({sl_pct:.2%})"
        )
        
        # Callback
        if self._on_position_change:
            try:
                self._on_position_change(position)
            except Exception as e:
                logger.error(f"Error in position change callback: {e}")
        
        return position
    
    def close_position(self, symbol: str, price: Optional[float] = None) -> Dict:
        """
        Chiudi una posizione.
        
        Args:
            symbol: Simbolo
            price: Prezzo uscita (None = prezzo corrente)
            
        Returns:
            Risultato chiusura
        """
        symbol = symbol.upper().replace('/', '')
        
        # Ottieni prezzo corrente
        if price is None:
            price = self.websocket.get_price(symbol)
            if not price:
                raise ValueError(f"Cannot get price for {symbol}")
        
        # Chiudi posizione
        result = self.portfolio.close_position(symbol=symbol, price=price)
        
        # Rimuovi stop-loss
        if symbol in self.stop_loss_orders:
            del self.stop_loss_orders[symbol]
        
        logger.info(
            f"✅ Position closed: {symbol} @ {price:.4f} "
            f"PnL: {result['pnl']:+.2f}"
        )
        
        return result
    
    def get_current_prices(self) -> Dict[str, float]:
        """Ottieni prezzi correnti."""
        return self.websocket.get_all_prices()
    
    def get_ohlcv_data(self, symbol: str) -> Any:
        """Ottieni dati OHLCV per un simbolo."""
        return self.websocket.get_data(symbol)
    
    def get_portfolio_state(self) -> Dict:
        """Ottieni stato portfolio."""
        return self.portfolio.to_dict()
    
    def get_open_positions(self) -> List[Position]:
        """Ottieni posizioni aperte."""
        return self.portfolio.get_open_positions()
    
    def get_position_logs(self, symbol: Optional[str] = None, limit: int = 100) -> List[PositionLog]:
        """
        Ottieni log posizioni.
        
        Args:
            symbol: Filtra per simbolo (None = tutti)
            limit: Numero massimo di log
            
        Returns:
            Lista di PositionLog
        """
        logs = self.position_logs
        if symbol:
            logs = [l for l in logs if l.symbol == symbol.upper()]
        return logs[-limit:]
    
    def update_stop_loss(self, symbol: str, new_stop_price: float):
        """Aggiorna stop-loss per una posizione."""
        symbol = symbol.upper().replace('/', '')
        
        if symbol not in self.stop_loss_orders:
            logger.warning(f"No stop-loss order for {symbol}")
            return
        
        self.stop_loss_orders[symbol].stop_price = new_stop_price
        logger.info(f"Stop-loss updated: {symbol} @ {new_stop_price:.4f}")
    
    def is_ready(self) -> bool:
        """Verifica se lo streaming è pronto."""
        return self.websocket.is_ready(min_candles=10)


# Convenience function
def create_live_manager(
    symbols: List[str],
    initial_balance: float = 100000,
    testnet: bool = False,
) -> LiveStreamingManager:
    """
    Crea un LiveStreamingManager configurato.
    
    Args:
        symbols: Lista simboli
        initial_balance: Bilancio iniziale
        testnet: Usa testnet
        
    Returns:
        LiveStreamingManager configurato
    """
    return LiveStreamingManager(
        symbols=symbols,
        initial_balance=initial_balance,
        testnet=testnet,
        default_stop_loss_pct=0.02,
        enable_trailing_stop=True,
        trailing_stop_pct=0.015,
    )
