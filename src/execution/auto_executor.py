"""
Auto Executor Module
====================
Modulo per l'esecuzione automatica degli ordini generati dal DecisionEngine.
Include funzionalità di sicurezza: rate limiting, quantità minima, stop-loss.

Collega il modulo decisionale agli exchange reali (Binance, Coinbase, Bybit, ecc.).
"""

import time
import logging
from typing import List, Dict, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import threading
from collections import deque

# Import existing ExchangeClient
try:
    from src.execution import ExchangeClient
except ImportError:
    ExchangeClient = None

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """Stato possibile di un ordine."""
    PENDING = "pending"
    EXECUTED = "executed"
    SKIPPED = "skipped"
    FAILED = "failed"
    CANCELLED = "cancelled"


class OrderType(Enum):
    """Tipo di ordine."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


@dataclass
class ExecutedOrder:
    """Rappresenta un ordine eseguito."""
    order_id: str
    asset: str
    action: str  # BUY, SELL
    amount: float
    price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    exchange_response: Optional[Dict] = None
    error_message: Optional[str] = None
    confidence: float = 0.0
    monte_carlo: Optional[Dict] = None


@dataclass
class SafetyConfig:
    """Configurazione delle regole di sicurezza."""
    min_order_value: float = 10.0  # Valore minimo ordine in USDT
    max_order_value: float = 10000.0  # Valore massimo ordine in USDT
    max_orders_per_minute: int = 10  # Rate limit
    max_orders_per_hour: int = 100  # Rate limit orario
    max_daily_loss: float = 0.05  # 5% perdita massima giornaliera
    enable_stop_loss: bool = True
    default_stop_loss_pct: float = 0.02  # 2% stop-loss default
    enable_take_profit: bool = True
    default_take_profit_pct: float = 0.05  # 5% take-profit default
    dry_run: bool = True  # Se True, simula solo senza inviare davvero


class RateLimiter:
    """
    Gestisce il rate limiting per evitare di superare i limiti dell'exchange.
    """
    
    def __init__(self, max_per_minute: int = 10, max_per_hour: int = 100):
        self.max_per_minute = max_per_minute
        self.max_per_hour = max_per_hour
        self._minute_orders = deque()
        self._hour_orders = deque()
        self._lock = threading.Lock()
    
    def can_execute(self) -> bool:
        """Verifica se è possibile eseguire un nuovo ordine."""
        with self._lock:
            now = datetime.now()
            
            # Pulisci ordini vecchi
            minute_ago = now - timedelta(minutes=1)
            hour_ago = now - timedelta(hours=1)
            
            while self._minute_orders and self._minute_orders[0] < minute_ago:
                self._minute_orders.popleft()
            
            while self._hour_orders and self._hour_orders[0] < hour_ago:
                self._hour_orders.popleft()
            
            # Verifica limiti
            if len(self._minute_orders) >= self.max_per_minute:
                logger.warning("Rate limit: too many orders per minute")
                return False
            
            if len(self._hour_orders) >= self.max_per_hour:
                logger.warning("Rate limit: too many orders per hour")
                return False
            
            return True
    
    def record_order(self):
        """Registra un nuovo ordine eseguito."""
        with self._lock:
            now = datetime.now()
            self._minute_orders.append(now)
            self._hour_orders.append(now)


class SimulatedExchangeClient:
    """
    Client simulato per test e sviluppo.
    Simula le risposte dell'exchange senza inviare ordini reali.
    """
    
    def __init__(self, latency_ms: float = 100):
        self.latency_ms = latency_ms
        self._order_counter = 0
    
    def send_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "MARKET",
        price: Optional[float] = None
    ) -> Dict[str, Any]:
        """Simula l'invio di un ordine."""
        # Simula latenza
        time.sleep(self.latency_ms / 1000)
        
        self._order_counter += 1
        order_id = f"SIM_{self._order_counter:06d}"
        
        # Simula prezzo (in produzione verrebbe dall'exchange)
        simulated_price = price if price else 100.0  # Placeholder
        
        return {
            "status": "success",
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "type": order_type,
            "price": simulated_price,
            "timestamp": datetime.now().isoformat(),
            "simulated": True
        }
    
    def get_balance(self, asset: str = "USDT") -> float:
        """Simula il recupero del saldo."""
        return 100000.0  # Saldo simulato
    
    def get_price(self, symbol: str) -> float:
        """Simula il recupero del prezzo."""
        return 100.0  # Prezzo simulato


class AutoExecutor:
    """
    Esecutore automatico di ordini.
    
    Riceve ordini dal DecisionEngine e li invia agli exchange
    rispettando regole di sicurezza e rate limiting.
    """
    
    def __init__(
        self,
        exchange_client: Optional[Any] = None,
        safety_config: Optional[SafetyConfig] = None,
        on_order_executed: Optional[Callable[[ExecutedOrder], None]] = None
    ):
        """
        Inizializza l'AutoExecutor.
        
        Args:
            exchange_client: Client per l'exchange (reale o simulato)
            safety_config: Configurazione sicurezza
            on_order_executed: Callback chiamato dopo ogni ordine
        """
        self.client = exchange_client or SimulatedExchangeClient()
        self.safety = safety_config or SafetyConfig()
        self.rate_limiter = RateLimiter(
            max_per_minute=self.safety.max_orders_per_minute,
            max_per_hour=self.safety.max_orders_per_hour
        )
        self.on_order_executed = on_order_executed
        
        # Statistiche
        self.stats = {
            "total_orders": 0,
            "executed_orders": 0,
            "skipped_orders": 0,
            "failed_orders": 0,
            "total_volume": 0.0,
            "daily_pnl": 0.0
        }
        
        # Storico ordini
        self.order_history: List[ExecutedOrder] = []
        
        # Posizioni aperte con stop-loss
        self.open_positions: Dict[str, Dict] = {}
    
    def _validate_order(self, order: Dict) -> tuple[bool, str]:
        """
        Valida un ordine prima dell'esecuzione.
        
        Args:
            order: Dizionario con i dati dell'ordine
            
        Returns:
            Tupla (valido, motivo)
        """
        # Verifica action
        action = order.get("action", "").upper()
        if action not in ["BUY", "SELL"]:
            return False, f"Invalid action: {action}"
        
        # Verifica amount
        amount = order.get("amount", 0)
        if amount <= 0:
            return False, f"Invalid amount: {amount}"
        
        # Verifica valore minimo
        if amount < self.safety.min_order_value:
            return False, f"Amount {amount} below minimum {self.safety.min_order_value}"
        
        # Verifica valore massimo
        if amount > self.safety.max_order_value:
            return False, f"Amount {amount} above maximum {self.safety.max_order_value}"
        
        # Verifica rate limit
        if not self.rate_limiter.can_execute():
            return False, "Rate limit exceeded"
        
        # Verifica perdita giornaliera
        if self.stats["daily_pnl"] < -self.safety.max_daily_loss:
            return False, "Daily loss limit exceeded"
        
        return True, "OK"
    
    def _generate_order_id(self) -> str:
        """Genera un ID univoco per l'ordine."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"ORD_{timestamp}_{self.stats['total_orders']:06d}"
    
    def execute_order(self, order: Dict) -> ExecutedOrder:
        """
        Esegue un singolo ordine.
        
        Args:
            order: Dizionario con i dati dell'ordine dal DecisionEngine
            
        Returns:
            ExecutedOrder con il risultato
        """
        self.stats["total_orders"] += 1
        
        # Crea oggetto ordine
        executed = ExecutedOrder(
            order_id=self._generate_order_id(),
            asset=order.get("asset", "Unknown"),
            action=order.get("action", "HOLD"),
            amount=order.get("amount", 0),
            confidence=order.get("confidence", 0),
            monte_carlo=order.get("monte_carlo")
        )
        
        # Valida ordine
        valid, reason = self._validate_order(order)
        if not valid:
            executed.status = OrderStatus.SKIPPED
            executed.error_message = reason
            self.stats["skipped_orders"] += 1
            logger.warning(f"Order skipped: {reason}")
            self.order_history.append(executed)
            return executed
        
        # Esegui ordine
        try:
            if self.safety.dry_run:
                # Modalità simulazione
                result = self.client.send_order(
                    symbol=order["asset"],
                    side=order["action"],
                    quantity=order["amount"]
                )
            else:
                # Modalità reale - usa ExchangeClient
                if hasattr(self.client, 'place_market_order_quote'):
                    result = self.client.place_market_order_quote(
                        symbol=order["asset"],
                        side=order["action"],
                        quote_orderQty=order["amount"]
                    )
                else:
                    result = self.client.send_order(
                        symbol=order["asset"],
                        side=order["action"],
                        quantity=order["amount"]
                    )
            
            if result and result.get("status") == "success":
                executed.status = OrderStatus.EXECUTED
                executed.exchange_response = result
                executed.price = result.get("price")
                self.stats["executed_orders"] += 1
                self.stats["total_volume"] += order["amount"]
                self.rate_limiter.record_order()
                
                # Registra posizione per stop-loss
                self._register_position(executed, order)
                
                logger.info(
                    f"✅ Order executed: {executed.action} {executed.amount:.2f} USDT of {executed.asset}"
                )
            else:
                executed.status = OrderStatus.FAILED
                executed.error_message = result.get("error", "Unknown error") if result else "No response"
                self.stats["failed_orders"] += 1
                logger.error(f"❌ Order failed: {executed.error_message}")
                
        except Exception as e:
            executed.status = OrderStatus.FAILED
            executed.error_message = str(e)
            self.stats["failed_orders"] += 1
            logger.exception(f"❌ Order exception: {e}")
        
        # Registra in storico
        self.order_history.append(executed)
        
        # Callback
        if self.on_order_executed:
            try:
                self.on_order_executed(executed)
            except Exception as e:
                logger.error(f"Callback error: {e}")
        
        return executed
    
    def execute_orders(self, orders: List[Dict]) -> List[ExecutedOrder]:
        """
        Esegue una lista di ordini dal DecisionEngine.
        
        Args:
            orders: Lista di ordini dal DecisionEngine
            
        Returns:
            Lista di ExecutedOrder con i risultati
        """
        executed_orders = []
        
        logger.info(f"Executing {len(orders)} orders...")
        
        for order in orders:
            executed = self.execute_order(order)
            executed_orders.append(executed)
            
            # Pausa tra ordini per evitare rate limiting
            if not self.safety.dry_run:
                time.sleep(0.1)
        
        # Riepilogo
        executed_count = sum(1 for o in executed_orders if o.status == OrderStatus.EXECUTED)
        logger.info(
            f"Execution complete: {executed_count}/{len(orders)} orders executed"
        )
        
        return executed_orders
    
    def _register_position(self, executed: ExecutedOrder, order: Dict):
        """
        Registra una posizione aperta con stop-loss.
        
        Args:
            executed: Ordine eseguito
            order: Ordine originale con dati Monte Carlo
        """
        asset = executed.asset
        
        if executed.action == "BUY":
            # Calcola stop-loss
            stop_loss_pct = self.safety.default_stop_loss_pct
            if order.get("monte_carlo"):
                # Usa VaR per determinare stop-loss più preciso
                var = order["monte_carlo"].get("var", -stop_loss_pct)
                stop_loss_pct = min(abs(var) * 0.8, stop_loss_pct * 2)  # Conservative
            
            self.open_positions[asset] = {
                "entry_price": executed.price,
                "amount": executed.amount,
                "stop_loss_pct": stop_loss_pct,
                "stop_loss_price": executed.price * (1 - stop_loss_pct) if executed.price else None,
                "take_profit_pct": self.safety.default_take_profit_pct,
                "timestamp": executed.timestamp,
                "order_id": executed.order_id
            }
            
            logger.info(
                f"Position registered: {asset} with stop-loss at {stop_loss_pct:.2%}"
            )
        
        elif executed.action == "SELL" and asset in self.open_positions:
            # Chiudi posizione
            del self.open_positions[asset]
            logger.info(f"Position closed: {asset}")
    
    def check_stop_losses(self, current_prices: Dict[str, float]) -> List[Dict]:
        """
        Verifica se qualche posizione ha raggiunto lo stop-loss.
        
        Args:
            current_prices: Dizionario {symbol: price}
            
        Returns:
            Lista di ordini di vendita da eseguire
        """
        stop_orders = []
        
        for asset, position in self.open_positions.items():
            current_price = current_prices.get(asset)
            if not current_price:
                continue
            
            stop_loss_price = position.get("stop_loss_price")
            if stop_loss_price and current_price <= stop_loss_price:
                # Trigger stop-loss
                logger.warning(
                    f"🛑 Stop-loss triggered for {asset} at {current_price} "
                    f"(stop: {stop_loss_price})"
                )
                stop_orders.append({
                    "asset": asset,
                    "action": "SELL",
                    "amount": position["amount"],
                    "reason": "stop_loss",
                    "entry_price": position["entry_price"]
                })
        
        return stop_orders
    
    def get_summary(self) -> Dict:
        """
        Ritorna un riepilogo dell'esecuzione.
        
        Returns:
            Dizionario con statistiche
        """
        return {
            "stats": self.stats.copy(),
            "open_positions": len(self.open_positions),
            "order_history_count": len(self.order_history),
            "safety_config": {
                "min_order_value": self.safety.min_order_value,
                "max_order_value": self.safety.max_order_value,
                "dry_run": self.safety.dry_run
            }
        }
    
    def reset_daily_stats(self):
        """Reset delle statistiche giornaliere."""
        self.stats["daily_pnl"] = 0.0
        logger.info("Daily stats reset")


# === Funzioni helper per integrazione ===

def create_executor_from_config(config: Dict) -> AutoExecutor:
    """
    Crea un AutoExecutor dalla configurazione.
    
    Args:
        config: Dizionario con configurazione
        
    Returns:
        AutoExecutor configurato
    """
    safety = SafetyConfig(
        min_order_value=config.get("min_order_value", 10.0),
        max_order_value=config.get("max_order_value", 10000.0),
        max_orders_per_minute=config.get("max_orders_per_minute", 10),
        dry_run=config.get("dry_run", True)
    )
    
    # Crea client exchange se configurato
    client = None
    if not safety.dry_run and ExchangeClient:
        client = ExchangeClient(
            api_key=config.get("api_key", ""),
            api_secret=config.get("api_secret", ""),
            testnet=config.get("testnet", False),
            exchange=config.get("exchange", "binance")
        )
    else:
        client = SimulatedExchangeClient()
    
    return AutoExecutor(exchange_client=client, safety_config=safety)


# === Esempio di utilizzo ===
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Configurazione sicurezza
    safety = SafetyConfig(
        min_order_value=10.0,
        max_order_value=5000.0,
        dry_run=True  # Modalità simulazione
    )
    
    # Crea executor
    executor = AutoExecutor(safety_config=safety)
    
    # Ordini di esempio (dal DecisionEngine)
    orders = [
        {
            "asset": "BTCUSDT",
            "action": "BUY",
            "amount": 500.0,
            "confidence": 0.75,
            "monte_carlo": {
                "var": -0.03,
                "prob_profit": 0.65
            }
        },
        {
            "asset": "ETHUSDT",
            "action": "BUY",
            "amount": 300.0,
            "confidence": 0.60,
            "monte_carlo": {
                "var": -0.04,
                "prob_profit": 0.55
            }
        },
        {
            "asset": "XRPUSDT",
            "action": "SELL",
            "amount": 0,  # Verrà saltato
            "confidence": 0.0
        }
    ]
    
    # Esegui ordini
    print("\n" + "=" * 60)
    print("AUTO EXECUTOR - Esecuzione Ordini")
    print("=" * 60)
    
    executed = executor.execute_orders(orders)
    
    # Stampa risultati
    print("\nOrdini eseguiti:")
    for order in executed:
        status_icon = "✅" if order.status == OrderStatus.EXECUTED else "⏭️" if order.status == OrderStatus.SKIPPED else "❌"
        print(f"  {status_icon} {order.action:4} {order.amount:>8.2f} USDT of {order.asset:10} | {order.status.value}")
        if order.error_message:
            print(f"      Error: {order.error_message}")
    
    # Riepilogo
    print("\n" + "=" * 60)
    print("RIEPILOGO")
    print("=" * 60)
    summary = executor.get_summary()
    print(f"  Totale ordini: {summary['stats']['total_orders']}")
    print(f"  Eseguiti: {summary['stats']['executed_orders']}")
    print(f"  Saltati: {summary['stats']['skipped_orders']}")
    print(f"  Falliti: {summary['stats']['failed_orders']}")
    print(f"  Volume totale: {summary['stats']['total_volume']:,.2f} USDT")
    print(f"  Posizioni aperte: {summary['open_positions']}")
    print("=" * 60)
