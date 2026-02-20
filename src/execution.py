"""
Execution Layer - Exchange Client
==================================
Ponte tra il motore di trading e i conti degli utenti (Binance/Bybit/OKX).

Modello: L'utente collega il suo conto tramite API key.
Il sistema può solo tradare, NON prelevare fondi.
"""

import time
import logging
from typing import Optional, Dict, Any, List

# Optional imports for exchange connectivity
try:
    from binance.client import Client
    from binance.enums import SIDE_BUY, SIDE_SELL, ORDER_TYPE_MARKET, ORDER_TYPE_LIMIT
    from binance.exceptions import BinanceAPIException
    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False
    # Define placeholders for when binance is not installed
    SIDE_BUY = "BUY"
    SIDE_SELL = "SELL"
    ORDER_TYPE_MARKET = "MARKET"
    ORDER_TYPE_LIMIT = "LIMIT"
    class BinanceAPIException(Exception):
        pass
    class Client:
        def __init__(self, *args, **kwargs):
            raise ImportError("binance package not installed. Install with: pip install python-binance")

logger = logging.getLogger(__name__)


class ExchangeClient:
    """
    Client per connettersi agli exchange tramite API.
    Supporta Binance (futures/spot), estendibile a Bybit/OKX.
    
    L'utente fornisce API Key e Secret - il sistema può solo tradare,
    non prelevare fondi.
    """
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False, exchange: str = "binance"):
        """
        Inizializza il client per l'exchange.
        
        Args:
            api_key: API Key dell'utente
            api_secret: API Secret dell'utente
            testnet: Usa testnet per i test
            exchange: Exchange da usare (binance, bybit, okx)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.exchange = exchange
        
        # Inizializza client per l'exchange selezionato
        self.client = None
        exchange_lower = exchange.lower()
        
        if exchange_lower == "binance":
            if testnet:
                self.client = Client(api_key, api_secret, testnet=True)
            else:
                self.client = Client(api_key, api_secret)
        elif exchange_lower == "bybit":
            try:
                from pybit.unified_trading import HTTP
                self.client = HTTP(
                    testnet=testnet,
                    api_key=api_key,
                    api_secret=api_secret
                )
            except ImportError:
                logger.warning("pybit not installed. Install with: pip install pybit")
                # Fallback: use ccxt
                import ccxt
                self.client = ccxt.bybit({
                    'apiKey': api_key,
                    'secret': api_secret,
                    'sandbox': testnet,
                })
        elif exchange_lower == "okx":
            try:
                import ccxt
                self.client = ccxt.okx({
                    'apiKey': api_key,
                    'secret': api_secret,
                    'sandbox': testnet,
                })
            except ImportError:
                logger.warning("ccxt not installed. Install with: pip install ccxt")
                raise
        else:
            # Generic ccxt fallback for any supported exchange
            try:
                import ccxt
                exchange_class = getattr(ccxt, exchange_lower, None)
                if exchange_class is None:
                    raise ValueError(f"Exchange {exchange} not supported by ccxt")
                self.client = exchange_class({
                    'apiKey': api_key,
                    'secret': api_secret,
                    'sandbox': testnet,
                })
            except ImportError:
                raise ImportError(
                    f"Exchange {exchange} requires ccxt. Install with: pip install ccxt"
                )
    
    # ======================
    # INFO BASE
    # ======================
    
    def get_balance(self, asset: str = "USDT") -> float:
        """
        Ritorna il saldo disponibile per un asset.
        
        Args:
            asset: Simbolo dell'asset (default: USDT)
            
        Returns:
            Saldo disponibile
        """
        try:
            account = self.client.get_account()
            for bal in account.get("balances", []):
                if bal["asset"] == asset:
                    return float(bal["free"])
            return 0.0
        except BinanceAPIException as e:
            logger.error(f"Errore nel recupero bilancio: {e}")
            return 0.0
    
    def get_all_balances(self) -> Dict[str, float]:
        """
        Ritorna tutti i saldi disponibili.
        
        Returns:
            Dizionario {asset: balance}
        """
        try:
            account = self.client.get_account()
            balances = {}
            for bal in account.get("balances", []):
                free = float(bal["free"])
                locked = float(bal["locked"])
                if free > 0 or locked > 0:
                    balances[bal["asset"]] = {"free": free, "locked": locked}
            return balances
        except BinanceAPIException as e:
            logger.error(f"Errore nel recupero bilanci: {e}")
            return {}
    
    def get_price(self, symbol: str) -> float:
        """
        Ritorna il prezzo corrente di un simbolo.
        
        Args:
            symbol: Simbolo (es. BTCUSDT)
            
        Returns:
            Prezzo corrente
        """
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker["price"])
        except BinanceAPIException as e:
            logger.error(f"Errore nel recupero prezzo: {e}")
            return 0.0
    
    def get_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Ritorna i prezzi per più simboli.
        
        Args:
            symbols: Lista di simboli
            
        Returns:
            Dizionario {symbol: price}
        """
        try:
            tickers = self.client.get_all_tickers()
            prices = {}
            for t in tickers:
                if t["symbol"] in symbols:
                    prices[t["symbol"]] = float(t["price"])
            return prices
        except BinanceAPIException as e:
            logger.error(f"Errore nel recupero prezzi: {e}")
            return {}
    
    # ======================
    # ORDINI MARKET
    # ======================
    
    def place_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Place un ordine market.
        
        Args:
            symbol: Simbolo (es. BTCUSDT)
            side: 'BUY' o 'SELL'
            quantity: Quantità
            
        Returns:
            Dizionario con info ordine o None se errore
        """
        try:
            order = self.client.create_order(
                symbol=symbol,
                side=SIDE_BUY if side == "BUY" else SIDE_SELL,
                type=ORDER_TYPE_MARKET,
                quantity=quantity
            )
            logger.info(f"📝 Ordine Market eseguito: {side} {quantity} {symbol}")
            return order
        except BinanceAPIException as e:
            logger.error(f"❌ Errore placing order: {e}")
            return None
    
    def place_market_order_quote(
        self,
        symbol: str,
        side: str,
        quote_orderQty: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Place un ordine market specificando il valore in quote asset.
        Es: compra per 100 USDT di BTC.
        
        Args:
            symbol: Simbolo (es. BTCUSDT)
            side: 'BUY' o 'SELL'
            quote_orderQty: Valore in asset quote (es. USDT)
            
        Returns:
            Dizionario con info ordine o None se errore
        """
        try:
            order = self.client.create_order(
                symbol=symbol,
                side=SIDE_BUY if side == "BUY" else SIDE_SELL,
                type=ORDER_TYPE_MARKET,
                quoteOrderQty=quote_orderQty
            )
            logger.info(f"📝 Ordine Market (quote) eseguito: {side} {quote_orderQty} {symbol}")
            return order
        except BinanceAPIException as e:
            logger.error(f"❌ Errore placing order: {e}")
            return None
    
    # ======================
    # ORDINI LIMIT
    # ======================
    
    def place_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        timeInForce: str = "GTC",
    ) -> Optional[Dict[str, Any]]:
        """
        Place un ordine limit.
        
        Args:
            symbol: Simbolo
            side: 'BUY' o 'SELL'
            quantity: Quantità
            price: Prezzo limite
            timeInForce: GTC, IOC, FOK
            
        Returns:
            Dizionario con info ordine o None se errore
        """
        try:
            order = self.client.create_order(
                symbol=symbol,
                side=SIDE_BUY if side == "BUY" else SIDE_SELL,
                type=ORDER_TYPE_LIMIT,
                quantity=quantity,
                price=price,
                timeInForce=timeInForce
            )
            logger.info(f"📝 Ordine Limit impostato: {side} {quantity} {symbol} @ {price}")
            return order
        except BinanceAPIException as e:
            logger.error(f"❌ Errore placing limit order: {e}")
            return None
    
    # ======================
    # GESTIONE POSIZIONI (FUTURES)
    # ======================
    
    def get_position(self, symbol: str) -> float:
        """
        Ritorna la size della posizione (per futures).
        Per spot, usa get_asset_balance.
        
        Args:
            symbol: Simbolo
            
        Returns:
            Size della posizione (positiva = long, negativa = short)
        """
        try:
            positions = self.client.futures_account()["positions"]
            for pos in positions:
                if pos["symbol"] == symbol:
                    return float(pos["positionAmt"])
            return 0.0
        except BinanceAPIException as e:
            logger.error(f"Errore nel recupero posizione: {e}")
            return 0.0
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Ritorna tutte le posizioni aperte (futures).
        
        Returns:
            Lista di posizioni
        """
        try:
            positions = self.client.futures_account()["positions"]
            open_positions = []
            for pos in positions:
                if float(pos["positionAmt"]) != 0:
                    open_positions.append({
                        "symbol": pos["symbol"],
                        "size": float(pos["positionAmt"]),
                        "entry_price": float(pos["entryPrice"]),
                        "unrealized_pnl": float(pos["unrealizedProfit"]),
                        "leverage": int(pos["leverage"]),
                    })
            return open_positions
        except BinanceAPIException as e:
            logger.error(f"Errore nel recupero posizioni: {e}")
            return []
    
    def close_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Chiude la posizione aprendo l'ordine opposto.
        
        Args:
            symbol: Simbolo
            
        Returns:
            Ordine di chiusura o None
        """
        position_size = self.get_position(symbol)
        if position_size == 0:
            logger.info(f"Nessuna posizione aperta su {symbol}")
            return None
        
        side = "SELL" if position_size > 0 else "BUY"
        quantity = abs(position_size)
        
        logger.info(f"🔄 Chiudo posizione {symbol}: {side} {quantity}")
        return self.place_market_order(symbol, side, quantity)
    
    def close_all_positions(self) -> List[Dict[str, Any]]:
        """
        Chiude tutte le posizioni aperte.
        
        Returns:
            Lista degli ordini eseguiti
        """
        positions = self.get_positions()
        closed = []
        
        for pos in positions:
            symbol = pos["symbol"]
            result = self.close_position(symbol)
            if result:
                closed.append(result)
            time.sleep(0.1)  # Rate limit
        
        return closed
    
    # ======================
    # ORDINI STOP-LOSS / TAKE-PROFIT
    # ======================
    
    def place_stop_loss(
        self,
        symbol: str,
        side: str,
        quantity: float,
        stop_price: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Place un ordine stop-loss.
        
        Args:
            symbol: Simbolo
            side: 'SELL' per chiudere long, 'BUY' per chiudere short
            quantity: Quantità
            stop_price: Prezzo di stop
            
        Returns:
            Ordine creato o None
        """
        try:
            order = self.client.create_order(
                symbol=symbol,
                side=SIDE_BUY if side == "BUY" else SIDE_SELL,
                type="STOP_LOSS",
                quantity=quantity,
                stopPrice=stop_price
            )
            logger.info(f"🛡️ Stop-Loss impostato: {side} {quantity} {symbol} @ {stop_price}")
            return order
        except BinanceAPIException as e:
            logger.error(f"❌ Errore stop-loss: {e}")
            return None
    
    def place_take_profit(
        self,
        symbol: str,
        side: str,
        quantity: float,
        stop_price: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Place un ordine take-profit.
        
        Args:
            symbol: Simbolo
            side: 'SELL' per chiudere long, 'BUY' per chiudere short
            quantity: Quantità
            stop_price: Prezzo di take-profit
            
        Returns:
            Ordine creato o None
        """
        try:
            order = self.client.create_order(
                symbol=symbol,
                side=SIDE_BUY if side == "BUY" else SIDE_SELL,
                type="TAKE_PROFIT",
                quantity=quantity,
                stopPrice=stop_price
            )
            logger.info(f"💰 Take-Profit impostato: {side} {quantity} {symbol} @ {stop_price}")
            return order
        except BinanceAPIException as e:
            logger.error(f"❌ Errore take-profit: {e}")
            return None
    
    # ======================
    # STORICO ORDINI
    # ======================
    
    def get_order_history(self, symbol: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Recupera lo storico ordini.
        
        Args:
            symbol: Filtra per simbolo (opzionale)
            limit: Numero massimo di ordini
            
        Returns:
            Lista ordini
        """
        try:
            if symbol:
                return self.client.get_all_orders(symbol=symbol, limit=limit)
            else:
                return self.client.get_all_orders(limit=limit)
        except BinanceAPIException as e:
            logger.error(f"Errore recupero storico: {e}")
            return []
    
    def get_trades(self, symbol: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Recupera i trade eseguiti.
        
        Args:
            symbol: Simbolo
            limit: Numero massimo di trade
            
        Returns:
            Lista trade
        """
        try:
            return self.client.get_my_trades(symbol=symbol, limit=limit)
        except BinanceAPIException as e:
            logger.error(f"Errore recupero trades: {e}")
            return []
    
    # ======================
    # VALIDAZIONE API KEY
    # ======================
    
    def validate_api_key(self) -> bool:
        """
        Valida che l'API key sia corretta e abbia i permessi necessari.
        
        Returns:
            True se valida, False altrimenti
        """
        try:
            self.client.get_account()
            return True
        except BinanceAPIException as e:
            logger.error(f"API Key non valida: {e}")
            return False
    
    def check_trading_permission(self) -> bool:
        """
        Verifica che l'API key abbia i permessi di trading.
        
        Returns:
            True se può tradare, False altrimenti
        """
        try:
            account = self.client.get_account()
            # Check se canTrade è True
            return account.get("canTrade", False)
        except BinanceAPIException as e:
            logger.error(f"Errore verifica permessi: {e}")
            return False


class LiveTradingEngine:
    """
    Motore di trading live che integra segnali + esecuzione.
    """
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        self.client = ExchangeClient(api_key, api_secret, testnet=testnet)
        self.positions = {}
    
    def execute_signal(
        self,
        symbol: str,
        signal: int,
        risk_capital: float = 100,
        close_on_flat: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Esegue un segnale di trading.
        
        Args:
            symbol: Simbolo (es. BTCUSDT)
            signal: 1 = BUY, -1 = SELL, 0 = FLAT
            risk_capital: Capitale a rischio per trade
            close_on_flat: Chiudi posizione quando signal = 0
            
        Returns:
            Ordine eseguito o None
        """
        price = self.client.get_price(symbol)
        
        if signal == 1:
            # LONG
            quantity = risk_capital / price
            return self.client.place_market_order(symbol, "BUY", quantity)
        
        elif signal == -1:
            # SHORT
            quantity = risk_capital / price
            return self.client.place_market_order(symbol, "SELL", quantity)
        
        elif signal == 0 and close_on_flat:
            # FLAT - chiudi posizione
            return self.client.close_position(symbol)
        
        return None
    
    def get_equity(self, quote_asset: str = "USDT") -> float:
        """
        Ritorna l'equity totale del conto.
        
        Args:
            quote_asset: Asset quotazione
            
        Returns:
            Equity totale
        """
        balance = self.client.get_balance(quote_asset)
        
        # Aggiungi PnL non realizzato delle posizioni
        positions = self.client.get_positions()
        unrealized_pnl = sum(p["unrealized_pnl"] for p in positions)
        
        return balance + unrealized_pnl


# ======================
# ESEMPIO DI UTILIZZO
# ======================

if __name__ == "__main__":
    # Test con API key fittizie (testnet)
    API_KEY = "test_api_key"
    API_SECRET = "test_api_secret"
    
    client = ExchangeClient(API_KEY, API_SECRET, testnet=True)
    
    # Test connessione
    if client.validate_api_key():
        print("✅ Connessione API valida")
        
        # Get balance
        balance = client.get_balance("USDT")
        print(f"💰 Balance USDT: {balance}")
        
        # Get price
        price = client.get_price("BTCUSDT")
        print(f"📊 BTCUSDT: ${price}")
    else:
        print("❌ API Key non valida")


# ======================
# ROUTER PER MULTI-ASSET
# ======================

class BaseRouter:
    """
    Classe base per i router degli exchange.
    Fornisce un'interfaccia comune per l'esecuzione degli ordini.
    """
    
    def __init__(self, portfolio, testnet: bool = True):
        """
        Inizializza il router.
        
        Args:
            portfolio: Istanza del portafoglio
            testnet: Usa testnet per i test
        """
        self.portfolio = portfolio
        self.testnet = testnet
        self.client = None
    
    def place_order(self, asset: str, side: str, quantity: float) -> Optional[Dict[str, Any]]:
        """
        Esegue un ordine. Da implementare nelle sottoclassi.
        
        Args:
            asset: Simbolo dell'asset (es. BTCUSDT)
            side: "BUY" o "SELL"
            quantity: Quantità da tradare
            
        Returns:
            Risultato dell'ordine o None
        """
        raise NotImplementedError("Subclasses must implement place_order")
    
    def get_symbol(self, asset: str) -> str:
        """
        Converte il simbolo dell'asset nel formato dell'exchange.
        
        Args:
            asset: Simbolo dell'asset (es. BTC, ETH)
            
        Returns:
            Simbolo nel formato dell'exchange (es. BTCUSDT)
        """
        # Se già nel formato corretto, ritorna così
        if "USDT" in asset:
            return asset
        return f"{asset}USDT"


class BinanceRouter(BaseRouter):
    """
    Router per l'exchange Binance.
    """
    
    def __init__(self, portfolio, testnet: bool = True, api_key: str = None, api_secret: str = None):
        """
        Inizializza il router Binance.
        
        Args:
            portfolio: Istanza del portafoglio
            testnet: Usa testnet per i test
            api_key: API key (opzionale, usa env vars se non fornita)
            api_secret: API secret (opzionale, usa env vars se non fornita)
        """
        super().__init__(portfolio, testnet)
        
        # Usa API key da environment se non fornite
        import os
        self.api_key = api_key or os.getenv("BINANCE_API_KEY", "")
        self.api_secret = api_secret or os.getenv("BINANCE_API_SECRET", "")
        
        if self.api_key and self.api_secret:
            self.client = Client(self.api_key, self.api_secret, testnet=testnet)
        else:
            logger.warning("Binance API keys not configured. Router will operate in simulation mode.")
            self.client = None
    
    def place_order(self, asset: str, side: str, quantity: float) -> Optional[Dict[str, Any]]:
        """
        Esegue un ordine su Binance.
        
        Args:
            asset: Simbolo dell'asset
            side: "BUY" o "SELL"
            quantity: Quantità da tradare
            
        Returns:
            Risultato dell'ordine o None
        """
        symbol = self.get_symbol(asset)
        
        if self.client is None:
            # Modalità simulazione
            logger.info(f"[SIMULATION] Binance: {side} {quantity} {symbol}")
            return {"status": "SIMULATED", "symbol": symbol, "side": side, "quantity": quantity}
        
        try:
            order = self.client.create_order(
                symbol=symbol,
                side=SIDE_BUY if side == "BUY" else SIDE_SELL,
                type=ORDER_TYPE_MARKET,
                quoteOrderQty=quantity if side == "BUY" else None,
                quantity=quantity if side == "SELL" else None
            )
            logger.info(f"✅ Binance order executed: {side} {quantity} {symbol}")
            return order
        except BinanceAPIException as e:
            logger.error(f"❌ Binance order failed: {e}")
            return None


class BybitRouter(BaseRouter):
    """
    Router per l'exchange Bybit.
    """
    
    def __init__(self, portfolio, testnet: bool = True, api_key: str = None, api_secret: str = None):
        """
        Inizializza il router Bybit.
        
        Args:
            portfolio: Istanza del portafoglio
            testnet: Usa testnet per i test
            api_key: API key (opzionale)
            api_secret: API secret (opzionale)
        """
        super().__init__(portfolio, testnet)
        
        import os
        self.api_key = api_key or os.getenv("BYBIT_API_KEY", "")
        self.api_secret = api_secret or os.getenv("BYBIT_API_SECRET", "")
        
        if self.api_key and self.api_secret:
            try:
                from pybit.unified_trading import HTTP
                self.client = HTTP(
                    testnet=testnet,
                    api_key=self.api_key,
                    api_secret=self.api_secret
                )
            except ImportError:
                logger.warning("pybit not installed. Using ccxt fallback.")
                try:
                    import ccxt
                    self.client = ccxt.bybit({
                        'apiKey': self.api_key,
                        'secret': self.api_secret,
                        'sandbox': testnet,
                    })
                except ImportError:
                    logger.error("ccxt not installed. Router will operate in simulation mode.")
                    self.client = None
        else:
            logger.warning("Bybit API keys not configured. Router will operate in simulation mode.")
            self.client = None
    
    def place_order(self, asset: str, side: str, quantity: float) -> Optional[Dict[str, Any]]:
        """
        Esegue un ordine su Bybit.
        
        Args:
            asset: Simbolo dell'asset
            side: "BUY" o "SELL"
            quantity: Quantità da tradare
            
        Returns:
            Risultato dell'ordine o None
        """
        symbol = self.get_symbol(asset)
        
        if self.client is None:
            # Modalità simulazione
            logger.info(f"[SIMULATION] Bybit: {side} {quantity} {symbol}")
            return {"status": "SIMULATED", "symbol": symbol, "side": side, "quantity": quantity}
        
        try:
            # Pybit format
            if hasattr(self.client, 'place_order'):
                order = self.client.place_order(
                    category="spot",
                    symbol=symbol,
                    side=side.capitalize(),
                    orderType="Market",
                    qty=quantity
                )
            else:
                # CCXT format
                order = self.client.create_market_order(
                    symbol=symbol,
                    side=side.lower(),
                    amount=quantity
                )
            logger.info(f"✅ Bybit order executed: {side} {quantity} {symbol}")
            return order
        except Exception as e:
            logger.error(f"❌ Bybit order failed: {e}")
            return None


class OKXRouter(BaseRouter):
    """
    Router per l'exchange OKX.
    """
    
    def __init__(self, portfolio, testnet: bool = True, api_key: str = None, api_secret: str = None, passphrase: str = None):
        """
        Inizializza il router OKX.
        
        Args:
            portfolio: Istanza del portafoglio
            testnet: Usa testnet per i test
            api_key: API key (opzionale)
            api_secret: API secret (opzionale)
            passphrase: Passphrase API (opzionale)
        """
        super().__init__(portfolio, testnet)
        
        import os
        self.api_key = api_key or os.getenv("OKX_API_KEY", "")
        self.api_secret = api_secret or os.getenv("OKX_API_SECRET", "")
        self.passphrase = passphrase or os.getenv("OKX_PASSPHRASE", "")
        
        if self.api_key and self.api_secret and self.passphrase:
            try:
                import ccxt
                self.client = ccxt.okx({
                    'apiKey': self.api_key,
                    'secret': self.api_secret,
                    'password': self.passphrase,
                    'sandbox': testnet,
                })
            except ImportError:
                logger.error("ccxt not installed. Router will operate in simulation mode.")
                self.client = None
        else:
            logger.warning("OKX API keys not configured. Router will operate in simulation mode.")
            self.client = None
    
    def place_order(self, asset: str, side: str, quantity: float) -> Optional[Dict[str, Any]]:
        """
        Esegue un ordine su OKX.
        
        Args:
            asset: Simbolo dell'asset
            side: "BUY" o "SELL"
            quantity: Quantità da tradare
            
        Returns:
            Risultato dell'ordine o None
        """
        # OKX usa formato diverso (BTC-USDT)
        symbol = self.get_symbol(asset).replace("USDT", "-USDT")
        
        if self.client is None:
            # Modalità simulazione
            logger.info(f"[SIMULATION] OKX: {side} {quantity} {symbol}")
            return {"status": "SIMULATED", "symbol": symbol, "side": side, "quantity": quantity}
        
        try:
            order = self.client.create_market_order(
                symbol=symbol,
                side=side.lower(),
                amount=quantity
            )
            logger.info(f"✅ OKX order executed: {side} {quantity} {symbol}")
            return order
        except Exception as e:
            logger.error(f"❌ OKX order failed: {e}")
            return None
    
    def get_symbol(self, asset: str) -> str:
        """
        Converte il simbolo dell'asset nel formato OKX.
        
        Args:
            asset: Simbolo dell'asset (es. BTC, ETH)
            
        Returns:
            Simbolo nel formato OKX (es. BTC-USDT)
        """
        if "-" in asset:
            return asset
        if "USDT" in asset:
            return asset.replace("USDT", "-USDT")
        return f"{asset}-USDT"
