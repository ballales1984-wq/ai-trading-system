"""
Logical Portfolio Module
========================
Modulo per la gestione logica del portafoglio multi-asset.
Contiene classi per la gestione news, portafoglio e analisi logica.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class NewsItem:
    """Rappresenta una notizia con titolo e fonte."""
    title: str
    source: str
    timestamp: Optional[datetime] = None
    sentiment: Optional[float] = None
    asset: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class Portfolio:
    """
    Gestisce un portafoglio multi-asset con bilanci in diverse criptovalute.
    """
    
    def __init__(self, balances: Optional[Dict[str, float]] = None):
        """
        Inizializza il portafoglio con i bilanci forniti.
        
        Args:
            balances: Dizionario {asset: quantità} es. {"BTC": 1.5, "ETH": 10.0}
        """
        self.balances = balances or {}
        self._prices: Dict[str, float] = {}
        self._trade_history: List[Dict] = []
    
    def total_value(self) -> float:
        """
        Calcola il valore totale del portafoglio in USDT.
        
        Returns:
            Valore totale stimato in USDT
        """
        total = 0.0
        for asset, qty in self.balances.items():
            if asset == "USDT":
                total += qty
            else:
                # Usa il prezzo se disponibile, altrimenti stima
                price = self._prices.get(asset, 0)
                total += qty * price
        return total
    
    def get_balance(self, asset: str) -> float:
        """Ritorna il saldo di un asset specifico."""
        return self.balances.get(asset, 0.0)
    
    def update_balance(self, asset: str, quantity: float):
        """Aggiorna il saldo di un asset."""
        if asset in self.balances:
            self.balances[asset] += quantity
        else:
            self.balances[asset] = quantity
        
        # Registra nel trade history
        self._trade_history.append({
            "timestamp": datetime.now(),
            "asset": asset,
            "quantity": quantity,
            "new_balance": self.balances[asset]
        })
    
    def set_price(self, asset: str, price: float):
        """Imposta il prezzo di un asset per il calcolo del valore."""
        self._prices[asset] = price
    
    def can_execute(self, asset: str, confidence: float, max_allocation: float) -> bool:
        """
        Determina se un'operazione può essere eseguita.
        
        Args:
            asset: Asset da tradare
            confidence: Livello di confidenza del segnale (0-1)
            max_allocation: Allocazione massima consentita in USDT
            
        Returns:
            True se l'operazione può essere eseguita
        """
        # Verifica che ci sia abbastanza capitale
        total = self.total_value()
        if total <= 0:
            return False
        
        # Verifica confidenza minima
        if confidence < 0.3:
            logger.info(f"Confidenza troppo bassa: {confidence}")
            return False
        
        # Verifica che l'allocazione non superi il massimo consentito
        if max_allocation > total * 0.5:  # Max 50% del portafoglio
            logger.warning(f"Allocazione richiesta troppo alta: {max_allocation}")
            return False
        
        return True
    
    def log_portfolio(self):
        """Registra lo stato attuale del portafoglio."""
        logger.info("=" * 50)
        logger.info("PORTFOLIO STATUS")
        logger.info("=" * 50)
        for asset, qty in self.balances.items():
            price = self._prices.get(asset, "N/A")
            value = qty * price if isinstance(price, (int, float)) else "N/A"
            logger.info(f"  {asset}: {qty} @ {price} = {value} USDT")
        logger.info(f"  TOTAL VALUE: {self.total_value()} USDT")
        logger.info("=" * 50)
    
    def get_trade_history(self) -> List[Dict]:
        """Ritorna lo storico delle operazioni."""
        return self._trade_history.copy()


class LogicalPortfolioEngine:
    """
    Engine per l'analisi logica delle news e generazione di segnali di trading.
    """
    
    # Mappatura parole chiave -> asset
    ASSET_KEYWORDS = {
        "bitcoin": "BTC",
        "btc": "BTC",
        "ethereum": "ETH",
        "eth": "ETH",
        "solana": "SOL",
        "sol": "SOL",
        "cardano": "ADA",
        "ada": "ADA",
        "ripple": "XRP",
        "xrp": "XRP",
        "dogecoin": "DOGE",
        "doge": "DOGE",
        "polkadot": "DOT",
        "dot": "DOT",
        "avalanche": "AVAX",
        "avax": "AVAX",
    }
    
    # Parole chiave sentiment positivo
    BULLISH_KEYWORDS = [
        "surge", "surges", "rally", "bullish", "gain", "gains", "rise", "rises",
        "high", "record", "breakout", "upgrade", "adoption", "launch", "approve",
        "approval", "etf", "inflow", "inflows", "boost", "positive", "growth"
    ]
    
    # Parole chiave sentiment negativo
    BEARISH_KEYWORDS = [
        "crash", "drop", "drops", "bearish", "fall", "falls", "decline", "sell",
        "selling", "ban", "regulation", "hack", "exploit", "fraud", "lawsuit",
        "sec", "lawsuit", "concern", "fear", "outflow", "outflows"
    ]
    
    def __init__(self, portfolio: Portfolio):
        """
        Inizializza l'engine con un portafoglio.
        
        Args:
            portfolio: Istanza del portafoglio da gestire
        """
        self.portfolio = portfolio
    
    def _extract_asset(self, title: str) -> str:
        """
        Estrae l'asset dalla notizia.
        
        Args:
            title: Titolo della notizia
            
        Returns:
            Simbolo dell'asset o "UNKNOWN"
        """
        title_lower = title.lower()
        for keyword, asset in self.ASSET_KEYWORDS.items():
            if keyword in title_lower:
                return asset
        return "UNKNOWN"
    
    def _calculate_sentiment(self, title: str) -> float:
        """
        Calcola il sentiment della notizia.
        
        Args:
            title: Titolo della notizia
            
        Returns:
            Score di sentiment da -1 (bearish) a 1 (bullish)
        """
        title_lower = title.lower()
        bullish_count = sum(1 for kw in self.BULLISH_KEYWORDS if kw in title_lower)
        bearish_count = sum(1 for kw in self.BEARISH_KEYWORDS if kw in title_lower)
        
        total = bullish_count + bearish_count
        if total == 0:
            return 0.0
        
        return (bullish_count - bearish_count) / total
    
    def _determine_signal(self, sentiment: float) -> str:
        """
        Determina il segnale di trading basato sul sentiment.
        
        Args:
            sentiment: Score di sentiment
            
        Returns:
            "BUY", "SELL", o "HOLD"
        """
        if sentiment > 0.3:
            return "BUY"
        elif sentiment < -0.3:
            return "SELL"
        else:
            return "HOLD"
    
    def _calculate_confidence(self, sentiment: float, source: str) -> float:
        """
        Calcola la confidenza del segnale.
        
        Args:
            sentiment: Score di sentiment
            source: Fonte della notizia
            
        Returns:
            Livello di confidenza da 0 a 1
        """
        # Base confidence dal sentiment
        base_confidence = abs(sentiment)
        
        # Bonus per fonti affidabili
        reliable_sources = ["reuters", "bloomberg", "coindesk", "cointelegraph", "the block"]
        source_bonus = 0.1 if source.lower() in reliable_sources else 0
        
        return min(1.0, base_confidence + source_bonus)
    
    def analyze_news(self, news_feed: List[NewsItem]) -> List[Dict]:
        """
        Analizza un feed di notizie e genera segnali di trading.
        
        Args:
            news_feed: Lista di NewsItem da analizzare
            
        Returns:
            Lista di dizionari con segnali
        """
        signals = []
        
        for news in news_feed:
            # Estrai asset
            asset = news.asset or self._extract_asset(news.title)
            
            # Calcola sentiment
            sentiment = news.sentiment if news.sentiment is not None else self._calculate_sentiment(news.title)
            
            # Determina segnale
            signal = self._determine_signal(sentiment)
            
            # Calcola confidenza
            confidence = self._calculate_confidence(sentiment, news.source)
            
            signals.append({
                "asset": asset,
                "signal": signal,
                "sentiment": round(sentiment, 2),
                "confidence": round(confidence, 2),
                "source": news.source,
                "title": news.title
            })
            
            logger.info(f"News analyzed: {news.title[:50]}... -> {asset} {signal} (conf: {confidence:.2f})")
        
        return signals


# === Esempio di utilizzo ===
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Crea portafoglio
    portfolio = Portfolio(balances={"BTC": 1.0, "ETH": 10.0, "USDT": 50000})
    portfolio.set_price("BTC", 95000)
    portfolio.set_price("ETH", 3500)
    
    # Crea engine
    engine = LogicalPortfolioEngine(portfolio)
    
    # Analizza news
    news_feed = [
        NewsItem("Bitcoin Surges Past $95K on ETF Inflows", "CoinDesk"),
        NewsItem("Ethereum Upgrade Boosts Network Activity", "CoinTelegraph"),
        NewsItem("Solana DeFi TVL Reaches New High", "The Block"),
        NewsItem("Fed Signals Potential Rate Cut in March", "Reuters"),
    ]
    
    signals = engine.analyze_news(news_feed)
    for s in signals:
        print(s)
    
    portfolio.log_portfolio()
