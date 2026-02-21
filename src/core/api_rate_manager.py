"""
API Rate Manager - Gestione intelligente delle chiamate API
============================================================
Gestisce i limiti di chiamata per ogni API e evita duplicazioni.

Limiti API:
- NewsAPI: 100 richieste/giorno (free tier)
- Alpha Vantage: 5 richieste/minuto, 500/giorno
- Binance: 1200 richieste/minuto
- CoinGecko: 10-50 richieste/minuto
"""

import os
import time
import json
import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from threading import Lock
import requests

logger = logging.getLogger(__name__)

# Directory per cache
CACHE_DIR = Path(__file__).parent.parent.parent / "cache" / "api"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class APILimits:
    """Limiti per una API."""
    name: str
    requests_per_minute: int
    requests_per_day: int
    min_interval_seconds: float = 0.0
    
    # Stato attuale
    minute_count: int = 0
    day_count: int = 0
    last_minute_reset: float = field(default_factory=time.time)
    last_day_reset: float = field(default_factory=time.time)
    last_request_time: float = 0.0
    
    def can_make_request(self) -> bool:
        """Verifica se possiamo fare una richiesta."""
        now = time.time()
        
        # Reset contatore minuto
        if now - self.last_minute_reset >= 60:
            self.minute_count = 0
            self.last_minute_reset = now
        
        # Reset contatore giorno
        if now - self.last_day_reset >= 86400:
            self.day_count = 0
            self.last_day_reset = now
        
        # Verifica limiti
        if self.minute_count >= self.requests_per_minute:
            return False
        if self.day_count >= self.requests_per_day:
            return False
        if self.min_interval_seconds > 0 and now - self.last_request_time < self.min_interval_seconds:
            return False
        
        return True
    
    def record_request(self):
        """Registra una richiesta effettuata."""
        self.minute_count += 1
        self.day_count += 1
        self.last_request_time = time.time()
    
    def wait_time(self) -> float:
        """Tempo da aspettare prima della prossima richiesta."""
        if self.can_make_request():
            return 0.0
        
        now = time.time()
        
        # Aspetta per il limite minuto
        if self.minute_count >= self.requests_per_minute:
            return 60 - (now - self.last_minute_reset) + 0.1
        
        # Aspetta per l'intervallo minimo
        if self.min_interval_seconds > 0:
            return max(0, self.min_interval_seconds - (now - self.last_request_time))
        
        return 0.0


class APIRateManager:
    """
    Gestore centralizzato per tutte le API.
    Implementa rate limiting, caching e deduplicazione.
    """
    
    # Limiti predefiniti per ogni API
    DEFAULT_LIMITS = {
        'newsapi': APILimits(
            name='newsapi',
            requests_per_minute=1,  # Molto conservativo
            requests_per_day=100,   # Limite free tier
            min_interval_seconds=60  # 1 richiesta al minuto max
        ),
        'alpha_vantage': APILimits(
            name='alpha_vantage',
            requests_per_minute=5,
            requests_per_day=500,
            min_interval_seconds=12  # 5 al minuto = 1 ogni 12 secondi
        ),
        'binance': APILimits(
            name='binance',
            requests_per_minute=100,  # Conservativo (limite reale 1200)
            requests_per_day=10000,
            min_interval_seconds=0.1
        ),
        'coingecko': APILimits(
            name='coingecko',
            requests_per_minute=10,  # Conservativo
            requests_per_day=5000,
            min_interval_seconds=6
        ),
        'open_meteo': APILimits(
            name='open_meteo',
            requests_per_minute=60,
            requests_per_day=10000,
            min_interval_seconds=1
        ),
    }
    
    def __init__(self, cache_ttl_seconds: int = 300):
        """
        Inizializza il gestore.
        
        Args:
            cache_ttl_seconds: Tempo di vita della cache (default 5 minuti)
        """
        self.limits = {k: APILimits(**v.__dict__) for k, v in self.DEFAULT_LIMITS.items()}
        self.cache_ttl = cache_ttl_seconds
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.news_hashes: set = set()  # Per deduplicazione notizie
        self.lock = Lock()
        
        # Carica stato precedente
        self._load_state()
    
    def _get_cache_key(self, api_name: str, endpoint: str, params: Dict = None) -> str:
        """Genera una chiave di cache."""
        params_str = json.dumps(params or {}, sort_keys=True)
        hash_input = f"{api_name}:{endpoint}:{params_str}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def _get_cache_file(self, cache_key: str) -> Path:
        """Ottiene il file di cache."""
        return CACHE_DIR / f"{cache_key}.json"
    
    def _load_cache(self, cache_key: str) -> Optional[Dict]:
        """Carica dati dalla cache."""
        cache_file = self._get_cache_file(cache_key)
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Verifica TTL
                    if time.time() - data.get('timestamp', 0) < self.cache_ttl:
                        return data.get('content')
            except Exception as e:
                logger.debug(f"Cache load error: {e}")
        return None
    
    def _save_cache(self, cache_key: str, content: Any):
        """Salva dati in cache."""
        cache_file = self._get_cache_file(cache_key)
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'timestamp': time.time(),
                    'content': content
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.debug(f"Cache save error: {e}")
    
    def _load_state(self):
        """Carica lo stato precedente (contatori, hash notizie)."""
        state_file = CACHE_DIR / "api_state.json"
        if state_file.exists():
            try:
                with open(state_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Carica hash notizie (solo ultime 24h)
                    cutoff = time.time() - 86400
                    self.news_hashes = {
                        h for h in data.get('news_hashes', [])
                        if isinstance(h, str)
                    }
                    logger.info(f"Loaded state: {len(self.news_hashes)} news hashes")
            except Exception as e:
                logger.debug(f"State load error: {e}")
    
    def _save_state(self):
        """Salva lo stato attuale."""
        state_file = CACHE_DIR / "api_state.json"
        try:
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'news_hashes': list(self.news_hashes)[-1000:],  # Ultimi 1000
                    'timestamp': time.time()
                }, f)
        except Exception as e:
            logger.debug(f"State save error: {e}")
    
    def _hash_news(self, title: str, source: str = '') -> str:
        """Genera hash per una notizia."""
        content = f"{title.lower().strip()}:{source.lower()}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def is_news_duplicate(self, title: str, source: str = '') -> bool:
        """Verifica se una notizia è duplicata."""
        news_hash = self._hash_news(title, source)
        return news_hash in self.news_hashes
    
    def mark_news_seen(self, title: str, source: str = ''):
        """Marca una notizia come vista."""
        news_hash = self._hash_news(title, source)
        self.news_hashes.add(news_hash)
    
    def filter_duplicate_news(self, news_list: List[Dict], title_key: str = 'title', source_key: str = 'source') -> List[Dict]:
        """
        Filtra notizie duplicate.
        
        Args:
            news_list: Lista di notizie
            title_key: Chiave del titolo
            source_key: Chiave della fonte
            
        Returns:
            Lista filtrata senza duplicati
        """
        unique_news = []
        for news in news_list:
            title = news.get(title_key, '')
            source = news.get(source_key, {}).get('name', '') if isinstance(news.get(source_key), dict) else news.get(source_key, '')
            
            if not self.is_news_duplicate(title, source):
                self.mark_news_seen(title, source)
                unique_news.append(news)
        
        self._save_state()
        logger.info(f"Filtered {len(news_list) - len(unique_news)} duplicate news")
        return unique_news
    
    def can_request(self, api_name: str) -> bool:
        """Verifica se possiamo fare una richiesta."""
        if api_name not in self.limits:
            return True
        return self.limits[api_name].can_make_request()
    
    def wait_if_needed(self, api_name: str) -> float:
        """Aspetta se necessario, ritorna il tempo aspettato."""
        if api_name not in self.limits:
            return 0.0
        
        wait_time = self.limits[api_name].wait_time()
        if wait_time > 0:
            logger.info(f"Waiting {wait_time:.1f}s for {api_name} rate limit")
            time.sleep(wait_time)
        return wait_time
    
    def request(self, api_name: str, url: str, params: Dict = None, 
                method: str = 'GET', timeout: int = 15,
                use_cache: bool = True) -> Optional[Dict]:
        """
        Esegue una richiesta con rate limiting e caching.
        
        Args:
            api_name: Nome dell'API
            url: URL della richiesta
            params: Parametri
            method: Metodo HTTP
            timeout: Timeout in secondi
            use_cache: Se usare la cache
            
        Returns:
            Risposta JSON o None se errore
        """
        # Controlla cache
        cache_key = self._get_cache_key(api_name, url, params)
        if use_cache:
            cached = self._load_cache(cache_key)
            if cached is not None:
                logger.debug(f"Cache hit for {api_name}")
                return cached
        
        # Rate limiting
        with self.lock:
            if api_name in self.limits:
                self.wait_if_needed(api_name)
                self.limits[api_name].record_request()
        
        # Esegui richiesta
        try:
            if method.upper() == 'GET':
                r = requests.get(url, params=params, timeout=timeout)
            else:
                r = requests.post(url, json=params, timeout=timeout)
            
            if r.status_code == 200:
                data = r.json()
                if use_cache:
                    self._save_cache(cache_key, data)
                return data
            elif r.status_code == 429:
                logger.warning(f"Rate limit hit for {api_name}")
                # Ritorna cache anche se scaduta
                return self._load_cache(cache_key)
            else:
                logger.warning(f"API error {api_name}: HTTP {r.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Request error {api_name}: {e}")
            return None
    
    def get_status(self) -> Dict[str, Dict]:
        """Ottiene lo stato di tutte le API."""
        status = {}
        for name, limits in self.limits.items():
            status[name] = {
                'minute_usage': f"{limits.minute_count}/{limits.requests_per_minute}",
                'day_usage': f"{limits.day_count}/{limits.requests_per_day}",
                'can_request': limits.can_make_request(),
                'wait_time': limits.wait_time()
            }
        return status
    
    def reset_daily(self):
        """Reset dei contatori giornalieri."""
        for limits in self.limits.values():
            limits.day_count = 0
            limits.last_day_reset = time.time()
        logger.info("Daily API counters reset")


# Istanza globale
_api_manager: Optional[APIRateManager] = None


def get_api_manager() -> APIRateManager:
    """Ottiene l'istanza globale del gestore API."""
    global _api_manager
    if _api_manager is None:
        _api_manager = APIRateManager()
    return _api_manager


# Funzioni di convenienza
def api_request(api_name: str, url: str, params: Dict = None, 
                use_cache: bool = True) -> Optional[Dict]:
    """Funzione di convenienza per richieste API."""
    return get_api_manager().request(api_name, url, params, use_cache=use_cache)


def filter_duplicate_news(news_list: List[Dict]) -> List[Dict]:
    """Funzione di convenienza per filtrare notizie duplicate."""
    return get_api_manager().filter_duplicate_news(news_list)


if __name__ == "__main__":
    # Test
    manager = APIRateManager()
    
    print("=== API Rate Manager Test ===")
    print()
    
    # Test stato
    print("API Status:")
    for api, status in manager.get_status().items():
        print(f"  {api}: {status}")
    
    print()
    
    # Test deduplicazione notizie
    print("News deduplication test:")
    test_news = [
        {'title': 'Bitcoin hits new high', 'source': {'name': 'CoinDesk'}},
        {'title': 'Bitcoin hits new high', 'source': {'name': 'CoinDesk'}},  # Duplicato
        {'title': 'Ethereum upgrade announced', 'source': {'name': 'CoinTelegraph'}},
        {'title': 'Bitcoin hits new high', 'source': {'name': 'Reuters'}},  # Fonte diversa
    ]
    
    filtered = manager.filter_duplicate_news(test_news)
    print(f"  Original: {len(test_news)} news")
    print(f"  Filtered: {len(filtered)} unique news")
    for n in filtered:
        print(f"    - {n['title']}")
