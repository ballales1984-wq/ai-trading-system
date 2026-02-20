#!/usr/bin/env python3
"""
Mont Blanck Strategy Module
===========================

Strategia di trading basata su previsione di picchi mediante regressione polinomiale.
Supporta multi-asset, paper trading e live trading.

Features:
- Previsione picchi di prezzo con regressione polinomiale
- Segnali BUY/SELL basati su threshold configurabili
- Integrazione con TradingLedger per tracking performance
- Supporto multi-asset e multi-exchange
"""

import numpy as np
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger("MontBlanck")


class Signal(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class MontBlanckConfig:
    """Configurazione della strategia Mont Blanck."""
    window_size: int = 4          # Finestra di osservazione
    future_steps: int = 3         # Step di previsione
    poly_degree: int = 3          # Grado del polinomio
    buy_threshold: float = 0.01   # Soglia per BUY (1%)
    sell_threshold: float = -0.01 # Soglia per SELL (-1%)
    min_confidence: float = 0.6   # Confidenza minima per segnale


@dataclass
class Prediction:
    """Risultato della previsione."""
    current_price: float
    predicted_peak: float
    predicted_change: float
    signal: Signal
    confidence: float
    trend: str  # "UP", "DOWN", "SIDEWAYS"


class MontBlanck:
    """
    Strategia Mont Blanck per previsione picchi di prezzo.
    
    Utilizza regressione polinomiale per stimare la direzione del prezzo
    e generare segnali di trading.
    """
    
    def __init__(self, 
                 window_size: int = 4,
                 future_steps: int = 3,
                 poly_degree: int = 3,
                 buy_threshold: float = 0.01,
                 sell_threshold: float = -0.01,
                 min_confidence: float = 0.6):
        """
        Inizializza la strategia.
        
        Args:
            window_size: Numero di punti da considerare per la regressione
            future_steps: Step futuri da prevedere
            poly_degree: Grado del polinomio per la regressione
            buy_threshold: Soglia percentuale per generare BUY
            sell_threshold: Soglia percentuale per generare SELL
            min_confidence: Confidenza minima per validare un segnale
        """
        self.window_size = window_size
        self.future_steps = future_steps
        self.poly_degree = poly_degree
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.min_confidence = min_confidence
        
        # Stato interno
        self.price_history: List[float] = []
        self.last_prediction: Optional[Prediction] = None
        self.coefficients: Optional[np.ndarray] = None
        
        logger.info(f"MontBlanck initialized: window={window_size}, degree={poly_degree}, "
                   f"buy_threshold={buy_threshold:.2%}, sell_threshold={sell_threshold:.2%}")
    
    def _fit_polynomial(self, prices: List[float]) -> np.ndarray:
        """
        Calcola i coefficienti del polinomio di regressione.
        
        Args:
            prices: Lista di prezzi
            
        Returns:
            Coefficienti del polinomio
        """
        x = np.arange(len(prices))
        y = np.array(prices)
        
        # Regressione polinomiale
        coefficients = np.polyfit(x, y, self.poly_degree)
        return coefficients
    
    def _predict_future(self, coefficients: np.ndarray, current_idx: int) -> float:
        """
        Prevede il prezzo futuro usando il polinomio.
        
        Args:
            coefficients: Coefficienti del polinomio
            current_idx: Indice corrente
            
        Returns:
            Prezzo previsto
        """
        future_idx = current_idx + self.future_steps
        poly = np.poly1d(coefficients)
        return poly(future_idx)
    
    def _calculate_confidence(self, prices: List[float], coefficients: np.ndarray) -> float:
        """
        Calcola la confidenza della previsione basata su R².
        
        Args:
            prices: Prezzi osservati
            coefficients: Coefficienti del polinomio
            
        Returns:
            Confidenza (0-1)
        """
        x = np.arange(len(prices))
        y = np.array(prices)
        poly = np.poly1d(coefficients)
        y_pred = poly(x)
        
        # R² score
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        if ss_tot == 0:
            return 0.0
        
        r_squared = 1 - (ss_res / ss_tot)
        
        # Normalizza tra 0 e 1
        confidence = max(0, min(1, r_squared))
        return confidence
    
    def _determine_trend(self, coefficients: np.ndarray) -> str:
        """
        Determina il trend basato sulla derivata del polinomio.
        
        Args:
            coefficients: Coefficienti del polinomio
            
        Returns:
            "UP", "DOWN", o "SIDEWAYS"
        """
        # Derivata del polinomio
        deriv_coeffs = np.polyder(coefficients)
        deriv = np.poly1d(deriv_coeffs)
        
        # Valuta la derivata all'ultimo punto
        last_idx = self.window_size - 1
        slope = deriv(last_idx)
        
        if slope > 0.01:
            return "UP"
        elif slope < -0.01:
            return "DOWN"
        else:
            return "SIDEWAYS"
    
    def predict(self, prices: List[float]) -> Prediction:
        """
        Genera una previsione basata sui prezzi forniti.
        
        Args:
            prices: Lista di prezzi (almeno window_size elementi)
            
        Returns:
            Prediction con segnale e confidenza
        """
        if len(prices) < self.window_size:
            return Prediction(
                current_price=prices[-1] if prices else 0,
                predicted_peak=0,
                predicted_change=0,
                signal=Signal.HOLD,
                confidence=0,
                trend="SIDEWAYS"
            )
        
        # Prendi gli ultimi window_size prezzi
        window = prices[-self.window_size:]
        current_price = window[-1]
        
        # Calcola regressione polinomiale
        self.coefficients = self._fit_polynomial(window)
        
        # Prevedi il picco futuro
        predicted_peak = self._predict_future(self.coefficients, len(window) - 1)
        
        # Calcola il cambiamento percentuale previsto
        predicted_change = (predicted_peak - current_price) / current_price
        
        # Calcola confidenza
        confidence = self._calculate_confidence(window, self.coefficients)
        
        # Determina trend
        trend = self._determine_trend(self.coefficients)
        
        # Genera segnale
        signal = Signal.HOLD
        if confidence >= self.min_confidence:
            if predicted_change >= self.buy_threshold:
                signal = Signal.BUY
            elif predicted_change <= self.sell_threshold:
                signal = Signal.SELL
        
        prediction = Prediction(
            current_price=current_price,
            predicted_peak=predicted_peak,
            predicted_change=predicted_change,
            signal=signal,
            confidence=confidence,
            trend=trend
        )
        
        self.last_prediction = prediction
        self.price_history = prices
        
        return prediction
    
    def genera_segnale(self, prices: List[float], last_trade: Optional[Dict] = None) -> str:
        """
        Genera un segnale di trading (compatibilità con versione precedente).
        
        Args:
            prices: Lista di prezzi
            last_trade: Ultimo trade eseguito (opzionale)
            
        Returns:
            "BUY", "SELL", o "HOLD"
        """
        prediction = self.predict(prices)
        
        # Se abbiamo già una posizione BUY, controlla se vendere
        if last_trade is not None and last_trade.get("tipo") == "BUY":
            if prediction.signal == Signal.SELL:
                return "SELL"
            return "HOLD"
        
        # Se non abbiamo posizione, controlla se comprare
        if prediction.signal == Signal.BUY:
            return "BUY"
        
        return "HOLD"
    
    def prevedi_picco(self, prices: List[float]) -> float:
        """
        Prevede il picco di prezzo (compatibilità con versione precedente).
        
        Args:
            prices: Lista di prezzi
            
        Returns:
            Prezzo previsto
        """
        prediction = self.predict(prices)
        return prediction.predicted_peak
    
    def get_analysis(self) -> Dict:
        """
        Restituisce un'analisi dettagliata dell'ultima previsione.
        
        Returns:
            Dizionario con analisi completa
        """
        if self.last_prediction is None:
            return {"error": "Nessuna previsione disponibile"}
        
        p = self.last_prediction
        return {
            "current_price": p.current_price,
            "predicted_peak": p.predicted_peak,
            "predicted_change_pct": p.predicted_change * 100,
            "signal": p.signal.value,
            "confidence": p.confidence,
            "trend": p.trend,
            "should_act": p.confidence >= self.min_confidence and p.signal != Signal.HOLD
        }


class MontBlanckMultiAsset:
    """
    Gestore multi-asset per la strategia Mont Blanck.
    """
    
    def __init__(self, config: MontBlanckConfig = None):
        """
        Inizializza il gestore multi-asset.
        
        Args:
            config: Configurazione condivisa (opzionale)
        """
        self.config = config or MontBlanckConfig()
        self.strategies: Dict[str, MontBlanck] = {}
        self.predictions: Dict[str, Prediction] = {}
    
    def add_asset(self, symbol: str, custom_config: MontBlanckConfig = None):
        """
        Aggiunge un asset da monitorare.
        
        Args:
            symbol: Simbolo dell'asset
            custom_config: Configurazione personalizzata (opzionale)
        """
        cfg = custom_config or self.config
        self.strategies[symbol] = MontBlanck(
            window_size=cfg.window_size,
            future_steps=cfg.future_steps,
            poly_degree=cfg.poly_degree,
            buy_threshold=cfg.buy_threshold,
            sell_threshold=cfg.sell_threshold,
            min_confidence=cfg.min_confidence
        )
        logger.info(f"Added asset: {symbol}")
    
    def update(self, symbol: str, prices: List[float]) -> Prediction:
        """
        Aggiorna la previsione per un asset.
        
        Args:
            symbol: Simbolo dell'asset
            prices: Lista di prezzi
            
        Returns:
            Previsione aggiornata
        """
        if symbol not in self.strategies:
            self.add_asset(symbol)
        
        prediction = self.strategies[symbol].predict(prices)
        self.predictions[symbol] = prediction
        return prediction
    
    def get_signals(self) -> Dict[str, str]:
        """
        Restituisce tutti i segnali attivi.
        
        Returns:
            Dizionario symbol -> signal
        """
        return {
            symbol: pred.signal.value 
            for symbol, pred in self.predictions.items()
            if pred.signal != Signal.HOLD
        }
    
    def get_best_opportunities(self, top_n: int = 3) -> List[Tuple[str, Prediction]]:
        """
        Restituisce le migliori opportunità di trading.
        
        Args:
            top_n: Numero massimo di opportunità
            
        Returns:
            Lista di (symbol, prediction) ordinati per confidenza
        """
        opportunities = [
            (symbol, pred) 
            for symbol, pred in self.predictions.items()
            if pred.signal != Signal.HOLD and pred.confidence >= self.config.min_confidence
        ]
        
        # Ordina per confidenza decrescente
        opportunities.sort(key=lambda x: x[1].confidence, reverse=True)
        
        return opportunities[:top_n]


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Mont Blanck Strategy")
    parser.add_argument("--test", "-t", action="store_true", help="Run test with sample data")
    args = parser.parse_args()
    
    if args.test:
        print("=" * 60)
        print("  MONT BLANCK STRATEGY - TEST")
        print("=" * 60)
        
        # Dati di test
        test_prices = [
            100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
            110, 109, 108, 107, 106, 105, 104, 103, 102, 101
        ]
        
        strategy = MontBlanck(window_size=5, poly_degree=2)
        
        print("\nTest con prezzi di esempio:")
        print(f"Prezzi: {test_prices}")
        
        for i in range(5, len(test_prices)):
            window = test_prices[:i+1]
            pred = strategy.predict(window)
            
            print(f"\n[{i}] Prezzo: {pred.current_price:.2f} | "
                  f"Picco previsto: {pred.predicted_peak:.2f} | "
                  f"Variazione: {pred.predicted_change*100:.2f}% | "
                  f"Segnale: {pred.signal.value} | "
                  f"Confidenza: {pred.confidence:.2f}")
        
        print("\n" + "=" * 60)
